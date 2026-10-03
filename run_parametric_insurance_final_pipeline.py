#!/usr/bin/env python3
import argparse, csv, glob, json, math, os, re
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261003
EARTH_KM = 6371.0088
PRIMARY_RADIUS_KM = 175
RADII_KM = (100, 175, 250)
SEVERITY_THRESHOLDS = (50_000, 100_000, 250_000, 500_000)
FIXED_WIND_TRIGGER_KPH = 119.0  # ~= 64 kt


def decode_char_array(arr):
    return b''.join(np.asarray(arr).tolist()).decode('utf-8', 'ignore').strip('\x00 ').strip()


def load_ibtracs_index(path):
    """Return SID -> storm row index without loading large arrays."""
    out = {}
    with h5py.File(path, 'r') as f:
        sid = f['sid']
        for i in range(sid.shape[0]):
            out[decode_char_array(sid[i])] = i
    return out


def read_track(path, sid_index, sid, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    idx = sid_index[sid]
    rows = []
    with h5py.File(path, 'r') as f:
        for j in range(f['iso_time'].shape[1]):
            ts = decode_char_array(f['iso_time'][idx, j])
            if not ts:
                continue
            t = pd.Timestamp(ts)
            if start <= t <= end:
                lat = float(f['lat'][idx, j]); lon = float(f['lon'][idx, j])
                if lat <= -90 or lat >= 90 or lon <= -360 or lon >= 360:
                    continue
                rows.append((t, lat, lon))
    if not rows:
        raise ValueError(f'No IBTrACS track points for {sid} in {start}..{end}')
    return pd.DataFrame(rows, columns=['time','lat','lon'])


def discover_chirps(root):
    """Find monthly CHIRPS NetCDF files and choose one deterministic path per YYYY.MM."""
    files = glob.glob(os.path.join(root, '**', 'chirps-v2.0.*.days_p05*.nc'), recursive=True)
    patt = re.compile(r'chirps-v2\.0\.(\d{4})\.(\d{2})\.days_p05')
    groups = {}
    for p in files:
        m = patt.search(os.path.basename(p))
        if not m:
            continue
        key = f'{m.group(1)}.{m.group(2)}'
        groups.setdefault(key, []).append(p)
    chosen = {}
    for key, paths in groups.items():
        # Prefer direct conversation uploads in /mnt/data over nested working copies;
        # then shortest path, then lexicographic for deterministic selection.
        paths = sorted(paths, key=lambda p: (0 if Path(p).parent == Path(root) else 1, len(p), p))
        chosen[key] = paths[0]
    return chosen


def haversine_min_grid(lat_vals, lon_vals, track_lat, track_lon):
    lat2, lon2 = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    min_d = np.full(lat2.shape, np.inf, dtype=np.float64)
    lat2r = np.radians(lat2)
    for la, lo in zip(track_lat, track_lon):
        dlat = lat2r - math.radians(float(la))
        dlon = np.radians(lon2 - float(lo))
        a = np.sin(dlat/2.0)**2 + np.cos(lat2r)*math.cos(math.radians(float(la))) * np.sin(dlon/2.0)**2
        d = EARTH_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        min_d = np.minimum(min_d, d)
    return min_d


def month_keys(start, end):
    cur = pd.Timestamp(start).normalize().replace(day=1)
    endm = pd.Timestamp(end).normalize().replace(day=1)
    out=[]
    while cur <= endm:
        out.append(cur.strftime('%Y.%m'))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def chirps_time_dates(f):
    units = f['time'].attrs.get('units', b'days since 1980-1-1 0:0:0')
    if isinstance(units, bytes): units = units.decode()
    m = re.search(r'days since\s+(\d{4})-(\d{1,2})-(\d{1,2})', str(units))
    if not m:
        raise ValueError(f'Unsupported CHIRPS time units: {units}')
    base = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return [base + pd.Timedelta(days=float(x)) for x in f['time'][:]]


def compute_event_rain_all_radii(track, start, end, monthly_files, radii_km=RADII_KM):
    event_dates = list(pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D'))
    track_lat = track['lat'].to_numpy(float); track_lon = track['lon'].to_numpy(float)
    max_radius = max(radii_km)
    lat_pad = max_radius / 111.0 + 0.2
    mid_lat = float(np.mean(track_lat))
    lon_pad = max_radius / (111.0 * max(0.20, math.cos(math.radians(mid_lat)))) + 0.2
    lat_min, lat_max = track_lat.min()-lat_pad, track_lat.max()+lat_pad
    lon_min, lon_max = track_lon.min()-lon_pad, track_lon.max()+lon_pad

    first_key = month_keys(start, end)[0]
    if first_key not in monthly_files:
        raise FileNotFoundError(f'Missing CHIRPS month {first_key}')
    with h5py.File(monthly_files[first_key], 'r') as f0:
        lats_all = f0['latitude'][:]; lons_all = f0['longitude'][:]
    yi = np.where((lats_all >= lat_min) & (lats_all <= lat_max))[0]
    xi = np.where((lons_all >= lon_min) & (lons_all <= lon_max))[0]
    if len(yi)==0 or len(xi)==0:
        raise ValueError('Empty CHIRPS crop')
    lats = lats_all[yi]; lons = lons_all[xi]
    min_dist = haversine_min_grid(lats, lons, track_lat, track_lon)
    masks = {rad: (min_dist <= rad) for rad in radii_km}
    cum = {rad: np.zeros(min_dist.shape, dtype=np.float64) for rad in radii_km}
    valid_any = {rad: np.zeros(min_dist.shape, dtype=bool) for rad in radii_km}
    daily_means = {rad: [] for rad in radii_km}
    daily_maxes = {rad: [] for rad in radii_km}
    used_dates=[]
    event_date_set = {d.normalize() for d in event_dates}

    for key in month_keys(start, end):
        if key not in monthly_files:
            raise FileNotFoundError(f'Missing CHIRPS month {key}')
        with h5py.File(monthly_files[key], 'r') as f:
            dates = chirps_time_dates(f)
            for ti, dt in enumerate(dates):
                if dt.normalize() not in event_date_set:
                    continue
                arr = f['precip'][ti, yi[0]:yi[-1]+1, xi[0]:xi[-1]+1].astype(np.float64)
                arr[arr <= -9990] = np.nan
                any_used=False
                for rad in radii_km:
                    mask=masks[rad]
                    vals = np.where(mask, arr, np.nan)
                    if np.all(np.isnan(vals)):
                        continue
                    daily_means[rad].append(float(np.nanmean(vals)))
                    daily_maxes[rad].append(float(np.nanmax(vals)))
                    cum[rad] += np.nan_to_num(vals, nan=0.0)
                    valid_any[rad] |= mask & np.isfinite(arr)
                    any_used=True
                if any_used:
                    used_dates.append(dt.normalize())

    if not used_dates:
        raise ValueError('No CHIRPS days used')
    out={}
    for rad in radii_km:
        vals=cum[rad][valid_any[rad]]
        out[rad]={
            'days_requested': len(event_dates),'days_used': len(set(used_dates)),
            'grid_cells_valid': int(valid_any[rad].sum()),
            'rain_cum_mean_mm': float(np.mean(vals)),'rain_cum_median_mm': float(np.median(vals)),
            'rain_cum_p90_mm': float(np.quantile(vals, 0.90)),'rain_cum_p95_mm': float(np.quantile(vals, 0.95)),
            'rain_cum_max_mm': float(np.max(vals)),'rain_daily_mean_peak_mm': float(np.max(daily_means[rad])),
            'rain_daily_cell_max_mm': float(np.max(daily_maxes[rad])),
            'pct_cells_cum_gt100': float(100*np.mean(vals > 100.0)),
            'pct_cells_cum_gt200': float(100*np.mean(vals > 200.0)),
        }
    return out


def choose_wind_threshold(train_wind, train_y):
    x = np.asarray(train_wind, float); y=np.asarray(train_y, int)
    ux=np.unique(x)
    candidates=np.r_[ux[0]-1e-6, (ux[:-1]+ux[1:])/2.0, ux[-1]+1e-6]
    best=None
    for k in candidates:
        p=(x>=k).astype(int)
        mismatch=np.mean(p!=y)
        bal=balanced_accuracy_score(y,p) if len(np.unique(y))>1 else accuracy_score(y,p)
        # deterministic tie-break: lower mismatch, higher balanced accuracy, then threshold nearest median wind
        score=(mismatch,-bal,abs(k-np.median(x)),k)
        if best is None or score<best[0]: best=(score,k)
    return float(best[1])


def model_predictions(df, threshold, rain_col):
    d=df.copy().reset_index(drop=True)
    y=(d['emdat_total_affected'].astype(float)>=threshold).astype(int).to_numpy()
    features=['max_wind_local_kph','min_pressure_local_mb','min_dist_land_local_km',rain_col]
    X=d[features].astype(float).to_numpy()
    n=len(d)
    preds={
        'M0_fixed_wind_119': np.zeros(n,dtype=int),
        'M1_calibrated_wind': np.zeros(n,dtype=int),
        'M2_logistic_multihazard': np.zeros(n,dtype=int),
        'M3_random_forest_multihazard': np.zeros(n,dtype=int),
    }
    probs={'M2_logistic_multihazard':np.full(n,np.nan),'M3_random_forest_multihazard':np.full(n,np.nan)}
    ks=[]
    for i in range(n):
        tr=np.arange(n)!=i
        preds['M0_fixed_wind_119'][i]=int(d.loc[i,'max_wind_local_kph']>=FIXED_WIND_TRIGGER_KPH)
        k=choose_wind_threshold(d.loc[tr,'max_wind_local_kph'].to_numpy(),y[tr]); ks.append(k)
        preds['M1_calibrated_wind'][i]=int(d.loc[i,'max_wind_local_kph']>=k)
        # If a training fold has only one class, use its constant class.
        if len(np.unique(y[tr]))<2:
            for name in ['M2_logistic_multihazard','M3_random_forest_multihazard']:
                preds[name][i]=int(y[tr][0]); probs[name][i]=float(y[tr][0])
            continue
        lr=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',random_state=SEED,max_iter=5000))])
        rf=RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=-1)
        for name,model in [('M2_logistic_multihazard',lr),('M3_random_forest_multihazard',rf)]:
            model.fit(X[tr],y[tr])
            preds[name][i]=int(model.predict(X[[i]])[0])
            probs[name][i]=float(model.predict_proba(X[[i]])[0,1])
    return y,preds,probs,ks


def metric_row(y,p, model, threshold, rain_radius):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {
        'impact_threshold_people':int(threshold),'model':model,'rain_corridor_km':rain_radius,
        'n_events':len(y),'n_severe':int(y.sum()),'n_nonsevere':int((1-y).sum()),
        'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),
        'precision_severe':precision_score(y,p,zero_division=0),'recall_severe':recall_score(y,p,zero_division=0),
        'f1_severe':f1_score(y,p,zero_division=0),'basis_risk_mismatch':float(np.mean(y!=p)),
        'false_negative_rate_severe':fn/(fn+tp) if (fn+tp) else np.nan,
        'false_positive_rate_nonsevere':fp/(fp+tn) if (fp+tn) else np.nan,
        'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)
    }


def wilson_ci(k,n,z=1.959963984540054):
    if n==0: return (np.nan,np.nan)
    ph=k/n; den=1+z*z/n
    center=(ph+z*z/(2*n))/den
    half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)


def exact_mcnemar_p(a_correct,b_correct):
    b=int(np.sum(a_correct & ~b_correct)); c=int(np.sum(~a_correct & b_correct)); n=b+c
    if n==0:return 1.0,b,c
    # two-sided exact binomial with p=.5
    from math import comb
    k=min(b,c)
    p=2*sum(comb(n,i)*(0.5**n) for i in range(k+1))
    return min(1.0,p),b,c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master',default='/mnt/data/integrated_event_master_final.csv')
    ap.add_argument('--ibtracs',default='/mnt/data/parametric/IBTrACS.since1980.v04r01.nc')
    ap.add_argument('--chirps-root',default='/mnt/data')
    ap.add_argument('--out-dir',default='/mnt/data/final_reproducible_results')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    master=pd.read_csv(args.master)
    sid_index=load_ibtracs_index(args.ibtracs)
    chirps=discover_chirps(args.chirps_root)

    # Verify all required months exist.
    required=set()
    for _,r in master.iterrows(): required.update(month_keys(r['local_start'],r['local_end']))
    missing=sorted(required-set(chirps))
    if missing: raise FileNotFoundError('Missing CHIRPS months: '+', '.join(missing))

    rain_records=[]
    for _,r in master.iterrows():
        track=read_track(args.ibtracs,sid_index,r['sid'],r['local_start'],r['local_end'])
        rec={'storm_key':r['storm_key'],'year':int(r['year']),'sid':r['sid'],'local_start':r['local_start'],'local_end':r['local_end'],'track_points':len(track)}
        all_rain=compute_event_rain_all_radii(track,r['local_start'],r['local_end'],chirps,RADII_KM)
        for rad,z in all_rain.items():
            for k,v in z.items(): rec[f'{k}_{rad}km']=v
        rain_records.append(rec)
        print(f"processed {r['storm_key']}")
    rain=pd.DataFrame(rain_records)
    rain.to_csv(out/'chirps_metrics_uniform.csv',index=False)

    # Merge uniform rainfall back to master.
    final=master.drop(columns=[c for c in master.columns if c.startswith('rain_mean_unified_')],errors='ignore').merge(rain,on=['storm_key','year','sid','local_start','local_end'],how='left')
    # Canonical 175 km aliases used in models and tables.
    alias_keys=['rain_cum_mean_mm','rain_cum_median_mm','rain_cum_p90_mm','rain_cum_p95_mm','rain_cum_max_mm','rain_daily_mean_peak_mm','rain_daily_cell_max_mm','pct_cells_cum_gt100','pct_cells_cum_gt200']
    for k in alias_keys:
        final[f'{k}_uniform']=final[f'{k}_{PRIMARY_RADIUS_KM}km']
    final.to_csv(out/'event_master_uniform.csv',index=False)

    analysis=final[final['emdat_total_affected'].notna()].copy().reset_index(drop=True)
    test_rows=[]; pred_rows=[]
    rain_col=f'rain_cum_mean_mm_{PRIMARY_RADIUS_KM}km'
    for thr in SEVERITY_THRESHOLDS:
        y,preds,probs,ks=model_predictions(analysis,thr,rain_col)
        for model,p in preds.items():
            row=metric_row(y,p,model,thr,PRIMARY_RADIUS_KM if 'multihazard' in model else '')
            lo,hi=wilson_ci(int(np.sum(y!=p)),len(y)); row['mismatch_ci95_low']=lo; row['mismatch_ci95_high']=hi
            if model=='M1_calibrated_wind': row['median_training_wind_threshold_kph']=float(np.median(ks))
            else: row['median_training_wind_threshold_kph']=np.nan
            test_rows.append(row)
        for i,ev in analysis.iterrows():
            pr={'impact_threshold_people':thr,'storm_key':ev['storm_key'],'emdat_total_affected':ev['emdat_total_affected'],'observed_severe':int(y[i]),
                'wind_kph':ev['max_wind_local_kph'],'pressure_mb':ev['min_pressure_local_mb'],'distance_land_km':ev['min_dist_land_local_km'],'rain_mean_175km_mm':ev[rain_col]}
            for model,p in preds.items(): pr[f'pred_{model}']=int(p[i])
            for model,pv in probs.items(): pr[f'prob_{model}']=pv[i]
            pred_rows.append(pr)
    tests=pd.DataFrame(test_rows); preds_df=pd.DataFrame(pred_rows)
    tests.to_csv(out/'trigger_tests_uniform.csv',index=False); preds_df.to_csv(out/'model_predictions_uniform.csv',index=False)

    # Spatial sensitivity for multihazard models, primary threshold only.
    spatial=[]
    for rad in RADII_KM:
        y,preds,_,_=model_predictions(analysis,100_000,f'rain_cum_mean_mm_{rad}km')
        for model in ['M2_logistic_multihazard','M3_random_forest_multihazard']:
            row=metric_row(y,preds[model],model,100_000,rad)
            lo,hi=wilson_ci(int(np.sum(y!=preds[model])),len(y)); row['mismatch_ci95_low']=lo; row['mismatch_ci95_high']=hi
            spatial.append(row)
    pd.DataFrame(spatial).to_csv(out/'spatial_sensitivity_uniform.csv',index=False)

    # Paired comparisons at the primary threshold.
    primary=preds_df[preds_df['impact_threshold_people']==100_000].copy().reset_index(drop=True)
    y=primary['observed_severe'].to_numpy(int)
    comp=[]
    base=(primary['pred_M1_calibrated_wind'].to_numpy(int)==y)
    for model in ['M0_fixed_wind_119','M2_logistic_multihazard','M3_random_forest_multihazard']:
        corr=(primary[f'pred_{model}'].to_numpy(int)==y)
        p,b,c=exact_mcnemar_p(base,corr)
        comp.append({'reference_model':'M1_calibrated_wind','comparison_model':model,'mcnemar_exact_p':p,
                     'reference_correct_comparison_wrong':b,'reference_wrong_comparison_correct':c})
    pd.DataFrame(comp).to_csv(out/'paired_model_comparisons.csv',index=False)

    # Feature/impact associations (descriptive only).
    from scipy.stats import spearmanr
    assoc=[]
    yy=np.log1p(analysis['emdat_total_affected'].astype(float))
    for col in ['max_wind_local_kph','min_pressure_local_mb','min_dist_land_local_km',rain_col]:
        rho,p=spearmanr(analysis[col].astype(float),yy,nan_policy='omit')
        assoc.append({'feature':col,'spearman_rho_with_log1p_affected':rho,'p_value':p,'n':len(analysis)})
    pd.DataFrame(assoc).to_csv(out/'hazard_impact_associations.csv',index=False)

    # Manifest: exact files and fixed analysis choices.
    manifest={
        'pipeline_version':'1.0-final-unified','seed':SEED,'created_utc':pd.Timestamp.utcnow().isoformat(),
        'input_master':str(Path(args.master).resolve()),'ibtracs':str(Path(args.ibtracs).resolve()),
        'chirps_month_files':chirps,'required_chirps_months':sorted(required),
        'primary_rain_corridor_km':PRIMARY_RADIUS_KM,'sensitivity_rain_corridors_km':list(RADII_KM),
        'severity_thresholds_people':list(SEVERITY_THRESHOLDS),'primary_severity_threshold_people':100000,
        'fixed_wind_trigger_kph':FIXED_WIND_TRIGGER_KPH,
        'validation':'leave-one-cyclone-out',
        'multihazard_features':['max_wind_local_kph','min_pressure_local_mb','min_dist_land_local_km','rain_cum_mean_mm_<radius>km'],
        'logistic':'StandardScaler + LogisticRegression(C=1, class_weight=balanced, liblinear)',
        'random_forest':'500 trees, max_depth=3, min_samples_leaf=2, max_features=sqrt, class_weight=balanced, fixed seed',
        'rainfall_definition':'Daily CHIRPS precipitation summed per grid cell for all calendar dates from IBTrACS local_start through local_end inclusive; corridor = cells <= radius km from any native IBTrACS local track point; summary metrics across valid corridor cells.'
    }
    with open(out/'reproducibility_manifest.json','w',encoding='utf-8') as f: json.dump(manifest,f,indent=2,ensure_ascii=False)

    # Compact final summary.
    t100=tests[tests.impact_threshold_people==100_000].set_index('model')
    s175=pd.DataFrame(spatial); s175=s175[s175.rain_corridor_km==175].set_index('model')
    lines=[]
    for model in ['M0_fixed_wind_119','M1_calibrated_wind','M2_logistic_multihazard','M3_random_forest_multihazard']:
        r=t100.loc[model]
        lines.append({'analysis':'primary_100k','model':model,'mismatch':r.basis_risk_mismatch,'balanced_accuracy':r.balanced_accuracy,'recall_severe':r.recall_severe,'f1_severe':r.f1_severe})
    pd.DataFrame(lines).to_csv(out/'final_primary_results.csv',index=False)
    print('DONE',out)

if __name__=='__main__':
    main()
