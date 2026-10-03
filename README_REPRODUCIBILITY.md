# Parametric Cyclone Insurance in Mozambique — Final Reproducible Analysis

## Purpose
This folder contains the **uniformly reprocessed** empirical results used for the final analysis. Earlier incremental CHIRPS results must not be used in the paper because some events were processed with slightly different spatial aggregation during exploratory work.

## Primary analysis
- **Classification sample:** 21 EM-DAT cyclone events with non-missing `Total Affected`.
- **Uniform CHIRPS coverage:** 27 cyclone tracks/events.
- **Primary severe-event definition:** `Total Affected >= 100,000`.
- **Sensitivity thresholds:** 50,000; 250,000; 500,000 affected.
- **Validation:** Leave-One-Cyclone-Out (LOOCV). A cyclone is never simultaneously in training and test data.

## CHIRPS preprocessing rule
1. Use IBTrACS `local_start` and `local_end` for each cyclone.
2. Include every **calendar day** from start through end, inclusive.
3. Use native IBTrACS local track points.
4. A CHIRPS grid cell belongs to the event corridor when it is within `R` km of **any** local track point.
5. Main corridor: **175 km**. Spatial sensitivity: **100 km and 250 km**.
6. Sum daily CHIRPS precipitation per grid cell over the event window.
7. Compute mean, median, p90, p95, maximum cumulative rainfall, peak daily corridor mean, peak daily cell value, and percentages of cells above 100 and 200 mm.

## Models
- **M0 — fixed wind:** payout/severe trigger at wind >= 119 km/h (about 64 kt).
- **M1 — calibrated wind:** wind-only threshold estimated inside each training fold to minimize classification mismatch, with deterministic tie-breaking.
- **M2 — logistic multihazard:** standardized L2 logistic regression using wind, pressure, distance to land, and CHIRPS cumulative mean rainfall. `class_weight=balanced`.
- **M3 — random forest multihazard:** 500 trees, max depth 3, minimum leaf size 2, sqrt feature selection, balanced class weights, fixed seed `20261003`.

## Main result (>=100,000 affected)
The uniformly reprocessed results do **not** establish that the multihazard/ML models reduce basis risk relative to the calibrated wind trigger:
- M0 fixed wind: mismatch 19.0%; balanced accuracy 79.2%; severe-event recall 91.7%.
- M1 calibrated wind: mismatch 19.0%; balanced accuracy 79.2%; severe-event recall 91.7%.
- M2 logistic + rainfall: mismatch 23.8%; balanced accuracy 75.0%; recall 83.3%.
- M3 random forest + rainfall: mismatch 28.6%; balanced accuracy 70.8%; recall 75.0%.

Exact McNemar comparisons versus M1 are non-significant in this small sample. At the >=500,000 threshold, multihazard models recover some severe events missed by M1, but there are only five severe cases, so this is a sensitivity finding rather than a general conclusion.

## Scientific interpretation
The defensible conclusion is **not** that AI generally outperforms conventional parametric triggers. Rainfall provides additional hazard information, but the small Mozambique event sample does not show a robust reduction in classification-based basis risk across severity definitions and spatial corridor widths. This negative/conditional result is itself useful for designing parsimonious parametric products in data-scarce markets.

## Key files
- `event_master_uniform.csv` — final integrated event table with uniformly recalculated CHIRPS variables.
- `chirps_metrics_uniform.csv` — uniform rainfall metrics for all 27 events at 100/175/250 km.
- `trigger_tests_uniform.csv` — M0–M3 results at all severity thresholds.
- `model_predictions_uniform.csv` — event-level LOOCV predictions.
- `spatial_sensitivity_uniform.csv` — 100/175/250 km sensitivity for multihazard models.
- `paired_model_comparisons.csv` — exact paired comparisons versus M1.
- `hazard_impact_associations.csv` — descriptive Spearman associations with log affected population.
- `chirps_uniformization_audit.csv` — comparison of exploratory vs uniform CHIRPS processing.
- `input_file_manifest.csv` — exact input file paths/sizes used in the reproducible run.
- `reproducibility_manifest.json` — fixed methodological choices and model specifications.
- `run_parametric_insurance_final_pipeline.py` — analysis source code.

## Rule for the paper
Only numbers in this final reproducible package should be reported in the manuscript. Earlier exploratory CHIRPS values and model tables are superseded.
