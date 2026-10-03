# Cyclone Parametric Insurance in Mozambique

Reproducibility repository for the study:

**Testing Multi-Hazard and Machine-Learning Triggers for Cyclone Parametric Insurance: A Practical and Reproducible Evaluation of Basis Risk in Mozambique**

## Authors

1. **Valter Tito Manjate** — Faculty of Economics, Universidade Eduardo Mondlane, Mozambique  
2. **Leandro Tito Manjate** — Department of Informatics, Universidade da Beira Interior, Portugal

## Purpose

This repository contains the processed analytical datasets, model outputs, and Python code used to evaluate whether multi-hazard and machine-learning triggers reduce a classification-based proxy of basis risk in cyclone parametric insurance for Mozambique.

The final analysis uses uniformly reprocessed CHIRPS rainfall metrics and leave-one-cyclone-out validation. Earlier exploratory outputs are not used in the manuscript.

## Main analysis

- Primary impact threshold: **100,000 people affected**
- Sensitivity thresholds: **50,000**, **250,000**, and **500,000**
- Primary rainfall corridor: **175 km** from the IBTrACS cyclone track
- Spatial sensitivity: **100 km** and **250 km**
- Validation: **Leave-One-Cyclone-Out (LOOCV)**
- Fixed random seed: **20261003**

Models:

- **M0:** fixed wind trigger at 119 km/h
- **M1:** wind-only threshold calibrated inside each training fold
- **M2:** standardized L2 logistic regression using wind, pressure, distance to land, and CHIRPS rainfall
- **M3:** random forest using the same multi-hazard predictors

## Main result

For the primary threshold of at least 100,000 affected people, the simple wind triggers produced lower mismatch than the multi-hazard models. The study therefore does **not** claim that machine learning generally reduces basis risk. Instead, it shows that additional hazard information can be useful, but added model complexity does not guarantee better trigger-impact alignment in a small, data-scarce cyclone sample.

## Repository structure

All public files are stored directly in the repository root:

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── DATA_NOTICE.md
├── requirements.txt
├── SHA256SUMS.txt
├── README_REPRODUCIBILITY.md
├── run_parametric_insurance_final_pipeline.py
├── event_master_uniform.csv
├── chirps_metrics_uniform.csv
├── trigger_tests_uniform.csv
├── model_predictions_uniform.csv
├── spatial_sensitivity_uniform.csv
├── paired_model_comparisons.csv
├── hazard_impact_associations.csv
├── final_primary_results.csv
├── chirps_uniformization_audit.csv
└── input_file_manifest_public.csv
```

## Reproducing the analysis

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The final pipeline expects three inputs:

1. the integrated event master table;
2. the IBTrACS NetCDF file;
3. a directory containing the required monthly CHIRPS NetCDF files.

Example:

```bash
python run_parametric_insurance_final_pipeline.py \
  --master /path/to/integrated_event_master_final.csv \
  --ibtracs /path/to/IBTrACS.since1980.v04r01.nc \
  --chirps-root /path/to/chirps/files \
  --out-dir ./results
```

The repository already contains the final processed analytical tables used for the manuscript, so the reported model results can be inspected without redistributing large third-party raw datasets.

## Data sources and redistribution

The analysis combines information derived from:

- IBTrACS
- CHIRPS
- EM-DAT
- DesInventar
- African Risk Capacity (ARC)
- IMF World Economic Outlook (WEO)

Raw third-party source files are **not redistributed in this repository**. Users who want to reproduce the full raw-data pipeline should obtain those files from the original providers under their current access and licensing terms.

## Important interpretation

The study uses trigger-impact mismatch as a **classification-based proxy for basis risk** because complete event-level insured monetary losses are not available for every cyclone. It should not be interpreted as a complete actuarial estimate of contract basis risk.

## Integrity

`SHA256SUMS.txt` contains SHA-256 hashes for the public files in this repository package.

## License

The analysis code is released under the MIT License. Third-party data and data derived from third-party sources remain subject to the terms of the original data providers. See `DATA_NOTICE.md`.

## Citation

Please cite the associated article once published. A `CITATION.cff` file is included for repository citation.
