# Project Map - dengue_paper

## Project purpose
A dual-scale dengue early warning research framework for Bangladesh linking individual clinical diagnosis (illness-day kinetics) to climate-driven district outbreak forecasting, evaluated under rolling-origin temporal and spatial holdout validation to quantify the Optimism Gap.

## Structure
- `dengue_bangladesh_paper.md` — High-level research plan, competitive positioning against Shiddik 2026, datasets, and hypotheses
- `dengue_bangladesh_methodology.md` — Detailed statistical methodology, nested ladders (C0-C3, P0-P5), BYM2/INLA equations, validation matrix
- `notebooks/` — Kaggle-ready execution pipeline
  - `notebooks/01_data_assembly_and_panel_engineering.ipynb` — Scrapes/merges DGHS cases, NASA POWER climate, BBS shapefiles, spatial lags
  - `notebooks/02_clinical_diagnostic_models.ipynb` — Arm A nested clinical ladder (C0-C3), illness-day kinetics, Pakistan benchmarking
  - `notebooks/03_population_ml_forecasting_and_validation.ipynb` — Arm B ML models (P0-P5), rolling-origin & spatial holdout CV, SHAP/ALE
  - `notebooks/04_bayesian_inla_spatiotemporal.R` — Arm B Bayesian spatio-temporal BYM2/RW modeling in R-INLA, Optimism Gap matrix

## Relationships
- `dengue_bangladesh_methodology.md` formalizes and expands `dengue_bangladesh_paper.md`
- `notebooks/01_data_assembly_and_panel_engineering.ipynb` outputs master panels for `notebooks/02_*`, `03_*`, and `04_*`
- `notebooks/02_clinical_diagnostic_models.ipynb` produces local catchment clinical leading indicators for `notebooks/03_*` (Arm C)
- `notebooks/03_population_ml_forecasting_and_validation.ipynb` and `notebooks/04_bayesian_inla_spatiotemporal.R` jointly populate the 2x2 validation matrix and Optimism Gap
