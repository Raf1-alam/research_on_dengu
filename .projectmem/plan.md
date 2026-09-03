# dengue_paper — plan

## Active plans
- [x] Build Notebook 1: Master Data Assembly & Panel Engineering (`notebooks/01_data_assembly_and_panel_engineering.ipynb` and `.py`)
  - [x] Fetch/preprocess NASA POWER climate for 64 district centroids
  - [x] Process DGHS case data and Kaggle historical backfill
  - [x] Generate Queen-contiguity spatial adjacency matrix from BBS shapefiles
  - [x] Construct temporal lags (1-8w), spatial lags (1-4w), and district baselines
  - [x] Export clean Parquet panels for downstream modeling
- [ ] Build Notebook 2: Clinical Diagnostic Models (Arm A)
- [ ] Build Notebook 3: Population ML Forecasting & Validation Harness (Arm B + Arm D)
- [ ] Build Notebook 4: Bayesian Spatio-Temporal Modeling with R-INLA (Arm B Bayesian)

## Next
- Execute Notebook 1 on Kaggle to generate initial panel artifacts
- Upload pre-processed dataset to Kaggle as a reusable input dataset

## Shipped
- [x] Internalized methodology and research design requirements
- [x] Defined 4-notebook Kaggle architecture and open data strategy
