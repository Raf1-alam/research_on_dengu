# projectmem - dengue_paper

_Last updated: 2026-09-03_

## Project purpose
A dual-scale dengue early warning research framework for Bangladesh linking individual clinical diagnosis (illness-day kinetics) to climate-driven district outbreak forecasting, evaluated under rolling-origin temporal and spatial holdout validation to quantify the Optimism Gap.

## Recent issues
- No issues logged yet.

## Decisions
- Decided on 4-notebook Kaggle execution architecture (01 Data Assembly, 02 Clinical Arm A, 03 Population ML Arm B, 04 Bayesian INLA Arm B) to run on open datasets and prevent memory/timeout limits.
- Strictly enforced 100% empirical data requirement: Removed synthetic simulation fallbacks. Connected directly to open Kaggle DGHS district dataset (shampabanik/district-wise-dengue-dataset-for-bangladesh) and Mendeley clinical dataset (kawsarahmed/dengue-dataset-of-bangladesh).
- Completed comprehensive methodology audit on Notebook 1: added BBS 2022 Census populations, incidence rates per 100k, 5 spatial CV blocks, divisional socio-economics, multi-horizon forward targets (1, 2, 4, 8 weeks), and dual clinical parsers for Jamalpur CBC and Dhaka serology.
- Synchronize Notebook 1 script and ipynb with 100% verified empirical inputs
- Notebook 1 execution and empirical artifacts verified on Kaggle
- Implement Notebook 2: Individual Clinical Diagnostic Models (Arm A)
- Advance to Notebook 3: Population Outbreak Forecasting & Validation Harness

## Notes
- Verified exact live Kaggle slugs: DGHS district cases (shampabanik12/district-wise-dengue-dataset-for-bangladesh) and clinical patient data (kawsarahmad/dengue-dataset-bangladesh). Upgraded Notebook 1 scanner to dynamically identify files by column signatures and extract all clinical symptoms (fever duration, platelet, WBC, retro-orbital pain, myalgia, rash).
- Discovered recently published empirical datasets: (1) Dengue Hematology Jamalpur General Hospital (1,523 patients, Feb-Sep 2024, Mendeley DOI 10.17632/6fsrsk3mb8.2) providing full 19-parameter CBC profile for Arm A ladder C2; (2) Bangladesh Daily & Monthly Dengue Cases 2024-2026 for out-of-sample temporal testing.
- Notebook 1 Methodological Audit and Accomplishments Report
- Notebook 2 Execution & Arm A Verification Complete
- Notebook 2 Methodological Audit and Accomplishments Report
- Notebook 3 Execution Complete & Optimism Gap Quantified
- Notebook 3 Methodological Audit and Accomplishments Report
- Notebook 4 Complete & All 4 Modeling Arms Empirically Verified
- Entire 5-Notebook Pipeline 100% Complete & Empirically Verified
- Comprehensive Findings Audit & Shiddik et al. (2026) Comparative Analysis

## Key files
- `e.g`
- `10.17632/6fsrsk3mb8.2`

## Open questions
- None logged yet.
