# Fever and Forecast: A Dual-Scale Framework Linking Clinical Dengue Diagnosis to Climate-Driven Outbreak Early Warning in Bangladesh

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Empirical Data: Bangladesh](https://img.shields.io/badge/Data-100%25%20Empirical-success.svg)](#)

> **Official Repository** for the research paper:  
> *"Fever and Forecast: A Dual-Scale Framework Linking Clinical Dengue Diagnosis to Climate-Driven Outbreak Early Warning in Bangladesh"*

---

## 📌 Executive Summary

Dengue surveillance systems traditionally operate at a single spatial or temporal scale, creating a persistent gap between patient bedside diagnostic triage and population outbreak early warning. Furthermore, previous machine learning studies in South Asia have reported near-perfect prospective forecasting accuracy ($\approx 0.99$ ROC-AUC) using retrospective random train/test splits.

This repository provides an end-to-end, 100% empirical, dual-scale framework combining:
1. **Scale 1 (Individual Clinical Triage - Arm A)**: A 4-tier diagnostic ladder ($C_0 \to C_3$) validated on $2,523$ real Bangladeshi hospital patients (Jamalpur General Hospital $n=1,523$; Dhaka Medical Center $n=1,000$). Validates **Hypothesis H1** (NS1 viremia vs. IgM seroconversion kinetics, identifying a diagnostic crossover at **~Day 3.8 of illness**).
2. **Scale 2 (Regional Population Early Warning - Arm B)**: Multi-horizon outbreak forecasting ($h \in \{1, 2, 4, 8\}$ weeks ahead) and Bayesian spatio-temporal modeling (BYM2 spatial graph prior on the $64 \times 64$ Queen contiguity Laplacian + RW1 temporal random walks) across **1,266 district-weeks** (2019–2023).
3. **Multimodal Cross-Scale Coupling (Arm C)**: Feeds local hospital diagnostic test positivity ($\widehat{S}_{d,t}$) into district early warning models, reducing forecast count error by **22.4%**.
4. **The Quantified Optimism Gap (§4.6)**: Proves mathematically across a formal $2 \times 2$ Combined Validation Matrix that retrospective random splits inflate performance by **$\Delta \text{AUC} = +0.4086$** and underestimate count error by **$4.5\times$** ($+610.5$ cases/week) compared to honest space-time holdouts.

---

## 🗂️ Repository Structure

```text
dengue_paper/
│
├── notebooks/                                # Complete 5-Notebook Pipeline
│   ├── 01_data_assembly_and_panel_engineering.ipynb   # Merges DGHS, NASA POWER, ERA5, BBS Census
│   ├── 01_data_assembly_and_panel_engineering.py
│   ├── 02_clinical_diagnostic_models.ipynb            # Clinical Ladder C0->C3 & Hypothesis H1
│   ├── 02_clinical_diagnostic_models.py
│   ├── 03_population_outbreak_models.ipynb            # ML Hierarchy P0->P5 & Optimism Gap
│   ├── 03_population_outbreak_models.py
│   ├── 04_bayesian_spatiotemporal_inla.ipynb          # BYM2 ICAR Spatial + RW1 Temporal Model
│   ├── 04_bayesian_spatiotemporal_inla.py
│   ├── 05_results_synthesis_and_publication_figures.ipynb # Compiles Tables 1-5 & Figures 1-5
│   └── 05_results_synthesis_and_publication_figures.py
│
├── dengue_bangladesh_paper.md                # Comprehensive Paper Manuscript Draft
├── dengue_bangladesh_methodology.md          # Formal Mathematical & Methodological Protocol
├── comprehensive_findings_audit_and_shiddik_comparison.md # Audit vs. Shiddik et al. (2026)
│
├── notebook1_accomplishments_and_audit.md    # Methodological audit for Notebook 1
├── notebook2_accomplishments_and_audit.md    # Methodological audit for Notebook 2
├── notebook3_accomplishments_and_audit.md    # Methodological audit for Notebook 3
├── notebook4_accomplishments_and_audit.md    # Methodological audit for Notebook 4
├── notebook5_accomplishments_and_audit.md    # Methodological audit for Notebook 5
│
└── .gitignore                                # Repository hygiene
```

---

## 🔬 Core Scientific Findings

### 1. The Quantified Optimism Gap (Table 4 & Figure 4)

$$\mathbf{\text{Optimism Gap}} = \mathbf{\text{Condition 1 (Single Split)} - \text{Condition 4 (Space-Time Holdout)} = +0.4086 \text{ ROC-AUC}}$$
$$\mathbf{\Delta \text{RMSE}} = \mathbf{+610.5 \text{ cases/week (4.5x Error Multiplier)}}$$

| Validation Condition | Outbreak ROC-AUC | Count RMSE (Cases/Week) | Interpretation |
|---|:---:|:---:|---|
| **Condition 1 (Single Random Split)** | **0.9505** | **171.6** | Flattering, inflated retrospective baseline (comparable to Shiddik et al. 2026). |
| **Condition 2 (Temporal Rolling-Origin Holdout)** | **0.6927** | **360.0** | Honest prospective time-series evaluation across 2021–2023. |
| **Condition 3 (Spatial 5-Block Leave-Out)** | **0.8858** | **439.1** | Geographic transferability to unseen administrative blocks. |
| **Condition 4 (Combined Space + Time Holdout)** | **0.5419** | **782.1** | **Fully honest operational deployment** in unseen districts during unseen future years. |

### 2. Clinical Diagnostic Ladder (Arm A, Table 2)

* **Tier $C_0$ (Pre-Test Symptoms, $n=1,000$)**: ROC-AUC **0.9996**, Sensitivity **99.1%**, Specificity **98.7%**.
* **Tier $C_1$ (Single NS1 Antigen)**: ROC-AUC **0.9999**, Accuracy **99.8%**, Specificity **100.0%**.
* **Tier $C_2$ (Combined Serology)**: ROC-AUC **1.0000** (NS1 + IgM).
* **Tier $C_3$ (Extended CBC + NLR/PLR in Resource-Limited Hospital, $n=1,523$)**: ROC-AUC **0.6878**, PR-AUC **0.7822**, **Sensitivity 95.1%**, F1-score **0.851**. Outperformed external Pakistani benchmark (Qaiser et al. 2024: 88.0% sensitivity, 79.0% specificity).

### 3. Bayesian Spatio-Temporal Model (Arm B Bayesian, Table 5)

* **Model Selection**: Model $B_2$ (BYM2 Spatial Prior + RW1 Temporal Random Walk) was decisively selected by DIC, slashing deviance by **$2,553.6$ points** over fixed effects alone.
* **Key Meteorological Drivers**:
  * **Relative Humidity**: $\text{RR} = 1.313$ ($95\% \text{ CI: } [1.214, 1.420]$) -> $+31.3\%$ risk increase per $+1\text{ SD}$.
  * **Maximum Temperature**: $\text{RR} = 1.179$ ($95\% \text{ CI: } [1.090, 1.276]$).
  * **Mean Temperature**: $\text{RR} = 1.172$ ($95\% \text{ CI: } [1.084, 1.268]$).
* **Endemic Reservoirs**: Controlling for population scale and hospital beds, the southern coastal belt (**Barishal $\zeta = 2.10$, Khulna $\zeta = 1.33$**) exhibits more than **twice the residual ecological transmission risk** of northern districts.

---

## 🚀 Quickstart & Execution

All code runs sequentially without requiring external database servers or GPU acceleration.

### Option 1: Local Execution
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd dengue_paper

# Install dependencies
pip install numpy pandas scipy scikit-learn xgboost shap matplotlib seaborn pyarrow

# Run pipeline scripts
python notebooks/01_data_assembly_and_panel_engineering.py
python notebooks/02_clinical_diagnostic_models.py
python notebooks/03_population_outbreak_models.py
python notebooks/04_bayesian_spatiotemporal_inla.py
python notebooks/05_results_synthesis_and_publication_figures.py
```

### Option 2: Interactive Jupyter Notebooks
Run the notebooks in sequential order (`01` $\to$ `05`) in VS Code, JupyterLab, or on Kaggle.

---

## 📜 Citation & Attribution

If you use this codebase, methodology, or empirical findings, please cite:
```bibtex
@article{alam2026feverandforecast,
  title={Fever and Forecast: A Dual-Scale Framework Linking Clinical Dengue Diagnosis to Climate-Driven Outbreak Early Warning in Bangladesh},
  author={Alam, Syed Rafi and Collaborators},
  journal={Working Paper / Under Review},
  year={2026}
}
```

---
*Developed with rigorous epidemiological standards, zero synthetic imputation, and prospective validation integrity.*
