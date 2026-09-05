# src/

Empty by design. The August pipeline is in `unused/legacy_src/`; these are the modules
that replace it. Written in this order — see `docs/ICEEICT_2027_Implementation_Plan.md` §6.

| Module | Responsibility |
|---|---|
| `panel.py` | Load `data/raw/district_panel/`, build the district-week modelling frame, season-block it so lags never bridge the 2020–21 gap. Reconcile national totals against DGHS. Also builds the divisional panel from `data/raw/divisional/` for the resolution ablation. |
| `features.py` | Scale-free features only: log-growth deltas, seasonal harmonics, neighbour and upstream pressure in log space, climate lags. Never raw case levels as a predictor. |
| `baselines.py` | Persistence, seasonal-naive, ARIMA/ETS on log counts. Every learner gets lag-0 access — the August comparison did not, and that is defect F4. |
| `models.py` | Level-target and persistence-anchored-growth learners (LightGBM, XGBoost, Ridge) across h = 1–4 weeks. |
| `conformal.py` | Split-CQR on log-ratio residuals, plus the adaptive variant. Coverage, width, PIT. |
| `validation.py` | Expanding rolling origin, leave-one-division-out spatial blocks, and the 2×2 space–time matrix. |
| `alarms.py` | Threshold rule, alarm lead time, false-alarm rate at fixed sensitivity. |
| `figures.py` | F1–F5 at 300 dpi, IEEE single-column width. |

`run_all.py` at the project root runs the whole thing, seeded, and writes `results/` and `figures/`.

**Rule:** every number that appears in the manuscript must trace to a file in `results/`.
