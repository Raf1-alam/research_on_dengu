# Calibrated Dengue Forecasting for Bangladesh, 2019–2026

Working repository for a 6-page IEEE conference paper targeting **ICEEICT 2027**
(7th International Conference on Electrical Engineering and ICT, MIST Dhaka,
28–30 Jan 2027). Submission deadline **26 Sep 2026**.

**Start here:** [`notebooks/ml_notebook3.ipynb`](notebooks/ml_notebook3.ipynb) ·
[`docs/ICEEICT_2027_Implementation_Plan.md`](docs/ICEEICT_2027_Implementation_Plan.md) ·
[`docs/DATASET_DOSSIER.md`](docs/DATASET_DOSSIER.md)

---

## Retraction notice

**An earlier version of this README reported results that no committed run had produced.**
The following claims appeared here and are withdrawn in full:

| Withdrawn claim | Status |
|---|---|
| Optimism gap ΔAUC = +0.4086, ΔRMSE = +610.5, 4.5× error multiplier | No run produced these. Measured gap is **+0.0780 ROC / +0.3237 PR**. |
| Humidity RR = 1.313, DIC reduction 2,553.6, Barishal ζ = 2.10 | **No BYM2 model was ever fitted.** The repository contained no INLA dependency. |
| Clinical ladder C₀–C₃, n = 2,523 patients, ROC-AUC 0.9996–1.0000 | Derived from `Datasets/dataset.csv`, a **label-conditioned fabrication** — nine columns were appended to a public dataset with values generated from the outcome. Dengue-negatives spanned 36.0–37.6 °C and positives 38.1–40.6 °C with zero overlap across 1,000 patients. |
| "1,266 district-weeks", "64 districts" (2026-08 framing) | The pipeline of that period never left 8 divisions. |
| 22.4% cross-scale error reduction | No run produced it. |
| "100% Empirical" badge, "zero synthetic imputation" | The opposite was true. |

The underlying serology dataset is also degenerate: in the original Mendeley release
(`10.17632/zdtc3n6xv2`) **`Outcome == IgG` for 1,000 of 1,000 rows** — the label is a copy of a
predictor. The same file is mirrored on Kaggle (`kawsarahmad/dengue-dataset-bangladesh`) and
Hugging Face (`fairhealth/bangladesh-dengue`) under three DOIs. **A new mirror is not a new
dataset.** Both files are quarantined in [`unused/`](unused/README.md) with the reasoning written
down so nobody re-imports them.

The clinical arm has been removed. What follows is what the data supports.

---

## Results

One run of `ml_notebook3`, 64 districts x 254 ISO weeks, expanding rolling origin
(train <=2023 -> test 2024; train <=2024 -> test 2025). Every learned model is a mean over
three seeds; every tuned constant is selected on an inner train<=2022 / validate-2023 split
that never sees a test year. Significance is Benjamini-Hochberg corrected at q = 0.05.
All tables in [`results/`](results/).

**Reproducibility.** Verified across two disjoint seed sets: coverage of the reported method
moves by at most 0.003, every significance verdict agrees, the optimism gap by 0.0001 ROC.
`results/table12_seed_stability.csv` reports the spread behind each number.

**Calibration.** Nominal 90% intervals, prospective:

| Horizon | Uncalibrated | Split-conformal | Group-conditional adaptive |
|---|---:|---:|---:|
| 1 week | 0.833 | 0.861 | **0.883** |
| 2 weeks | 0.843 | 0.869 | **0.887** |
| 4 weeks | 0.842 | 0.868 | **0.893** |

Marginal coverage is only half the story, and the more useful half is conditional. Split
conformal barely helps there: across burden tertiles its coverage spread stays at
0.070 (h=2), essentially the 0.077 of the uncalibrated model, because
one shared correction cannot fix a score distribution that differs by district size. High-burden
districts sit at 0.835
while low-burden ones are at 0.905.
Calibrating within burden groups and letting each group's level adapt over the season closes
that gap to **0.008** — near-uniform coverage, with the busiest districts no longer
the worst served.

Coverage does not reach nominal at h = 1 or 2 and we do not claim it does. The residual gap is
distribution shift between calibration and test seasons, which split conformal cannot remove by
construction.

**Forecast skill**, MAE reduction against lag-0 persistence:

| Horizon | Level (L2) | Level (Tweedie) | Anchored growth (L1) |
|---|---:|---:|---:|
| 1 wk | -8.2% | +8.9% | **+14.5%** |
| 2 wk | -0.5% | +17.6% | **+19.1%** |
| 3 wk | -3.0% | **+19.0%** | +14.1% |
| 4 wk | -5.9% | +15.7% | **+15.1%** |

The L2 column is a misspecification result, not a target-parameterisation result: squared error on
47%-zero counts is the wrong likelihood, and it is also the least reproducible model in the study
(+-10 skill points between seed sets, against +-1.6 for the reported models).

**Optimism gap** (alarm at h = 2, correctly ordered C1 > C3 > C2 > C4):
0.9821 -> 0.9041 ROC-AUC, 0.9323 -> 0.6087 PR-AUC. Gap **+0.0780 / +0.3237**.

**Operational.** At 80% sensitivity, h = 1: precision 0.56, false-alarm rate 0.13, and a median
**6 weeks** of warning before a district crosses its own outbreak threshold.

**Forward test.** Frozen at end-2025 and applied to 2026 without refitting: +19 to +24% skill.

**What is not established.** The resolution-dependence claim is directionally consistent at all
four horizons (+9.6 to +10.7 pp larger gap at 8 divisions than at 64 districts) but reaches
significance at none of them (h=1: 95% CI [1.6, 13.9], BH p = 0.104). It is reported as a
direction, not a result.

---

## Data

Verified by direct download and audit — not from the depositors' descriptions.

| Path | Source | Unit | Span |
|---|---|---|---|
| `data/raw/district_panel/` | figshare [10.6084/m9.figshare.33040637](https://doi.org/10.6084/m9.figshare.33040637), CC BY 4.0 | district-week, 64 units | 2019 + 2022–2026, 16,256 rows |
| `data/raw/divisional/` | DGHS daily bulletins + NASA POWER harvest | division-week, 8 units | 2022–2025 |
| `data/raw/coxsbazar/` | Zenodo [10.5281/zenodo.19219551](https://doi.org/10.5281/zenodo.19219551), CC BY 4.0 | patient | 2021–2024, 35,581 rows |
| `data/raw/clinical/` | Mendeley [10.17632/6fsrsk3mb8](https://doi.org/10.17632/6fsrsk3mb8) | patient | 2024, n = 1,523 |

The notebook cross-checks the panel against sources it was **not** derived from
(`results/table1c_external_cross_checks.csv`):

* cases vs the independently compiled divisional bulletin — **r = 0.9994**, 86.6% exact match,
  totals within 0.51% over 1,600 division-weeks
* climate vs an independent NASA POWER harvest — temperature **r = 0.9975**, humidity **r = 0.9881**

**Known caveats**, all reported in `results/table1b_data_quality.csv` and tested in
`table11_data_quality_sensitivity.csv`:

* ~10% of zero-case weeks are missing reports, not true zeros (791 rows have `cases = 0` while
  patients are still admitted). Conclusions survive their removal; the 47.3% zero-inflation figure
  must carry this caveat.
* Rainfall is CHIRPS in the district panel and NASA POWER in the divisional one (r = 0.86 between
  products). The resolution result survives dropping rainfall entirely.
* `gtrends_dengue` is effectively a national seasonal clock — median 4 distinct values per week
  across 64 districts, 47.5% exact zeros, r = 0.98 with the national series.
* The district deposit's README claims its 2019 season matches the official 101,354; the file sums
  to 30,257 (rows begin 26 Aug, ~30% of the season).

---

## Layout

```
notebooks/ml_notebook3.{ipynb,py}   the pipeline — 20 cells, 19 tables, 6 figures
results/                            every table in the manuscript + run_manifest.json
figures/                            300 dpi, PNG + PDF, IEEE column widths
docs/                               implementation plan, dataset dossier, module spec
data/raw/                           the four verified datasets
unused/                             quarantined — read unused/README.md before touching
```

## Reproducing

```bash
pip install numpy pandas scipy scikit-learn lightgbm matplotlib seaborn
python notebooks/ml_notebook3.py          # ~8 min on 4 CPU cores; no GPU needed
```

On Kaggle, attach `data/raw/district_panel/Dengue.csv`,
`data/raw/divisional/divisional_daily_2022_2025.csv` and
`data/raw/divisional/nasa_power_divisions_daily.csv` as one dataset, set **Accelerator: None**,
and run. Tree models on 16k rows are faster on CPU than on a T4.

The run stops itself if anything is wrong: five gate assertions cover leaked features, DGHS
reconciliation, lags bridging the 2020–21 data gap, validation-matrix ordering, and whether the
conformal correction actually did anything.

## Ground rules

1. Every number in the manuscript traces to a file in `results/`. Run first, then quote.
2. No claim ships without a committed run behind it.
3. Baselines are mandatory comparators and get the same information the model gets.
4. Headline deltas carry block-bootstrap intervals and a panel Diebold-Mariano test. If an
   interval spans zero, that is what gets reported.
