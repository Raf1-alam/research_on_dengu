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

> **These numbers are one commit behind the notebook.** They are the last canonical
> Kaggle run and are internally consistent, but the notebook has since been corrected in
> three ways that move them: the h=4 conformal calibration window was being chosen by a
> tie-break between two failed evaluations (both scored |coverage - 0.9| = 0.9 because NaN
> targets made the conformal quantile NaN); the burden tertiles in the conditional-coverage
> table were derived over a different year window than the ones the Mondrian method is
> calibrated on, so two districts sat in the wrong group; and `table5d` now reports partial
> correlations. Re-run and regenerate this section before quoting any of it in the
> manuscript. Do not mix numbers from this section with numbers from a new run.

**Canonical run:** Kaggle, 5 Sep 2026, `ml_notebook3` at commit `7d76129`
(Python 3.12.13, pandas 2.3.3, numpy 2.0.2, LightGBM 4.6.0). Every number below and every file
in `results/` comes from that single execution. Cross-checked against an independent run on a
different stack (Python 3.13.5 / pandas 3.0.3 / LightGBM 4.7.0): coverage of the reported method
agreed to **0.0015**, the optimism gap to **0.0024**, skill to **1.83 points**, the ARIMA row to
**0.08 points**, and all 18 calibration and 4 resolution verdicts were identical.

64 districts x 254 ISO weeks, expanding rolling origin (train <=2023 -> test 2024;
train <=2024 -> test 2025). Every learned model is a mean over three seeds; every tuned constant
is selected on an inner train<=2022 / validate-2023 split that never sees a test year.
Significance is Benjamini-Hochberg corrected at q = 0.05. All tables in [`results/`](results/).

**Calibration — marginal coverage is not the useful number.** Nominal 90% intervals, prospective:

| Horizon | Uncalibrated | Split-conformal | Group-conditional adaptive |
|---|---:|---:|---:|
| 1 week | 0.822 | 0.868 | **0.884** |
| 2 weeks | 0.844 | 0.873 | **0.887** |
| 4 weeks | 0.834 | 0.868 | **0.892** |

Coverage does not reach nominal at h = 1 or 2 and we do not claim it does; the residual gap is
distribution shift between calibration and test seasons, which split conformal cannot remove by
construction.

The finding is conditional, not marginal. Across district burden tertiles at h = 2, split
conformal leaves a spread of **0.069** — barely below the uncalibrated model's
**0.073** — with high-burden districts at 0.839 against
0.908 for low-burden. One shared correction cannot fix a score
distribution that differs by district size. Calibrating within burden groups and letting each
group's level adapt over the season closes the spread to **0.011**, so the busiest
districts are no longer the worst served.

Prior work has reported *marginal* conformal coverage for dengue (Rio de Janeiro, 2025); this is
the multi-unit result a single-city study cannot produce. See `docs/RELATED_WORK_NOTES.md`.

**The conditional failure, formally tested.** Two-proportion test of low- versus
high-burden coverage, h = 2:

| Method | Low burden | High burden | Difference | z | p (BH) |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 0.883 | 0.810 | 0.073 | 6.86 | <1e-10 |
| Split-conformal | 0.908 | 0.839 | **0.069** | **7.03** | <1e-10 |
| Group-conditional adaptive | 0.883 | 0.887 | **-0.004** | -0.41 | **0.87** |

Split conformal leaves a burden gap statistically indistinguishable from doing nothing.
The group-conditional method eliminates it, at h = 1, 2 and 4 alike. Spread reduction
0.052-0.061, district-resampling bootstrap p = 0.004-0.019.

Mechanism: per-district coverage against log burden rho = -0.604 (p < 1e-6), against log
population rho = -0.164 (p = 0.20, **not** associated). District size is not the explanation.

**Target parameterisation is horizon-dependent, not uniformly better.** Twenty seeds, each
fitting both models on the same folds so seed is a blocking factor:

| Horizon | Anchored | Level (Tweedie) | Paired difference | 95% CI | Cohen dz | Verdict |
|---|---:|---:|---:|---|---:|---|
| 1 wk | 13.76% | 8.63% | **+5.13** | [4.49, 5.76] | 3.53 | anchored better |
| 2 wk | 18.37% | 17.51% | +0.86 | [-0.30, 2.02] | 0.33 | no difference |
| 3 wk | 14.13% | 18.55% | **-4.42** | [-5.71, -3.13] | -1.50 | **level better** |
| 4 wk | 16.69% | 15.54% | +1.15 | [-0.49, 2.79] | 0.31 | no difference |

Anchoring on the last observation helps while the anchor is fresh and hurts once it is
stale. This is a crossover, and the paper must not be framed as "our reparameterisation
wins".

**Forecast skill**, MAE reduction against lag-0 persistence, with baselines:

| Horizon | Seasonal naive | ARIMA(2,1,2) | Level (L2) | Level (Tweedie) | Anchored growth |
|---|---:|---:|---:|---:|---:|
| 1 wk | -409.2% | -47.6% | -36.8% | +9.4% | **+13.4%** |
| 2 wk | -278.8% | -41.6% | -20.3% | +17.5% | **+20.1%** |
| 3 wk | -196.8% | -34.1% | -11.4% | **+20.8%** | +16.0% |
| 4 wk | -149.3% | -30.0% | -11.0% | **+16.3%** | +16.1% |

The ARIMA order is Naher et al.'s, selected on national *monthly* data; applied here at
district-week resolution with 47% zeros it is out of regime. Read that row as *the published
statistical benchmark does not transfer to operational resolution*, not as a win — see
`table0_assumptions_register.csv`. The L2 column is likewise a misspecification result, and the
least reproducible model in the study.

**Uncertainty on the comparisons.** `table3` reports effects with block-bootstrap intervals and
BH-corrected p-values, and deliberately carries **no significance label**: three such labels
flipped between platforms while every point estimate agreed to 0.291 MAE and all 20 intervals
overlapped. Of 16 model comparisons the bootstrap excludes zero in 8 — two test seasons cannot
resolve differences of this size, and the paper says so.

**Optimism gap** (alarm at h = 2, correctly ordered C1 > C3 > C2 > C4):
0.9822 -> 0.9046 ROC-AUC,
0.9326 -> 0.6114 PR-AUC.
Gap **+0.0776 / +0.3213**.

**Operational.** At 80% sensitivity, h = 1: precision 0.56, false-alarm rate 0.13, median
**6 weeks** of warning before a district crosses its own outbreak threshold.

**Forward test.** Frozen at end-2025, applied to 2026 without refitting: +18.6 to +23.5% skill.

**What is not established.** Resolution dependence is directionally consistent at all four
horizons but significant at none (h=1 BH p = 0.104). No individual climate variable's
contribution is identifiable: leave-one-out deltas sit inside seed noise and **change sign
between platforms for 4 of 10 variables**, so the redundancy question cannot be settled here in
either direction.

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

`notebooks/ml_notebook3.py` is the source of record. `notebooks/ml_notebook3.ipynb` is
generated from it for Kaggle and must never be edited directly.

```bash
pip install numpy pandas scipy scikit-learn lightgbm statsmodels matplotlib

python notebooks/ml_notebook3.py          # ~25 min on 4 CPU cores; no GPU needed
python scripts/build_notebook.py          # regenerate the .ipynb after any source edit
python scripts/build_notebook.py --check  # non-zero if the .ipynb has drifted
```

`ICEEICT_PAIR_SEEDS` (default 20) and `ICEEICT_SEEDS` (default `42,7,1`) shorten a smoke
test. Both change the numbers; canonical runs use the defaults. A local run writes to
`artifacts/`, never over the committed `results/`.

On Kaggle, attach `data/raw/district_panel/Dengue.csv`,
`data/raw/divisional/divisional_daily_2022_2025.csv` and
`data/raw/divisional/nasa_power_divisions_daily.csv` as one dataset, set **Accelerator: None**,
and run. Tree models on 16k rows are faster on CPU than on a T4.

The run stops itself if anything is wrong: seven gate assertions cover leaked features, DGHS
reconciliation, lags bridging the 2020–21 data gap, validation-matrix ordering, whether the
conformal correction actually did anything, whether the threshold sweep reproduces the gap it
perturbs, and whether the quantile and floor sweeps agree where they overlap.

`results/run_manifest.json` records the library versions, both seed sets, the alarm constants
and a SHA-256 of every input file. A table whose manifest does not match the committed one was
produced by different code or different data.

## Ground rules

1. Every number in the manuscript traces to a file in `results/`. Run first, then quote.
2. No claim ships without a committed run behind it.
3. Baselines are mandatory comparators and get the same information the model gets.
4. Headline deltas carry block-bootstrap intervals and a panel Diebold-Mariano test. If an
   interval spans zero, that is what gets reported.
