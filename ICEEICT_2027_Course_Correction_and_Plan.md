# Dengue Paper — Course Correction & Implementation Plan

**Target:** 7th ICEEICT, MIST Dhaka, 28–30 Jan 2027 · IEEE Xplore + Scopus · Conf. ID 72922 · submit via CMT
**Submission deadline (extended):** 26 Sep 2026 — **22 days from 4 Sep 2026**
**Audited:** `Raf1-alam/research_on_dengu` @ `2684901` + `E:\3-2\Machine Learning Lab`
**Web version:** https://claude.ai/code/artifact/1f833a56-42f1-4507-a6c2-591fbd5e44aa

---

## 1. Verdict

**Half right.** The *framing* is correct and matches a genuinely open gap: Bangladeshi dengue ML papers report near-perfect accuracy from random splits and almost none validate the way an operational system is judged. Your literature review already names the finding that should anchor the paper. The *execution* rests on two datasets that cannot support the claims made on them.

Two disagreeing bodies of work exist:

- **The GitHub repo** — a five-notebook "dual-scale" framework (64 districts, BYM2/INLA, 2×2 validation matrix, cross-scale coupling). Ambitious, **never executed**.
- **The local `src/` + `results/` pipeline** — smaller, actually run 18 Aug, producing numbers that contradict the repo's.

**Recommendation:**
1. **Delete the clinical diagnostic arm.** Both serology datasets are unusable; one was fabricated.
2. **Keep and deepen the forecasting arm.** The divisional panel is real, reproduces official DGHS totals, and covers 2025 — a season the literature has barely touched.
3. **Change the claim from accuracy to honesty.** Don't compete on a bigger AUC. Compete on being the first Bangladesh dengue forecast evaluated against persistence, prospectively, with calibrated uncertainty. Pilot already run — §4.

---

## 2. Blockers

### F1 — BLOCKER · `Datasets/dataset.csv` is a label-conditioned fabrication

First nine columns are byte-identical to `Datasets/Not Used/dataset (2).csv` (the original Mendeley/Kaggle release). Nine columns were appended — `Fever_Duration`, `Body_Temperature`, `Platelet_Count`, `WBC_Count`, and five symptom flags. These are exactly the columns carrying the "H1 illness-day kinetics" hypothesis, and they were generated *from* the label:

```
                     Outcome = 0            Outcome = 1        overlap
Body_Temperature     36.0 - 37.6 C          38.1 - 40.6 C      none
Fever_Duration       0 - 3 days             3 - 7 days         1 value
Platelet_Count       117,931 - 362,426      26,885 - 119,280   1,349 units
WBC_Count            3,256 - 10,229         1,740 - 4,070      814 units
```

Across 1,000 patients, not one dengue-negative febrile patient has a temperature above 37.6 °C. This is why `results/table_clinical_tier_1_symptoms.csv` reports ROC-AUC = 1.000 for all six models — the table is a readout of the generator, not a finding.

Your own literature review states it correctly: *"No publicly indexed Bangladeshi dataset identified in this review … records exact date of symptom onset."* That includes this one. H1 cannot be tested with data in hand.

### F2 — BLOCKER · The original serology dataset is degenerate: the label *is* the IgG column

In `Not Used/dataset (2).csv`, `Outcome == IgG` for **1,000 of 1,000 rows**.

```
NS1 IgM IgG | Outcome=0  Outcome=1
 0   0   0  |     251          0
 0   1   0  |     216          0     <-- IgM-positive, labelled negative
 0   0   1  |       0         13
 1   0   1  |       0        261
 1   1   1  |       0        258
```

Any model given serology returns AUC 1.0 trivially; remove serology and only demographics remain. **Both** versions of the clinical dataset are dead, not just the fabricated one.

### F3 — BLOCKER · No repo headline number is reproducible from anything committed

All five pipeline notebooks contain **zero executed output cells**. `notebooks/ml-notebook1.ipynb`, committed as "executed master Kaggle notebook with outputs", is **0 bytes**. There is no district-level data, no `R-INLA` dependency, and no BYM2 fit that has ever run.

Unsupported README claims: ΔAUC = +0.4086 · humidity RR = 1.313 · DIC −2,553.6 · 1,266 district-weeks across 64 districts · n = 2,523 patients · 22.4% cross-scale reduction · Barishal ζ = 2.10.

Some are internally impossible — the local pipeline never leaves 8 divisions. The 2×2 matrix I actually ran (§4) gives a gap of **0.060**, not 0.409.

### F4 — FIX · The ML forecaster is handicapped against its own baseline

In `src/feature_engineering.py` autoregressive features begin at `cases_lag_7d`, while the persistence baseline in `forecasting_models.py` uses cases at time *t*. For a *t*+7 target the ML models are effectively forecasting 14 days out against a 7-day baseline. That is much of why persistence wins in `table_macro_rolling_origin_forecasting.csv`. Fix before claiming "ML loses to persistence."

### F5 / F6 — FIX · Two smaller defects

- `Vital_Index = (Body_Temperature − 98.6) × (Fever_Duration+1)` subtracts a Fahrenheit constant from Celsius readings. Moot after F1, but reviewers find these.
- `early_warning_matrix.py` returns a hand-written alert table. Fine as an operational translation layer; it is **authored prose, not a model output**, and must never be presented as a result.

---

## 3. Assets

### F7 — KEEP · `Datasets/Dataset (1).csv` is real DGHS surveillance, and is your entire paper

11,200 daily records: 8 divisions × 1,400 days, **1 Jan 2022 → 31 Oct 2025**, with same-day max/min temperature, rainfall, humidity.

```
Annual national totals      this file      published DGHS
2022                          61,227          ~62,400
2023                         320,644         ~321,200
2024                         101,844         ~101,200
2025 (to 31 Oct)              69,777          in-season

Cross-check vs Not Used/dengu dataset.csv (independent source)
  3,368 overlapping division-days · 98.4% exact agreement · r = 0.9986
```

Four seasons including the record 2023 epidemic and a near-complete 2025. **2025 is the novelty window** — Shiddik et al. stop at 2024, Rahman et al. at 2021, Liu et al. at 2023. Weekly aggregation: 201 weeks × 8 divisions = 1,608 division-weeks, 22.9% zero-weeks (vs 42.3% zero-days).

### F8 — KEEP, DEMOTED · Jamalpur hematology dataset is real

n = 1,523, 19 CBC parameters, 68.4% positive. Your honest result (AUC ≈ 0.684, sens 0.95, spec 0.39) is plausible for CBC-only screening. Not strong enough to headline. Either cut it, or keep a half-page framed as *cost-sensitive rural triage*. **Recommendation: cut**, spend the page on calibration.

### F9 — KEEP · The literature review is good and already contains the paper's best idea

Section VIII cites *SpatialEpiBench* (2026): under standardized rolling-origin evaluation, sophisticated spatial/ML methods beat naive persistence **less than half the time**. Section X concludes credible Bangladeshi claims require rolling-origin plus a naive comparator. **That sentence is your paper.**

*Prose caveat:* several passages read as paraphrase-of-paraphrase and a few sentences restate their predecessor. Both will trip a similarity check. Rewrite in your own voice at the LaTeX stage.

---

## 4. Pilot evidence (run on your data, 4 Sep 2026)

**Protocol.** Divisional weekly panel (weeks ending Saturday), 8 divisions × 201 weeks. Expanding rolling origin: train ≤2023 → test 2024; train ≤2024 → test 2025. Metrics pooled. Features scale-free: log-growth deltas, seasonal harmonics, climate lags 2–12 weeks.

### Table A — Level-space ML does not reliably beat persistence

| Horizon | Model | MAE | RMSE | wMAPE % | Skill vs persistence |
|---|---|---:|---:|---:|---:|
| 1 wk | Persistence | 41.4 | 105.2 | 18.2 | — |
| | LightGBM, level target | 43.5 | 116.1 | 19.1 | **−5.1%** |
| | **LightGBM, anchored growth** | **37.0** | **96.8** | **16.3** | **+10.6%** |
| 2 wk | Persistence | 66.6 | 184.8 | 29.0 | — |
| | LightGBM, level target | 64.5 | 167.3 | 28.1 | +3.1% |
| | **LightGBM, anchored growth** | **60.7** | 171.8 | **26.5** | **+8.8%** |
| 3 wk | Persistence | 92.9 | 262.7 | 40.1 | — |
| | LightGBM, level target | 91.7 | 253.6 | 39.5 | +1.3% |
| | **LightGBM, anchored growth** | **85.6** | **246.7** | **36.9** | **+7.9%** |
| 4 wk | Persistence | 119.6 | 335.1 | 51.1 | — |
| | LightGBM, level target | 111.1 | 298.3 | 47.4 | +7.1% |
| | **LightGBM, anchored growth** | **99.0** | **257.9** | **42.3** | **+17.3%** |

Seasonal-naive is catastrophic (MAE 345–359 at every horizon) because 2023 was ~3× every neighbouring season — itself a finding: the post-2022 regime breaks year-over-year baselines.

**Mechanism.** A level-space learner trained on 2022–2023 has learned the *magnitude* of the 2023 epidemic and carries it into 2024–2025. Predicting the log growth ratio and anchoring on the last observed count removes scale from the learning problem; the model only learns direction and rate, which transfer across regimes.

### Table B — Intervals are badly over-confident; conformal calibration repairs them

| Horizon | Quantile LGBM coverage | + split-conformal | Median width, raw | Median width, conformal |
|---|---:|---:|---:|---:|
| 1 wk | **0.738** | 0.837 | 32 | 41 |
| 2 wk | **0.737** | 0.852 | 40 | 51 |
| 4 wk | **0.697** | **0.921** | 53 | 109 |

Nominal is 0.90. A 90% interval containing truth 70% of the time is worse than no interval. Split-conformal on held-out log-ratio residuals (last 26 weeks of each training window) restores 0.84–0.92. Residual under-coverage at short horizons is real — conformal assumes exchangeability and epidemic onset weeks violate it. Report it; point at adaptive conformal as the fix.

**No Bangladeshi dengue forecasting paper in your review reports interval coverage at all.** Cleanest available novelty, ~40 lines of code.

### Table C — The honest optimism gap (alarm classification, h = 2 wk, identical model & features)

| Condition | Holds out | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| C1 · Random 80/20 | nothing meaningful | 0.9750 | 0.9567 |
| C2 · Rolling origin | future seasons | 0.9294 | 0.8876 |
| C3 · Leave-division-out | unseen geography | 0.9493 | 0.9122 |
| **C4 · Space + time** | both | **0.9151** | **0.8491** |

Gap = **0.0599 ROC-AUC**, **0.1076 PR-AUC**. Smaller than the claimed 0.4086, but true — and PR-AUC (the metric that matters for a rare-alarm task) degrades nearly twice as fast. C4 at 0.915 is *good*: honesty does not require concluding everything fails.

---

## 5. The paper to write

**Working title:** *Persistence-Anchored Growth Forecasting with Calibrated Uncertainty for Dengue Early Warning in Bangladesh, 2022–2025*

**Thesis:** Under honest rolling-origin evaluation on Bangladesh's post-2022 dengue regime, level-space machine learning does not reliably beat a one-week persistence baseline. Reparameterising the target as persistence-anchored log growth recovers real skill at every horizon from one to four weeks, and conformal calibration repairs prediction intervals that are otherwise dangerously over-confident.

**Contributions**

1. **First rolling-origin benchmark on the 2022–2025 divisional series, including 2025.** Persistence and seasonal-naive as mandatory comparators. Extends every prior Bangladeshi study by ≥1 epidemic season.
2. **Persistence-anchored log-growth reparameterisation.** +7.9% to +17.3% MAE reduction vs persistence across h = 1–4 wk, where the conventional level target gives −5.1% to +7.1%.
3. **Prospective calibration audit.** Nominal 90% intervals achieve 70–74% coverage; split-conformal restores 84–92%. No prior study in this literature reports coverage.
4. **Honest optimism gap + operational alarm evaluation.** 2×2 space–time matrix on identical features; alarm lead-time / false-alarm-rate curves at fixed sensitivity.

**Six-page IEEE budget**

| Section | Content | Pages |
|---|---|---:|
| I. Introduction | Post-2022 regime shift; validation problem; four contributions | 0.75 |
| II. Related work | Shiddik/Rahman/Liu; rolling-origin & SpatialEpiBench; conformal prediction | 0.75 |
| III. Data & problem | Panel, provenance, DGHS cross-validation, zero-inflation, formal statement | 0.75 |
| IV. Method | Anchored-growth reparameterisation; features; conformal; alarm rule; validation matrix | 1.25 |
| V. Results | Tables 1–4, Figures 2–4 | 1.75 |
| VI–VII. Discussion & Conclusion | Operational reading, limitations, references | 0.75 |

*Verify the page limit on CMT before drafting — the current site does not publish it. Prior editions ran 6 pages IEEEtran two-column with over-length charges.*

**Tables:** T1 panel description & DGHS cross-validation · T2 skill vs persistence · T3 interval coverage & width · T4 optimism-gap matrix
**Figures (300 dpi, single-column):** F1 pipeline & rolling-origin schematic · F2 national weekly cases with test folds shaded · F3 forecast vs actual with conformal bands (Dhaka + one low-burden division) · F4 reliability diagram before/after · F5 alarm lead time vs false-alarm rate by division

---

## 6. Implementation

```
src/
  panel.py            # daily CSV -> weekly divisional panel; DGHS cross-check
  features.py         # scale-free: log-growth deltas, harmonics, climate lags 2-12wk
  baselines.py        # persistence, seasonal-naive, ARIMA/ETS on log counts
  models.py           # level-target and anchored-growth learners (LGBM, XGB, Ridge)
  conformal.py        # split-CQR on log-ratio residuals + adaptive variant
  validation.py       # rolling origin, leave-division-out, 2x2 space-time matrix
  alarms.py           # threshold rule, lead time, FAR at fixed sensitivity
  figures.py          # F1-F5 at 300 dpi, IEEE column widths
run_all.py            # one command -> results/*.csv + figures/*.png, seeded

DELETE
  notebooks/ (all five, never executed)
  src/clinical_models.py, src/early_warning_matrix.py
  src/explainability.py  (rebuild for the macro model only)
  Datasets/dataset.csv               (fabricated)
  Datasets/Not Used/dataset (2).csv  (degenerate label)
  results/table_clinical_*.csv
  figures/fig2_biomarker_kinetics_curve.png
```

**Experiments**

| ID | Experiment | Produces | Est. |
|---|---|---|---:|
| E1 | Build weekly panel; cross-validate vs independent 2019–23 source; document zero-inflation and the 2023 regime break | T1, F2 | 1 d |
| E2 | Baselines: persistence, seasonal-naive, ARIMA/ETS on log counts. **Fix F4 first** — give every learner lag-0 access | T2 rows | 1 d |
| E3 | Level vs anchored-growth targets, h = 1–4, three learner families, rolling origin over 2024 and 2025 | T2, F3 | 1.5 d |
| E4 | Quantile intervals → split-conformal → adaptive conformal; coverage, width, PIT histogram | T3, F4 | 1.5 d |
| E5 | 2×2 validation matrix; alarm lead time and FAR at fixed sensitivity | T4, F5 | 1 d |
| E6 | Ablations (climate on/off, seasonality on/off, per-division); block-bootstrap CIs on every headline delta | robustness ¶ | 1 d |

**Three non-negotiables**

- **Uncertainty on every headline number.** Two test seasons is small. Block-bootstrap CIs on skill deltas, or a reviewer will ask whether +7.9% is noise. If an interval crosses zero, say so.
- **One seeded command reproduces everything.** `python run_all.py` regenerates every table and figure. Commit notebook outputs, or don't commit notebooks.
- **Every number in the manuscript traces to a file in `results/`.** The failure mode that produced F3 was writing prose ahead of runs. Invert it.

**Optional, only if early:** splice the 2019 season from `Not Used/dengu dataset.csv` as an out-of-regime stress test (train 2022–2025, forecast 2019). Coverage is partial (127/365 days in 2019) — sensitivity analysis only. Skip 64-district and BYM2/INLA entirely; you have no district data, and reaching for them produced the unbacked claims.

---

## 7. Schedule — 22 days, two hard gates

| Dates | Phase | Work |
|---|---|---|
| **Sep 4–6** (Fri–Sun) | Demolition & foundation | Delete clinical arm; retract unbacked README claims (§8) before anyone else reads the repo; build `panel.py` + `features.py`; lock and freeze the weekly panel; run E1 → T1, F2 |
| **Sep 7–10** (Mon–Thu) | Core result | E2 then E3 (fix F4 first); T2 and F3 final — **the paper's spine**; draft Sections III–IV in parallel |
| **Sep 10** | **GATE 1 — go/no-go** | If anchored growth beats persistence at ≥3 of 4 horizons with bootstrap intervals clear of zero → proceed. If not → pivot headline to the calibration result (E4), demote forecasting skill. Decide on the 10th, not the 20th. |
| **Sep 11–14** (Fri–Mon) | Calibration, alarms, robustness | E4 → T3/F4 · E5 → T4/F5 · E6 → bootstrap CIs; draft Section V around finished tables |
| **Sep 15–16** (Tue–Wed) | Freeze | **Results freeze** — no new experiments after the 16th; regenerate all figures at 300 dpi, IEEE column width, check print legibility |
| **Sep 17–21** (Thu–Mon) | Full LaTeX draft | IEEEtran, all six sections, abstract last; rewrite lit-review prose in your own voice; references incl. Shiddik 2026, Rahman 2025, Liu 2025, Bhowmik 2026, Sarker 2024, SpatialEpiBench 2026, Romano et al. (CQR), Gibbs & Candès (adaptive conformal) |
| **Sep 22–24** (Tue–Thu) | **GATE 2 — adversarial review** | Supervisor + one outside reader; trace every number in the PDF to a file in `results/`; similarity check (IEEE CrossCheck — aim <15%, no single source >3%) |
| **Sep 25** (Fri) | Submit | PDF eXpress, CMT submission, authors/affiliations final. Submit a day early; the extended deadline will not extend again. |

---

## 8. Risk register & retraction checklist

**Anticipated objections**

| Objection | Your answer |
|---|---|
| "Eight divisions is coarse. Why not districts?" | Limitations: divisional resolution is what public daily surveillance provides for 2022–2025. The methodological claim (target parameterisation and calibration) is resolution-independent. Say so. |
| "Only two test seasons." | Pre-empted by E6: block-bootstrap intervals on every delta plus a per-division breakdown showing the effect is not Dhaka-driven. |
| "Log-ratio anchoring is not new." | Correct — don't claim otherwise. The contribution is the demonstration on a national dengue system against honest baselines with the calibration audit attached. Cite prior use and position as transfer + evidence. |
| "Where is the deep learning?" | Answer with SpatialEpiBench and your own C1-vs-C4 gap. Add one LSTM/ConvLSTM row to Table 2 if space allows — cheap, and it forecloses the question. |
| "Why no BYM2 / spatial model?" | Eight areal units cannot identify a structured spatial prior. One sentence; stronger than a fitted model. |
| "Climate variables barely help." | If E6 shows that, report it. A negative result about climate features under honest validation is more interesting to this venue than another confirmation. |

**Retraction checklist — do before the repo is read outside the team**

- [ ] README: remove ΔAUC = +0.4086, ΔRMSE = +610.5, the 4.5× error multiplier, and the Condition 1–4 table. Replace with §4 Table C once re-run.
- [ ] README: remove the Bayesian section — humidity RR = 1.313, DIC −2,553.6, Barishal ζ = 2.10. No BYM2 model was ever fitted.
- [ ] README: remove "n = 2,523 patients", the C₀–C₃ ladder results, "1,266 district-weeks", "64 districts", the 22.4% cross-scale figure.
- [ ] README: remove the "100% Empirical" badge and the "zero synthetic imputation" line. Given F1 they are the opposite of true.
- [ ] Delete `notebooks/ml-notebook1.ipynb` (0 bytes); correct the commit message claiming it contains outputs.
- [ ] Delete or quarantine `Datasets/dataset.csv` with a README note explaining why, so nobody on the team reuses it.
- [ ] Update or archive under `deprecated/`: `dengue_bangladesh_paper.md`, `dengue_bangladesh_methodology.md`, and the five notebook audit files.

---

**The one thing to take away.** You do not have a weak project. You have a strong dataset attached to claims it cannot support. The 2022–2025 divisional panel is real, reproduces official totals, covers a season nobody has published on, and the honest result it yields is genuinely interesting: the standard approach loses to a baseline, a simple reparameterisation fixes it, and everyone's confidence intervals are wrong. That is a better ICEEICT paper than a dual-scale framework with a perfect AUC — and it is the one you can finish by the 25th.

---

*Audit performed 4 Sep 2026. Pilot results in §4 computed directly from `Datasets/Dataset (1).csv` under the protocol described there — indicative, not final; regenerate via `run_all.py` before use in the manuscript. Conference details from iceeict.mist.ac.bd as of 4 Sep 2026 — confirm page limit and template version on CMT before drafting.*
