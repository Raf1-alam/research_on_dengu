# Course Correction & Implementation Plan: ML Notebook 1 Limitations & Audit
### ICEEICT 2027 · MIST Dhaka
**Prepared:** 4 Sep 2026  
**Core Strategy:** Cut the clinical arm, keep the forecast, win on honest validation  

> **Executive Summary:**  
> Your surveillance data is real and genuinely valuable. Your clinical data is fabricated, and your headline numbers do not exist in any committed run. This is what to delete, what to keep, and the paper you can actually get into IEEE Xplore in the twenty-two days you have left.

---

### Project & Submission Metadata
* **Repo:** `Raf1-alam/research_on_dengu` @ `2684901`
* **Working Directory:** `E:\3-2\Machine Learning Lab`
* **Audited:** 6 datasets, 5 notebooks, 7 modules, 9 result tables
* **Submission Deadline:** **26 Sep 2026** (Extended. 22 days from today)
* **Notification:** 20 Nov 2026
* **Camera-ready:** 20 Dec 2026
* **Conference:** 28–30 Jan 2027 · 7th ICEEICT, MIST, Dhaka
* **Indexing:** IEEE Xplore + Scopus · Conf. ID 72922 · Submit via CMT

---

## 01. Verdict
*Direction, in one page.*

### Are you heading in the right direction?
**Half.** The framing is right and matches a real, currently-open gap in the literature — Bangladeshi dengue ML papers report near-perfect accuracy from random splits, and almost none of them validate the way an operational system would be judged. Your own literature review already names the finding that should anchor the paper. The execution rests on two datasets that cannot support the claims being made on them.

Two separate bodies of work exist under this project, and they disagree with each other:
1. **The GitHub repo** — a five-notebook "dual-scale" framework covering 64 districts, BYM2/INLA Bayesian spatio-temporal modelling, a 2×2 validation matrix, and cross-scale coupling. Ambitious, and never executed.
2. **The local `src/` + `results/` pipeline** — smaller, actually run on 18 August, and producing numbers that contradict the repo's.

The repo's README is where the paper's identity currently lives, and it is the part with no evidence behind it. The local pipeline is the part that ran — and what it honestly reports (ROC-AUC = 1.000 across six clinical models; a naive persistence baseline beating every ML forecaster) is a diagnosis, not a result.

### The Short Recommendation
* **Delete the clinical diagnostic arm.** Both serology datasets are unusable, one of them because it was fabricated. Nothing built on them is publishable, and a reviewer who runs one `describe()` will see it.
* **Keep and deepen the forecasting arm.** The divisional surveillance panel is real, reproduces official DGHS totals, and covers 2025 — a season the published literature has barely touched. That alone is a paper.
* **Change the claim from accuracy to honesty.** Don't compete on a bigger AUC. Compete on being the first Bangladesh dengue forecast evaluated the way an early-warning system is actually judged: against persistence, prospectively, with calibrated uncertainty. I have already run the pilot and the result holds — see §04.

---

## 02. Blockers
*Three things that must be fixed before any writing starts.*

### What the audit found
Every claim below is reproducible from the files in `Datasets/` and the repo as cloned. I ran each check directly rather than reading the documentation.

---

### [Blocker F1] `Datasets/dataset.csv` is a label-conditioned fabrication
Its first nine columns are byte-identical to `Datasets/Not Used/dataset (2).csv`, the original Mendeley/Kaggle release. Nine columns were then appended — `Fever_Duration`, `Body_Temperature`, `Platelet_Count`, `WBC_Count`, and five symptom flags. Those nine are exactly the columns carrying the paper's illness-day "H1" hypothesis, and they were generated from the label:

| Variable | Outcome = 0 | Outcome = 1 | Overlap |
|---|---|---|---|
| **Body_Temperature** | 36.0 – 37.6 °C | 38.1 – 40.6 °C | **None** |
| **Fever_Duration** | 0 – 3 days | 3 – 7 days | **1 value** |
| **Platelet_Count** | 117,931 – 362,426 | 26,885 – 119,280 | 1,349 units |
| **WBC_Count** | 3,256 – 10,229 | 1,740 – 4,070 | 814 units |

Across 1,000 patients, not one dengue-negative febrile patient has a temperature above 37.6 °C. Real triage data does not look like this; generated data does. This is why `results/table_clinical_tier_1_symptoms.csv` reports ROC-AUC = 1.000 for all six models — the table is a readout of the generator, not a finding.

Your own literature review states it correctly: *"No publicly indexed Bangladeshi dataset identified in this review … records exact date of symptom onset."* That includes this one. The H1 hypothesis cannot be tested with data in hand.

---

### [Blocker F2] Even the original serology dataset is degenerate — the label is the IgG column
In `Not Used/dataset (2).csv`, `Outcome == IgG` for 1,000 of 1,000 rows. The "diagnosis" is not a diagnosis; it is one of the predictors copied into the target.

```text
NS1  IgM  IgG | Outcome=0   Outcome=1
 0    0    0  |     251           0
 0    1    0  |     216           0     <-- IgM-positive, labelled negative
 0    0    1  |       0          13
 1    0    1  |       0         261
 1    1    1  |       0         258
```

Any model given serology returns AUC 1.0 trivially. Remove serology and only demographics remain. There is no honest classification task in this file at all — which means both versions of the clinical dataset are dead, not just the fabricated one.

---

### [Blocker F3] None of the repo's headline numbers are reproducible from anything committed
All five pipeline notebooks contain zero executed output cells. `notebooks/ml-notebook1.ipynb`, committed as "executed master Kaggle notebook with outputs", is 0 bytes. The repository contains no district-level data, no R-INLA dependency, and no BYM2 fit that has ever run.

So these README claims currently have no evidence behind them:
* $\Delta\text{AUC} = +0.4086$ optimism gap
* Humidity $\text{RR} = 1.313$
* $\text{DIC}$ reduction of $2,553.6$
* $1,266$ district-weeks across 64 districts
* $n = 2,523$ patients
* $22.4\%$ cross-scale error reduction
* Barishal $\zeta = 2.10$

Some are also internally impossible: the local pipeline never leaves 8 divisions, and the 2×2 matrix in §04 below — which I actually ran on your data — gives a gap of 0.060, not 0.409. Publishing 0.409 would be indefensible under review; publishing 0.060 is a legitimate, if smaller, contribution.

---

### [Fix F4] The ML forecaster is handicapped against its own baseline
In `src/feature_engineering.py` the autoregressive features begin at `cases_lag_7d`, while the persistence baseline in `forecasting_models.py` predicts using cases at time $t$. For a $t+7$ target the ML models are effectively forecasting 14 days out against a 7-day baseline. That is a large part of why persistence wins in `table_macro_rolling_origin_forecasting.csv` — the comparison was never fair. It has to be corrected before "ML loses to persistence" can be claimed as a finding.

---

### [Fix F5 · F6] Two smaller defects worth naming
* **F5**: `Vital_Index = (Body_Temperature − 98.6) × (Fever_Duration + 1)` subtracts a Fahrenheit constant from Celsius readings, so the index is negative for every patient. Moot once F1 is resolved, but it is the kind of thing a reviewer finds.
* **F6**: `early_warning_matrix.py` returns a hand-written table of alert thresholds. It is a reasonable operational translation layer, but it is authored prose, not a model output, and must never be presented as a result.

---

## 03. Assets
*What survives, and why it is enough.*

### What you actually have

#### [Keep F7] `Datasets/Dataset (1).csv` — Real DGHS surveillance, and your entire paper
11,200 daily records: 8 divisions × 1,400 days, 1 Jan 2022 → 31 Oct 2025, each row carrying same-day max/min temperature, rainfall and humidity. Two independent checks say it is genuine:

| Annual National Totals | This File | Published DGHS |
|---|:---:|:---:|
| **2022** | 61,227 | ~62,400 |
| **2023** | 320,644 | ~321,200 |
| **2024** | 101,844 | ~101,200 |
| **2025 (to 31 Oct)** | 69,777 | in-season |

```text
Cross-check vs Not Used/dengu dataset.csv (independent source):
  3,368 overlapping division-days — 98.4% exact agreement — r = 0.9986
```

Four dengue seasons, including the record 2023 epidemic and a near-complete 2025 season. **2025 is the novelty window** — Shiddik et al. stop at 2024, Rahman et al. at 2021, Liu et al. at 2023. Weekly aggregation gives 201 weeks × 8 divisions = 1,608 division-weeks, with 22.9% zero-weeks (against 42.3% zero-days) — enough signal to model, sparse enough that the count distribution is a genuine methodological problem worth writing about.

#### [Keep, Demoted F8] The Jamalpur hematology dataset is real
`Dengue Fever Hematological Dataset.csv`, $n = 1,523$, 19 CBC parameters, 68.4% positive. Your own honest result — $\text{AUC} \approx 0.684$, sensitivity 0.95, specificity 0.39 — is entirely plausible for CBC-only dengue screening, and stands in useful contrast to the fabricated file's 1.000. Not strong enough to headline a paper. Two options: cut it, or keep it as a half-page supporting result framed as cost-sensitive rural triage (high sensitivity at a stated false-positive cost) rather than as a diagnostic claim. **My recommendation:** cut it, and spend the page on calibration instead — see §05.

#### [Keep F9] The literature review is good, and already contains the paper's best idea
Section VIII cites the 2026 SpatialEpiBench result: under standardized rolling-origin evaluation, sophisticated spatial and ML methods beat a naive persistence baseline less than half the time. Your Section X then concludes that credible Bangladeshi claims require rolling-origin plus a naive comparator — *"a standard that a significant portion of existing research on machine learning in Bangladesh has yet to reach."*

**That sentence is your paper.** You wrote the correct thesis in the review and then built a framework that does not test it. Build the one that does.

Two prose issues before this ships: several passages read as paraphrase-of-paraphrase (*"It seems like something is coming back after a long time, about 20 years"*), and a few sentences restate their own preceding sentence. Both will trip a similarity check, and both read as machine-generated. Rewrite in your own voice at the LaTeX stage.

---

## 04. Pilot
*The reframe, already tested on your data.*

### Evidence that the new paper works
I did not want to hand you a plan built on hope, so I ran the core experiment on `Dataset (1).csv` before writing this. The numbers below are real pilot output from your data under the honest protocol. They are what the paper would report, subject to the tightening in §06.

#### Protocol
Divisional weekly panel (weeks ending Saturday), 8 divisions × 201 weeks. Expanding rolling origin: train $\le 2023 \to$ test 2024; train $\le 2024 \to$ test 2025. Metrics pooled across both test seasons. Features are scale-free by construction — log-growth deltas, seasonal harmonics, and climate lags at 2–12 weeks.

---

### Result 1 — Level-space ML does not reliably beat persistence

#### Table A · Weekly divisional forecast error, pooled over the 2024 and 2025 test seasons

| Horizon | Model | MAE | RMSE | wMAPE % | Skill vs Persistence |
|---|---|:---:|:---:|:---:|:---:|
| **1 week** | Persistence | 41.4 | 105.2 | 18.2 | — |
| | LightGBM, level target | 43.5 | 116.1 | 19.1 | −5.1% |
| | **LightGBM, anchored growth** | **37.0** | **96.8** | **16.3** | **+10.6%** |
| **2 weeks** | Persistence | 66.6 | 184.8 | 29.0 | — |
| | LightGBM, level target | 64.5 | 167.3 | 28.1 | +3.1% |
| | **LightGBM, anchored growth** | **60.7** | **171.8** | **26.5** | **+8.8%** |
| **3 weeks** | Persistence | 92.9 | 262.7 | 40.1 | — |
| | LightGBM, level target | 91.7 | 253.6 | 39.5 | +1.3% |
| | **LightGBM, anchored growth** | **85.6** | **246.7** | **36.9** | **+7.9%** |
| **4 weeks** | Persistence | 119.6 | 335.1 | 51.1 | — |
| | LightGBM, level target | 111.1 | 298.3 | 47.4 | +7.1% |
| | **LightGBM, anchored growth** | **99.0** | **257.9** | **42.3** | **+17.3%** |

Seasonal-naive (same week last year) is catastrophic here — MAE 345–359 at every horizon — because 2023 was roughly 3× every neighbouring season. That is itself a finding: Bangladesh's post-2022 regime breaks the year-over-year baselines that dengue forecasting normally leans on.

The mechanism is simple and worth a paragraph in the paper. A level-space learner trained on 2022–2023 has learned the magnitude of the 2023 epidemic and carries it into 2024–2025, which were far smaller. Predicting the log growth ratio and anchoring it on the last observed count removes scale from the learning problem entirely; the model only has to learn direction and rate, which transfer across regimes. Cheap, principled, and it wins at every horizon.

---

### Result 2 — The intervals are badly over-confident, and conformal calibration repairs them

#### Table B · Empirical coverage of nominal 90% prediction intervals, prospective

| Horizon | Quantile LGBM | + Split-Conformal | Median Width, Raw | Median Width, Conformal |
|---|:---:|:---:|:---:|:---:|
| **1 week** | 0.738 | **0.837** | 32 | 41 |
| **2 weeks** | 0.737 | **0.852** | 40 | 51 |
| **4 weeks** | 0.697 | **0.921** | 53 | 109 |

A 90% interval that contains the truth 70% of the time is worse than no interval — it tells a health directorate a surge is impossible when it is merely unlikely. Split-conformal calibration on held-out log-ratio residuals (the last 26 weeks of each training window) moves coverage to 0.84–0.92 for a modest width penalty. The residual under-coverage at short horizons is real and should be reported, not hidden: conformal prediction assumes exchangeability, and epidemic onset weeks violate it. Naming that limitation, and pointing at adaptive conformal as the fix, is stronger than pretending it is solved.

No Bangladeshi dengue forecasting paper in your review reports interval coverage at all. This is the cleanest novelty available to you, and it costs about forty lines of code.

---

### Result 3 — The honest optimism gap, re-derived

#### Table C · Alarm classification at $h = 2$ weeks, four validation regimes, identical model and features

| Condition | What It Holds Out | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|
| **C1 · Random 80/20** | Nothing meaningful | 0.9750 | 0.9567 |
| **C2 · Rolling Origin** | Future seasons | 0.9294 | 0.8876 |
| **C3 · Leave-Division-Out** | Unseen geography | 0.9493 | 0.9122 |
| **C4 · Space + Time** | Both | **0.9151** | **0.8491** |

$$\text{Optimism Gap} = \mathbf{0.0599 \text{ ROC-AUC}}, \quad \mathbf{0.1076 \text{ PR-AUC}}$$

Report exactly this. It is smaller than the repo's claimed 0.4086, but it is true — and PR-AUC, the metric that matters for a rare-alarm task, degrades nearly twice as fast as ROC-AUC, which is a genuinely useful message for this venue's audience. Note also that C4 at 0.915 is good: the honest framing does not require you to conclude that everything fails.

---

## 05. The Paper
*Title, thesis, contributions, page budget.*

### The Paper to Write
* **Working Title:** *Persistence-Anchored Growth Forecasting with Calibrated Uncertainty for Dengue Early Warning in Bangladesh, 2022–2025*
* **Core Thesis:** Under honest rolling-origin evaluation on Bangladesh's post-2022 dengue regime, level-space machine learning does not reliably beat a one-week persistence baseline. Reparameterising the target as persistence-anchored log growth recovers real skill at every horizon from one to four weeks, and conformal calibration repairs prediction intervals that are otherwise dangerously over-confident.

### Four Major Contributions
1. **First rolling-origin benchmark on the 2022–2025 Bangladeshi divisional series, including the 2025 season.**  
   Persistence and seasonal-naive baselines are mandatory comparators, not afterthoughts. Extends every prior Bangladeshi forecasting study by at least one full epidemic season.
2. **Persistence-anchored log-growth reparameterisation.**  
   $+7.9\%$ to $+17.3\%$ MAE reduction against persistence across $h = 1\text{–}4$ weeks, where the conventional level-target formulation delivers $-5.1\%$ to $+7.1\%$. Directly addresses the SpatialEpiBench caution on a national dataset.
3. **A prospective calibration audit of dengue forecast uncertainty in Bangladesh.**  
   Nominal 90% intervals achieve 70–74% coverage; split-conformal calibration restores 84–92%. No prior study in this literature reports coverage at all.
4. **Honest quantification of the optimism gap, plus operational alarm evaluation.**  
   A 2×2 space–time validation matrix on identical features, and alarm lead-time / false-alarm-rate curves at fixed sensitivity — the form in which DGHS would actually consume the output.

---

### Six-Page IEEE Budget

| Section | Content | Target Pages |
|---|---|:---:|
| **I. Introduction** | Post-2022 regime shift; the validation problem; four contributions | 0.75 |
| **II. Related Work** | Bangladeshi forecasting (Shiddik, Rahman, Liu); rolling-origin and SpatialEpiBench; conformal prediction. Gap stated in two sentences. | 0.75 |
| **III. Data & Problem** | Divisional panel, provenance and DGHS cross-validation, zero-inflation, formal forecast statement | 0.75 |
| **IV. Method** | Anchored-growth reparameterisation; feature set; conformal calibration; alarm rule; validation matrix | 1.25 |
| **V. Results** | Tables 1–4, Figures 2–4 | 1.75 |
| **VI. Discussion + VII. Conclusion** | Operational reading, limitations stated plainly, references | 0.75 |

> **Action Item:** Verify the page limit on CMT before drafting — the current ICEEICT site does not publish it. Prior editions ran 6 pages in IEEEtran two-column with over-length charges. Write to 6 and be ready to cut to 5.

---

### Deliverables Inventory
* **Tables:**
  * **T1**: Panel description & DGHS cross-validation
  * **T2**: Skill vs persistence, $h = 1\text{–}4$ (Table A)
  * **T3**: Interval coverage & width (Table B)
  * **T4**: Optimism-gap matrix (Table C)
* **Figures (300 DPI, single-column IEEE):**
  * **F1**: Pipeline & rolling-origin schematic
  * **F2**: National weekly cases 2022–25, test folds shaded
  * **F3**: Forecast vs actual with conformal bands, Dhaka + one low-burden division
  * **F4**: Reliability diagram: nominal vs empirical coverage, before and after
  * **F5**: Alarm lead time vs false-alarm rate by division

---

## 06. Build
*Modules, experiments, and what to delete.*

### Target Architecture (After the Cut)
```text
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
```

### Components to Delete / Quarantine
* `notebooks/` (all five, never executed)
* `src/clinical_models.py`, `src/early_warning_matrix.py`
* `src/explainability.py` (rebuild for the macro model only)
* `Datasets/dataset.csv` (fabricated)
* `Datasets/Not Used/dataset (2).csv` (degenerate label)
* `results/table_clinical_*.csv`
* `figures/fig2_biomarker_kinetics_curve.png`

---

### Experiments Roadmap

| ID | Experiment | Produces | Est. Time |
|---|---|---|:---:|
| **E1** | Build the weekly panel; cross-validate against independent 2019–23 source; document zero-inflation and 2023 regime break | T1, F2 | 1 day |
| **E2** | Baseline suite: persistence, seasonal-naive, ARIMA/ETS on log counts. Fix F4 first — give every learner lag-0 access so comparison is fair. | T2 rows | 1 day |
| **E3** | Level vs anchored-growth targets, $h = 1\text{–}4$, three learner families, rolling origin over 2024 and 2025 | T2, F3 | 1.5 days |
| **E4** | Quantile intervals $\to$ split-conformal $\to$ adaptive conformal; coverage, width, PIT histogram | T3, F4 | 1.5 days |
| **E5** | 2×2 validation matrix; alarm lead time and false-alarm rate at fixed sensitivity | T4, F5 | 1 day |
| **E6** | Ablations: climate lags on/off, seasonality on/off, per-division breakdown; block-bootstrap CIs on every headline delta | Robustness ¶ | 1 day |

### Three Non-Negotiables
1. **Uncertainty on every headline number.** Two test seasons is a small sample. Report block-bootstrap confidence intervals on the skill deltas, or a reviewer will ask whether $+7.9\%$ is noise. If an interval crosses zero, say so.
2. **One seeded command reproduces everything.** `python run_all.py` regenerates every table and figure. Commit notebook outputs this time, or do not commit notebooks.
3. **Every number in the manuscript traces to a file in `results/`.** The failure mode that produced F3 was writing prose ahead of runs. Invert it: run first, then quote.

#### Optional (Only if you finish early)
Splice the 2019 season from `Not Used/dengu dataset.csv` as an out-of-regime stress test — train on 2022–2025, forecast the 2019 epidemic, and show whether anchored growth transfers backwards. Coverage there is partial (127 of 365 days in 2019), so treat it as a sensitivity analysis, never as a main result. Skip the 64-district and BYM2/INLA ambitions entirely: you do not have district data, and reaching for them is what produced the unbacked claims in the first place.

---

## 07. Schedule
*22 days. Two hard gates.*

The binding constraint is 22 days. The plan is scoped so a working paper exists by 21 September, and the last five days are review rather than rescue.

```text
Timeline:
Sep 4–6   [Fri–Sun]  Demolition & Foundation (delete clinical, retract claims, build panel.py/features.py, run E1 -> T1, F2)
Sep 7–10  [Mon–Thu]  Core Result (E2 & E3 level-vs-anchored comparison -> T2, F3 final; draft Sec III & IV)
Sep 10    [GATE 1]   Go / No-Go: If anchored growth beats persistence at >=3 horizons with bootstrap CIs > 0 -> proceed; else pivot to calibration
Sep 11–14 [Fri–Mon]  Calibration, Alarms, Robustness (E4 -> T3, F4; E5 -> T4, F5; E6 bootstrap CIs; draft Sec V)
Sep 15–16 [Tue–Wed]  FREEZE: Results freeze; regenerate all 300 DPI figures
Sep 17–21 [Thu–Mon]  Full LaTeX Draft (IEEEtran 6 pages; rewrite lit review in own voice; abstract last)
Sep 22–24 [GATE 2]   Adversarial Review (Supervisor + outside reader; trace numbers to results/; CrossCheck similarity <15%)
Sep 25    [Fri]      SUBMIT: PDF eXpress check, CMT submission 1 full day early
```

---

## 08. Risk
*What reviewers will say, and the retraction list.*

### Anticipated Objections & Rebuttals

| Objection | Your Answer |
|---|---|
| *"Eight divisions is coarse. Why not districts?"* | State it plainly in Limitations: divisional resolution is what public daily surveillance provides for 2022–2025. The methodological claim — target parameterisation and calibration — is resolution-independent, and you say so. |
| *"Only two test seasons."* | Pre-empted by E6: block-bootstrap intervals on every delta, plus a per-division breakdown showing the effect is not driven by Dhaka alone. |
| *"Log-ratio anchoring is not new."* | Correct, and do not claim otherwise. The contribution is the demonstration on a national dengue system, against honest baselines, with the calibration audit attached. Cite the forecasting literature that already uses it and position as transfer plus evidence. |
| *"Where is the deep learning?"* | Answer with the SpatialEpiBench result and your own C1-vs-C4 gap. Add one LSTM or ConvLSTM row to Table 2 if space allows — it is cheap and it forecloses the question. |
| *"Why no BYM2 / spatial model?"* | Eight areal units cannot identify a structured spatial prior. Say that in one sentence; it is a stronger answer than a fitted model would be. |
| *"Climate variables barely help."* | If the E6 ablation shows that, report it. A negative result about climate features under honest validation is more interesting to this venue than another confirmation. |

---

### Retraction Checklist
*Do this before the repo is read by anyone outside the team. Every item below is currently public and unsupported.*

1. **README:** Remove $\Delta\text{AUC} = +0.4086$, $\Delta\text{RMSE} = +610.5$, the $4.5\times$ error multiplier, and the whole Condition 1–4 table. Replace with §04 Table C once re-run.
2. **README:** Remove the Bayesian section — humidity $\text{RR} = 1.313$, $\text{DIC} -2,553.6$, Barishal $\zeta = 2.10$. No BYM2 model was ever fitted.
3. **README:** Remove "$n = 2,523$ patients", the $C_0\text{–}C_3$ ladder results, "1,266 district-weeks", "64 districts", and the $22.4\%$ cross-scale figure.
4. **README:** Remove the "100% Empirical" badge and the "zero synthetic imputation" line. Given F1, they are the opposite of true.
5. **Delete `notebooks/ml-notebook1.ipynb`** (0 bytes) and correct the commit message claiming it contains outputs.
6. **Delete or clearly quarantine `Datasets/dataset.csv`** with a README note explaining why, so nobody on the team reuses it.
7. **Update `dengue_bangladesh_paper.md`, `dengue_bangladesh_methodology.md`** and the five audit files — or archive them under `deprecated/`.

---

### The One Thing to Take Away
> **You do not have a weak project. You have a strong dataset attached to claims it cannot support.**  
> The 2022–2025 divisional panel is real, it reproduces official totals, it covers a season nobody has published on, and the honest result it yields is genuinely interesting: the standard approach loses to a baseline, a simple reparameterisation fixes it, and everyone's confidence intervals are wrong. That is a better ICEEICT paper than a dual-scale framework with a perfect AUC — and it is the one you can finish by the 25th.

---
*Audit performed 4 Sep 2026 against `Raf1-alam/research_on_dengu` @ `2684901` and `E:\3-2\Machine Learning Lab`.*  
*Pilot results in section 4 were computed directly from `Datasets/Dataset (1).csv` under the protocol described there. They are indicative, not final, and must be regenerated by `run_all.py` before use in the manuscript.*  
*Conference details from `iceeict.mist.ac.bd` as of 4 Sep 2026 — confirm the page limit and template version on CMT before drafting.*
