# Implementation Plan v2 — ICEEICT 2027

**Target:** 7th ICEEICT, MIST Dhaka, 28–30 Jan 2027 · IEEE Xplore + Scopus · Conf. ID 72922 · CMT
**Deadline:** 26 Sep 2026 (extended) — **22 days from 4 Sep 2026**
**Status:** v2, 4 Sep 2026. Supersedes v1 (archived at `unused/legacy_docs/ICEEICT_plan_v1_2026-09-04.md`).
**Web version:** https://claude.ai/code/artifact/1f833a56-42f1-4507-a6c2-591fbd5e44aa
**Dataset provenance:** [`DATASET_DOSSIER.md`](DATASET_DOSSIER.md)

---

## 1. What changed since v1

v1 said: cut the clinical arm, keep the divisional forecast, compete on honesty. The first and third still hold. The second was upgraded, and the specific honest claim moved.

**Done — the repository has been restructured.** Verified data under `data/raw/` in four labelled folders. The August pipeline, its outputs, the dual-scale documents and both contaminated CSVs are quarantined in `unused/` with written reasons. Nothing deleted. `src/` is empty with a README naming the eight modules to write.

**New — the divisional panel is now the ablation, not the dataset.** The figshare district panel gives 64 districts × 254 weeks with climate, satellite indices, Google Trends and a mobility graph, reconciling to published DGHS totals within 0.6% / 0.1% / 0.2% for 2022 / 2023 / 2024, and extending into 2026.

**Moved — v1's headline does not survive at district resolution.** v1's thesis was that level-space ML loses to persistence and that anchoring the target fixes it. Re-run on the district panel: anchoring still wins, **but so does the plain level model now** (+7.4% to +19.5% over persistence, where at 8 divisions it *lost* 5.1% at one week).

Building a paper on "ML loses to persistence" would be overclaiming from one resolution. But the reason it moved is itself a better finding: **the benefit of persistence-anchoring depends on how much data and how many spatial units you have.** That can only be shown by running both panels — and you have both.

### Revised recommendation

1. **Lead with calibration.** Cleanest, most novel, most operationally serious; it *strengthened* at district resolution, and no paper in this literature reports it. Prospective 90% intervals cover 77–79%; conformal restores 89–96%.
2. **Support with the resolution-dependence ablation.** Two panels, one method, one honest conclusion about when target parameterisation matters.
3. **Bank Cox's Bazar.** 35,581 patients with real symptom-onset dates is the stronger scientific find, and that is exactly why it should not be compressed into six pages with three weeks left. Paper 2, and the Gate 1 fallback.

---

## 2. Pilot evidence, at both resolutions

Expanding rolling origin: train ≤2023 → test 2024; train ≤2024 → test 2025, pooled. Season-blocked so lags never bridge the 2020–21 gap. Same features, same learner, same protocol at both resolutions — only the spatial unit differs.

### Result 1 — calibration (the headline)

Empirical coverage of nominal 90% prediction intervals, prospective:

| Horizon | Districts, raw | Districts, conformal | Divisions, raw | Divisions, conformal |
|---|---:|---:|---:|---:|
| 1 week | **0.783** | **0.893** | 0.738 | 0.837 |
| 2 weeks | **0.787** | **0.914** | 0.737 | 0.852 |
| 4 weeks | **0.766** | 0.955 | 0.697 | 0.921 |

A stated 90% interval that holds 77% of the time tells a health directorate a surge is impossible when it is merely unlikely. Split-conformal on held-out log-ratio residuals fixes it, landing almost exactly on nominal at h=1,2 at district resolution. Over-coverage at h=4 (0.955, median width 26 cases vs 11 raw) is the honest cost — report it.

Report the residual gap too: conformal assumes exchangeability, epidemic onset weeks violate it, which is why h=1 lands at 0.893 not 0.900. Naming the limitation and pointing at adaptive conformal beats pretending it's solved.

### Result 2 — skill depends on resolution

MAE reduction against persistence:

| Horizon | Districts, level | Districts, anchored | Divisions, level | Divisions, anchored |
|---|---:|---:|---:|---:|
| 1 week | +7.4% | **+11.0%** | **−5.1%** | +10.6% |
| 2 weeks | +12.9% | **+18.0%** | +3.1% | +8.8% |
| 3 weeks | +16.7% | **+19.0%** | +1.3% | +7.9% |
| 4 weeks | **+19.5%** | +19.2% | +7.1% | +17.3% |

At 8 divisions, anchoring is the difference between beating persistence and not. At 64 districts — 5× the training rows, and case magnitudes far more homogeneous across units — the level model recovers and the anchoring advantage narrows from 15.7 points at h=1 to nothing at h=4.

Mechanism, worth a paragraph: a level-space learner trained across 2022–2023 absorbs the *magnitude* of the 2023 epidemic and carries it into much smaller seasons. That error is proportionally devastating when 8 coarse units each span three orders of magnitude, and much milder when 64 finer units are more alike. **Persistence-anchoring is a small-sample, heterogeneous-scale correction, not a universal improvement.**

One more trade-off for Discussion:

```
anchored wins MAE and wMAPE at every horizon,
but LEVEL wins RMSE at h=3 (51.6 vs 55.6) and h=4 (51.4 vs 67.1).

L1-fitted anchored growth tracks the typical week;
Tweedie-fitted level tracks the peak. Peaks are what a surge alert is for.
```

### Result 3 — the optimism gap at district resolution

Alarm classification at h=2 wk, identical model and features, 13,952 district-weeks, 32.1% base rate:

| Condition | Holds out | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| C1 · Random 80/20 | nothing meaningful | 0.9742 | 0.9512 |
| C2 · Rolling origin | future seasons | 0.9515 | 0.9251 |
| C3 · Leave-division-out | unseen geography | 0.9609 | 0.9198 |
| **C4 · Space + time** | both | **0.9355** | **0.8897** |

Gap = **0.0387 ROC-AUC, 0.0615 PR-AUC** — smaller than divisional (0.0599 / 0.1076) and far smaller than the 0.4086 the old README claimed. Report the district number as primary with the divisional one alongside: the gap *shrinking* as you add data and spatial units is consistent with Result 2 and reinforces it. PR-AUC still degrades ~1.6× faster than ROC-AUC.

---

## 3. The paper

**Working title:** *Calibrated Dengue Forecasting for Bangladesh: Conformal Prediction Intervals and the Resolution-Dependence of Forecast Skill, 2019–2026*

**Thesis:** Machine-learning dengue forecasts for Bangladesh are reported without prediction intervals, and the intervals they would produce are badly over-confident: nominal 90% bounds hold 77–79% under prospective evaluation. Split-conformal calibration restores near-nominal coverage. Point-forecast skill, meanwhile, is not a property of the model alone — it depends on spatial resolution, and the target parameterisation that rescues a coarse eight-division forecast is nearly redundant at 64 districts.

**Contributions**

1. **A prospective calibration audit of dengue forecast uncertainty in Bangladesh, and a conformal fix.** Nominal 90% intervals achieve 0.766–0.787 coverage at district resolution; split-CQR on log-ratio residuals restores 0.893–0.955. No study in this literature reports coverage at all.
2. **Evidence that forecast skill is resolution-dependent, from one method on two panels.** Anchored growth gains +11.0% to +19.2% at 64 districts and +7.9% to +17.3% at 8 divisions — but the conventional level target ranges from −5.1% to +7.1% at divisions and +7.4% to +19.5% at districts.
3. **An honest 2×2 space–time optimism gap, measured twice.** 0.0387 / 0.0615 at districts, 0.0599 / 0.1076 at divisions, on identical features.
4. **The first rolling-origin benchmark on Bangladeshi surveillance through 2026.** Persistence and seasonal-naive as mandatory comparators. Extends Shiddik et al. (to 2024), Liu et al. (to 2023) and Rahman et al. (to 2021) by at least two epidemic seasons.

**Six-page IEEE budget**

| Section | Content | Pages |
|---|---|---:|
| I. Introduction | Post-2022 regime shift; forecasts without uncertainty; four contributions | 0.75 |
| II. Related work | Shiddik, Rahman, Liu, Sarker; rolling-origin & SpatialEpiBench; conformal prediction | 0.75 |
| III. Data | District panel + provenance and DGHS reconciliation; divisional panel; OpenDengue as independent context; formal forecast statement | 0.75 |
| IV. Method | Anchored-growth reparameterisation; scale-free features; split-CQR; alarm rule; 2×2 matrix | 1.25 |
| V. Results | Tables 1–4, Figures 2–4 | 1.75 |
| VI–VII. Discussion & Conclusion | Operational reading, limitations, references | 0.75 |

*Verify the page limit on CMT before drafting — the ICEEICT site does not publish it. Prior editions ran 6 pages IEEEtran two-column with over-length charges. Write to 6, be ready to cut to 5.*

**Tables:** T1 both panels + DGHS reconciliation · T2 interval coverage and width, both resolutions · T3 skill vs persistence, 2 targets × 2 resolutions · T4 optimism-gap matrix

**Figures (300 dpi, single-column):** F1 pipeline & rolling-origin schematic · F2 national weekly cases 2019–2026, test folds shaded, 2020–21 gap marked · F3 forecast vs actual with conformal bands (Dhaka + one low-burden district) · F4 reliability diagram, before/after, both resolutions · F5 alarm lead time vs false-alarm rate by division

---

## 4. Implementation

### Repository as restructured

```
data/
  raw/
    district_panel/   district_week_panel.csv (16,256 x 54), districts.csv,
                      district_population.csv, edges_static.csv,
                      edges_dynamic_weekly.csv, SOURCE_README.md    [PRIMARY]
    coxsbazar/        coxsbazar_dengue_2021_2024.xlsx (35,581 x 84) [PAPER 2]
    divisional/       divisional_daily_2022_2025.csv,
                      divisional_daily_2019_2023.csv                [ABLATION]
    clinical/         jamalpur_hematology_n1523.csv                 [OPTIONAL]
  processed/          built by src/panel.py
src/                  empty; see src/README.md for the eight modules
results/  figures/    regenerated by run_all.py
docs/                 this plan, dataset dossier, literature review
unused/               contaminated_datasets/ legacy_src/ legacy_results/
                      legacy_figures/ legacy_docs/ course_admin/
```

**Two cautions on the district panel.** It ships a pre-computed `ignition` target built for spatial-spillover forecasting — **do not inherit it**; use `new_cases` and `incidence` and define your own threshold, or you inherit someone else's research question. And its README claims the 2019 season matches the official 101,354 total; the file sums to 30,257 because rows begin 26 August. Model on 2022–2026 and treat 2019 as a partial-season extra.

### Experiments

| ID | Experiment | Produces | Est. |
|---|---|---|---:|
| E1 | Build both panels in `panel.py`; season-block; reconcile national totals; document the 2020–21 gap and zero-inflation | T1, F2 | 1 d |
| E2 | Baselines: persistence, seasonal-naive, ARIMA/ETS on log counts. *Every learner gets lag-0 access* — defect F4 in the old pipeline was that they did not | T3 rows | 1 d |
| E3 | Level vs anchored targets × both resolutions × h = 1–4. Report MAE, RMSE and wMAPE separately — they disagree, and that is Result 2's trade-off | T3, F3 | 1.5 d |
| **E4** | **Priority.** Quantile → split-conformal → adaptive conformal; coverage, width, PIT histogram, reliability diagram, both resolutions | T2, F4 | 1.5 d |
| E5 | 2×2 validation matrix at both resolutions; alarm lead time and FAR at fixed sensitivity | T4, F5 | 1 d |
| E6 | Ablations: climate / satellite / Google Trends / mobility on-off, per-district breakdown; block-bootstrap CIs on every headline delta | robustness ¶ | 1 d |

E6 now has real teeth. The district panel carries NDVI, NDWI, LST, VIIRS night lights, Google Trends and a mobility graph — none of which the divisional data had. If the ablation shows satellite and mobility layers add little once case history and climate are in, **report that**. A negative result about fashionable covariates under honest validation is more interesting to this venue than another confirmation.

### Three non-negotiables

- **Uncertainty on every headline number.** Two test seasons is a small sample. Block-bootstrap CIs on skill deltas; if an interval crosses zero, say so. This matters more now that some deltas are only 3–4 points apart.
- **One seeded command reproduces everything.** `python run_all.py` regenerates every table and figure.
- **Every number in the manuscript traces to a file in `results/`.** The failure that produced the old README was writing prose ahead of runs. Invert it.

---

## 5. Schedule — 22 days, two gates

| Dates | Phase | Work |
|---|---|---|
| **Sep 4** ✅ | Audit, dataset search, restructure | Clinical arm cut, contaminated files quarantined with reasons; district panel and Cox's Bazar located, downloaded, reconciled against DGHS; repository restructured; pilot re-run at both resolutions |
| **Sep 5–7** (Sat–Mon) | Foundation | Retract unbacked README claims (§6) before anyone outside the team reads the repo; write `panel.py` + `features.py`; build and freeze both panels; E1 → T1, F2 |
| **Sep 8–10** (Tue–Thu) | Calibration first | E4 before E2/E3 — it is the headline, so it gets the freshest hours; T2 and F4 final at both resolutions, adaptive-conformal variant attempted; draft Sections III–IV in parallel |
| **Sep 10** | **GATE 1** | Conformal coverage within ~3 points of nominal at h=1,2 with bootstrap intervals excluding raw-quantile coverage → proceed. If not → promote resolution-dependence (E3) to headline. If both fail → pivot to Cox's Bazar illness-day severity triage (35,581 patients, 10.5% Group B/C, self-contained, two weeks). Decide on the 10th, not the 20th. |
| **Sep 11–14** (Fri–Mon) | Skill, alarms, robustness | E2, E3 → T3, F3 · E5 → T4, F5 · E6 → bootstrap CIs and covariate ablations; draft Section V |
| **Sep 15–16** (Tue–Wed) | Freeze | **Results freeze** — no new experiments after the 16th; regenerate all figures at 300 dpi, IEEE column width, check print legibility |
| **Sep 17–21** (Thu–Mon) | Full LaTeX draft | IEEEtran, six sections, abstract last; rewrite lit-review prose in your own voice; references incl. Shiddik 2026, Rahman 2025, Liu 2025, Bhowmik 2026, Sarker 2024, SpatialEpiBench 2026, OpenDengue (Sci Data 2024), Romano et al. (CQR), Gibbs & Candès (adaptive conformal); data-availability statement citing both deposit DOIs and CC BY 4.0 |
| **Sep 22–24** (Tue–Thu) | **GATE 2 — adversarial review** | Supervisor + one outside reader; trace every number in the PDF to a file in `results/`; similarity check (CrossCheck — under 15%, no single source above 3%) |
| **Sep 25** (Fri) | Submit | PDF eXpress, CMT, authors final. Submit a day early. |

---

## 6. Risk register & retraction checklist

| Objection | Your answer |
|---|---|
| "You did not collect this data." | Correct, and Section III says so with both DOIs. Yours is the reconciliation against official totals, the two-resolution comparison, the calibration audit and the validation matrix. Note publicly that the 2019 rows do not match the deposit's stated total — catching that is evidence of care. |
| "Conformal prediction is not new." | It is not; do not claim it. The contribution is that nobody has measured dengue forecast interval coverage in Bangladesh, that it is 77–79% when stated as 90%, and that the fix is cheap. |
| "Only two test seasons." | E6: block-bootstrap intervals on every delta plus a per-district breakdown. With 64 units the per-unit breakdown is genuinely informative. |
| "Where is the deep learning?" | SpatialEpiBench plus your own C1-to-C4 gap. Add one LSTM row to Table 3 if space allows. |
| "Why no BYM2 / spatial Bayesian model?" | You now have 64 areal units and a Queen adjacency graph, so "cannot identify it" no longer holds. The honest answer is scope: the question is calibration and target parameterisation; BYM2 is future work. **Do not bolt one on in three weeks.** |
| "Your skill numbers disagree between panels." | That is the finding, not a flaw. Table 3 is built to show it. |

**Retraction checklist — still outstanding, all currently public in the GitHub repo**

- [ ] README: remove ΔAUC = +0.4086, ΔRMSE = +610.5, the 4.5× error multiplier and the Condition 1–4 table. Replace with §2 Result 3.
- [ ] README: remove the Bayesian section — humidity RR = 1.313, DIC −2,553.6, Barishal ζ = 2.10. No BYM2 model was ever fitted.
- [ ] README: remove "n = 2,523 patients", the C₀–C₃ ladder results, "1,266 district-weeks", the 22.4% cross-scale figure. You now genuinely have 64 districts and 16,256 district-weeks — which makes leaving the fabricated version up worse, not better.
- [ ] README: remove the "100% Empirical" badge and the "zero synthetic imputation" line.
- [ ] Delete `notebooks/ml-notebook1.ipynb` (0 bytes); correct the commit message claiming it holds outputs.
- [ ] Push the restructured tree including `unused/README.md`, so the quarantine reasoning is on the record.
- [ ] Consider a short note to the Mendeley depositors of `zdtc3n6xv2`. The `Outcome == IgG` defect is mirrored on Kaggle and Hugging Face under three DOIs and papers are being built on it.

---

**Where this leaves you.** This morning the project had a fabricated clinical dataset, a degenerate serology file, a README full of numbers no run had produced, and eight divisions of real surveillance. Tonight it has 16,256 verified district-weeks reconciled against DGHS, 35,581 patients with real symptom-onset dates banked for the next paper, a clean tree, and three measured results — a calibration failure and its fix, a resolution-dependence finding, and an honest optimism gap.

None of those results is spectacular. All of them are true, and all of them are things this literature has not reported. That is a sufficient ICEEICT paper, and it is finishable by the 25th.

---

*Pilot figures in §2 were computed from `data/raw/district_panel/` and `data/raw/divisional/` under the protocol stated there. Indicative, not final — regenerate via `run_all.py` before quoting in the manuscript.*
