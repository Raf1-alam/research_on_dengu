# Methodology

### Fever and Forecast: A Dual-Scale Framework Linking Clinical Dengue Diagnosis to Climate-Driven Outbreak Early Warning in Bangladesh

*Full methodology — expands §3–§5 of the research plan (dengue_bangladesh_research_plan.md), incorporating the literature review's methodological precedents*

---

## M1. Study Design Overview

This is an observational, retrospective-data study with a **prospectively-evaluated forecasting component**. It has two linked analytical arms plus a validation layer that sits above both:

| Arm | Unit of analysis | Outcome | Core question |
|---|---|---|---|
| **A. Individual diagnostic model** | Patient, at time of test | P(dengue \| markers, illness day) | RQ1 |
| **B. Population forecasting model** | District/upazila-week | Case count / outbreak status | RQ2 |
| **C. Linkage analysis** | District(s) with clinical coverage, week | Does A improve B locally? | RQ3 |
| **D. Explainability layer** | Applied to both A and B | Mechanism, not ranking | RQ4 |

All four arms are evaluated under the validation protocol in §M8, which is the study's central methodological contribution given that no prior Bangladesh dengue-ML study (including Shiddik et al., 2026, the current state of the art) has applied rolling-origin or spatial-holdout evaluation.

---

## M2. Data Sources and Variable Definitions

### M2.1 Population-scale panel

**Spatial unit:** District (64 units) as the primary/baseline resolution, matching Shiddik et al. (2026) and Sarker et al. (2024) for direct comparability. Upazila-level (~495 units) as a secondary, extended resolution if case-count sparsity (see M5.2) doesn't destabilize the spatial random-effects model — this is the resolution gap both Shiddik et al. (2026) and Salim et al.'s (2025) Yogyakarta precedent point to as the natural next step.

**Temporal unit:** Weekly, aggregated to monthly for comparability with Shiddik et al.'s (2026) monthly models where needed. Weekly is preferred as the primary resolution because it supports a more operationally meaningful 2–8 week forecast horizon (matching the lead times found necessary for vector-control response in the Singapore EWS literature).

**Panel construction — variables:**

| Variable | Source | Definition / notes |
|---|---|---|
| Case count (outcome) | DGHS Dengue Dashboard + archived press releases | Weekly confirmed cases per district; back-filled from Kaggle 2008–2022 panel for historical years where DGHS archives are incomplete |
| Mean/min/max temperature | NASA POWER API | Daily, aggregated to weekly mean, per district centroid |
| Rainfall | NASA POWER API | Daily accumulated, summed to weekly total |
| Relative humidity | NASA POWER API | Daily mean, aggregated to weekly mean |
| Lagged case count (own district) | Derived | Lags of 1, 2, 3, 4, 6, 8 weeks — lag selection informed by Shiddik et al.'s (2026) 1- and 1–3-period lag structure, extended further given the 2–8 week forecast horizon target |
| Lagged case count (neighboring districts) | Derived, using BBS adjacency | Queen-contiguity neighbor case counts, lag 1–4 weeks — tests spatial spillover, not just local climate |
| Population | BBS / World Bank | For incidence-rate normalization (cases per 100,000) |
| Socio-economic covariates | World Bank Open Data, WHO GHO | GDP proxy, poverty headcount, hospital beds per capita, health expenditure — same category as Shiddik et al. (2026); division-level, held constant within division if district-level unavailable |
| Land use / urbanization | Remote sensing (Sentinel-2, following Salim et al.'s 2025 approach) | Optional extension; supports upazila-level heterogeneity if pursued |
| Administrative geometry | HDX `bgd_adm_bbs_20201113_SHP` | Same shapefile used by Sarker et al. (2024), enabling direct adjacency-matrix and hotspot-map comparability |

**Threshold/label construction:** an "outbreak" binary label is derived **per district/upazila**, relative to that unit's own historical distribution (see §M5.3 for the full justification and the comparison protocol against Shiddik et al.'s pooled-percentile definition).

### M2.2 Individual-scale clinical data

**Plan A (preferred): hospital-sourced, illness-day-tracked data.**

| Variable | Definition |
|---|---|
| NS1 antigen result | Positive/negative, rapid test or ELISA |
| IgM result | Positive/negative |
| IgG result | Positive/negative |
| Day of illness at test | Days since self-reported symptom onset — **the variable absent from every existing public Bangladesh dataset, and the core justification for pursuing primary data** |
| Age, sex | Demographics |
| Hematology (if available) | Platelet count, WBC count, hematocrit, MCH, MCHC, MCV, RDW-CV — this exact feature set is validated as predictive in the Bangladeshi hematology-ML literature (Jamalpur CBC/XGBoost study; GJO-XGBoost benchmark study) even though those studies didn't have illness-day |
| Ground truth | RT-PCR result where available (gold standard), otherwise WHO clinical case definition |

**Plan B (contingency, triggered per the 4-week decision rule in the research plan §9): multi-disease febrile-illness differentiation.**

| Variable | Definition |
|---|---|
| NS1/IgM/IgG (dengue panel) | As above |
| Confirmatory test for co-circulating pathogens | Chikungunya IgM/RT-PCR, Zika RT-PCR — following the multiplex RT-PCR design used in the 2024 Dhaka febrile-cohort study (PMC12900302), which found chikungunya (31.2%) more prevalent than dengue (14.1%) in a real febrile population |
| Demographics, basic clinical presentation | Age, sex, fever duration, presenting symptoms |
| Outcome (multi-class) | Dengue / Chikungunya / Zika / co-infection / none — modeled as multinomial classification, following the differential-diagnosis template of the tropical-infections-differentiation study (dengue vs. malaria/leptospirosis/scrub typhus) |

---

## M3. Data Preprocessing Pipeline

**Population panel:**
1. Pull DGHS case data; cross-validate totals against independently published year-end figures (WHO DON, peer-reviewed outbreak summaries) to catch transcription/scraping errors before modeling.
2. Query NASA POWER for each district centroid (or upazila centroid, if extending resolution), full available date range.
3. Spatial join climate + case + shapefile + socio-economic layers on district/upazila code.
4. Construct lag features (§M2.1).
5. Missing data: for climate variables, NASA POWER has near-complete coverage so missingness should be minimal; for socio-economic covariates available only at division level, carry forward with an explicit flag variable rather than silently imputing at finer resolution than the data supports.
6. Multicollinearity check (variance inflation factor) across all covariates before model fitting — temperature/humidity/rainfall are often correlated and this must be checked and reported, not assumed away.

**Clinical data:**
1. De-identify at source (or immediately on receipt) per the ethics approval terms (§M11).
2. Standardize test result coding (positive/negative/equivocal — decide how equivocal results are handled and state this explicitly, since dropping them silently biases the sample toward clear-cut cases).
3. For Plan A: validate illness-day field for internal consistency (e.g., test date − reported symptom-onset date; flag and review outliers such as negative values or implausibly long delays).
4. For Plan B: harmonize case definitions across the three pathogens using the source hospital's diagnostic protocol, documented explicitly.

---

## M4. Individual-Scale Diagnostic Model (Arm A)

### M4.1 Illness-day stratification design

Bin definitions are grounded directly in the diagnostic-kinetics literature rather than chosen arbitrarily:

| Bin | Rationale |
|---|---|
| **Day 1–3** | NS1/RT-PCR-dominant window; foundational kinetic panel work reports NS1 sensitivity of 81.8–91.1% in the first seven days, with viremia (and therefore NS1/RT-PCR detectability) declining after this window |
| **Day 4–7** | The NS1-decline / IgM-rise crossover; IgM reported detectable from ~day 3 (42.9% positive) rising toward 100% by day 8; this is the window where single-marker models are expected to be least reliable and where combining markers should show the largest relative gain (H1) |
| **Day 8+** | IgM/IgG-dominant window; IgG reported detectable from ~day 5, reaching 100% by day 15; primary vs. secondary infection divergence becomes most relevant here (secondary infection accelerates both IgM and IgG response by several days) |

Report bin boundaries as a sensitivity analysis (e.g., 1–2/3–5/6+ as an alternative binning) rather than treating the chosen bins as the only valid specification.

### M4.2 Models — nested clinical ladder

Structure Arm A as a nested ladder, mirroring the population model's hierarchy (§M5.1), so each added feature's incremental contribution is isolated:

| Model | Inputs | Isolates |
|---|---|---|
| C0 | NS1 + IgM + IgG | Raw biomarker signal alone |
| C1 | C0 + illness day | Value of illness-day awareness |
| C2 | C1 + biomarker × illness-day interactions | Whether the interaction pattern (not illness day alone) carries the signal — this is where H1's predicted day 4–7 crossover effect should be most visible |
| C3 | C2 + hematology / demographics / symptoms | Value of routine clinical data beyond serology |

1. **Interpretable baseline (C0–C1):** logistic regression, `P(dengue) ~ NS1 + IgM + IgG + age + sex`, fit separately within each illness-day bin.
2. **Continuous illness-day model (C2):** `P(dengue) ~ NS1 + IgM + IgG + age + sex + f(illness_day)` with illness-day entered via a spline or as an interaction term with each marker, so the model produces a smooth, plottable diagnostic-probability curve over illness day rather than three disconnected bin-specific models — **this curve is the headline result for RQ1**.
3. **Gradient boosting:** XGBoost / LightGBM, same feature progression C0→C3, tuned via nested cross-validation (inner loop for hyperparameters, outer loop for the validation protocol in §M8.1).
4. **Small feed-forward network:** 1–2 hidden layers, following the Pakistan precedent (Qaiser et al., 2024) so architecture and reported metrics are directly comparable to that benchmark.
5. **C3 specifically:** the extended feature set (platelet, WBC, hematocrit, MCH, MCHC, MCV, RDW-CV) follows the Bangladeshi CBC-XGBoost precedent; report whether it adds predictive value beyond serology (C2) alone.
6. **Sensitivity check on imperfect gold standard:** if RT-PCR is unavailable for the full sample, consider a Bayesian latent-class model (following the approach used to re-estimate NS1/IgM/IgG accuracy without assuming a perfect reference test) as a secondary analysis, since pooled point-estimates that assume any single assay is a perfect gold standard are known to understate true marker sensitivity.

### M4.3 Plan B model (if triggered)

Multinomial classification (dengue / chikungunya / Zika / co-infection / none) using the same model family progression (logistic regression → gradient boosting → small neural network), following the differential-diagnosis modeling template from the tropical-infections-differentiation literature. No illness-day stratification is required here (H1 is dropped under Plan B), but age, sex, and presenting symptom profile become the primary discriminating covariates in place of illness-day.

### M4.4 External benchmarking

Report your model's performance against the **Pakistan study (Qaiser et al., 2024, *Advances in Virology*)** as the primary diagnostic benchmark: SVM best among six models (logistic regression, XGBoost, LightGBM, random forest, SVM, CatBoost), RT-PCR ground truth, n=300, NS1/IgM/IgG + hematology inputs — this is the closest published precedent for the actual task your model performs (dengue-positive vs. -negative from routine clinical presentation).

**Correction:** the Taiwan study (Huang, Tsai et al., 2020, *PLOS NTD*) is **not** a diagnostic benchmark — verified directly against the paper, it is a severity/prognosis model built on patients already confirmed dengue-positive (predicting which dengue patients progress to severe disease), not a dengue-vs-not-dengue classifier. Cite it only if you add a severity-prediction extension to Arm A, not as a comparator for diagnostic accuracy.

This benchmarking is itself part of the contribution, since no Bangladesh-specific number currently exists to compare against the Pakistan study.

---

## M5. Population-Scale Outbreak Forecasting Model (Arm B)

### M5.1 Baseline replication and nested model hierarchy

Replicate a simplified version of Shiddik et al.'s (2026) pipeline (XGBoost with `objective='count:poisson'`, an MLP, SHAP) on your own assembled panel, and treat this explicitly as a **control condition for comparability, not part of the contribution** — the actual scientific content is the validation design (§M8), the outbreak-definition fix (§M5.4), the model ladder below, and whichever of serotype/clinical-signal layers you're able to add on top.

Beyond this baseline, structure the full population model as a **nested hierarchy**, so the incremental value of each information layer is isolated and testable rather than only ever comparing full-complexity models against each other:

| Model | Inputs added | Isolates |
|---|---|---|
| P0 | Seasonal baseline (historical same-week pattern) only | The floor every model must beat |
| P1 | + lagged case counts (surveillance) | Value of surveillance history alone |
| P2 | + climate | Value of climate beyond surveillance |
| P3a | + geographic spatial structure (BYM2, adjacency-based, §M5.3) | Value of spatial dependence beyond climate |
| P3b | + connectivity-weighted spatial structure (e.g., Meta Social Connectedness Index in place of raw adjacency) | Whether social-connectivity-derived spatial coupling generalizes better than geographic adjacency — run only if connectivity data access/resolution is confirmed for Bangladesh, and report as an explicit comparison against P3a, never as a silent substitute. Do not treat SCI as literal observed mobility. |
| P4 | + serotype share | Value of pathogen surveillance — see priority note below |
| P5 | + local clinical signal (Arm C, §M7) | RQ3 — the full integrated model |

**Serotype (P4) — priority data-access question, not a footnote.** Actively pursue serotype data access (via the same IEDCR/hospital channel as the clinical-data request, §M2.2) rather than deciding after the fact it wasn't available. The Singapore literature (*Nature Communications*, 2025) shows climate-plus-serotype-competition models materially outperform climate-only models at 2–8 week horizons — precisely this study's target horizon — making serotype the highest-value optional layer here. Represent it as a simple share variable, `SerotypeShare_{k,t} = SequencedCases_{k,t} / TotalSequencedCases_t`, at whatever resolution the underlying sequencing genuinely supports (realistically national/quarterly, given how patchy published Bangladeshi serotyping coverage is) rather than forcing a district-week resolution the data can't back up.

Fit and validate **every level** under the full protocol in §M8, and report the complete ladder in results, not just P5. A ladder showing where each layer stops adding value (e.g., "P3a barely improves on P2") is itself a defensible, citable finding.

### M5.2 Distributional specification

Dengue case counts are non-negative, overdispersed, and — critically at finer spatial/temporal resolution — zero-inflated. A 2025 Bayesian-hybrid modeling study specifically addressing Bangladeshi dengue counts reports approximately 42% zero-case days in the daily 2022–2025 surveillance series and explicitly recommends zero-inflated negative binomial (ZINB) or endemic–epidemic (HHH) mixture frameworks over plain Poisson or standard negative binomial models for this reason. Specification:
- **Bayesian/INLA model:** Negative Binomial likelihood as default (matching Shiddik et al., 2026, who adopted NB after diagnosing overdispersion and zero-inflation in their own district-level data); fit a ZINB variant as a formal alternative and compare via WAIC/DIC, rather than assuming one is correct.
- **Gradient boosting:** Poisson objective, not squared error.
- **Structural-zero handling:** before running spatial leave-out folds (§M8.2) on low-incidence districts/upazilas, confirm the INLA/BYM2 specification's behavior on all-zero training folds — a naive Poisson/NB fit can produce degenerate estimates here, so this needs to be checked, not assumed.

### M5.3 Spatial-temporal model — full specification

Following the Bangladesh-specific precedent of Sarker et al. (2024) — who confirmed significant positive spatial autocorrelation across all 64 districts via Moran's I and Geary's C, then compared Poisson-Gamma, Poisson-Lognormal, CAR, Convolution, and BYM2 formulations — and the extension in Shiddik et al. (2026), who paired BYM2 with RW1/RW2 temporal structure:

**Model equation (district *i*, week *t*):**

```
Y_it ~ NegBinomial(μ_it, θ)
log(μ_it) = log(E_it) + β₀ + Xᵢₜβ + uᵢ + vᵢ + γₜ + δᵢₜ
```

Where:
- `E_it` = expected count offset (population-based, or historical-baseline-based — see §M5.4)
- `Xᵢₜβ` = fixed effects for climate, lagged cases, socio-economic covariates
- `uᵢ` = structured spatial random effect (ICAR component of BYM2, using the BBS adjacency matrix)
- `vᵢ` = unstructured spatial random effect (BYM2's iid component)
- `γₜ` = temporal random effect (RW1 or RW2, compare both)
- `δᵢₜ` = optional space-time interaction term, included if model comparison (WAIC) favors it

Fit via **R-INLA**, following Moraga's *Geospatial Health Data* and Konstantinoudis's BYM2/INLA tutorials directly, using the same `bgd_adm_bbs_20201113_SHP`-derived adjacency structure as Sarker et al. (2024) for direct comparability.

**International transferability check:** Salim et al.'s (2025) BYM2-RW2 INLA application across 78 Yogyakarta sub-districts, using the identical NASA POWER + official-shapefile + remote-sensing data stack proposed here, is the closest published precedent for extending this exact model to sub-district (upazila) resolution — cite and, where feasible, mirror its diagnostic checks for spatial random-effect stability at finer resolution.

### M5.4 Outbreak-threshold definition

Define "outbreak" **relative to each district/upazila's own historical baseline** (e.g., weekly incidence rate exceeding that unit's own 90th percentile, or cases per 100,000 exceeding a unit-specific threshold), not a single pooled percentile across all districts. A pooled percentile is dominated by Dhaka's scale and can leave a genuine local surge in a small rural district undetected while flagging Dhaka as "outbreak" almost by default. Report a sensitivity analysis replicating Shiddik et al.'s (2026) pooled P25/P50/P75/P90 approach for direct comparability, but treat the per-unit relative threshold as the primary specification — this is a citable methodological improvement, not just a technical adjustment, and should be described as such in the manuscript.

### M5.5 Forecast horizon

Primary target: 2–8 weeks ahead, following the operational lead-time literature (Singapore's EWS work identifies roughly two months as the horizon needed for vector-control response to meaningfully act on a forecast). Report performance at each horizon separately (2, 4, 6, 8 weeks) rather than a single pooled horizon metric, since accuracy is expected to degrade with horizon length and this degradation curve is itself informative for operational deployment.

---

## M6. Explainability Analysis (Arm D)

Applied to both Arm A and Arm B models, with an explicit mandate to go beyond a static feature-importance ranking (Shiddik et al., 2026 already published SHAP rankings for essentially this covariate set, so replicating one adds little):

1. **SHAP dependence plots** for each climate covariate, checking specifically against the threshold values already reported in the Bangladesh literature as a form of replication/extension rather than starting from nothing — Rahman et al. (2025) report inflection points at mean temperature 27°C, minimum temperature 22°C, maximum temperature 32°C, and relative humidity 82%; test whether your independently assembled panel and model reproduce these thresholds.
2. **ALE (accumulated local effects) plots** as a robustness check on SHAP dependence plots, particularly where covariates are correlated (temperature and humidity commonly are).
3. **Interaction detection:** SHAP interaction values or 2D dependence plots to test compound conditions (e.g., "risk rises sharply once weekly rainfall exceeds X mm, but only within a specific temperature band") rather than reporting single-variable thresholds in isolation.
4. **Dynamic explainability for Arm A:** SHAP values computed separately within each illness-day bin (§M4.1), plotted as a small-multiples or animated sequence showing how the relative importance of NS1 vs. IgM vs. IgG shifts across the illness course — this dynamic view, not a single static ranking, is the explainability contribution for the diagnostic model.

---

## M7. Cross-Scale Linkage Analysis (Arm C)

**The research question is general; the current data source is not.** RQ3 asks whether locally observed clinical diagnostic information provides incremental predictive information beyond surveillance, climate, and spatial structure, when geographically matched to its catchment — it is not defined as "does Dhaka's signal predict Dhaka." Dhaka is simply the first empirical test of this question, because it's the most realistic near-term data source (§M2.2). Framing it this way means obtaining clinical data from a second site later (Chattogram, Sylhet, etc.) extends the same analysis rather than requiring the research question to be rewritten.

**Scope constraint on the current empirical test:** this analysis is bounded to the district(s) actually covered by your Arm A clinical data — most realistically Dhaka if working with a single tertiary hospital. It is not tested or claimed at national scale.

**Procedure:**
1. Aggregate Arm A model outputs to a weekly local signal: proportion of tested patients flagged high-probability-dengue in the relevant district, using the illness-day-aware model from §M4.2.
2. Add this signal as an additional covariate to the Arm B population model, **restricted to the matching district(s)**.
3. Compare forecast performance (§M9) with vs. without this signal, under the same rolling-origin validation protocol as the rest of Arm B.
4. If clinical data is later obtained from additional sites, repeat as separate per-district local tests — do not pool across sites unless clinical data coverage becomes genuinely broad enough to support a multi-district or national claim.

---

## M8. Validation Protocol (the study's central methodological contribution)

### M8.1 Rolling-origin (walk-forward) temporal validation

Following Hyndman & Athanasopoulos's standard time-series cross-validation methodology, and directly modeled on its recent infectious-disease applications (malaria incidence forecasting in Mumbai; monkeypox forecasting in Thailand; the 2026 rolling-origin evaluation of the live Bundibugyo virus outbreak):

```
Fold 1:  Train 2015–2019  →  Test 2020
Fold 2:  Train 2015–2020  →  Test 2021
Fold 3:  Train 2015–2021  →  Test 2022
Fold 4:  Train 2015–2022  →  Test 2023
Fold 5:  Train 2015–2023  →  Test 2024
Fold 6:  Train 2015–2024  →  Test 2025
```

Retrain at each fold; evaluate only on genuinely subsequent, not-yet-seen data. Report metrics per fold, not just averaged — 2023 and 2019 are known structural-break years (large, abrupt outbreaks against a lower baseline), and a 2026 rolling-origin benchmark of global COVID-19 forecasting found model rankings are highly sensitive to exactly this kind of structural change, so per-fold reporting is necessary to show whether your model holds up specifically across Bangladesh's own outbreak-year transitions, not just on average.

### M8.2 Spatial leave-out (spatial cross-validation)

Use **predefined, reproducible geographic blocks** rather than an arbitrary district rotation — e.g., five blocks aligned to Bangladesh's broad geography (central, northern, eastern, western, southern/coastal), fixed in advance from geographic criteria and not adjusted after seeing how well any particular split validates. Train on the remaining blocks, test only on the held-out one, rotating through all five. This directly tests whether the model has learned a transferable climate–transmission relationship versus simply memorizing that Dhaka is always high.

### M8.3 Leakage audit (procedural, not optional)

At every rolling-origin fold (§M8.1) and every spatial-block fold (§M8.2), explicitly confirm — as a documented audit step, not an assumption — that the following are fit using only data available strictly before that fold's forecast origin: feature selection, missing-data imputation, feature scaling/normalization, outbreak-threshold selection (§M5.4), hyperparameter tuning, and probability calibration. This is the most common way a nominally rigorous rolling-origin design quietly reverts to a leaky one — via a scaler, imputer, or feature-selection step fit once on the full dataset before the fold loop even begins. Log which data window each preprocessing step was fit on, per fold, as part of the pipeline's output — not just the final model — so this is auditable after the fact, not just asserted.

### M8.4 Mandatory baseline comparator

**Every model must be reported against a naive baseline** (seasonal-naive / same-week-last-year persistence), not just against other ML models. This is not optional: the 2026 SpatialEpiBench benchmarking study, evaluating spatial epidemic-forecasting methods across multiple diseases and regions under standardized rolling-origin conditions, found that sophisticated spatial/ML methods beat a naive persistence baseline **less than half the time** once evaluated honestly. Report a skill score (e.g., 1 − [model error / naive-baseline error]) alongside raw accuracy metrics for exactly this reason — a model that doesn't beat the naive baseline under honest evaluation is a genuine, reportable finding, not a failure to hide.

### M8.5 Combined validation matrix

Report every population model's performance across all four combinations, not just one:

| | Single split (comparable to Shiddik 2026) | Rolling-origin |
|---|---|---|
| **National (all districts)** | (1) | (2) |
| **Spatial leave-out** | (3) | (4) — the fully honest condition |

The gap between cell (1) and cell (4) is itself a headline finding, directly testing whether Shiddik et al.'s (2026) reported 0.99 ROC-AUC survives fully honest conditions. **Name and report this explicitly as the Optimism Gap:** `Optimism Gap = Performance(single-split) − Performance(rolling-origin + spatial leave-out)`. Expect it to be substantial — plausibly on the order of a high-0.90s AUC under condition (1) degrading toward the 0.75–0.85 range under condition (4), based on the pattern seen elsewhere in the forecasting-validation literature (§M8.1). A large, well-quantified optimism gap is the single most differentiating, most citable number in the paper relative to Shiddik et al. (2026) — treat it as a headline result, not an embarrassing caveat.

**Levels of prospective evidence — be precise about which one you're claiming.** (1) Retrospective rolling-origin validation (§M8.1); (2) prospective shadow prediction (§M8.6); (3) prospective clinical/operational deployment; (4) prospective impact evaluation. This study targets Levels 1–2. Do not use "operationally deployed" or "clinically validated" language in the abstract or title — that would claim Levels 3–4, which this design does not attempt.

### M8.6 Prospective shadow validation (optional, strongest evidence available — pursue if the timeline allows)

Once the model is finalized on retrospective data (§M8.1–M8.5), a further, strictly stronger validation step: freeze the model (weights, thresholds, feature definitions), generate forecasts on a real, ongoing rolling basis, timestamp and store each prediction *before* the corresponding DGHS surveillance data for that period is released, and score only after outcomes land. This removes any possibility — even unconscious — of the model or its evaluation having been shaped by knowledge of the outcome, which no purely retrospective design (however careful) can fully rule out. It requires no new data sources, only a few additional weeks at the end of the project timeline to accumulate enough shadow-forecast periods to evaluate. If your timeline genuinely cannot accommodate this, say so explicitly as a limitation rather than omitting it silently — reviewers in this space increasingly expect at least a discussion of why prospective evaluation wasn't feasible.

---

## M9. Evaluation Metrics

**Population forecasting (Arm B), count/regression framing:**
- RMSE, MAE — standard, but interpret cautiously given the zero-inflated distribution (report alongside a metric less sensitive to zero-heavy data)
- MAPE — useful for relative comparison to Liu, Hossain & Hossain's (2025) reported Dhaka-division benchmark (RMSE 109, MAPE 12.9%), but unstable near zero counts; report with that caveat
- CRPS (continuous ranked probability score) or log-score — for probabilistic calibration, since an "early warning system" framing requires calibrated probabilities, not just point accuracy
- Skill score vs. naive baseline (§M8.3)

**Population forecasting, outbreak-alert (binary) framing:**
- Sensitivity, specificity, ROC-AUC, precision-recall AUC (more informative than ROC-AUC under class imbalance, which is expected given outbreak weeks are a minority class)
- Calibration plot (predicted probability vs. observed frequency) — report explicitly; Shiddik et al.'s (2026) reported metrics do not include calibration, and a well-calibrated but modest-AUC model is more operationally useful than a poorly-calibrated high-AUC one

**Individual diagnostic model (Arm A):**
- Sensitivity, specificity, AUC, balanced accuracy — reported per illness-day bin (§M4.1), not only pooled, and per model in the C0→C3 ladder (§M4.2)
- Negative predictive value — clinically the most decision-relevant metric for ruling out dengue early in illness
- Direct comparison to the Pakistan benchmark (§M4.4)

**Optional — uncertainty decomposition.** If time and statistical capacity allow, go beyond a single predictive interval (e.g., "80% risk, 95% CI 65–90%") and decompose *where* the uncertainty comes from: parameter uncertainty, climate-input uncertainty, clinical-signal uncertainty (if Arm C is included), reporting/surveillance uncertainty, and spatial uncertainty. The BYM2/INLA model (§M5.3) already produces posterior uncertainty as a byproduct, so this extends infrastructure you're already building rather than requiring a new pipeline — but treat it as a stretch goal that strengthens the Bayesian component's payoff, not as required scope. Don't let it delay the core deliverable.

---

## M10. Software and Computational Tools

| Task | Tool |
|---|---|
| Data assembly, panel construction | Python (pandas), R |
| Climate data retrieval | NASA POWER API (direct HTTP calls) |
| Spatial operations, adjacency matrices | R (`sf`, `spdep`), Python (`geopandas`) |
| Gradient boosting | Python (`xgboost`, `lightgbm`) |
| Neural network models | Python (`pytorch` or `keras`) |
| Bayesian spatio-temporal modeling | R (`R-INLA`), following Moraga's and Konstantinoudis's tutorials |
| Explainability | Python (`shap`, `PyALE` or `ALEPlot` in R) |
| Rolling-origin / spatial CV harness | Custom implementation (no off-the-shelf library correctly handles both spatial and temporal holdout simultaneously for this design — build and unit-test this carefully) |

---

## M11. Ethical Considerations

- Any patient-level data (Plan A or Plan B) requires ethics approval — institutional IRB, and BMRC (Bangladesh Medical Research Council) if partnering with a hospital.
- De-identification at source; no patient addresses or names retained beyond what's needed for illness-day validation, and only under the approved protocol.
- Data-sharing agreement with the source hospital/IEDCR should specify permitted use, retention period, and publication rights explicitly before data transfer.
- Aggregated DGHS/NASA/BBS data used for the population model carries no individual-level ethics burden but should still be cited and attributed per each source's terms of use.

---

## M12. Anticipated Limitations (state these upfront, don't wait for reviewers to find them)

1. If Plan B is triggered, H1 (illness-day stratification) is not testable, and this should be stated as a scope change, not buried.
2. The linkage analysis (Arm C) is geographically bounded to wherever clinical data is obtained — explicitly not a national claim (§M7).
3. Socio-economic covariates are likely only available at division level, coarser than the district/upazila resolution of the rest of the panel — flag any resolution mismatch explicitly rather than silently downscaling.
4. Rolling-origin validation reduces effective training data in early folds (e.g., Fold 1 trains on only 2015–2019) — report whether performance in early folds is noisier due to smaller training windows, so this isn't mistaken for genuine model instability.
5. Spatial leave-out validation on low-incidence districts/upazilas may produce wide uncertainty intervals simply due to small case counts, not model failure — report interval width alongside point estimates.
