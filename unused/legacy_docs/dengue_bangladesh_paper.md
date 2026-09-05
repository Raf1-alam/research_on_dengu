# Fever and Forecast: Full Research Plan
### Multimodal, Multi-Scale Dengue Early Warning for Bangladesh — Clinical Diagnosis × Climate-Driven Outbreak Forecasting

*Working plan v1 — August 2026*

---

## 0. Positioning statement (read this first)

As of March 2026, the population-level half of your original idea (climate + case data → district-level ML/DL/SHAP/Bayesian outbreak forecasting for Bangladesh) has already been published in detail: Shiddik, Toshi, Yesmin & Rahman, *"District-Level Dengue Early Warning Prediction System in Bangladesh Using Hybrid Explainable AI and Bayesian Deep Learning,"* Tropical Medicine and Infectious Disease 11(3):73 (2026) — hereafter **"Shiddik 2026."** It covers all 64 districts, 2017–2024, DGHS case data + NASA climate + World Bank/BBS socio-economic data + WHO healthcare indicators, four ML models, four DL models, SHAP, and Bayesian spatio-temporal (BYM2/INLA) modeling with lag effects, and it publishes district-wise alerts through 2026.

**This does not kill the project. It sharpens it.** Your revised job is to (a) explicitly position against this paper, and (b) build the two things it does not have: a validated individual-diagnosis component, and a genuinely prospective/spatially-honest validation of the outbreak model (which Shiddik 2026's own limitations section admits it lacks).

**Working title:** *Fever and Forecast: A Dual-Scale, Prospectively Validated Framework Linking Clinical Dengue Diagnosis to Climate-Driven Outbreak Early Warning in Bangladesh*

**One-line novelty statement to put in your proposal/abstract:**
> Recent Bangladesh dengue-forecasting studies have reported strong retrospective performance, but the generalizability of these models to geographically disjoint settings and their performance under genuinely prospective conditions remain insufficiently established. This study (1) links patient-level serological diagnosis to district/upazila-level outbreak forecasting in one framework, and (2) evaluates the outbreak model under rolling-origin temporal validation and spatial leave-out validation, directly testing whether previously reported near-perfect accuracy (e.g., ROC-AUC 0.99 in Shiddik 2026) survives honest out-of-sample conditions.

---

## 1. What to read before you write anything

Read these in this order. Take notes on **exactly what data, what model, what validation, and what they found** for each — you'll need this for your literature review and, more importantly, to make sure you don't accidentally rebuild something that already exists.

### 1.1 Direct competitors (Bangladesh, dengue, ML/climate) — read in full
- **Shiddik et al. 2026**, *Trop. Med. Infect. Dis.* 11(3):73 — the paper you must differentiate from. Read the Methods and Limitations sections closely.
- **Bhowmik et al. 2026**, *"Two Decades of Dengue in Bangladesh (2001–2024): Epidemiologic Trends, Geographic Spread and Climatic Drivers,"* Tropical Medicine & International Health — a 24-year narrative synthesis of DGHS/IEDCR surveillance data; good for your background/introduction numbers and for cross-checking case counts.
- Bari, Shailee, Joyanti & Islam (2025), *"Dengue Outbreak Prediction in Bangladesh Based on Meteorological Factors,"* QPAIN 2025 — smaller ensemble-regression study using the same 2008–2022 monthly DGHS/BMD dataset available on Kaggle. Useful as a simpler baseline to beat.

### 1.2 Clinical/serological diagnosis literature (your differentiator — study these closely)
- **Correction (verified directly against the paper):** Huang, Tsai et al., PLOS NTD 2020, *"Assessing the risk of dengue severity using demographic information and laboratory test results with machine learning"* — 1,581 Taiwan patients — is a **severity/prognosis** model built on patients **already confirmed dengue-positive**, not a dengue-vs-not-dengue diagnostic model. Cite it as precedent for clinical-ML-on-serology-plus-labs generally, and as a possible severity-prediction extension of your work, but **not** as your main diagnostic benchmark — an earlier version of this plan mischaracterized it that way.
- **Qaiser et al. 2024** (*Advances in Virology*), *"Support Vector Machine Outperforms Other Machine Learning Models in Early Diagnosis of Dengue Using Routine Clinical Data"* — 300 Pakistani patients, NS1/IgM/IgG + hematology, RT-PCR as ground truth, compares six ML models (SVM best). **This is your closer, primary diagnostic benchmark** — it actually predicts dengue-positive vs. -negative from routine clinical presentation, which is the task your model performs.
- **Nature Scientific Reports 2025**, *"Dual level dengue diagnosis using lightweight multilayer perceptron with XAI in fog computing environment"* — shows what an XAI-diagnosis paper with a small (~1,000-record) dataset can still get published on; useful for scoping what's achievable if you're stuck with a modest dataset.
- **Sabchareon et al. framework on NS1/IgM/RT-PCR kinetics** (search: "dengue diagnostic window NS1 IgM IgG day of illness") — foundational virology on why illness-day matters: NS1 and RT-PCR dominate days 1–3, IgM rises after day ~4–5, IgG differentiates primary vs. secondary infection. You need this cited to justify the whole illness-day-stratified design.
- **"Artificial intelligence in differentiating tropical infections: A step ahead,"** PLOS NTD — builds an ML tool distinguishing dengue from malaria, leptospirosis, and scrub typhus using clinical/lab presentation. **Read this closely — it's your template for Plan B** (see §4.2) if illness-day-tracked hospital data doesn't come through.

### 1.3 Bangladesh dengue virology/epidemiology context (2023–2026, for your background section)
- Systematic review, *IJERPH* 2026, *"Epidemiological Characteristics of Dengue Infection in Bangladesh"* — national demographic/serotype synthesis.
- PMC12016812 (2026) — DENV-2 Cosmopolitan genotype resurgence, Southwest Bangladesh, 2023.
- PMC10222234 — Evercare Hospital Dhaka, serotype vs. severity 2018–2022 (useful if you want a severity angle).
- PMC11985886 — IEDCR serotype distribution 2023–2024 (DENV-2 dominant both years per this source).
- **Note for your lit review:** sources disagree on exact 2023 serotype proportions (Shiddik 2026 says DENV-4 was 27% of 2023 infections; IEDCR-sourced sources say DENV-2 was ~70–74%). Don't silently pick one — flag the discrepancy and cite both; it's a legitimate observation about surveillance data quality that strengthens your paper's credibility.
- PMC12900302 (2024) — Zika/Chikungunya co-circulation in Dhaka; relevant if you want to discuss differential diagnosis / clinical overlap.

### 1.4 International methodology precedents — these are your validation-design template
- **Shi et al. 2016**, *Environmental Health Perspectives*, Singapore's operational 3-month LASSO-based dengue early warning system — the gold-standard example of a forecasting system built and evaluated the way an operational EWS should be.
- **Nature Communications 2025**, *"Climate variation and serotype competition drive dengue outbreak dynamics in Singapore"* — recent (Nov 2025), forecasts outbreaks up to 2 months ahead using climate + serotype data; closest published analogue to what a rigorous version of your outbreak model should look like.
- **PLOS NTD 2022**, *"Deep learning models for forecasting dengue fever based on climate data in Vietnam"* — explicitly discusses degraded accuracy under genuine future-looking prediction vs. retrospective fit; useful citation for why your validation design matters.
- **D-MOSS** (Vietnam, operational since 2019; also Malaysia) — a real operational climate-driven dengue EWS; cite as precedent that this is implementable, not just academic.

### 1.5 Methodology references to actually build from
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Ch. 5.10, "Time series cross-validation" (free online, otexts.com/fpp3/tscv.html) — the standard reference for rolling-origin evaluation; cite this for your validation methodology.
- Moraga, P., *Geospatial Health Data: Modeling and Visualization with R-INLA and Shiny* (free online book) — practical, step-by-step for the Bayesian spatio-temporal side.
- Konstantinoudis, G. — public BYM2/INLA tutorials (disease mapping, ecological regression) — hands-on code you can adapt directly.

---

## 2. Research questions and hypotheses

**Overarching question:** Can a dual-scale framework that links illness-day-stratified serological diagnosis to spatially- and temporally-honest climate-driven outbreak forecasting provide clinically and operationally actionable dengue early warning in Bangladesh?

**RQ1 (individual scale):** How does the diagnostic performance of NS1/IgM/IgG (individually and combined) change as a function of day of illness, and does incorporating illness-day improve classification over biomarker-only models?
- H1: Combined-marker models incorporating illness day will outperform any single-marker model, with the largest relative gain occurring 4–7 days post-onset (the NS1-decline/IgM-rise crossover window).

**RQ2 (population scale):** Can district/upazila-level dengue surges be forecast 2–8 weeks ahead using lagged climate and case data, with performance that holds under rolling-origin and spatial-holdout validation (not just a single historical train–test split)?
- H2: Model performance under rolling-origin + spatial-holdout validation will be measurably lower than the same model evaluated with a single train–test split — testing this gap explicitly is itself a contribution, given what Shiddik 2026 reports and flags as unvalidated.

**RQ3 (linkage):** Does locally observed clinical diagnostic information provide incremental predictive information beyond surveillance, climate, and spatial structure, when geographically matched to its catchment? *(The empirical test of this is currently Dhaka, because that's the most realistic first data source — but the question itself is stated generally, so getting clinical data from a second site later, e.g. Chattogram or Sylhet, extends the same framework rather than requiring you to rewrite it.)*
- H3: Adding a locally matched clinical signal will provide measurable incremental predictive information beyond surveillance, climate, and spatial structure, expressed through discrimination, calibration, or warning lead time. *(Deliberately not directional — "no measurable improvement" is a valid, reportable answer to this hypothesis, not a failed experiment.)*

**RQ4 (mechanism, not just prediction):** What nonlinear thresholds and lag structures characterize the climate–dengue relationship in Bangladesh, and do they differ from the aggregate SHAP rankings reported in prior work?

---

## 3. Datasets — what to use for what

| Purpose | Dataset / Source | What it gives you | Access | Caveats |
|---|---|---|---|---|
| Current & historical case counts | **DGHS Dengue Dashboard** (dashboard.dghs.gov.bd) + archived daily press releases (old.dghs.gov.bd) | Weekly/daily national and district case & death counts, current through 2026 | Public, scrape/manually log weekly | No line-level (patient) data; aggregated only |
| Ready-made climate+case panel | **Kaggle: "Dengue-Climate Dataset of Bangladesh (2008–2022)"** (Bari et al.) | Monthly min/max temp, rainfall, humidity, dengue counts, pre-merged | Free download | Monthly only, national/limited spatial granularity — use as a quick-start baseline, not your primary dataset |
| Climate covariates (build your own, district/upazila level) | **NASA POWER API** (power.larc.nasa.gov) | Daily/monthly temperature, rainfall, humidity, surface pressure by exact coordinates, 1981–near-present | Free, no auth, direct API calls per lat/lon | This is what Shiddik 2026 itself used — matching it lets you do a fair comparison, and querying per-upazila centroid is exactly how you'd go finer-grained than their district-level design |
| Administrative boundaries / spatial structure | **HDX "Bangladesh - Subnational Administrative Boundaries"** (BBS-sourced, COD-AB) and the **`bangladesh` R package** | Shapefiles at admin levels 0–4 (division/district/upazila/union), needed for adjacency matrices (BYM2), choropleths, spatial joins | Free | Same source (BBS) Shiddik 2026 used at district level — use their exact `bgd_adm_bbs_20201113_SHP` file if you want directly comparable maps, or the finer upazila layer if you go sub-district |
| Socio-economic / healthcare covariates | World Bank Open Data, WHO Global Health Observatory | GDP, poverty headcount, hospital beds, health expenditure, UHC index — same category Shiddik 2026 used | Free, public APIs | Mostly available at national/division level, not district; you may need to interpolate or drop this layer if going upazila-level |
| Long-run epidemiological context | Bhowmik et al. 2026 review (2001–2024 synthesis) | Narrative trend data, useful for cross-checking your own aggregated numbers, not a raw downloadable dataset | Read for context, request data from corresponding author if needed | — |
| **Clinical/serological (individual level)** | **Mendeley: "A Comprehensive Dengue Dataset of Bangladesh"** | NS1, IgG, IgM, age, sex, house type, area classification, from a Dhaka-region cross-sectional study | Free download | **Important limitation:** survey-based (hospital visitors + community members self-reporting), not confirmed-diagnosis hospital lab records with tracked illness-day-of-test. Good for a pilot/proof-of-concept, not sufficient on its own for the illness-day-stratified analysis that's your key differentiator. |
| Clinical benchmark comparators (not reusable, cite only) | Qaiser et al. 2024 (Pakistan, n=300, RT-PCR ground truth — your primary diagnostic benchmark); Huang/Tsai et al. 2020 (Taiwan, n=1,581 — a severity/prognosis benchmark, not diagnostic, see §1.2) | External performance numbers to benchmark your model against | Published results only — not public raw data | Use for framing/comparison in your discussion, not for training |
| **Real hospital-sourced clinical data (the one you likely need to actively pursue)** | A tertiary hospital or IEDCR, via formal data-sharing request | Confirmed NS1/IgM/IgG results **with date of symptom onset and date of each test**, ideally with RT-PCR or clinical outcome as ground truth | Requires institutional request + likely ethics approval | This is the single highest-leverage thing you can do for this project. The 2023 outbreak paper (Zaman et al., IJID) obtained exactly this kind of patient-address dataset from MoHFW's Management Information System by direct request — it's precedent that Bangladeshi authorities will share data for research under the right approvals. |

---

## 4. Methodology

### 4.1 Overall architecture

```
                         DENGUE — DUAL SCALE FRAMEWORK
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                             │
      INDIVIDUAL SCALE                             POPULATION SCALE
   (illness-day diagnostic model)              (district/upazila forecast)
              │                                             │
   NS1, IgM, IgG, day of illness,          Lagged temp/rainfall/humidity,
   age, sex, symptoms                       lagged case counts, neighboring
              │                             district cases, land use
              ▼                                             ▼
   P(dengue | markers, day) —              Rolling-origin + spatial-holdout
   time-varying diagnostic curve            validated ML/DL + Bayesian BYM2
              │                                             │
              └─────────────────┬───────────────────────────┘
                                 ▼
                  Aggregated local suspected-case
                  signal feeds into population model
                  as an additional early-warning input
                                 │
                                 ▼
                    SHAP / ALE for MECHANISM
                 (thresholds, interactions, lags —
                    not just a feature ranking)
                                 │
                                 ▼
              District/upazila-level early warning output,
              reported with honest, validated uncertainty
```

### 4.2 Individual-scale diagnostic model

**Data prep:** For each patient record, construct: NS1 (pos/neg), IgM (pos/neg), IgG (pos/neg), day of illness at time of test, age, sex, and (if available) basic hematology (platelet count, WBC, hematocrit — these appeared as useful predictors in the Pakistan study). Ground truth ideally RT-PCR or clinical case definition; if only survey data is available, treat this explicitly as a limitation and consider it a pilot analysis.

**Modeling — structured as a nested ladder, mirroring the population model's hierarchy (§4.3), so the incremental value of each added feature is isolated rather than only ever comparing full models:**

| Model | Inputs | Isolates |
|---|---|---|
| C0 | NS1 + IgM + IgG | Raw biomarker signal, no time-awareness |
| C1 | C0 + illness day | Value of knowing when in the illness course the test was taken |
| C2 | C1 + biomarker × illness-day interactions | Whether the *combination* pattern, not just illness day alone, matters — this is where H1's predicted 4–7 day crossover effect should show up most clearly |
| C3 | C2 + hematology / demographics / symptoms | Value of routine clinical data beyond serology, following the Bangladeshi CBC-XGBoost precedent |

1. Fit C0→C3 at each illness-day bin (days 1–3, 4–7, 8+) using logistic regression as the interpretable baseline, then gradient boosting (XGBoost/LightGBM) and a small feed-forward network, following the Taiwan and Pakistan precedents so your results are directly comparable to theirs.
2. Fit one continuous model with day-of-illness as a smooth interaction term (C2, continuous version): `P(dengue | NS1, IgM, IgG, day, age, sex)`, and plot the fitted probability curves by day of illness — this plot is your headline result for RQ1.
3. Validate with stratified k-fold (by patient, not by record) plus, if sample size allows, a held-out hospital/site for external validation.

**Plan B — if illness-day-tracked hospital data does not come through within your 4-week decision window (see §9):** the Mendeley/survey-style data cannot support H1 as written, because it doesn't reliably capture day-of-illness at time of testing. Don't force it. Pivot the individual-scale question instead to: *"Can NS1/IgM/IgG plus demographics distinguish dengue from other co-circulating febrile illnesses (Chikungunya, Zika) in Bangladesh?"* This is directly supported by two things already in your reading list — the PLOS NTD tropical-infections-differentiation paper (dengue vs. malaria/leptospirosis/scrub typhus) as a methodological template, and the 2024 Dhaka study documenting real DENV/CHIKV/ZIKV co-circulation and co-infection in febrile patients (§1.3) as your epidemiological justification. Note this requires a dataset with *multiple* disease labels, not just dengue-positive/negative — the Mendeley set alone won't have this, so this pivot still needs its own data source (a febrile-illness panel from a hospital lab, ideally the same one you're already approaching for the main request).

### 4.3 Population-scale outbreak model

**Data prep:** Assemble a district-week or district-month panel: case counts (target), lagged case counts (own and neighboring districts, lags 1–8 weeks), lagged temperature/rainfall/humidity (following Shiddik 2026's lag structure of 1, and 1–3 combined, as a starting point, then test your own), and static covariates (population, land use, if going that route).

**Distributional specification — decide this before you fit anything:** dengue case counts are non-negative integers, overdispersed, and zero-inflated in many districts/upazilas during dry-season months. Fitting standard squared-error regression (or XGBoost's default objective) on raw counts will produce biased, sometimes negative, predictions and will systematically underfit outbreak spikes. Specify:
- **INLA/Bayesian model:** Negative Binomial likelihood as the default (this is also what Shiddik 2026 used, having confirmed overdispersion and zero-inflation in their own count data); test Zero-Inflated Negative Binomial (ZINB) as an alternative if structural zeros (districts/upazilas with true near-zero transmission, not just underreporting) are common in your panel.
- **XGBoost:** use `objective='count:poisson'`, not squared-error.
- **Note for spatial leave-out validation specifically:** some rural districts/upazilas will have zero cases for months at a time. Read up on how INLA/BYM2 handles these structural zeros before you run leave-out folds on sparse-case regions — a naive Poisson/NB fit can behave poorly on all-zero training folds.

**Modeling:**

Structure the population model as a **nested hierarchy**, so that the predictive contribution of each additional information layer is isolated and testable rather than only ever comparing full-complexity models against each other:

| Model | Inputs | Tests |
|---|---|---|
| P0 | Seasonal baseline only (historical same-week pattern) | The floor any model must beat |
| P1 | P0 + lagged case counts (surveillance) | Value of surveillance history alone |
| P2 | P1 + climate | Value of climate beyond surveillance |
| P3a | P2 + geographic spatial structure (BYM2, adjacency-based) | Value of spatial dependence beyond climate |
| P3b | P2 + connectivity-weighted spatial structure (e.g., Meta Social Connectedness Index in place of raw adjacency) | Whether *how people actually connect* generalizes better than *who's geographically next to whom* — run this only if connectivity data availability is confirmed, and always report it as an explicit comparison against P3a, never as a silent replacement. Don't treat SCI as literal weekly mobility — describe it as social-connectivity-derived spatial coupling. |
| P4 | P3 (a or whichever wins) + serotype share | Value of pathogen surveillance |
| P5 | P3/P4 + local clinical signal, *scoped per §4.5* | RQ3 — the full integrated model |

**On serotype specifically — treat this as a priority data-access question to actively pursue, not a footnote you leave to chance.** Of the optional extensions in this table, serotype is the one most likely to be worth the effort: the Singapore literature (Nature Communications, 2025) shows climate-plus-serotype-competition models materially outperform climate-only models at 2–8 week horizons, which is exactly your target horizon. Represent it simply — a serotype-share variable `SerotypeShare_{k,t} = SequencedCases_{k,t} / TotalSequencedCases_t` at whatever temporal/spatial resolution the data genuinely support (likely national/quarterly at best, given how patchy Bangladesh's published serotyping coverage is across the studies in §1.3) — rather than attempting a finer-grained genomic layer the underlying sequencing can't support. Make a real inquiry to IEDCR or a hospital virology department about serotype data access alongside your main clinical-data request (§9), rather than deciding after the fact that it wasn't available.

Fit and validate every level of this hierarchy under the same protocol (§5 below), and report the full ladder, not just P5's number. A result showing "P2 beats P1 clearly, but P3 barely beats P2" is itself a finding worth a sentence — it tells you which layer of complexity is actually earning its place, which is a more defensible framing than a single black-box model comparison.

1. **Benchmark, not centerpiece:** replicate a simplified version of Shiddik 2026's approach (XGBoost, MLP, SHAP) on your own assembled panel. Treat this explicitly as a **control condition for comparability**, not as part of the contribution — the actual scientific content is the validation design, the outbreak-definition fix, the model ladder, and (if pursued) serotype and clinical-signal layers on top of it. Don't let the write-up spend more space on reproducing Shiddik than on what you did beyond it.
2. Add Bayesian spatio-temporal modeling with **R-INLA**, BYM2 spatial structure, RW1/RW2 or AR1 temporal structure, Negative Binomial likelihood (see above), following Moraga's book and Konstantinoudis's tutorials directly — this is the more statistically rigorous half and doubles as your uncertainty quantification.
3. **This is the critical difference from prior work:** evaluate every model under **both**:
   - **Rolling-origin (walk-forward) validation** — train on 2015–2019, test 2020; expand, retrain, test 2021; repeat through your most recent complete year. This is standard practice in forecasting (see Hyndman & Athanasopoulos) and is exactly what Shiddik 2026's own limitations section says was missing.
   - **Spatial leave-out validation** — train on a subset of districts/upazilas, test on excluded ones, rotate. This tests whether the model generalizes geographically or is just memorizing Dhaka's dominant signal (Dhaka accounts for a disproportionate share of national cases in every recent year, which can make a model look good while actually just learning "Dhaka is always high").
4. Report both the single-split numbers (for direct comparability with Shiddik 2026) **and** the rolling-origin/spatial-holdout numbers side by side. The gap between them, if there is one, is itself a finding worth a paper section.

### 4.4 Explainability — mechanism, not leaderboard

Don't stop at a SHAP importance bar chart (Shiddik 2026 already published one for essentially this exact covariate set). Instead:
- Use SHAP dependence plots / ALE plots to find **thresholds and interaction effects** — e.g., "risk rises sharply once weekly rainfall exceeds X mm, but only when minimum temperature stays within Y–Z °C" — this is the kind of finding the original strategy document flagged as the real scientific contribution.
- For the diagnostic model, use SHAP to show how the *relative* importance of NS1 vs. IgM vs. IgG shifts across illness-day bins — a dynamic, not static, explainability result.

### 4.5 Linking the two scales — mind the spatial mismatch

**Important scoping constraint:** if your clinical data comes from one or two tertiary hospitals (most realistic scenario — likely Dhaka), you cannot use their suspected-case signal to predict outbreaks in Khulna or Rajshahi. A hospital's catchment is not the same as a district, let alone the whole country. Don't build RQ3 as a national-scale claim — it will not survive review.

**Correct scoping:** define RQ3 explicitly as a **local** test, bounded to the district(s) your clinical data actually covers: *"Does a Dhaka-hospital-derived combined signal improve Dhaka(-district)-level outbreak forecasting, relative to climate-only?"* Aggregate the individual-level model's outputs (e.g., weekly proportion of tested patients flagged high-probability) as a candidate leading indicator, and test it only against the population model's forecasts for the matching district. This is a smaller claim than "combining scales improves national early warning," but it's a real, defensible one — and it's still the piece that doesn't exist anywhere in the current literature. If you later obtain clinical data from more than one site, you can extend the linkage test to each site's own district, but treat each as a separate local test, not a pooled national one, unless you have clinical data with genuinely broad geographic coverage.

**The general principle behind this fix, worth stating explicitly in your methods section:** the most dangerous failure mode for the linkage analysis specifically is treating a tertiary-hospital signal as if it were population prevalence. The fix isn't more modeling sophistication — it's residence/catchment information, geographic restriction of the claim, and a sensitivity analysis on how sample volume affects signal stability. Keep this principle in mind if you extend to more sites later, not just for the initial single-hospital case.

**On optional extensions (serotype-share time series, mobility/connectivity layers such as Meta's Social Connectedness Index):** these can genuinely strengthen the model if data availability supports them, but neither has been verified for Bangladesh at this project's required spatial/temporal resolution — don't plan them into the core scope until you've confirmed access and granularity, the same way NASA POWER, the HDX shapefiles, and DGHS access were confirmed earlier in this plan. Treat them as "add if the data turns out to support it," not as assumed components.

---

## 4.6 How to frame the paper (narrative, evidence level, and one named result to build toward)

**Levels of prospective evidence — know which one you're claiming.** Not all "prospective" claims are equal, and conflating them is an easy way to overclaim:
1. Retrospective rolling-origin validation (§5)
2. Prospective shadow prediction (freeze the model, timestamp forecasts, score against real future outcomes — §9's timeline permitting)
3. Prospective clinical/operational deployment (the model actually used in practice)
4. Prospective impact evaluation (deployment measurably changes outcomes)

**Your first paper should target Levels 1 and 2.** That's a strong, honest, achievable claim. Don't reach for "operationally deployed" or "clinically validated" language in the abstract/title — you haven't done Levels 3–4, and claiming them undermines the credibility of the (already strong) Levels 1–2 result.

**The narrative arc for the paper**, roughly: *first, establish how the diagnostic information carried by dengue biomarkers changes over illness time; second, quantify how much climate improves surveillance-based forecasting; third, test whether spatial, pathogen, and clinical information add further predictive value; finally, determine how much of this performance survives geographically and temporally honest validation.* That's a scientific story, not "we built an ensemble model."

**Name your headline result: the Optimism Gap.** Define it explicitly — `Optimism Gap = Performance(retrospective single-split) − Performance(fully honest validation)` — and expect it to be sizeable (plausibly something like AUC 0.98 on a random retrospective split degrading toward 0.75–0.85 under combined temporal + spatial holdout, based on the pattern seen elsewhere in the forecasting-validation literature, §8). A large, well-quantified optimism gap is not a weaker result than a high retrospective AUC — it's a more scientifically important one, and it's the single most citable, most differentiating number in the paper relative to Shiddik 2026.

**Optional stretch — uncertainty decomposition.** If time and statistical capacity allow, go beyond reporting a single predictive interval (e.g., "80% risk, 95% CI 65–90%") and decompose *where* the uncertainty comes from — parameter uncertainty, climate-input uncertainty, clinical-signal uncertainty (if Arm C is included), reporting/surveillance uncertainty, and spatial uncertainty. The BYM2/INLA model already produces posterior uncertainty as a byproduct, so this is a genuine extension of infrastructure you're already building, not a new pipeline — but treat it as a nice-to-have that strengthens the Bayesian component's payoff, not as required scope. Don't let it delay the core deliverable.

---

- [ ] Rolling-origin temporal validation for all outbreak models (not a single split)
- [ ] **Leakage audit at every rolling-origin fold:** confirm feature selection, imputation, scaling, threshold selection, hyperparameter tuning, and calibration are all fit using only data available *before* that fold's forecast origin — never on the full dataset first. This is the single most common way a "rigorous" rolling-origin design quietly turns back into a leaky one; audit it explicitly rather than assuming your pipeline order is correct.
- [ ] Spatial leave-out validation, using **predefined, reproducible geographic blocks** (e.g., central / northern / eastern / western / southern-coastal divisions) rather than an arbitrary or performance-tuned district rotation — blocks must be fixed in advance from geographic criteria, not chosen because they happen to validate well.
- [ ] **Prospective shadow validation, if your timeline allows it:** once the model is finalized, freeze it (weights, thresholds, feature definitions), generate real forecasts on a rolling basis, timestamp and store each prediction *before* the corresponding DGHS outcome data is released, and score only after outcomes land. This is strictly stronger evidence than any retrospective rolling-origin result, since it removes any possibility of the model or its evaluation having been shaped, even unconsciously, by knowledge of the outcome. Doesn't require new data — just discipline and a few extra weeks at the end of the timeline.
- [ ] **Outbreak definition — use a per-district/upazila relative threshold, not a single pooled percentile.** Shiddik 2026 defined "outbreak" as districts above the cross-sectional mean/percentile for that year — but a pooled percentile is dominated by Dhaka's scale, so a small rural district's genuine local surge (e.g., 5 cases against a historical baseline near zero) may never cross a national-level cutoff, while Dhaka may register as "outbreak" almost by default. Define outbreak instead relative to *each district/upazila's own historical baseline* (e.g., cases exceeding that unit's own 90th percentile, or an incidence rate per 100,000 population) so smaller districts' real surges are detectable. This is also a legitimate, citable methodological improvement over Shiddik 2026, not just a technical fix — say so explicitly in your methods section.
- [ ] Sensitivity analysis on outbreak-threshold definition (Shiddik 2026 tested P25/P50/P75/P90 pooled across districts — replicate for comparability, but report your per-unit relative threshold as the primary analysis)
- [ ] Confirm how your Bayesian model handles all-zero training folds before running spatial leave-out validation on low-incidence districts/upazilas (see §4.3 count-data note)
- [ ] Multicollinearity check (VIF) on covariates before modeling
- [ ] Missing-data strategy stated explicitly and justified
- [ ] External comparator for the diagnostic model (Taiwan/Pakistan benchmarks) reported alongside your own numbers
- [ ] Explicit statement of what's aggregated vs. patient-level, to avoid ecological-fallacy overreach
- [ ] Report calibration (not just AUC/accuracy) for anything framed as "early warning" — a well-calibrated probability matters more operationally than raw accuracy

---

## 6. Suggested timeline (adjust to your actual constraints)

| Phase | Duration | Deliverable |
|---|---|---|
| 1. Literature + gap mapping | 2–3 weeks | Annotated bibliography, finalized novelty statement, explicit comparison table vs. Shiddik 2026 |
| 2. Data assembly (population side) | 3–4 weeks | Clean district/upazila-week panel: cases + NASA POWER climate + shapefiles merged |
| 3. Data assembly (clinical side) | 4–8 weeks (longer if pursuing real hospital data) | Either a formal data-sharing agreement in motion, or the Mendeley pilot dataset cleaned and documented with limitations noted |
| 4. Population model build + validation | 4–6 weeks | ML/DL baseline + BYM2 Bayesian model, both under rolling-origin and spatial-holdout validation |
| 5. Diagnostic model build | 3–4 weeks | Illness-day-stratified model(s), benchmarked against Taiwan/Pakistan numbers |
| 6. Linkage analysis | 2–3 weeks | Combined-signal test (RQ3) |
| 7. Explainability / mechanism analysis | 2 weeks | SHAP/ALE threshold and interaction findings |
| 8. Write-up | 4–6 weeks | Full manuscript, explicit differentiation section vs. Shiddik 2026 |

---

## 7. Practical / ethical logistics

- **Ethics approval:** required for any patient-level clinical data, even retrospective/de-identified. Your institution's ethics review board, or BMRC (Bangladesh Medical Research Council) if working with a hospital.
- **Data request routes:** for real clinical data, approach a teaching hospital's medicine or virology department directly, or IEDCR/DGHS's Management Information System — there is direct precedent for this working (the 2023 outbreak geographic-distribution study obtained patient-address data from MoHFW's MIS by request).
- **Attribution:** if you end up building a district/upazila panel from NASA POWER + BBS shapefiles + DGHS case data, cite all three sources explicitly, the same way Shiddik 2026 credits DGHS, WHO, NASA, and the World Bank in their acknowledgments.

---

## 8. Realistic publication targets

Given the competitive landscape, in rough order of ambition:

1. **Solid, achievable tier:** *PLOS Neglected Tropical Diseases*, *Scientific Reports*, *BMC Public Health* / *BMC Infectious Diseases* — all have published very similar Bangladesh/regional dengue-ML work recently and are a realistic fit if you execute the plan above well.

2. **Two currently-open Special Issue targets worth prioritizing** (checked as of Aug 2026):
   - ***Tropical Medicine and Infectious Disease* — "Urban Vector-Borne Pathogens in Tropical Cities Under Climate Change,"** deadline **30 September 2026**. This is the exact same Special Issue Shiddik 2026 was published under — meaning the editors and likely reviewer pool are already primed on this precise topic, and you'd be submitting a direct, explicit rebuttal/extension into the venue where the paper you're differentiating from already lives. Tight deadline given your timeline in §6 — flag this to your supervisor early if you want to target it.
   - ***Tropical Medicine and Infectious Disease* — "Advances in Infectious Disease Surveillance: Climate-Sensitive, Biostatistical, and Simulation Modeling Approaches,"** deadline **31 December 2026**. Same journal, more relaxed timeline, strong thematic fit for the rolling-origin/spatial-validation angle specifically.

3. **Regional-focus strong option:** *Lancet Regional Health – Southeast Asia*.

4. **Correction on the Nature Portfolio Collection mentioned earlier:** the cross-journal "Digital Medicine for Infectious Diseases" Collection (Nature Communications, Nature Medicine, Communications Medicine, Communications Health, npj Digital Medicine) had a submission **deadline of June 3, 2026 — already passed**. It is not a live option. If your results end up strong enough for that tier, submit as a **standalone** paper to *npj Digital Medicine* or *Communications Medicine* directly (no Collection deadline pressure that way), rather than counting on that specific Collection.

5. Not realistic: *Nature* itself (the flagship journal) — say this plainly to your supervisor so expectations are calibrated from day one.

---

## 9. Immediate next steps (this week)

1. Read Shiddik 2026 in full (not just the abstract) and write a half-page comparison table: their data, their models, their validation, their stated limitations — this becomes the backbone of your "gap" section.
2. **Start the hospital/IEDCR data request now, and give yourself a hard 4-week deadline for a yes/no answer** (verbal commitment is enough to proceed with detailed planning; formal agreement can follow). Don't let this drift — it gates everything else. **While you're at it, make the same inquiry about serotype data access** (§4.3) — it's the single highest-value optional extension, and there's no reason to wait until the clinical-data question resolves to ask about it separately.
   - **If yes (illness-day-tracked data obtainable):** proceed with the full plan as written, RQ1 and the illness-day diagnostic model intact.
   - **If no, at the 4-week mark:** pivot immediately to the Plan B dengue-vs-co-circulating-febrile-illness framing described in §4.2, and start sourcing a multi-disease febrile-illness dataset instead of continuing to wait. A scoped-down project moving forward beats an ambitious one stalled on data that isn't coming.
3. Pull a test NASA POWER API query for 2–3 district centroids to confirm the data pipeline works before committing to the full build.
4. Download the HDX admin-boundary shapefiles and the Mendeley clinical dataset now, even before you've settled every detail — having the data in hand while you finalize scope saves time.
