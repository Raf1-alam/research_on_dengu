# Related-work notes

Eight papers from the Bangladeshi dengue literature, read 5 Sep 2026, with the
methodological detail that bears on this study. **The PDFs are deliberately not
committed** — the set mixes CC-BY and subscription titles, and redistributing the
latter from a public repository is a licensing problem. Retrieve them by DOI.

## Added 9 Sep 2026 — two Bangladesh papers that change what we may claim

Found while checking data sources. Both are 2026, both are Bangladesh, and neither was
in this file. **Read both before writing the introduction.**

### Fuad, Milki & Aziz (2026) · *PLoS ONE* 21(8):e0353069 · [10.1371/journal.pone.0353069](https://doi.org/10.1371/journal.pone.0353069)

*Tactical vs. strategic: an adaptable framework for horizon-dependent dengue forecasting
using data-driven approaches with serotype and climate covariates in Bangladesh.*

**This is the closest paper to our horizon-dependence claim and it was published first.**
11 districts, monthly, Jan 2017 – Dec 2023, 924 observations, horizons 1–6 months,
chronological train/validation/purge/test split with the test window Sep 2022 – Dec 2023.
Their conclusion: *"Model ranking changed with horizon, epidemic regime, district
contribution and whether the task was magnitude forecasting or alerting."* SARIMAX wins
one-step alerting (ROC-AUC 0.950); Prophet wins pooled magnitude at h=2–6.

**We cannot present "the best model depends on the horizon" as new.** What remains ours:
they compare *model families*, we test one *target parameterisation* against another with
seed as a blocking factor across 20 paired fits, giving an effect size and interval
(h=1 +5.13, dz=3.53; h=3 −4.42, dz=−1.50) rather than a ranking. Different claim, and it
must be worded as such, with this paper cited at the point we make it.

**Their calibration result is a gift to our motivation.** Only their TFT produces
intervals, at nominal 80%. Reported coverage is 0.17–0.24 in normal periods and
**exactly zero during outbreak periods** — *"none of the observed outbreak counts fell
within those intervals."* They document the failure and do not repair it. No conformal
prediction, no bootstrap intervals, no coverage conditional on any subgroup.

That is the strongest possible setup for our contribution: a published Bangladesh
district forecasting paper whose intervals collapse exactly when a forecast would be
used. Our group-conditional adaptive conformal reaches 0.884–0.892 against nominal 0.90
and removes the burden gradient. Cite this as the motivating failure, not as a rival.

### Shiddik, Toshi, Yesmin & Rahman (2026) · *Trop Med Infect Dis* 11(3):73 · [10.3390/tropicalmed11030073](https://doi.org/10.3390/tropicalmed11030073)

*District-level dengue early warning prediction system in Bangladesh using hybrid
explainable AI and Bayesian deep learning.*

All 64 districts, 2017–2024, DGHS + NASA + World Bank + BBS — the same sources we use.
Yearly split 2017–2023/2024 and monthly Jan 2022–Dec 2023/2024. Binary outbreak
classification against a mean-case threshold. MLP yearly **ROC-AUC 0.99**, ConvLSTM
monthly ROC-AUC 0.81; BYM2_RW2 with lagged effects, DIC 3671.055; humidity SHAP 0.314,
rainfall RR 1.303.

**Temporal holdout only. No spatial blocking, no prediction intervals, no coverage, no
conformal prediction.** Their own limitation section concedes validation rested
*"primarily on train–test splits within the same country"*.

**This is the empirical anchor for our optimism gap.** A yearly ROC-AUC of 0.99 across
64 districts is precisely the regime our C1 measures (0.9858). Adding both spatial and
temporal blocking costs 0.094 ROC and 0.475 PR. We can now say concretely: the published
district-level Bangladesh early-warning AUCs are obtained without spatial blocking, and
here is what that omission is worth. Argue it from our own matrix — do not assert their
number would fall, since we have not re-run their model.

**Housekeeping.** This is the "Shiddik" the legacy audit in
`unused/legacy_docs/comprehensive_findings_audit_and_shiddik_comparison.md` benchmarked
against under a different title. That legacy document also contains the withdrawn
clinical figures. Note for the supervisor: several withdrawn README claims (a BYM2 model
never fitted, humidity RR 1.313, a DIC reduction) sit close to this paper's published
BYM2 / RR 1.303 / DIC results. Whatever the origin, nothing resembling those numbers may
re-enter the manuscript except as a citation to this paper.

### Supporting, lower priority

| Paper | DOI | Why it matters |
|---|---|---|
| Faruk (2026), *Health Sci Rep* 9(4):e72207 | [10.1002/hsr2.72207](https://doi.org/10.1002/hsr2.72207) | Monthly national 2010–2024; ML models "showed significant test overfitting", SARIMAX generalised best. Independent support for our optimism argument. |
| Chowdhury (2026), *Health Sci Rep* 9(3):e72147 | [10.1002/hsr2.72147](https://doi.org/10.1002/hsr2.72147) | Monthly national 2000–2023, ARIMA vs XGBoost; useful only for long-run context. |
| Liu, Hossain & Hossain (2025), *Sci Rep* 15:35931 | [10.1038/s41598-025-19752-7](https://doi.org/10.1038/s41598-025-19752-7) | 5 divisions, monthly 2022–2023, XGBoost best. Coarser than us on every axis. |
| Bhuiyan et al. (2025), *J Trop Med* 2025:1709439 | [10.1155/jotm/1709439](https://doi.org/10.1155/jotm/1709439) | Symptom-based clinical classification, n=500, ANN 97.5%. Same genre as the clinical arm we dropped; cite if reviewers ask why we dropped it. |

**Resolution standing after these additions.** Shiddik is 64 districts yearly/monthly;
Fuad is 11 districts monthly; Liu is 5 divisions monthly. We are 64 districts **weekly**,
which is the finest spatio-temporal resolution in this literature and is defensible as
stated.

**Sources consulted 9 Sep 2026 via PubMed.**

---

## The closest prior work — read this before writing Section II

### Assessing dengue forecasting methods, Rio de Janeiro (2025) · [PMC11984044](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984044/)

**This paper already does most of our methodological stack and must be cited as closest
prior work.** Weekly dengue, 2016–2023, a 6-year moving-window validation (not a single
split), AR/MA/ARIMA/ETS/VAR/SARIMAX against SVM/RF/XGBoost/LSTM/Prophet, **adaptive
conformal prediction** for intervals, and **empirical 95% coverage reported by horizon**.

Consequences for our framing:

* **"First to validate interval coverage in dengue forecasting" is FALSE.** Do not write it.
  Conformal prediction, adaptive conformal, rolling-origin evaluation and empirical coverage
  reporting have all been done for dengue.
* What they cannot do is **stratify**: Rio de Janeiro is a single city, so coverage is
  aggregated over all observations with no grouping. Our contribution is what only a
  multi-unit panel can show — that marginal coverage conceals a large conditional failure
  (high-burden districts 0.841 against 0.907 for low-burden under split conformal, spread
  0.066, closed to 0.010 by group-conditional calibration).
* Position the paper as **extending** this work from one city to 64 districts, and as showing
  that the marginal number it reports is not sufficient in a multi-district system.
* Their LSTM/Prophet comparison also answers the "where is the deep learning?" objection
  without us having to run one.

## Bears directly on our claims

### Al Mobin (2024) · *Scientific Reports* 14:32073 · [10.1038/s41598-024-83770-0](https://doi.org/10.1038/s41598-024-83770-0)
"Forecasting dengue in Bangladesh using meteorological variables with a novel feature selection approach."
National **monthly** counts, 13 meteorological variables, a wrapper feature selector (SSFS),
reported **84.02% accuracy**, +12.63% accuracy and −70.82% MAPE from the selector.

Methods, quoted: *"Apply scaler … to the range [0,1] … The final dataset is split into three
parts namely: 70% for training set, 10% for validation set, and remaining 20% for test set."*

* The scaler is fitted on the **full dataset before the split** — normalisation leakage.
* No rolling-origin or prospective evaluation.
* No prediction intervals, so no coverage to validate.
* Lags out to 26–30 months on a monthly series.
* Claims **relative humidity is redundant** — our covariate ablation speaks to this directly.

**Use as the primary foil for the optimism gap in Section II.** It is current, high-profile and
Bangladesh-specific, which Shiddik is not to the same degree. Cite factually, not polemically.

### Naher et al. (2022) · *Health Science Reports* 5:e666 · [10.1002/hsr2.666](https://doi.org/10.1002/hsr2.666)
ARIMA vs ETS vs TBATS on national monthly counts, Jan 2008 – Jan 2020. Selects **ARIMA(2,1,2)**
by AIC/BIC. **Reports 80% and 95% prediction intervals.**

Two consequences:

1. **A claim had to be softened, twice.** "No study in this literature reports prediction
   intervals" is false — Naher reports them. And "coverage has never been evaluated for dengue"
   is also false: the Rio de Janeiro study above evaluates it. What survives is narrow and
   specific: within the **Bangladeshi** literature intervals are occasionally reported and never
   validated, and **nowhere in the dengue literature is coverage reported conditionally across
   spatial units.**
2. **A baseline had to be added.** ARIMA(2,1,2) is now in the suite (`table2`). Note the
   regime caveat in `table0_assumptions_register.csv`: the order was selected on national
   monthly data and is applied here at district-week resolution with 47% zeros. Report it as
   *the published benchmark does not transfer to operational resolution*, never as
   *we beat ARIMA* — the order was not re-selected for this data.

### Hossain et al. (2024) · *PLOS NTD* 18:e0012503 · [10.1371/journal.pntd.0012503](https://doi.org/10.1371/journal.pntd.0012503)
Spatio-temporal patterns 2019–2023 (2020 excluded), Moran's I and Anselin local Moran's I,
513,344 cumulative cases.

* Independently confirms **101,354 cases in 2019, Dhaka 51%** — corroborates our note that the
  district panel's 2019 rows cover only ~30% of that season.
* Establishes significant spatial autocorrelation — **cite to motivate the spatial holdout**
  rather than deriving it ourselves.
* Excludes 2020 for unavailable district-wise data, matching our panel's gap.

### Hasan et al. (2024) · *J Medical Entomology* 61:345–353 · [10.1093/jme/tjae001](https://doi.org/10.1093/jme/tjae001)
Two decades 2000–2022: trends, seasonality, monthly growth factors with CIs, temperature and
rainfall effects. Mean monthly growth factor 1.37 (SD 0.86), above 1 only in April–July.
Useful for Section I's seasonality framing.

## Epidemiological context (Introduction citations)

| Paper | DOI | Use |
|---|---|---|
| Hossain et al. (2023), *Trop Med Health* 51:37 | [10.1186/s41182-023-00528-6](https://doi.org/10.1186/s41182-023-00528-6) | 22 years of outbreaks, serotypes, future risk |
| Kayesh et al. (2023), *Trop Med Infect Dis* 8:32 | [10.3390/tropicalmed8010032](https://doi.org/10.3390/tropicalmed8010032) | increasing burden, severe dengue risk |
| Bhowmik et al. (2023), *Health Science Reports* 6:e1210 | [10.1002/hsr2.1210](https://doi.org/10.1002/hsr2.1210) | 2022 outbreak, public-health threat |
| Bonna et al. (2023), *IJID One Health* 1:100001 | [10.1016/j.ijidoh.2023.100001](https://doi.org/10.1016/j.ijidoh.2023.100001) | short communication, national picture |

## Still missing from the bibliography

The conformal references, which no Bangladeshi dengue paper cites and which the method section
needs: Romano et al. (CQR, NeurIPS 2019), Vovk (Mondrian conformal), Gibbs & Candès (adaptive
conformal, NeurIPS 2021), and Bates et al. or Angelopoulos & Bates for the general treatment.
