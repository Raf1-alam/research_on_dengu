# Related-work notes

Eight papers from the Bangladeshi dengue literature, read 5 Sep 2026, with the
methodological detail that bears on this study. **The PDFs are deliberately not
committed** — the set mixes CC-BY and subscription titles, and redistributing the
latter from a public repository is a licensing problem. Retrieve them by DOI.

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

1. **A claim had to be softened.** "No study in this literature reports prediction intervals" is
   false. The defensible and still-novel claim: intervals are occasionally reported but their
   **empirical coverage has never been evaluated** — Naher plots model-based ARIMA bands and
   never checks whether they contain the truth.
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
