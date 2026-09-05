# Dataset Search & Verification
### Companion to the ICEEICT Course-Correction Plan
**Date:** 4 Sep 2026  
**Focus:** Two datasets that change what this paper can be  

> **Summary:**  
> I searched DataCite, figshare, Zenodo, Mendeley, OpenDengue, Hugging Face, HDX and DGHS, then downloaded and audited the serious candidates rather than trusting their descriptions. Eighteen sources are catalogued below. Two of them are worth rewriting the plan around.  
> * **87** DataCite dataset records screened  
> * **6** downloaded and audited  
> * **3** flagged contaminated  

---

## 01. Headline
*What actually changed.*

### The Two Finds

#### 1 · A 64-district weekly panel exists, and it validates against DGHS
A figshare deposit from July 2026 provides **16,256 district-weeks** — all 64 districts × 254 ISO weeks, covering 2019 and 2022–2026 — with climate, satellite indices, Google Trends and a mobility graph attached. I downloaded it and reconciled the case counts against published national totals myself:
* **2022:** within 0.6%
* **2023:** within 0.1%
* **2024:** within 0.2%

This is a drop-in upgrade from your 8 divisions to 64 districts, and it extends coverage into 2026.

#### 2 · The illness-day dataset your review said does not exist — now does
A Zenodo deposit from March 2026 contains **35,581 patients** from Cox's Bazar (Rohingya camps and host community), October 2021 – December 2024, with both symptom-onset date and examination date, real NS1 and IgG/IgM rapid-test results, the full WHO symptom set, warning signs and severity grading.

Your literature review said no publicly indexed Bangladeshi dataset records symptom onset. That was true when you wrote it. It stopped being true in March. The **H1 hypothesis** you fabricated data to test can now be tested for real — and I tested it. It holds (see §03).

---

## 02. Verified
*Downloaded, audited, and safe to build on.*

### Tier 1 — Use These

---

### [Verified] Reconciled District-Week Panel of Dengue Surveillance, Environmental, Behavioural and Mobility-Graph Data for Bangladesh
* **Authors / Source:** Md. Muhtasim Rahman Mim · figshare · [10.6084/m9.figshare.33040637](https://doi.org/10.6084/m9.figshare.33040637) · CC BY 4.0 · published 20 Jul 2026
* **Unit:** District-week (64 districts)
* **Span:** 2019, 2022–2026 · 254 weeks
* **Rows:** 16,256 × 54 columns
* **Size:** 6.1 MB, direct download
* **Attributes:** Cases, deaths, admissions, incidence per 100k. Climate (rain, temp min/mean/max, humidity) with 2/3/4-week lags. Satellite: NDVI, NDWI, land-surface temperature, VIIRS night lights. Static: population density, river density, road density. Google Trends (district-varying plus national). Plus a directed weekly inter-district mobility graph and a static 148-edge adjacency graph.

#### Reconciliation Against Published DGHS National Totals

| Year | Panel Sum | Official DGHS Total | Ratio (%) | Notes |
|:---:|:---:|:---:|:---:|---|
| **2022** | 62,021 | 62,382 | 99.4% | Validated |
| **2023** | 321,593 | 321,179 | 100.1% | Validated |
| **2024** | 101,016 | 101,214 | 99.8% | Validated |
| **2025** | 102,562 | — | — | Full season present |
| **2026** | 6,871 | — | — | Partial, to 29 Jun |

> **One caveat I found:**  
> The README claims the 2019 season was *"validated to the exact official national total (101,354)"*. The file sums to 30,257 for 2019, because rows begin 26 Aug 2019 and capture only the tail of that season. Use **2022–2026** as your modelling span and treat 2019 as a partial-season extra, not a full year.

**Why it matters for you:**  
Your `panel.py` was going to build 1,608 division-weeks. This gives you 16,256 district-weeks for free, makes leave-one-district-out spatial CV meaningful (64 units, not 8), kills the *"why not districts?"* reviewer objection outright, and hands you 2026 data no competitor has. It also removes the only real justification for skipping a spatial model.

---

### [Verified] Dengue Upsurge in the Rohingya Refugee Camps and Host Communities, Cox's Bazar (2021–2024)
* **Source:** Zenodo · [10.5281/zenodo.19219551](https://doi.org/10.5281/zenodo.19219551) · published 25 Mar 2026 · single 11.8 MB XLSX
* **Unit:** Individual patient
* **Span:** 19 Oct 2021 – 31 Dec 2024
* **Rows:** 35,581 × 84 columns
* **Population:** 32,077 refugee / 3,504 host
* **Core Variables:**
  * **Symptom Onset and Date of Examination:** Both complete for all 35,581 rows, so exact illness day is computable. Median 3 days, full spread 0–20+.
  * **RDT for NS1 Ag:** 23,626 positive / 5,136 negative / 6,819 not done.
  * **RDT for IgG/IgM:** 13,499 positive / 3,135 negative / 18,947 not done.
  * **Clinical Features:** Full WHO symptom checklist, vitals, tourniquet test, haemorrhagic signs, seven warning-sign flags, eleven comorbidity flags.
  * **Severity Grading:** Group A mild (31,836) · Group B warning signs (3,587) · Group C severe (158). Plus prior-dengue history, recent travel, mosquito exposure, standing water, household clustering.
  * **Temporal Tracking:** EPI week and year, so it doubles as a local weekly incidence series for Cox's Bazar.

> **Three limits to state in any paper that uses it:**  
> 1. *Clinical Diagnosis is "Dengue Fever" for 99.4% of rows*, so this is a case series — you can estimate test sensitivity by illness day but never specificity.  
> 2. *Testing was not randomised:* The NS1-versus-IgM ordering pattern is itself protocol-driven.  
> 3. *The population is 90% Rohingya refugees in one district*, so generalisation to national practice needs an explicit caveat.

---

### [Verified Live] Open-Meteo Historical Reanalysis & NASA POWER — District-Level Daily Climate
* **APIs:** `archive-api.open-meteo.com/v1/archive` · `power.larc.nasa.gov/api` · Both free, no API key required
* I called both APIs during this search and both returned valid data for Dhaka. Open-Meteo serves ERA5-based daily reanalysis back to 1940 at ~9 km; NASA POWER serves daily agro-climate variables. Either gives you per-district temperature, rainfall and humidity for all 64 district centroids in a single scripted pass — roughly ten minutes of runtime.

```text
Sample Call Comparison (Dhaka, 2025-01-01 .. 2025-01-03):
Open-Meteo:  tmax: [24.9, 24.1, 25.1]  tmin: [14.9, 13.7, 13.5]  precip: [0.0, 0.0, 0.0]  RH: [77, 79, 76]
NASA POWER:  T2M:  [16.62, 16.92, 17.27] (Note: T2M is a 2 m daily mean, not a max)
```

Use one source or the other consistently — the two differ in what they report and in vertical reference, so mixing them mid-panel introduces a spurious break. **Open-Meteo is the easier fit** because it returns daily max/min/precipitation-sum/RH-mean directly.

---

## 03. Result: The H1 Test, on Real Data
*Your fabricated hypothesis was right.*

I computed the NS1-versus-antibody kinetics curve from the Cox's Bazar file — **28,762 real NS1 rapid tests** and **16,634 real IgG/IgM rapid tests**, stratified by exact illness day. This is the figure your project invented data to produce.

#### Rapid-Test Positivity by Day of Illness · Cox's Bazar 2021–2024
*(Computed 4 Sep 2026)*

| Illness Day | NS1 Tests | NS1 Positive (%) | IgG/IgM Tests | IgG/IgM Positive (%) |
|:---:|:---:|:---:|:---:|:---:|
| **0** | 452 | 100.0% | 40 | 0.0% |
| **1** | 5,323 | 95.6% | 1,097 | 45.6% |
| **2** | 8,717 | 94.7% | 2,292 | 46.3% |
| **3** | 5,851 | 88.1% | 2,584 | 73.0% |
| **4** | 3,634 | 69.2% | 3,528 | 90.2% |
| **5** | 2,070 | 57.1% | 2,787 | 95.3% |
| **6** | 1,279 | 47.8% | 1,839 | 96.8% |
| **7** | 822 | 44.4% | 1,157 | 96.8% |
| **8–12** | 526 | 0.0% | 1,128 | 100.0% |

> **Key Finding:**  
> Linear interpolation puts the diagnostic crossover at **illness day 3.4**. Your fabricated paper claimed **3.8**. The hypothesis was sound; only the evidence was invented. You can now report the real number, from 45,396 rapid tests, and cite it.

**Caution on Days 8+:**  
Do not report the day 8+ row without investigating it. NS1 at exactly 0/526 and antibodies at exactly 1,128/1,128 is too clean to be biology. It almost certainly reflects a clinic protocol that stops ordering NS1 and switches to antibody testing after day 7. Report it as a protocol artefact, or restrict the curve to days 0–7 where the sample is dense and the pattern is smooth. Getting this wrong is the one way to turn a genuine result back into an embarrassing one.

---

## 04. Supporting Sources
*Useful, with conditions (Tier 2).*

### [Peer-Reviewed] OpenDengue — Global Dengue Case-Count Database, v1.3
* **Source:** [opendengue.org/data.html](https://opendengue.org/data.html) · GitHub `OpenDengue/master-repo` · described in *Scientific Data* (2024), [10.1038/s41597-024-03120-7](https://doi.org/10.1038/s41597-024-03120-7)
* I pulled the Spatial extract and filtered to Bangladesh: 234 records, all Admin-0 national — monthly 2008–2025 (207 rows) plus annual 1980–2010. No subnational disaggregation for Bangladesh; the authors name South Asia as a known gap.
* **Role in paper:** It is not a modelling panel, but an independent, peer-reviewed, citable national series covering 17 years for Section III. Use it to establish the post-2022 regime break quantitatively and to cross-check your panel's national totals.

### [Check First] Comprehensive Dengue Hematology and Clinical Dataset from Bangladesh
* **Source:** Mendeley Data · [10.17632/jsbmtk8hty.4](https://doi.org/10.17632/jsbmtk8hty.4) · CC BY 4.0 · v4, 2025
* 1,329 patients from Shaheed Suhrawardy Medical College Hospital and Dhaka Medical College Hospital, July–November 2025. Gender, age, NS1 result, WBC, platelets, RBC, haematocrit, lymphocyte %, neutrophil %, ALT, AST.
* **Leakage risk:** Same shape as the one that sank your clinical arm. The class label is the NS1 test result, and NS1 also appears as a variable. Drop NS1 from the feature set and the task becomes legitimate and interesting: predict NS1 positivity from routine CBC and liver enzymes alone (triage before rapid test). Leave NS1 in and you repeat F2 (AUC = 1.0).
* **Value-add over Jamalpur:** ALT and AST liver enzymes are present.

### [Clean, Small] Structured Clinical and Hematological Dataset for Early Dengue Diagnosis
* **Source:** Mendeley Data · [10.17632/673swz9tb4.1](https://doi.org/10.17632/673swz9tb4.1) · CC BY 4.0 · 2025
* 1,018 records (697 positive, 321 non-dengue) from Life Aid General Hospital, Munshiganj. Gender, age, platelets, WBC, fever, duration of fever, headache, muscle pain, vomiting, rash, location.
* **Context:** This column list is almost exactly the schema that was grafted onto your `dataset.csv`. Whoever fabricated your file appears to have copied this dataset's structure and generated values against the label. Here the values are real, and there is a genuine non-dengue comparison group.

### [Covariates] Monthly Division-Level Dengue Cases & Mortality with Climate and Environmental Indicators
* **Source:** Mendeley Data · [10.17632/cgwjshkx5k.2](https://doi.org/10.17632/cgwjshkx5k.2) · CC BY 4.0 · Dec 2025
* 574 division-months, January 2020 – December 2025, 8 divisions. Carries pollution, population density and atmospheric pressure alongside temperature and humidity, plus mortality. Too coarse to model on, but covers the 2020–21 gap and supplies two extra covariates.

### [Geospatial] HDX Administrative Boundaries & WorldPop Population Counts for Bangladesh
* **Source:** [data.humdata.org](https://data.humdata.org/dataset/worldpop-population-counts-for-bangladesh) · [hub.worldpop.org](https://hub.worldpop.org/geodata/summary?id=94) · CC BY 4.0
* ADM1 (8 divisions) and ADM2 (64 districts) shapefiles — the same boundaries used by Sarker et al. (2024). WorldPop gives 100 m gridded population for district denominators. Needed for Figure 5 and incidence-per-100k normalisation.

### [Large, Unverified] Dengue Daily Data and Environmental Variable — Bangladesh
* **Source:** Zenodo · [10.5281/zenodo.17421169](https://doi.org/10.5281/zenodo.17421169) · CC BY 4.0 · Oct 2025 · 1.25 GB single ZIP
* Suggests raster environmental layers rather than a tabular panel. Not recommended for download inside the 3-week deadline unless a specific covariate is missing.

### [Primary Source, Currently Down] DGHS Daily Dengue Status Reports
* **URL:** `old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report`
* Host currently does not resolve and live dashboard returns 403. Internet Archive holds snapshots (June 2022 – October 2023). Do not attempt manual PDF scraping inside your deadline; the figshare panel has already done it.

---

## 05. Avoid
*Known-bad, and spreading.*

### ❌ [Degenerate Label] A Comprehensive Dengue Dataset of Bangladesh — and its Mirrors
* **Sources:** Mendeley [10.17632/zdtc3n6xv2.3](https://doi.org/10.17632/zdtc3n6xv2.3) · Kaggle `kawsarahmad/dengue-dataset-bangladesh` · Hugging Face `fairhealth/bangladesh-dengue`
* Source of your `Not Used/dataset (2).csv`, where `Outcome == IgG` for all 1,000 rows. The contamination is spreading across three repositories under three DOIs.
* **Action:** Document the reason in your README so nobody re-uses it.

### ❌ [Fabricated] Your Local `Datasets/dataset.csv`
* Locally augmented from the Mendeley file. Nine columns appended with values generated conditional on the label (zero overlap in body temperature between cases and controls).
* **Action:** Delete it, and note why.

### ⚠️ [Superseded] Older National Monthly Aggregates
* Kaggle `samiulbari/dengue-dataset-of-bangladesh` (2008–2022) · `fazlyrabbi/dengue-incidents-weather-of-bangladesh` · figshare [27176046](https://doi.org/10.6084/m9.figshare.27176046.v1)
* OpenDengue covers the same ground through 2025 with peer-reviewed provenance, so cite that instead.

---

## 06. Decision
*How the plan changes. And how it does not.*

### What this does to the 22-day plan
Two new datasets, three weeks, one submission. The temptation is to use both. **Don't — one is a drop-in upgrade and the other is a second paper.**

| Strategy | Action | Details |
|---|---|---|
| **Adopt** | **Swap divisional panel for district panel** | Same pipeline, same experiments, same thesis — 64 units instead of 8, plus 2025 and 2026. Leave-one-district-out spatial CV becomes real, "why not districts?" disappears, and seasonal-naive baseline has enough history. Give this until **8 Sep**: if reconciliation is clean, proceed; if not, fall back to divisional. |
| **Add** | **OpenDengue & Open-Meteo as supporting layers** | OpenDengue gives Section III an independent 17-year national series to establish the regime break. Open-Meteo fills climate gaps and covers 2020–21. |
| **Bank** | **Cox's Bazar is your NEXT paper, not this one** | It is a stronger scientific find that should not be crammed into a six-page ICT forecasting paper with three weeks left. It deserves its own ethics framing, refugee-health context, and dedicated literature. Start it the day after submitting ICEEICT. |
| **Fallback** | **Cox's Bazar as the Gate 1 escape hatch** | If Gate 1 fails on 10 Sep (anchored growth skill does not survive bootstrap CIs), you can pivot to an illness-day severity triage model on 35,581 patients (predicting Group B/C at 10.5% base rate with the kinetics curve as motivating figure). Keep it in your pocket. |

### Caution About the District Panel
The figshare panel was built for spatial-spillover "ignition" forecasting and ships a pre-computed ignition target. **Do not inherit that target.** Use the raw `new_cases`, incidence, climate, and satellite columns, and define your own outbreak threshold as your plan already specifies.

---
*Search performed 4 Sep 2026 across DataCite (87 Bangladesh dengue dataset records), figshare, Zenodo, Mendeley Data, OpenDengue v1.3, Hugging Face, HDX, WorldPop, and dghs.gov.bd.*  
*Six datasets downloaded and audited directly; reconciliation and kinetics figures computed from those files.*  
*Confirm licenses before redistribution, and cite every source in manuscript data-availability statement.*
