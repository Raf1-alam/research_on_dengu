# Dataset Dossier

Search and verification performed 4 Sep 2026 across DataCite (87 Bangladesh dengue dataset records), figshare, Zenodo, Mendeley Data, OpenDengue v1.3, Hugging Face, HDX, WorldPop and dghs.gov.bd. Six datasets were **downloaded and audited directly** — the reconciliation figures below are computed from the files, not taken from their descriptions.

**Web version:** https://claude.ai/code/artifact/e8522b30-550e-44c5-822d-1da52ce061fc

---

## Tier 1 — verified, in the tree

### District-week panel → `data/raw/district_panel/`

figshare [10.6084/m9.figshare.33040637](https://doi.org/10.6084/m9.figshare.33040637) · Md. Muhtasim Rahman Mim · CC BY 4.0 · published 20 Jul 2026

- **16,256 rows × 54 columns** = 64 districts × 254 ISO weeks, 2019 + 2022–2026
- Cases, deaths, admissions, incidence per 100k
- Climate: rain, temp min/mean/max, humidity, with 2/3/4-week lags
- Satellite: NDVI, NDWI, land-surface temperature, VIIRS night lights
- Static: population density, river density, road density
- Google Trends (district-varying + national), festival window and influx
- Directed weekly inter-district mobility graph (`edges_dynamic_weekly.csv`) + 148-edge static adjacency (`edges_static.csv`)

**My reconciliation against published DGHS national totals:**

| Year | Panel sum | Official | Ratio |
|---|---:|---:|---:|
| 2022 | 62,021 | 62,382 | 99.4% |
| 2023 | 321,593 | 321,179 | 100.1% |
| 2024 | 101,016 | 101,214 | 99.8% |
| 2025 | 102,562 | — | full season present |
| 2026 | 6,871 | — | partial, to 29 Jun |

> **Caveat I found.** The deposit's README claims the 2019 season was "validated to the exact official national total (101,354)". The file sums to **30,257** for 2019 because rows begin 26 Aug and capture only the season tail. Model on 2022–2026; treat 2019 as a partial-season extra.

> **Do not inherit the `ignition` target.** It was built for spatial-spillover forecasting with its own thresholds. Use `new_cases` / `incidence` and define your own, or you adopt someone else's research question.

### Cox's Bazar patient dataset → `data/raw/coxsbazar/`

Zenodo [10.5281/zenodo.19219551](https://doi.org/10.5281/zenodo.19219551) · CC BY 4.0 · published 25 Mar 2026

- **35,581 rows × 84 columns**, 19 Oct 2021 – 31 Dec 2024
- 32,077 Rohingya refugee / 3,504 host community
- **`Symptom Onset` and `Date of Examination`, both complete for all rows** → exact illness day is computable. Median 3 days, spread 0–20+.
- `RDT for NS1 Ag`: 23,626 pos / 5,136 neg / 6,819 not done. `RDT for IgG/IgM`: 13,499 / 3,135 / 18,947.
- Full WHO symptom checklist, vitals, tourniquet test, haemorrhagic signs, 7 warning-sign flags, 11 comorbidity flags
- Severity: Group A mild 31,836 · Group B warning signs 3,587 · Group C severe 158
- Prior dengue, recent travel, mosquito exposure, standing water, household clustering; EPI week and year

This is the dataset your literature review said did not exist. It was deposited after the review was written.

**The H1 test, run on it (28,762 NS1 and 16,634 IgG/IgM rapid tests):**

| Illness day | NS1 n | NS1 pos | IgG/IgM n | IgG/IgM pos |
|---:|---:|---:|---:|---:|
| 0 | 452 | 100.0% | 40 | 0.0% |
| 1 | 5,323 | 95.6% | 1,097 | 45.6% |
| 2 | 8,717 | 94.7% | 2,292 | 46.3% |
| 3 | 5,851 | 88.1% | 2,584 | 73.0% |
| 4 | 3,634 | 69.2% | 3,528 | 90.2% |
| 5 | 2,070 | 57.1% | 2,787 | 95.3% |
| 6 | 1,279 | 47.8% | 1,839 | 96.8% |
| 7 | 822 | 44.4% | 1,157 | 96.8% |
| 8–12 | 526 | 0.0% | 1,128 | 100.0% |

**Crossover at illness day 3.4.** The fabricated paper claimed 3.8 — the hypothesis was sound, only the evidence was invented.

> **Three limits to state in any paper using it.** Clinical Diagnosis is "Dengue Fever" for 99.4% of rows, so this is a case series — you can estimate test *sensitivity* by illness day but never specificity. Testing was not randomised; the NS1-versus-IgM ordering pattern is protocol-driven. Population is 90% Rohingya refugees in one district — generalisation needs an explicit caveat.
>
> **Do not report the day 8+ row without investigating it.** NS1 at exactly 0/526 and antibodies at exactly 1,128/1,128 is too clean to be biology — almost certainly a clinic protocol that stops ordering NS1 after day 7. Report it as an artefact, or restrict the curve to days 0–7.

### Divisional daily → `data/raw/divisional/`

DGHS daily bulletins via Kaggle mirrors. `divisional_daily_2022_2025.csv` (11,200 rows, 8 divisions × 1,400 days, 2022-01-01 → 2025-10-31, with daily max/min temp, rainfall, humidity) and `divisional_daily_2019_2023.csv` (4,776 rows, 2019-08 → 2023-08, incomplete day coverage pre-2022).

Cross-validated against each other: **3,368 overlapping division-days, 98.4% exact agreement, r = 0.9986.** Annual totals reproduce official DGHS figures to ~1%. Now the low-resolution arm of the resolution ablation.

### Jamalpur hematology → `data/raw/clinical/`

Mendeley [10.17632/6fsrsk3mb8](https://doi.org/10.17632/6fsrsk3mb8) · n = 1,523, 19 CBC parameters, 68.4% positive, Feb–Sep 2024. Honest AUC ≈ 0.684 (sens 0.95, spec 0.39) — plausible for CBC-only screening. Optional; not in the current paper plan.

### Climate APIs (verified live, free, no key)

- **Open-Meteo** `archive-api.open-meteo.com/v1/archive` — ERA5-based daily reanalysis back to 1940, ~9 km. Returns daily max/min/precipitation-sum/RH-mean directly.
- **NASA POWER** `power.larc.nasa.gov/api/temporal/daily/point` — daily agro-climate variables.

Both were called during this search and returned valid data for Dhaka. **Use one consistently** — they differ in what they report and in vertical reference, so mixing them mid-panel introduces a spurious break. Open-Meteo is the easier fit.

---

## Tier 2 — supporting, not in the tree

| Dataset | Source | What it gives you |
|---|---|---|
| **OpenDengue v1.3** | [opendengue.org](https://opendengue.org/data.html), *Sci Data* [10.1038/s41597-024-03120-7](https://doi.org/10.1038/s41597-024-03120-7) | Bangladesh: 234 records, **all Admin-0 national** — monthly 2008–2025 plus annual 1980–2010. No subnational data (South Asia is a stated gap). Not a modelling panel, but an independent peer-reviewed 17-year national series for Section III's regime-break argument. |
| **Dhaka CBC + LFT, n=1,329** | Mendeley [10.17632/jsbmtk8hty.4](https://doi.org/10.17632/jsbmtk8hty.4) | Shaheed Suhrawardy + Dhaka Medical College, Jul–Nov 2025. Adds **ALT and AST**, absent from Jamalpur. ⚠️ **Leakage risk:** the class label *is* the NS1 result and NS1 also appears as a variable. Drop NS1 from features and the task becomes legitimate — predict positivity from CBC + LFT before the rapid test. Leave it in and you repeat F2. |
| **Munshiganj clinical, n=1,018** | Mendeley [10.17632/673swz9tb4.1](https://doi.org/10.17632/673swz9tb4.1) | 697 positive / 321 non-dengue, with **duration of fever**. Note: this column list is almost exactly the schema grafted onto the fabricated `dataset.csv` — whoever built that file appears to have copied this structure and generated values against the label. Here the values are real and there is a genuine comparison group. |
| **Division-monthly + environment** | Mendeley [10.17632/cgwjshkx5k.2](https://doi.org/10.17632/cgwjshkx5k.2) | 574 division-months, Jan 2020 – Dec 2025, carrying **pollution, population density, atmospheric pressure** and mortality. Too coarse to model on, but covers the 2020–21 gap and supplies two covariates the district panel lacks. |
| **Boundaries + population** | [HDX](https://data.humdata.org/dataset/worldpop-population-counts-for-bangladesh), [WorldPop](https://hub.worldpop.org/geodata/summary?id=94) | ADM1/ADM2 shapefiles (same boundaries as Sarker et al. 2024 and the PLOS One divisional model, so maps are comparable) + 100 m gridded population. Needed for F5 and incidence normalisation. |
| **Environmental rasters** | Zenodo [10.5281/zenodo.17421169](https://doi.org/10.5281/zenodo.17421169) | 1.25 GB single ZIP, likely raster layers. Not downloaded — open only if the district panel is missing a covariate you specifically need. |

### DGHS primary source — currently down

`old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report` is the origin of every dataset above and the URL cited by Shiddik et al., the PLOS One divisional paper and the 2019–23 spatio-temporal study. **The host does not currently resolve.** The live DGHS site still links to it; the link is dead, and the DGHS dengue dashboard returns 403.

The Internet Archive holds snapshots (confirmed captures Jun 2022 – Oct 2023), though the service was intermittently offline during this search. If district data ever needs reconstructing: Wayback snapshots of the report index, then parse the daily PDFs. **Do not attempt this inside the deadline** — it is weeks of work and the figshare panel has already done it.

---

## Do not use

Quarantined in `unused/contaminated_datasets/` — see `unused/README.md`.

| File | Defect |
|---|---|
| `dataset_FABRICATED.csv` | Nine columns appended to the Mendeley file with values generated **conditional on the label**. Zero overlap in body temperature between cases and controls across 1,000 patients. Any model returns AUC ≈ 1.000. |
| `mendeley_zdtc3n6xv2_DEGENERATE_LABEL.csv` | `Outcome == IgG` for **1,000 of 1,000 rows**. The label is a copy of a predictor. |
| `national_monthly_2008_2018_SUPERSEDED.csv` | Fine, but OpenDengue covers the same ground through 2025 with peer-reviewed provenance. |

**The degenerate file is mirrored across three repositories under three DOIs** — Mendeley [`10.17632/zdtc3n6xv2`](https://doi.org/10.17632/zdtc3n6xv2), Kaggle `kawsarahmad/dengue-dataset-bangladesh`, and Hugging Face `fairhealth/bangladesh-dengue` (I pulled the HF copy and confirmed it is byte-identical). **A new mirror is not a new dataset.**

Superseded national monthly aggregates, for completeness: Kaggle `samiulbari/dengue-dataset-of-bangladesh` (2008–2022), `fazlyrabbi/dengue-incidents-weather-of-bangladesh`, figshare [27176046](https://doi.org/10.6084/m9.figshare.27176046.v1) (Jan 2008 – Sep 2023). Nothing wrong with them; OpenDengue is the better citation.

### Not verified

Kaggle dataset pages block scraping and the API needs credentials, so column-level detail could not be checked for `shampabanik12/district-wise-dengue-dataset-for-bangladesh` or `Bangladesh Daily & Monthly Dengue Cases` (DOI 10.34740/kaggle/dsv/16345294). Given the figshare panel already validates against official totals at district resolution, this is not worth the time.

---

*Licences stated as published by each depositor — confirm before redistribution, and cite every source in the manuscript's data-availability statement.*
