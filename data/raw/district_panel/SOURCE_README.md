# A Reconciled District-Week Panel of Dengue Surveillance, Environmental, Behavioural, and Mobility-Graph Data for Bangladesh (2019, 2022–2026)

## Dataset Description

This dataset is a validated, multi-source, weekly district-level panel constructed to support
spatial-spillover ("emergence") forecasting of dengue in Bangladesh — the problem of predicting
*which currently low-incidence district* will cross an outbreak threshold in the near term, as
opposed to conventional models that forecast case counts within already-endemic locations.

Bangladesh's 2023 dengue epidemic (321,179 cases; 1,705 deaths) was distinguished less by its
scale than by its geography: transmission that had historically concentrated in metropolitan Dhaka
redistributed into previously low-incidence peripheral districts, a pattern attributed in the
epidemiological literature to Eid-festival travel and rapid urbanisation. No publicly available,
machine-readable, district-level surveillance resource previously existed to study this
redistribution quantitatively; district-wise case counts are published only as daily PDF press
releases by the Directorate General of Health Services (DGHS), with no consolidated weekly
time series and no accompanying inter-district mobility structure.

This resource fills that gap. It provides a reconciled weekly case panel for all 64 districts of
Bangladesh spanning the 2019 season and 2022–2026 (2020–2021 are excluded, as district-wise
reporting was disrupted by the COVID-19 pandemic and is unusable, consistent with prior literature),
fused with satellite-derived environmental indices, climate variables with transmission-relevant
lags, static geographic and demographic covariates, digital-behavioural (search-interest) signals,
and — distinctively — a directed, time-varying district-to-district mobility-flow graph that
encodes festival-driven travel (Eid and Durga Puja) alongside static geographic adjacency. An
emergence/ignition target and a "naive-district" flag are pre-computed to directly support
spillover-focused modelling, and the resource additionally includes independent external-validation
layers (Facebook Social Connectedness Index; observed Meta Movement Distribution mobility) that can
be used to assess whether a candidate transmission graph reflects real inter-district connectivity.

Every case value is derived from official government surveillance records and reconciled against
published national totals; no synthetic, imputed, or simulated case counts are included. The
dataset is intended for research on spatial epidemic forecasting, graph-based spatio-temporal
modelling, human-mobility–disease coupling, and early-warning system design, and it is directly
reusable beyond dengue for other vector-borne or spatially-diffusing disease systems in Bangladesh.

Bangladesh district-week panel, **2019 + 2022–2026** (2020–21 excluded: COVID-suppressed,
district-wise data unavailable), 64 districts × 254 ISO-weeks. The 2019 season (Aug–Dec, the
first big geographic-expansion outbreak, 101,354 cases) was recovered from  DGHS PDFs and validated to the exact official national total (101,354).
Everything here is validated and ready to feed the model. Join key everywhere:
**`(district_id, week_start)`**.

## Files

### 1. `Dengue.csv` — NODES (features + target + labels)
One row = one district-week (16,256 rows = 64 districts × 254 ISO-weeks). Groups:

| Group | Columns |
|---|---|
| keys | district_id, district, division, year, iso_week, week_start |
| **target** | **ignition** (1 if a genuinely QUIET district crosses the outbreak threshold in the next 1–4 wks; NaN when not eligible), **eligible** (1 = quiet = not in outbreak now AND ≤20 cases in prior 6 wks — excludes re-flares), **naive** (1 = 0 cases in prior 8 wks; the strict spillover subset) |
| target (loose, for comparison) | ignition_loose / eligible_loose — the OLD "below threshold this week" definition (dominated by re-flare; kept only to show the retarget's effect) |
| **spillover features** | **upstream_pressure** (Σ over flow-graph sources of `w_ij(t)·cases_i(t)` — festival-flow-weighted incoming case pressure), **nbr_cases** (Σ cases of adjacent districts), + `_lag1/_lag2` of each |
| incidence | incidence (weekly cases/100k), population |
| satellite | ndvi, ndwi, lst, viirs, ndvi_filled, viirs_change |
| climate | rain, temp_mean/min/max, humidity, precip_power + rain/temp/humidity `_lag2/3/4` |
| static | pop_density, river_density, road_density |
| behavioral | gtrends_dengue (district-varying), gtrends_dengue_nat / gtrends_dengue_bn_nat / gtrends_fever_bn_nat (**NATIONAL — identical across all 64 districts each week; treat as a seasonal clock, ablate in the "no-national" run**), festival_window, **festival_influx** (Eid redistribution: − Dhaka drains, + rural home districts fill) |
| labels (aux) | new_cases, new_deaths, currently_admitted, cumulative_cases, cumulative_deaths |

Notes: features are **UNSCALED** — fit scaling on the TRAIN split only. Lag columns are NaN
in the first weeks of each season (unavoidable) and never bridge the 2020–21 gap. `ndvi_filled`
flags interpolated (cloud-gap) cells. Target knobs (in `scripts/17_retarget_and_features.py`):
θ=5/100k, min 10 cases, quiet-window 6 wk / ≤20 cases, horizon 4 wk. Ignition base rate ≈ 1.3%
of eligible (120 positives); **71 of them fall in 2023** (the Dhaka→periphery year), so use
**leave-one-season-out with 2023 as a test fold** — the 2025–26 blocked split has only 15.

### 2. `edges_static.csv` — STATIC graph (adjacency)
`source_id, target_id, adjacency, centroid_dist_km` — 148 undirected border pairs (296 directed).

### 3. `edges_dynamic_weekly.csv` — DYNAMIC flow graph (the novelty)
`source_id, target_id, week_start, weight` — directed, row-normalised per (target, week).
`w_ij(t) = origin_cases_i(t) · (local_ij + β·festival(t)·homeward_ij)`:
- local = adjacency + gravity (year-round neighbour spread)
- homeward = urban→rural, distance-agnostic, **activated during Eid/Puja** (Dhaka→periphery seeding)



### 4. `districts.csv` — node reference
district_id (1–64), district, division, lat, lon (centroids for graph geometry).

### 5. `district_population.csv` — population per district (WorldPop 2020 zonal sum).


## Provenance
Labels: DGHS daily press releases, parsed & **validated 235/235 vs national totals**
(2023 total 321,593 ≈ official 321,179). No dummy/synthetic values. 
