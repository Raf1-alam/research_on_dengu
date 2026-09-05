# Sources, Provenance, and Licence

This dataset is a derived research resource. Every layer is either a government-public record or an
openly-licensed product; the derived district–week panel and graphs are released under **CC-BY 4.0**.
Attribute the original sources as listed below.

## Files in this folder
| File | Description |
|---|---|
| `Dengue.csv` | Node panel — one row per (district × ISO-week), 16,256 rows (64 districts × 254 weeks), 2019 + 2022–2026. Features + emergence target + naive flag. |
| `edges_static.csv` | Static graph: shared-border adjacency (148 undirected / 296 directed pairs) + centroid distance. |
| `edges_dynamic_weekly.csv` | Dynamic festival-mobility flow graph, directed and row-normalised per (target, week). |
| `districts.csv` | Node reference: district_id, name, division, centroid lat/lon. |
| `district_population.csv` | Resident population per district (WorldPop 2020 zonal sum). |
| `README.md` | Full field dictionary and loading instructions. |

## Source attribution (per layer)
- **Dengue surveillance (labels):** DGHS Daily Dengue Press Releases (Directorate General of Health
  Services, Bangladesh) — government public record. Weekly cases derived by differencing cumulative totals;
  reconciled to official national totals on all 235 audited report days.
- **Satellite:** Sentinel-2 SR Harmonized (Copernicus, open); MODIS MOD11 LST (NASA, open); VIIRS
  night-lights (EOG/Colorado Mines, CC-BY 4.0).
- **Climate:** CHIRPS v2 (UCSB, open); NASA POWER (NASA, open).
- **Static / geography:** WorldPop (CC-BY 4.0); GADM v4.1 level-2 boundaries (free academic use);
  OpenStreetMap roads (ODbL); HydroSHEDS rivers (free, attribution).
- **Behavioural:** Google Trends (Google ToS, non-commercial research).


## Citation
If you use this dataset, please cite the accompanying manuscript and the deposited archive DOI.
