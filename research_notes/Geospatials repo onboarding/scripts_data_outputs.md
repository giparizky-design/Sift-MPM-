# Geospatials repo: execution scripts, configuration, data layout and outputs

Local-codebase study of /Users/gip/Geospatials, done 2026-11-08. Every claim cites a file path, plus a line number where useful. Read-only commands were used: `ls`, `du`, `grep`, `sed`, `git status`, `git log`, `git diff`. No secret values were printed. Values from `.env` and the CARTO key in git history are deliberately left out.

## 1. Script inventory, roles, entry points, live vs legacy

### Takeaway
`scripts/` holds 33 Python scripts, one shell script and one requirements file. The live operational spine is:
- `run_seven_pass_training.py`, the 2,985-line orchestrator for Kalimantan K=4 metals and K=3 coal.
- `dashboard_server.py`, which launches that orchestrator as a subprocess.
- `build_purged_ui.py`, which compiles the 77 MB UI bundle.
- The data builders `harvest_kalimantan_multimodal.py`, `build_east_kalimantan_250m.py` and `build_kutai_basin_100m.py`.

Roughly half the remaining scripts are one-off or legacy experiment drivers from 21–31 Oct (Yilgarn pilot, national Indonesia, 500 m/50 m island scans, Sangatta hotspots). Several of these depend on an unmounted external SSD.

### Cited Findings
**Canonical training and inference entry point**
- `run_seven_pass_training.py` is 2,985 lines, last modified 2026-11-08. Its argparse is titled "Kalimantan Dual-Domain (K=4 Metals + K=3 Coal) 7-Pass Training Engine". Flags:
  - `--domain {metals,coal,both}` (default `both`)
  - `--region` (default `KALIMANTAN`)
  - `--data-dir`, `--out-dir`, `--base-model-dir`
  - `--incremental` (default True)
  - `--reset-from-zero` / `--reset`
  - `--scan-only` (inference only: Pass 1 plus Passes 4–7)
  - `--on-schema-mismatch {retrain,prior}`
  - `--disable-unet`
  - `--train-10x` (500 estimators x 10 bags, 15 DAE epochs)
  - `--rounds` (1), `--trees-per-round` (50), `--bags` (3), `--dae-epochs` (2)
  - `--learning-rate` (0.05), `--max-depth` (6)
  - `--l-threshold` (0.50), `--min-area-ha` (5.0)
  - `--block-size-m`, `--active-heads`, `--run-label`, `--notes`

  — [scripts/run_seven_pass_training.py#L2626-L2780](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)
- Regions are restricted to a Kalimantan whitelist. Examples: KALIMANTAN, BORNEO, EAST_KALIMANTAN, KUTAI_BASIN_100M, KELIAN_CORRIDOR, SANGATTA_TREND, CUSTOM_POLYGON. Prefixes HOTSPOT_/KUTAI_/SCAN_/POLY_ are also allowed, and an explicit `--data-dir` overrides the check. Yilgarn therefore cannot run through this script. — [scripts/run_seven_pass_training.py#L2784-L2800](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)
- 18+ files reference the orchestrator. Wrappers that call it include `run_10_rounds_training.py`, `run_kutai_basin_3rounds_training.py`, `run_sangatta_10rounds_and_hotspot_scan.py`, `sagemaker_entry.py`, `run_kalimantan_full_scan.py`, `run_kalimantan_500m_and_100m_scans.py`, `run_indonesia_*` and `dashboard_server.py`. Unit and integration tests also reference it. — `grep -rlw run_seven_pass_training` over scripts/, tests/ and the docs
- On startup it archives the previous output directory with `archive_existing_run(out_dir, region_name)`. It copies `scanned_boundary.geojson` from the data directory, with `data/east_kalimantan/east_kalimantan_boundary.geojson` as fallback. Output goes to `maps/`, `targets/` and `models/` subdirectories. — [scripts/run_seven_pass_training.py#L551-L588](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)
- It writes:
  - `spatial_oof_metrics_{metals,coal}.json`
  - `spatial_oof_metrics.json`
  - `blind_holdout.json`, which is the same OOF payload
  - `seven_pass_summary.json`
  - a per-round history JSON

  — [scripts/run_seven_pass_training.py#L1795, L2272-2275, L2466-2469, L2583, L2618](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)
- It sets `OMP_NUM_THREADS=1` by default to avoid XGBoost/PyTorch OpenMP deadlocks. — [scripts/run_seven_pass_training.py#L21](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)

**UI and dashboard**
- `dashboard_server.py` (2,997 lines):
  - Flags: `--port 8765`, `--host 127.0.0.1`. — [L2984-2985](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
  - Launches `run_seven_pass_training.py` via `subprocess.Popen`, passing through region, domain, rounds, trees, bags, epochs, learning rate, depth, thresholds, block size, run label, `--scan-only`, `--reset-from-zero`, `--disable-unet`, `--active-heads` and `--notes`. — [L2392-2431](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
  - Before launching a polygon run, it rasterizes the polygon into a feature cube. — [L2380-2390](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
  - API routes: `/api/region|bundle`, `/api/satellite_layer`, `/api/dual_domain_metrics`, `/api/config|carto`, `/api/status`, `/api/performance`, `/api/model_introspect`, `/api/runs`, `/api/models`, `/api/polygons`, `/api/compare_runs`, `/api/run_bundle`, `/api/rerank`, `/api/validate_polygon`, `/api/train`, `/api/scan`, `/api/train/stop`, `/api/runs/rename`, `/api/runs/delete`, `/api/clean_old_training`. — [L2729-2973](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
  - The performance route shells out to `sysctl hw.memsize`, `vm_stat` and `pgrep -f run_kalimantan_50m_island_scan` (macOS-specific). — [L2536-2599](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- `build_purged_ui.py` is 6,490 lines. It compiles `ui/index.html` and the root `index.html`. Both are 77,059,389 bytes, because the data bundles are embedded. — [AGENTS.md §6](file:///Volumes/Seagate%202/Geospatials/AGENTS.md); `ls -la ui/` and the repo root
- It is called at the end of training by `run_seven_pass_training.py`, `run_10_rounds_training.py` and `run_sangatta_10rounds_and_hotspot_scan.py`. — script cross-reference grep

**Feature and dataset builders**
- `harvest_kalimantan_multimodal.py` (1,284 lines, modified 2026-11-08) runs seven workstreams:
  1. SAR (PALSAR-2)
  2. Sentinel-2
  3. Thermal (Landsat/ASTER)
  4. *not listed in the docstring excerpt*
  5. ESDM geology, GEM faults, GlobSed
  6. Gravity and EMAG2 magnetics
  7. Deposits, OSM transport, GFW

  Its only flag is `--enrich-only`. — [scripts/harvest_kalimantan_multimodal.py#L6-12, L1270-1284](file:///Volumes/Seagate%202/Geospatials/scripts/harvest_kalimantan_multimodal.py)
- `build_east_kalimantan_250m.py` (177 lines, no CLI) interpolates 21 channels from 500 m to 250 m for East Kalimantan. — [scripts/build_east_kalimantan_250m.py#L1-3](file:///Volumes/Seagate%202/Geospatials/scripts/build_east_kalimantan_250m.py)
- `build_kutai_basin_100m.py` (246 lines) builds the Kutai Basin cube: lon 115–118, lat −2.0 to 1.4, 100 m, 3,778 x 3,333 = 12.59 M cells. It is called by `run_kutai_basin_3rounds_training.py` (`--rebuild-cube` / `--reset`). — [scripts/build_kutai_basin_100m.py#L1-6](file:///Volumes/Seagate%202/Geospatials/scripts/build_kutai_basin_100m.py); [scripts/run_kutai_basin_3rounds_training.py#L110-111](file:///Volumes/Seagate%202/Geospatials/scripts/run_kutai_basin_3rounds_training.py)
- `add_real_sar_channels.py` (166 lines, untracked, 2026-11-08) adds or replaces `sar_*` channels from the JAXA PALSAR-2 annual 25 m mosaic, fetched through Planetary Computer STAC. Flags: `--data-dir` (accepts several), `--year` (default 2021). — [scripts/add_real_sar_channels.py#L1-5, L152-153](file:///Volumes/Seagate%202/Geospatials/scripts/add_real_sar_channels.py)
- `run_borneo_features.py` (240 lines) builds features for the Kelian 50 m corridor, and `train_borneo_pilot.py` (217 lines) runs 5-fold CV on it. — [AGENTS.md §2.2](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)

**Data acquisition (fetch_*)**
| Script | Lines | Mtime | Fetches |
|---|---|---|---|
| `fetch_real_pilot_data.py` | 311 | 10-21 | Yilgarn |
| `fetch_borneo_data.py` | 93 | 10-21 | MRDS Borneo |
| `fetch_kalimantan_iron_coal.py` | 195 | 10-21 | ESDM Fe/coal ArcGIS REST |
| `fetch_indonesia_geospatial_data.py` | 419 | 10-22 | national ESDM/GeoRIMA |
| `fetch_satellite_gravity.py` | 121 | 10-24 | Sandwell & Smith v33.1 from topex.ucsd.edu |

— per-file docstrings. Apart from gravity, none of these is called by another script. `fetch_satellite_gravity` is referenced from `src/features/geophysics.py`, and `fetch_indonesia_geospatial_data` from `dashboard_server.py`. — cross-reference grep

**Cloud training (untracked, 2026-10-27)**
- `launch_sagemaker_training.py` submits SageMaker jobs. Defaults: `--instance-type ml.m5.xlarge`, `--volume-size-gb 50`, `--max-runtime-hours 4`, `--region-target EAST_KALIMANTAN`. — [scripts/launch_sagemaker_training.py#L44-54](file:///Volumes/Seagate%202/Geospatials/scripts/launch_sagemaker_training.py)
- `sagemaker_entry.py` reads the `SM_*` environment channels. — [scripts/sagemaker_entry.py#L44-57](file:///Volumes/Seagate%202/Geospatials/scripts/sagemaker_entry.py)
- `ec2_train_bootstrap.sh` creates a 4 GB swapfile and `aws s3 sync`s data, models and code from a `sagemaker-ap-southeast-2-<acct>` bucket. — [scripts/ec2_train_bootstrap.sh](file:///Volumes/Seagate%202/Geospatials/scripts/ec2_train_bootstrap.sh)

**Legacy, one-off or experiment scripts (nothing references them, or they are single experiments)**
- Yilgarn pilot: `run_pilot.py` (623 lines; `--config config/pilot.yaml --stage all --version-tag v1`), `generate_pilot_data.py` (synthetic 10 km benchmark, `np.random.default_rng(2026)`), `run_data_audit.py`. — [scripts/run_pilot.py#L5, L581-583](file:///Volumes/Seagate%202/Geospatials/scripts/run_pilot.py); [scripts/generate_pilot_data.py#L45](file:///Volumes/Seagate%202/Geospatials/scripts/generate_pilot_data.py)
- Island and national scans:

  | Script | Lines | Notes |
  |---|---|---|
  | `run_kalimantan_full_scan.py` | 116 | ~1.1 km grid |
  | `run_kalimantan_500m_and_100m_scans.py` | 581 | |
  | `run_kalimantan_50m_island_scan.py` | 504 | 42 tiles, ~488 M cells, targets the external SSD |
  | `generate_kalimantan_50m_hd_polygons.py` | | reads `/Volumes/Seagate/Geospatials_Kalimantan_50m` |
  | `run_indonesia_national_pipeline.py` | | |
  | `run_indonesia_hd_upgrade.py` | | |
  | `run_viewport_7pass.py` | 420 | `--min-lon/--min-lat/--max-lon/--max-lat`, `--grid-size 200` |

  — docstrings; [scripts/generate_kalimantan_50m_hd_polygons.py#L21](file:///Volumes/Seagate%202/Geospatials/scripts/generate_kalimantan_50m_hd_polygons.py)
- Multi-round wrappers: `run_10_rounds_training.py` (140 lines, East Kalimantan 250 m), `run_kutai_basin_3rounds_training.py`, `run_sangatta_10rounds_and_hotspot_scan.py` (661 lines, five stages), and `train_unlabeled_25km_tile.py` (660 lines; `--tile_id`, `--r0`, `--c0`, `--seed 2026`). — docstrings
- Maintenance:
  - `move_to_geospatials_research.py` moves the RAG documents to `/Users/gip/Geospatials Research` and leaves symlinks behind. — [scripts/move_to_geospatials_research.py#L3](file:///Volumes/Seagate%202/Geospatials/scripts/move_to_geospatials_research.py)
  - `verify_rag.py` tests RAG retrieval.
- `/Volumes` currently contains only `Claude` and `Macintosh HD`. The Seagate SSD that `dashboard_server.py` (L2576-2610) and the 50 m scripts rely on is **not mounted**. — `ls /Volumes`

### Inferences
- Effective run modes:
  - **Train:** `run_seven_pass_training.py --region EAST_KALIMANTAN|KUTAI_BASIN_100M|KELIAN_CORRIDOR …`, run directly or through the dashboard.
  - **Rebuild data:** `harvest_kalimantan_multimodal.py`, then `build_east_kalimantan_250m.py`, then optionally `add_real_sar_channels.py`.
  - **View:** `dashboard_server.py`.
  - Yilgarn (`run_pilot.py`) is a separate, older pipeline driven by `pilot.yaml`.
- The 50 m whole-island and national-Indonesia scripts look frozen. They depend on the unmounted SSD and on Oct 22–23 data, and only show up in the UI through baked bundles.
- There is a lot of overlap: three multi-round wrappers re-implement loops that `--rounds` on the main orchestrator now provides.

### Gaps
- I did not trace whether `run_borneo_features.py` and `train_borneo_pilot.py` still run against the current `src/` after the v4 restructure. No tests were run, because the constraint was read-only.
- Workstream 4 of the harvest script was not confirmed.

## 2. Configuration (config/) and how it is consumed

### Takeaway
`config/` holds four YAML files:
- `pilot.yaml` (356 lines) drives only the Yilgarn `run_pilot.py` and `run_data_audit.py`.
- `borneo.yaml` (104 lines) and `logging.yaml` (39 lines) appear to have no Python consumers.
- `carto.yaml` (9 lines) is effectively superseded by environment variables read in `dashboard_server.py`.

The Kalimantan pipelines are configured entirely by CLI flags and constants hard-coded in the scripts.

### Cited Findings
- Sizes: `pilot.yaml` 356 lines, `borneo.yaml` 104, `logging.yaml` 39, `carto.yaml` 9. — `wc -l config/*`
- `pilot.yaml` has these sections: `pilot`, `grid`, `sources`, `features`, `pu_learning`, `evaluation`, `models`, `outputs`, `compute`. — [config/pilot.yaml](file:///Volumes/Seagate%202/Geospatials/config/pilot.yaml)
- Only `scripts/run_pilot.py` (L581) and `scripts/run_data_audit.py` (L47, L222) load `config/pilot.yaml`. `src/models/retrain.py#L72` mentions it in a docstring. No Python file references `borneo.yaml`, `logging.yaml` or `carto.yaml`. — grep over src/, scripts/, agents/, mcp_server/ and tests/
- `borneo.yaml` describes the Kelian 50 m AOI in EPSG:32750 (`aoi_bounds_geo [115.35,-0.15,115.55,0.05]`) and its sources: PALSAR-2, Copernicus GLO-30 at `s3://copernicus-dem-30m`, and others. — [config/borneo.yaml#L15-46](file:///Volumes/Seagate%202/Geospatials/config/borneo.yaml)
- `dashboard_server.py` answers `/api/config` and `/api/carto` from `CARTO_API_KEY`, `CARTO_RASTER_TILE_URL`, `CARTO_VECTOR_STYLE_URL` and `CARTO_DARK_MATTER_URL`, falling back to CARTO basemaps URLs. It does not read `carto.yaml`. — [scripts/dashboard_server.py#L2772-2792](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **Security:**
  - The committed HEAD version of `config/carto.yaml` contains a non-empty `api_key` and `?key=` query strings on the tile URLs.
  - The uncommitted working copy blanks them, a 4-line change.
  - **The key therefore remains in git history.**

  — `git diff config/carto.yaml`. Value deliberately not reproduced.
- Commit `8eb40c6` (2026-10-27) replaced the watermarked CARTO basemap with ESRI World Dark Gray. — `git log`

### Inferences
- If the repo is ever pushed or shared, the CARTO key should be rotated and/or purged from history.
- New contributors should treat `pilot.yaml` as Yilgarn-only. Kalimantan has no declarative config, so reproducibility depends on the CLI flags recorded in `seven_pass_summary.json` and the run catalog.

### Gaps
- I did not check whether `logging.yaml` is loaded indirectly, for example through a `src/` helper that builds the path dynamically.

## 3. Data layout, sizes, and git tracking

### Takeaway
`data/` is 13 GB and `outputs/` is 11 GB on disk. Most of that is in a few large binaries:
- `data/kalimantan/processed/feature_stack.joblib`: 2.3 GB
- `data/kalimantan/processed/dataset.parquet`: 1.4 GB
- `data/kutai_basin_100m`: 4.2 GB
- per-run `models/` directories: up to 2.1 GB each

`.git` is 1.6 GB, all loose objects with nothing packed. 503 files are tracked, including 246 `.tif`, 8 `.joblib`, 8 `.parquet` and 8 `.npy`. These raster and model binaries were committed **before** `.gitignore` excluded them, which is why `data/**/*.tif` shows as modified even though it is now ignored.

### Cited Findings
**Sizes**
- Top-level: data 13G, outputs 11G, .venv 1.7G, .uv_cache 1.1G, ui 74M, index.html 73M, docs 584K, audits 444K, agents 228K, src 1.9M, scripts 1.9M, tests 1.7M, mcp_server 80K, design-system 28K, notebooks 12K, app 12K. — `du -sh`
- Inside `data/`:

  | Path | Size | Contents |
  |---|---|---|
  | kalimantan | 6.6G | processed 6.2G, interim 403M, raw empty |
  | kutai_basin_100m | 4.2G | processed only |
  | east_kalimantan | 1.1G | processed 1.0G, interim 27M, boundary GeoJSON |
  | kalimantan_500m | 650M | |
  | pdfs | 177M | RAG corpus |
  | indonesia | 134M | |
  | hotspot_ek_* | 32–97M each | six 50 m hotspot dirs |
  | raw | 31M | |
  | borneo | 31M | |
  | unlabeled_tile*_25km_100m | ~30M each | |
  | sumbawa/sumatra/papua/sulawesi | 20–30M each | |
  | polygons | 24M | |
  | hotspot_* (Oct 22–23) | 14–20M each | older hotspot dirs |
  | processed | 11M | |
  | lancedb_store | 7.2M | |
  | raw_real | 3.6M | Yilgarn real data |
  | interim | 3.3M | |
  | markdown | 1.5M | |
  | external | empty | |

  — `du -sh data/*`
- In data/kalimantan, kutai_basin_100m and east_kalimantan, the files are 194 `.tif`, 3 `.parquet`, 3 `.joblib`, 3 `.geojson`, 2 `.json`, 1 `.npz`, 1 `.npy` and 1 `.gpkg`. — `find … | sort | uniq -c`
- Only one symlink exists under data/ and outputs/: `data/raw/crust -> crust1.0`. — `find -type l`
- `data/README.md` documents the provenance layout: `raw_real/` holds MINEDEX, WAMEX, GSWA geology/faults and the DEM. `borneo/` holds USGS MRDS (50 records: 28 genuine, 22 centroid artifacts), ESDM Fe (79) and coal (850). — [data/README.md](file:///Volumes/Seagate%202/Geospatials/data/README.md)

**Git**
- 503 tracked files. By extension: 246 tif, 72 json, 67 py, 34 gpkg, 24 csv, 8 tsx, 8 parquet, 8 npy, 8 md, 8 joblib, 4 yaml. — `git ls-files`
- `.git` is 1.6G. `git count-objects -vH` reports 1,195 loose objects totalling 1.59 GiB, in-pack 0. — `git count-objects -vH`
- `.gitignore`:
  - Ignores `data/raw/`, `data/interim/`, `data/external/`, `data/kalimantan_500m/`, `data/hotspot_*/` and `data/**/*.{joblib,parquet,npy,tif}`.
  - A comment says `data/raw_real/`, `data/borneo/`, `outputs/seven_pass_*/` and `outputs/reports/` are intentionally tracked for UI demo reproducibility.
  - The uncommitted diff adds `outputs/**/models/`, `outputs/run_history/**/models/`, `outputs/run_history/**/maps/*.{tif,vrt}`, `data/lancedb_store/`, `data/manifest.sqlite`, `data/markdown/`, `data/pdfs/` and `*.joblib`.

  — [.gitignore](file:///Volumes/Seagate%202/Geospatials/.gitignore); `git diff .gitignore`
- History: 32 commits from 2026-10-22 to 2026-10-28; latest is `ad5476a` on 10-28. Commits per day: 11 on 10-22, 6 on 10-23, 1 each on 10-24 and 10-25, 10 on 10-27, 3 on 10-28. — `git log`

**Uncommitted state, by status**
- 98 untracked (`??`), 67 modified (`M`), 43 deleted (`D`). — `git status --short | awk '{print $1}' | sort | uniq -c`

**Uncommitted state, by top-level directory**

| Directory | Untracked | Modified | Deleted |
|---|---|---|---|
| outputs | 46 | 23 | 40 |
| data | 7 | 21 | 3 |
| src | 11 | 10 | |
| scripts | 12 | 6 | |
| tests | 13 | 2 | |

Single items: `.gitignore`, `HANDOFF.md`, `config/carto.yaml`, `index.html` and `ui/` are modified. `AGENTS.md`, `CHANGELOG.md`, `CODEBASE_CORPUS.md`, `PROJECT.md`, `requirements.txt`, `audits/`, `agents/`, `mcp_server/` and `.antigravity/` are untracked. — `git status --short`

- What is untracked:
  - The entire v4.0 modular model layer: `src/models/{calibration,coal,conformal,fusion,metals}.py`, `src/schema/`, `src/evaluation/spatial_cv.py`, `src/features/{sar_palsar2,polygon_rasterizer,coal_and_multimodal_features}.py`, `src/outputs/model_introspection.py`.
  - 12 new unit-test files and `tests/integration/`.
  - Governance documents, including AGENTS.md and CHANGELOG.md.

  — `git status --short | grep '^??'`
- Deleted (`D`): the old archived runs `outputs/run_history/borneo/run_20261026_*` (manifests, maps, GPKGs), plus `data/kalimantan/processed/{dist_plate_boundary_km,dist_volcanic_arc_km,volcanic_arc_halo}.tif`. — git status snapshot

### Inferences
- About 11 days of substantive work (Oct 29 to Nov 8: the v4.0 upgrade, coal domain, SAR, polygon rasterizer, tests) exist only in the working tree. One disk failure or `git checkout .` would lose them. This is the main operational risk.
- `.git` will keep growing fast as long as every retrain of tracked `.tif`, `.parquet` and `.joblib` files is committed: it has no packs and holds 1.6 GB of binary blobs. Git LFS, DVC, or untracking the binaries (`git rm --cached`) would be the usual fix.
- The three deleted `data/kalimantan/processed` tracers (plate-boundary distance, volcanic-arc distance, arc halo) fit the leakage and proxy cleanup described in the audits. Removing them changes the feature schema, which is why `--on-schema-mismatch` exists.

### Gaps
- I did not measure the git-tracked byte total, for example with `git ls-files -z | xargs du`.
- I did not check whether `data/kalimantan/processed/dataset.parquet` (1.4 GB) is itself tracked. It is listed as `M` in git status, so it is.

## 4. Outputs and run artifacts

### Takeaway
Each region writes to `outputs/seven_pass_<region>/` with this layout:
- `maps/*.tif`: calibrated prospectivity, conformal 90% bounds, fused raw score, DAE reconstruction error, propensity e(x), and K=4/K=3 prospectivity, each with `_metals` and `_coal` variants
- `targets/`: ranked target GPKGs and site_cards JSON
- `models/`: `k4_ensemble.joblib`, `k3_coal_ensemble.joblib`, and `dae_{metals,coal}_fold_{0..4}.pt`
- metrics JSON and CSV files

The previous run is archived under `outputs/run_history/<region>/run_<ts>/`, and all runs are indexed in `training_runs_catalog.json`.

### Cited Findings
- Layout and sizes of `outputs/seven_pass_east_kalimantan`:
  - `models/` 2.1G: ten DAE fold `.pt` files plus two ensemble joblibs
  - `maps/` 106M
  - `targets/` 408K
  - also `seven_pass_summary.json`, `ten_rounds_history.json`, `spatial_oof_metrics*.json`, `spatial_cv_metrics*.csv`, `blind_holdout.json`, `scanned_boundary.geojson`

  — `find`, `du`
- Output directory sizes:

  | Directory | Size |
  |---|---|
  | seven_pass_east_kalimantan | 2.2G |
  | seven_pass_kelian_corridor | 2.1G |
  | seven_pass_kalimantan_50m | 1.8G |
  | seven_pass_kutai_basin_100m | 1.3G |
  | seven_pass_hotspot_east_kalimantan_50m | 951M |
  | seven_pass_borneo | 903M |
  | two unlabeled-tile runs | 722M each |
  | run_history | 634M |
  | everything else | under 15M each |

  The "everything else" group covers indonesia, papua, sumbawa, sumatra, sulawesi, yilgarn (672K), the hotspots and custom_viewport. — `du -sh outputs/*`
- `outputs/run_history/` contains east_kalimantan (2 runs, 231M), kelian_corridor (1 run, 1.1M), kutai_basin_100m (1 run, 402M) and `training_runs_catalog.json` (16K). — `ls`, `du`
- The catalog schema is `{version, updated_at, runs[], archived_runs_location}`. Each run records:
  - `run_id`, `region`, `label`/`display_name`, `relative_path`, `is_active`
  - `metrics` (`pr_auc_xgb`, `pr_auc_unet`, `pr_auc_fused`, `elkan_noto_c_hat`, `conformal_q_hat_90`)
  - `training_mode` (`standard` | `incremental_warm_start` | `transfer_adaptation`)
  - `n_ranked_targets`, `ore_categories`

  — [outputs/run_history/training_runs_catalog.json](file:///Volumes/Seagate%202/Geospatials/outputs/run_history/training_runs_catalog.json)
- Selected catalog entries:
  - EK 10-round run (2026-10-30): `pr_auc_fused` 0.3373.
  - `v4.0-smoke-test` (2026-11-07): `pr_auc_fused` 0.0311.
  - Currently active EK run `EXP_EK_250M_SANGATTA_CORRIDOR_V1vc` (2026-11-08, coal domain): `pr_auc_fused` 0.0504, 3 targets.
  - Kutai 100 m: 0.3766, then 0.1215 after the 10-31 rerun.
  - Kelian corridor (11-08): 0.1311 with `pr_auc_unet` null.

  — same file
- `seven_pass_summary.json` for the current EK run:
  - region EAST_KALIMANTAN, domain `coal`
  - `n_cells` 117,196; `n_positives` 4,160 (1,935 metal, 2,225 coal)
  - per-head coal PR-AUC: k0 0.0167, k1 0.0187, k2 0.1158
  - all metal head PR-AUCs null
  - `w_xgb` 0.3351, `w_unet` 0.6649; `elkan_noto_c_hat` 0.5; `conformal_q_hat_90` 0.5; 3 ranked targets
  - the top target card carries mineral-systems SHAP groups

  — [outputs/seven_pass_east_kalimantan/seven_pass_summary.json](file:///Volumes/Seagate%202/Geospatials/outputs/seven_pass_east_kalimantan/seven_pass_summary.json)
- `n_cells` = 117,196 contrasts with the documented 2.92 M valid cells for the 250 m East Kalimantan cube. — [AGENTS.md §2.2](file:///Volumes/Seagate%202/Geospatials/AGENTS.md); [HANDOFF.md#L39-52](file:///Volumes/Seagate%202/Geospatials/HANDOFF.md)
- HANDOFF says "All 29 historical training runs are cataloged" in the catalog. — [HANDOFF.md#L41](file:///Volumes/Seagate%202/Geospatials/HANDOFF.md)
- `outputs/reports/` holds Yilgarn-era `pipeline_2026102*.log` files (15), `blind_holdout_v{1,2_real}.json`, `spatial_cv_metrics_v{1,2_real}.csv` and `data_audit_report.md`. — `ls outputs/reports`
- `outputs/maps` and `outputs/targets` are empty (0B). — `du`

### Inferences
- `elkan_noto_c_hat` and `conformal_q_hat_90` both equal exactly 0.5 in several catalog entries. That looks like placeholder or fallback values, which is a concern under the AGENTS.md "no dummy 0.50" rule. Some `conformal_q_hat_90` values exceed 1.0 (1.17–1.18), which is odd for a probability-scale quantile.
- The current EK run's `n_cells` of 117,196, against 2.92 M documented, suggests it ran on a subsampled cube or a polygon subset. Worth checking before anyone treats it as the production run.
- The active EK output directory was overwritten by a coal-only run labelled "SANGATTA_CORRIDOR_V1vc". The metals maps in that folder may come from earlier runs, so the filenames are mixed-provenance.

### Gaps
- I counted 20 entries in the catalog excerpt printed. I did not reconcile that with HANDOFF's claim of 29 runs.
- I did not open the GPKG or site-card contents.

## 5. External data sources and credentials

### Takeaway
`.env.example` declares only the four CARTO variables. The real `.env` also defines UI-tool keys, an Elsevier key and `GEOLOGY_DATA_DIR`. Data comes from public REST, WFS, STAC and S3 endpoints: ESDM, GeoRIMA, GSWA SLIP, USGS MRDS, Copernicus DEM, AWS Terrarium, Scripps gravity, Planetary Computer PALSAR-2 and fraxen tectonic plates. Several "workstreams" that are labelled as real sensors actually generate seeded random fields.

### Cited Findings
- `.env.example` variable names: `CARTO_API_KEY`, `CARTO_RASTER_TILE_URL`, `CARTO_VECTOR_STYLE_URL`, `CARTO_DARK_MATTER_URL`. — [.env.example](file:///Volumes/Seagate%202/Geospatials/.env.example)
- `.env` variable names, values not read: the four CARTO variables plus `UI_UX_PRO_MAX_API_KEY`, `UI_UX_PRO_MAX_ESSENTIALS`, `UIUXPROMAX_API_KEY`, `UXKIN_API_KEY`, `ELSEVIER_API_KEY`, `GEOLOGY_DATA_DIR`. `.env` is mode 600 and gitignored. — `.env` key names only; [.gitignore](file:///Volumes/Seagate%202/Geospatials/.gitignore)
- Endpoint hosts found in the fetch, harvest and SAR scripts, with occurrence counts:

  | Endpoint | Count |
  |---|---|
  | `geoportal.esdm.go.id/gis4/rest/services` | 5 |
  | `geologi.esdm.go.id/georima` | 3, plus `/parsingv3/komoditas` |
  | `public-services.slip.wa.gov.au/public/rest/services` | 2 |
  | `topex.ucsd.edu/cgi-bin/get_data.cgi` | |
  | `s3.amazonaws.com/elevation-tiles-prod/terrarium` | |
  | `raw.githubusercontent.com/fraxen/tectonicplates` | |
  | `planetarycomputer.microsoft.com/api/stac/v1` | |
  | `mrdata.usgs.gov/services/wfs/mrds` | |
  | `copernicus-dem-30m.s3.eu-central-1.amazonaws.com` | |

  — grep over scripts/fetch_*.py, harvest_kalimantan_multimodal.py, add_real_sar_channels.py, run_indonesia_hd_upgrade.py, run_borneo_features.py and src/features/sar_palsar2.py
- `data/README.md` also lists the NOAA EMAG2v3 global magnetics GeoTIFF URL and the ESDM Fe and coal MapServer layers (79 Fe, 850 coal). — [data/README.md](file:///Volumes/Seagate%202/Geospatials/data/README.md)
- **Synthetic fields inside the "real" ingestion pipeline:**
  - Sentinel-2 workstream: NDVI and albedo are built from `np.random.default_rng(77)` Gaussian-filtered noise.
  - Thermal workstream: uses `default_rng(88)` noise.
  - "NOAA EMAG2v3 Magnetics" block: TMI is `gaussian_filter(rng.normal(0,45,…))` with `default_rng(555)`, and that field is then reduced to the pole.

  — [scripts/harvest_kalimantan_multimodal.py#L329-L345, L382, L563-L566](file:///Volumes/Seagate%202/Geospatials/scripts/harvest_kalimantan_multimodal.py)
- By contrast, the PALSAR-2 workstream says real data is used, and when it is unavailable the SAR channels are "omitted (not fabricated)". — [scripts/harvest_kalimantan_multimodal.py#L283-305, L960](file:///Volumes/Seagate%202/Geospatials/scripts/harvest_kalimantan_multimodal.py)
- MCP and agent credentials:
  - `mcp_server/geology_mcp_server.py` loads `.env` and uses `GEOLOGY_DATA_DIR`.
  - `agents/famous_literature_harvester.py` uses `ELSEVIER_API_KEY` for Scopus. Its docstring also lists "Library Genesis (LibGen) & Anna's Archive (via mirror resolvers)" as outlets.

  — [mcp_server/geology_mcp_server.py#L24](file:///Volumes/Seagate%202/Geospatials/mcp_server/geology_mcp_server.py); [agents/famous_literature_harvester.py#L1-12](file:///Volumes/Seagate%202/Geospatials/agents/famous_literature_harvester.py)
- AWS: `launch_sagemaker_training.py` hard-codes a SageMaker execution role ARN containing an account ID and uses an `ap-southeast-2` S3 bucket. — [scripts/launch_sagemaker_training.py#L36, L75-78](file:///Volumes/Seagate%202/Geospatials/scripts/launch_sagemaker_training.py)

### Inferences
- The synthetic Sentinel-2, thermal and magnetic layers contradict the "Provenance Honesty" guardrail in AGENTS.md §3. Any feature derived from them, such as RTP and the analytic signal of the random TMI, is noise that the model can still fit spatially. Audit these channels before trusting the metals PR-AUCs.
- The LibGen and Anna's Archive outlets in the literature harvester are a licensing/copyright risk.
- The hard-coded AWS account details, plus a CARTO key in git history, make the repo unsuitable for public release in its current state.

### Gaps
- I did not verify which of the processed `.tif` channels in `data/kalimantan/processed` or `data/east_kalimantan/processed` came from the synthetic code paths as opposed to real downloads.
- I did not check whether `sar_palsar2.py` has successfully downloaded real mosaics. There is no explicit log of that.

## 6. Other top-level directories (agents/, mcp_server/, notebooks/, app/, design-system/, docs/)

### Takeaway
These are side projects, not wired into the MPM pipeline:
- A geology-literature RAG system: `agents/` plus `mcp_server/`, exposed to the "Antigravity" IDE through `.antigravity/mcp.json`.
- A Next.js-style `app/` stub that would mount `ui/GeospatialWorkspace.tsx` but has no `package.json`.
- One Yilgarn data-audit notebook.
- Design-system markdown for UI agents.

### Cited Findings
- **`agents/`** (all untracked):
  - `famous_literature_harvester.py` (468 lines) targets 200 books and 100 papers.
  - `subagent_swarm_orchestrator.py` (329 lines) runs 6 subagents across 16 topics.
  - `master_geology_catalog.py` (363 lines, 78 KB of hard-coded book data) writes `master_geology_catalog.json` (91 KB).
  - `taxonomy_manifest.json` (18 KB).

  — `ls -la agents`; docstrings
- **`mcp_server/geology_mcp_server.py`** (689 lines) is a FastMCP server named "geology-rag-mcp". Tools:
  - `search_open_access_papers`, `search_usgs_publications`, `search_books_and_monographs`, `search_scopus_papers`
  - `download_and_ingest_pdf`
  - `query_geology_rag`, `get_rag_stats`

  It uses SQLite (`data/manifest.sqlite`), LanceDB (`data/lancedb_store`) and markdown chunking. — [mcp_server/geology_mcp_server.py#L45-671](file:///Volumes/Seagate%202/Geospatials/mcp_server/geology_mcp_server.py)
- `.antigravity/mcp.json` registers the server with `./.venv/bin/python mcp_server/geology_mcp_server.py`. `.antigravity/skills/` has two skills, `geology-harvester` and `geology-rag-expert`. — [.antigravity/mcp.json](file:///Volumes/Seagate%202/Geospatials/.antigravity/mcp.json)
- `requirements.txt` (untracked) lists only the RAG/MCP stack: `mcp[cli]`, `httpx`, `pymupdf`, `pymupdf4llm`, `lancedb`, `sentence-transformers`, `torch`, `pydantic`, `rich`, `typer`. The MPM stack is declared separately in `pyproject.toml` as project `mpm-yilgarn` v1.0.0: rasterio, geopandas, xgboost, shap, torch and others. — [requirements.txt](file:///Volumes/Seagate%202/Geospatials/requirements.txt); [pyproject.toml](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)
- RAG corpus on disk: `data/pdfs` (177M), `data/markdown` (1.5M), `data/lancedb_store` (7.2M), `data/manifest.sqlite` (20K). The `.gitignore` diff now ignores all of them. — `du`; `git diff .gitignore`
- **`app/`** has only `layout.tsx` (30 lines), `page.tsx` (17 lines) and `globals.css` (97 lines). `page.tsx` dynamically imports `../ui/GeospatialWorkspace` with `ssr:false`, and the layout title is "GEOSPATIALS // Autonomous Mineral Intelligence Workspace". There is no `package.json` or `next.config.*` at the repo root. — [app/page.tsx](file:///Volumes/Seagate%202/Geospatials/app/page.tsx); [app/layout.tsx](file:///Volumes/Seagate%202/Geospatials/app/layout.tsx); `ls`
- `ui/` contains `GeospatialWorkspace.tsx` (23 KB) and `components/`, `hooks/` and `types/` directories next to the compiled `index.html`. — `ls -la ui`
- `design-system/geospatials/` has `MASTER.md` (233 lines; page-specific files override it) and `pages/dashboard.md` (52 lines). — [design-system/geospatials/MASTER.md](file:///Volumes/Seagate%202/Geospatials/design-system/geospatials/MASTER.md)
- `notebooks/01_data_audit.ipynb` (292 lines) is the only notebook. — `ls notebooks`
- `docs/` contains `MPM_System_Design.md` (261 lines), `borneo_data_audit.md` (164 lines) and `session/transcript.jsonl` (414 lines). — `wc -l`

### Inferences
- The production UI is the compiled single-file `index.html`, served by `dashboard_server.py` or opened via `file://`. `app/` and `ui/*.tsx` are an abandoned or parallel React path: no build tooling exists in the repo.
- The RAG and agents system is a separate product sharing the venv. It only touches the MPM pipeline through shared `.venv` and `.env` and the `data/` directory.

### Gaps
- I did not open the `.antigravity/skills` contents or `docs/session/transcript.jsonl`.

## 7. Runtime and compute requirements

### Takeaway
Training runs on CPU on the developer's Mac by default. The U-Net uses CUDA only if available and never picks MPS automatically. A 250 m East Kalimantan incremental round takes about 11–12.5 minutes. A 50 m hotspot round takes about 50 minutes. Memory pressure comes from multi-GB joblib and parquet cubes. Cloud execution exists through SageMaker and EC2, with ml.m5.xlarge as the default.

### Cited Findings
- `get_best_device()` honours a `SIFT_DEVICE` override (cpu/cuda/mps). Otherwise it picks CUDA, then CPU, and "Defaults to CPU on macOS to eliminate Apple Silicon Metal command buffer synchronization stalls". — [src/models/unet_model.py#L57-69](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py)
- `OMP_NUM_THREADS=1` is forced in the orchestrator and in `model_introspection.py`. The PU isolation forest uses `n_jobs=-1`. — [scripts/run_seven_pass_training.py#L21](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py); [src/outputs/model_introspection.py#L27](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py); [src/labels/pu_negatives.py#L115](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py)
- Measured durations:
  - `ten_rounds_history.json` for East Kalimantan 250 m: rounds took 675–754 s each (about 2 h for 10 rounds), adding 50 trees per round; completed 2026-10-30.
  - `three_rounds_hotspot_history.json` for the EK 50 m hotspots: 2,956.6 to 3,084.7 s per round.
  - Another history file records 1,850.2 s.

  — [outputs/seven_pass_east_kalimantan/ten_rounds_history.json](file:///Volumes/Seagate%202/Geospatials/outputs/seven_pass_east_kalimantan/ten_rounds_history.json); [outputs/seven_pass_hotspot_east_kalimantan_50m/three_rounds_hotspot_history.json](file:///Volumes/Seagate%202/Geospatials/outputs/seven_pass_hotspot_east_kalimantan_50m/three_rounds_hotspot_history.json)
- `pilot.yaml` `compute` (Yilgarn): about 40 M active cells, a 12 GB feature matrix, a recommended "8-core CPU, 64 GB RAM, 500 GB SSD", GPU optional (A10, 24 GB) and only needed for the U-Net. — [config/pilot.yaml#L347-356](file:///Volumes/Seagate%202/Geospatials/config/pilot.yaml)
- Cloud:
  - SageMaker defaults: ml.m5.xlarge, 50 GB volume, 4 h max runtime. g4dn and p3 instances are handled specially (GPU).
  - The EC2 bootstrap adds a 4 GB swapfile "to prevent OOM".

  — [scripts/launch_sagemaker_training.py#L45-53, L70](file:///Volumes/Seagate%202/Geospatials/scripts/launch_sagemaker_training.py); [scripts/ec2_train_bootstrap.sh](file:///Volumes/Seagate%202/Geospatials/scripts/ec2_train_bootstrap.sh)
- Large in-memory artefacts: `feature_stack.joblib` is 2.3 GB and `dataset.parquet` is 1.4 GB in `data/kalimantan/processed`. Kutai 100 m has 12.59 M cells. The 50 m island scan claims about 488 M cells across 42 tiles. — `find -size +100M`; script docstrings

### Inferences
- A full dual-domain 10-round East Kalimantan run is a roughly 2-hour CPU job on the Mac. A 50 m hotspot 3-round run is roughly 2.5 hours. A machine with at least 32–64 GB RAM is advisable once multi-GB joblib and parquet files are loaded alongside the DAE folds.
- `n_jobs=-1` combined with `OMP_NUM_THREADS=1` means parallelism varies between passes.

### Gaps
- No peak-memory figures are recorded in any manifest I found.
- The dashboard samples memory live with `vm_stat` but does not persist it.
- Hardware specs of the development Mac were not checked.
