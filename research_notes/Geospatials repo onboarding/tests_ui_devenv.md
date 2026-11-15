# Geospatials repo: test suite, dev environment, and dashboard UI/server health (as of 2026-11-08)

All sources are local files under `/Users/gip/Geospatials` or read-only command output captured on 2026-11-08. "Source" links point to the file (with line numbers) or name the command that was run.

## Q1. Dev environment: Python, packaging, uv, heavy deps, lint/type config, CI, pre-commit

### Takeaway
The `.venv` works: Python 3.14.3 with an editable install of the `mpm-yilgarn` package, and all the heavy geospatial/ML deps import. Packaging is split across three dependency lists that don't agree with each other, there's no lockfile, no lint/type tools are installed, and there's no CI or pre-commit. The most urgent problem is the host machine itself: the data volume has about 537 MiB free (100% full), and that already breaks tests.

### Cited Findings
- `.venv` uses Python 3.14.3, created with the stdlib `venv` from Homebrew `python@3.14`, not uv (`command = /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv /Users/gip/Geospatials/.venv`). — [.venv/pyvenv.cfg](file:///Volumes/Seagate%202/Geospatials/.venv/pyvenv.cfg)
- `pyproject.toml` declares `requires-python = ">=3.11"`, but mypy targets `python_version = "3.11"` while the interpreter is 3.14, so the declared floor isn't exercised anywhere. — [pyproject.toml#L10, L80-83](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)
- The package is still named for the old pilot: `name = "mpm-yilgarn"`, `version = "1.0.0"`, described as "Orogenic Gold, Yilgarn Craton pilot". The `[project.scripts]` entry `run-pilot = "scripts.run_pilot:main"` exists. — [pyproject.toml#L6-8, L64-65](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)
- Editable install is present (`__editable__.mpm_yilgarn-1.0.0.pth` in `.venv/lib/python3.14/site-packages`), so `import src...` and `import scripts...` resolve. — command: `ls .venv/lib/python3.14/site-packages | grep editable`
- **Three disjoint dependency lists:**
  - `pyproject.toml` lists the geospatial/ML core (rasterio, geopandas, pyogrio, pyproj, shapely, xarray, numpy, scipy, scikit-image, scikit-learn, xgboost, shap, torch, pandas, pyarrow, joblib, matplotlib, seaborn, contextily, etc.). — [pyproject.toml#L14-49](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)
  - `requirements.txt` has NO geospatial/ML deps. It covers only the MCP/RAG server: `mcp[cli]<2`, httpx, pymupdf, pymupdf4llm, lancedb, sentence-transformers, torch, pydantic, rich, typer. — [requirements.txt#L1-15](file:///Volumes/Seagate%202/Geospatials/requirements.txt)
  - `scripts/requirements_sagemaker.txt` (untracked) is a third list: xgboost, joblib, pandas, pyarrow, scikit-learn, scipy, rasterio, geopandas, shapely, affine, scikit-image, tqdm. It has no torch. — [scripts/requirements_sagemaker.txt](file:///Volumes/Seagate%202/Geospatials/scripts/requirements_sagemaker.txt)
- **Undeclared runtime deps that happen to be installed:** `src/features/sar_palsar2.py` imports `pystac_client` and `planetary_computer` (installed: pystac-client 0.9.1, planetary-computer 1.0.0), and code imports `dotenv` (python-dotenv 1.2.4). None of the three lists declares these. — command: `grep -rhoE "^\s*(import|from) ..." src scripts mcp_server` + `pip list`
- Installed versions of the heavy deps: torch 2.14.1 (+torchvision 0.29.1), xgboost 3.4.1, rasterio 1.5.2, geopandas 1.2.0, shapely 2.2.0, pyproj 3.8.0, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, shap 0.54.1, scikit-image 0.26.0, xarray 2026.9.0, pyarrow 26.0.0. 153 lines of `pip list` output. — command: `.venv/bin/python -m pip list`
- GDAL Python bindings are NOT installed (`from osgeo import gdal` raises `ModuleNotFoundError`), and there's no `gdalinfo` on PATH. The code relies on rasterio/pyogrio wheels, which bundle GDAL. — command output
- `.venv` takes 1.7 GB on disk, and `.uv_cache` takes 1.1 GB. `uv` 0.13.1 is on PATH (`/opt/homebrew/bin/uv`), but there's no `uv.lock`, no `.python-version`, and the venv wasn't created by uv. `.uv_cache` is probably left over from `uv pip install` runs, and it isn't in `.gitignore`. — commands: `du -sh`, `find . -maxdepth 2 -name uv.lock`; [.gitignore](file:///Volumes/Seagate%202/Geospatials/.gitignore)
- **Lint/type config exists on paper only.** `[tool.ruff]` (line-length 100, select E,F,W,I,N,UP) and `[tool.mypy]` are configured, and the `dev` extra lists pytest-cov, ruff, mypy, ipykernel, jupyterlab. In the venv, ruff, mypy, pytest-cov, ipykernel, and jupyterlab are all MISSING, and black is absent too. Only pytest 9.1.1 (+anyio plugin) is installed. — [pyproject.toml#L51-59, L75-83](file:///Volumes/Seagate%202/Geospatials/pyproject.toml); command: `pip show` loop
- **No CI:** there's no `.github/` directory at all. **No pre-commit:** there's no `.pre-commit-config.yaml`. — command: `ls .github` → "No such file or directory"
- pytest config: `testpaths = ["tests"]`, `addopts = "--tb=short -v"`. `tests/conftest.py` only sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`. — [pyproject.toml#L71-73](file:///Volumes/Seagate%202/Geospatials/pyproject.toml); [tests/conftest.py](file:///Volumes/Seagate%202/Geospatials/tests/conftest.py)
- **Disk is effectively full:** `df -h` shows `/System/Volumes/Data` at 228Gi size, 199Gi used, **537Mi available, 100% capacity**. — command: `df -h /`
- Node v22.23.1 is available at `/Users/gip/.local/bin/node`. — command: `node --version`
- **Git hygiene:** HEAD is `ad5476a` (2026-10-28), while CHANGELOG runs to v4.4.1 dated 2026-11-08. `git status --porcelain` shows 98 untracked, 67 modified, and 43 deleted paths. These are untracked: `AGENTS.md`, `CHANGELOG.md`, `CODEBASE_CORPUS.md`, `audits/`, `src/schema/`, `src/models/{calibration,coal,conformal,fusion,metals}.py`, `src/evaluation/spatial_cv.py`, `src/features/{sar_palsar2,polygon_rasterizer,coal_and_multimodal_features}.py`, `tests/integration/`, and 12 of the 26 unit test files (including `test_v4_math_and_bugs.py`). — command: `git status --porcelain`; [CHANGELOG.md#L10](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md)
- The committed `HEAD:index.html` is 103,484,719 bytes (`git cat-file -s HEAD:index.html`), over GitHub's 100 MB per-file hard limit. The remote is `origin https://github.com/giparizky-design/Geospatials.git`. Loose object store is 1.59 GiB with 0 packs. — commands: `git cat-file -s`, `git count-objects -vH`, `git remote -v`
- `.gitignore` ignores `data/**/*.tif`, `*.parquet`, `*.npy`, and `*.joblib`, yet `data/kalimantan/processed/*.tif`, `dataset.parquet`, `labels.npy`, and `feature_stack.joblib` show as modified, so they were committed before the ignore rule was added. — [.gitignore#L24-27, L44](file:///Volumes/Seagate%202/Geospatials/.gitignore); git status

### Inferences
- A fresh clone + `pip install -e .` would NOT reproduce this venv. The MCP/RAG deps and the STAC/Planetary Computer deps come only from `requirements.txt` or from undeclared installs. Anyone recreating the environment needs `pip install -e ".[dev]" -r requirements.txt pystac-client planetary-computer python-dotenv` at minimum. Python 3.14 is bleeding-edge relative to the ">=3.11" floor.
- Because most of the v4 code (`src/models/metals.py`, `coal.py`, `calibration.py`, `conformal.py`, `src/schema/`), the governance docs, and most tests are untracked, the git remote does not represent the working system. A clone of `main` would likely fail to import modules that tracked files depend on (e.g. `src/models/pu_pipeline.py` is a facade re-exporting the untracked modules, per AGENTS.md §2.1). That wasn't verified by checkout.
- Any push of a commit including the current 77 MB `index.html` is under GitHub's 100 MB limit, but the existing HEAD blob of 103 MB suggests pushes may already be rejected (not verified).
- The full disk is the biggest operational risk. Each UI rebuild writes 2 × 77 MB, training archival needs ≥1 GB (`check_disk_space(min_free_gb=1.0)`), and four unit tests already fail because of it (see Q2).

### Gaps
- Whether `origin/main` has the 103 MB `index.html` or a push was ever rejected: I didn't contact the remote (read-only constraint).
- I didn't check whether uv was used for installs (no lockfile or uv metadata found beyond the cache dir).

## Q2. Tests: inventory, real run results vs. AGENTS.md claims, runtime, coverage gaps

### Takeaway
`tests/unit` now collects **226 tests, not 184**. A real run gave **222 passed, 4 failed, 7 warnings in 199.82 s**. All 4 failures are the same environmental `DiskSpaceError` (0.53 GB free < 1.0 GB minimum) in `archive_existing_run`, not logic regressions. `test_v4_math_and_bugs.py` does pass **13/13** (4.59 s), matching AGENTS.md. Nothing tests the dashboard UI build or the generated JS, and nothing tests the HTTP routes end to end.

### Cited Findings
- **Exact command and result (unit suite):**
  `cd /Users/gip/Geospatials && /usr/bin/time -p .venv/bin/pytest tests/unit -q -p no:cacheprovider`
  → `platform darwin -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0 ... plugins: anyio-4.15.1 ... collected 226 items`
  → `============ 4 failed, 222 passed, 7 warnings in 199.82s (0:03:19) =============` (`real 202.51`, exit code 1). — command output
- **Failures (verbatim, trimmed):**
  - `FAILED tests/unit/test_incremental_weights.py::test_archival_preserves_models_directory`
  - `FAILED tests/unit/test_milestone4_disk_governance.py::test_archive_existing_run_excludes_pt_checkpoints`
  - `FAILED tests/unit/test_milestone4_disk_governance.py::test_archive_existing_run_triggers_automatic_pruning`
  - `FAILED tests/unit/test_training_run_manager.py::test_archive_existing_run_success`
  - All four fail with: `src/models/training_run_manager.py:248: in archive_existing_run → check_disk_space(RUN_HISTORY_DIR, min_free_gb=1.0) → src/models/training_run_manager.py:63: raise DiskSpaceError(... 'Pre-flight disk space check failed on '/private/var/folders/.../pytest-70/...': 0.53 GB free, minimum required is 1.00 GB.`). — command output; [src/models/training_run_manager.py#L45-66](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py)
- **Warnings:** 4 × geopandas `UserWarning: Boolean Series key will be reindexed` (in `test_challenger_empirical_gate.py::test_empirical_sedimentary_coal_ranking_cliffs_vs_continuous_seams`), and 3 × `PendingDeprecationWarning: Use @ matmul instead of * mul operator` from `scripts/dashboard_server.py:145` (`src.transform * src.transform.scale(...)`, an affine 3.x deprecation). — command output; [scripts/dashboard_server.py#L145](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **Runtime:** about 200 s wall. Progress stalled for minutes inside `tests/unit/test_milestone2_algorithmic_integrity.py` (about 2.5 min elapsed at 104% CPU, around the 65% mark), so that file dominates runtime. No per-test durations were collected. — `ps` + progress output
- **v4 verification file:** `.venv/bin/pytest tests/unit/test_v4_math_and_bugs.py -q -p no:cacheprovider` → `13 passed in 4.59s`, which matches AGENTS.md's "13 passed". — command output; [AGENTS.md#L120](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)
- **Doc drift:** AGENTS.md says "Run Full Unit Test Suite (`184 passed`)" — [AGENTS.md#L116](file:///Volumes/Seagate%202/Geospatials/AGENTS.md). The CHANGELOG's v4.3.0 entry says "full 190-test unit suite pass (`190 passed`)" — [CHANGELOG.md#L91](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md). An older entry says "163 passed" — [CHANGELOG.md#L401](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md). Today's collection is 226 (208 `def test_` functions plus parametrization).
- **Integration suite (also run):** `.venv/bin/pytest tests/integration -q -p no:cacheprovider` → `1 failed, 17 passed in 6.35s`. The one failure is `test_empirical_challenger_2.py::test_challenge_5_disk_governance`, with the same `DiskSpaceError` ("0.60 GB free, minimum required is 1.00 GB"). — command output
- **Inventory (count of `def test_` / lines):**
  - unit: test_adversarial_v4_math_invariants 21/594; test_argon_leakage_remediation 8/322; test_aster_indices 12/348; test_borneo_phase1 14/419; test_borneo_phase2 9/317; test_borneo_phase3 5/286; test_challenger_empirical_gate 4/542; test_coal_and_multimodal_pipeline 8/401 (collects 18 via parametrize); test_fixes 14/263; test_geophysics 15/479; test_incremental_weights 4/191; test_leakage_and_provenance 4/125; test_metrics 5/103; test_milestone1_domain_isolation 7/174; test_milestone2_algorithmic_integrity 5/252; test_milestone3_execution_parity_and_full_raster 6/416; test_milestone4_disk_governance 10/328; test_outputs 2/83; test_polygon_rasterizer_and_api 6/210; test_pu_negatives 4/120; test_retrain 3/99; test_sar_palsar2_and_schema 7/143; test_seven_pass 9/450; test_spatial_splits 7/203; test_training_run_manager 6/207; test_v4_math_and_bugs 13/692.
  - integration: test_empirical_challenger_2 5/692; test_v4_challenger_2_adversarial 13/648. — command: `grep -cE "^\s*def test_"` per file, under [tests/](file:///Volumes/Seagate%202/Geospatials/tests/)
- **Server coverage is function-level only:** `test_leakage_and_provenance.py::test_rerank_is_read_only` imports `_recompute_targets_live` from `scripts.dashboard_server`, and `test_polygon_rasterizer_and_api.py::test_dashboard_server_polygonal_rgba_and_catalogs` imports server helpers. No test starts the HTTP server, hits a route, or checks CORS headers. — [tests/unit/test_leakage_and_provenance.py#L100-102](file:///Volumes/Seagate%202/Geospatials/tests/unit/test_leakage_and_provenance.py); [tests/unit/test_polygon_rasterizer_and_api.py#L159-161](file:///Volumes/Seagate%202/Geospatials/tests/unit/test_polygon_rasterizer_and_api.py)
- **No UI tests:** no test references `build_purged_ui`, `index.html`, or `apiFetch`. — command: `grep -rlE "dashboard_server|build_purged_ui|apiFetch|index\.html" tests` (only the two server-helper files above matched)
- **src modules with zero direct test references** (grep for module path / `from src.x import name` in tests): `src/features/sentinel2_indices.py` (226 lines), `src/ingestion/ga_download.py` (354), `src/ingestion/aster_download.py` (463), `src/harmonization/reproject.py` (475), `src/utils/io.py` (414), `src/models/xgboost_model.py` (437). Only one test file each references `sar_palsar2.py`, `cover_mask.py`, `aster_indices.py`, `deposit_labels.py`, `retrain.py`, and `polygon_rasterizer.py`. — command: per-module grep over `tests/`
- **Scripts:** tests reference only `scripts/run_seven_pass_training.py` (several), `scripts/harvest_kalimantan_multimodal.py`, and `scripts/dashboard_server.py` helpers. The other ~30 scripts in `scripts/` (fetchers, scan runners, SageMaker entry, `build_purged_ui.py`) have no tests. — command: `grep -rhoE "scripts/..." tests`

### Inferences
- The suite is healthy in logic terms. On a machine with ≥1 GB free, it would very likely be 226/226 unit and 18/18 integration (inferred: every failure is the pre-flight disk check). A new agent should free disk space before trusting a red run, or should not "fix" the tests by lowering `min_free_gb`.
- `xgboost_model.py` and `unet_model.py` may be exercised indirectly through the `pu_pipeline` facade, so "zero direct references" overstates the gap for `xgboost_model.py`. No coverage tool is installed, so I couldn't measure it.
- The most fragile, most recently broken surface (the generated 77 MB HTML/JS bundle, with the "JS syntax error" and "blank homepage" fixes) has zero automated guard.

### Gaps
- Line/branch coverage numbers: pytest-cov is not installed, and installing it would modify the venv.
- Per-test durations: I didn't run `--durations`, to avoid another 200 s run on a full disk.

## Q3. Dashboard UI: how `build_purged_ui.py` generates `ui/index.html` + `index.html`, size, embedding, features

### Takeaway
The UI is a single 6,490-line Python script holding the entire HTML/CSS/JS app as two giant triple-quoted strings. It splices in a JSON line `const EMBEDDED_BUNDLES = {...}` of about 76.7 MB, built by importing `scripts/dashboard_server.py` loaders, and writes identical 77,059,389-byte files to `ui/index.html` and `index.html`. The current inline JS passes `node --check` (exit 0), but no automated check runs it.

### Cited Findings
- **Build flow** in [scripts/build_purged_ui.py](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py):
  - L12-22: reads the *existing* `ui/index.html` and finds the `const EMBEDDED_BUNDLES = ` line (aborts with "Error: EMBEDDED_BUNDLES line not found!" if absent), so the build is self-referential and depends on a prior build artifact.
  - L31-61: `from scripts.dashboard_server import _load_region_bundle, _load_bundle_from_run_dir`, then loads bundles for EAST_KALIMANTAN, KALIMANTAN, BORNEO, 6 HOTSPOT_* 50 m domains, KUTAI_BASIN_100M, UNLABELED_TILE_25KM_100M, and UNLABELED_TILE2_25KM_100M with `force_reload=True`. If KALIMANTAN or BORNEO fail, it falls back to copying the EK bundle.
  - L65-80: also embeds *every* catalogued run in `outputs/run_history/training_runs_catalog.json` that has `maps/calibrated_prospectivity.tif` ("for 0ms instant switching").
  - L83-84: on any exception it prints `Warning: Could not update bundles` and silently reuses the old embedded line.
  - L86-112: builds the run-selector `<option>` list from the catalog.
  - L114+: `html_head = '''<!DOCTYPE html>...`; L3098 `html_script = '''...`; L6479 `full_html = html_head + '    ' + bundle_line + html_script`; L6481-6485 writes `open('ui/index.html','w')` and `open('index.html','w')`. Paths are **relative to CWD**, so it only works when run from the repo root.
- **Size:** `index.html` and `ui/index.html` are byte-identical (`cmp` → IDENTICAL), each **77,059,389 bytes, 6,375 lines**. Line 2996 alone is 76,747,535 bytes (the EMBEDDED_BUNDLES JSON). There's one inline `<script>` block of 76,910,960 chars, plus external Leaflet `https://unpkg.com/leaflet@1.9.4/dist/leaflet.js` (line 15). — commands: `wc`, `cmp`, `awk length`, Python regex extraction
- **Embedded bundle contents:** 31 keys. The largest are EAST_KALIMANTAN ≈8.9 MB, KUTAI_BASIN_100M ≈8.7 MB, HOTSPOT_EAST_KALIMANTAN_50M ≈8.5 MB, and KALIMANTAN ≈7.5 MB. 19 `RUN_*` archived-run bundles of 0.1–4.2 MB each hold base64 PNG heatmaps (`heatmap_png`, `conformal_l_png`, `coal_heatmap_png`), `predicted_targets_geojson`, `faults_geojson`, `site_cards`, `cv_folds`, and so on. Several runs are near-duplicates (e.g. three KUTAI_BASIN_100M runs at 4198 KB). — command: JSON parse of line 2996
- **`node --check` result:** extracted the inline script to the scratchpad and ran `node --check inline_0.js` → **exit 0** (parses). — command output
- **No automated JS validation:** `build_purged_ui.py` contains no `node`/`subprocess`/parse step (grep only matches CSS/JS identifiers like `.pulse-node`), there's no CI, and no tests touch the bundle. — command: `grep -nE "node|esprima|check_js|subprocess" scripts/build_purged_ui.py`
- **Recent UI fix history:** `adc71e2` "fix(ui): resolve blank homepage bug by restoring .app-shell DOM nesting" (2026-10-27); `0c4d6ef` "fix(ui): eliminate JS syntax error in generated bundle to restore all tab selectors" (2026-10-28; 12 lines in build_purged_ui.py, 14 in each html); `ad5476a` tooltip feature (167 lines builder, 95 lines server). — command: `git log`, `git show --stat`
- **Implicit rebuild trigger:** `scripts/run_seven_pass_training.py` L2973-2980 calls `build_purged_ui.main()` at the end of every CLI training run unless `MPM_SKIP_UI_REBUILD=1`, with failures only logged as warnings. The dashboard server sets `MPM_SKIP_UI_REBUILD=1` for UI-launched jobs (`scripts/dashboard_server.py` L2328). `run_10_rounds_training.py` and `run_sangatta_10rounds_and_hotspot_scan.py` also import it. — [scripts/run_seven_pass_training.py#L2973-2980](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py); [scripts/dashboard_server.py#L2328](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **App structure (title "SIFT // Spatial Inference & Fault Targeting"):** 3 top-level views with nav buttons `nav-btn-home`, `nav-btn-workspace`, and `nav-btn-training` (L1706-1714), with containers `view-home` (L1872), `view-workspace` (L2127), and `view-training` (L2563). Hash routing uses `#workspace` / `#training` (L6466-6472).
  - Home sections: "ACTIVE EXPLORATION THEATERS" (L1933), "THE 7-PASS GEOSPATIAL PIPELINE" (L2007), "21-CHANNEL MULTI-SENSOR STACK" (L2075).
  - Workspace: Leaflet map with "PIPELINE PARAMETERS" (L2141), "MAP OVERLAYS & SENSORS" (L2183; satellite layer selector incl. the new PALSAR-2 optgroup per CHANGELOG v4.4.1), target site card (L2388), "RANKED DRILL TARGET PORTFOLIO" (L2525), and analytics sub-tabs `conformal`, `prauc`, `unet`, and `xgbtree` (model "brain" views incl. an XGBoost root-to-leaf split trace, L3040, L5902).
  - Training view: "CUSTOMIZED TRAINING, MODEL IMPROVEMENT & POLYGONAL SCAN" (L2585), "REAL-TIME CONVERGENCE & PROCESS TELEMETRY" with a "STDOUT / STDERR STREAM CONSOLE" (L2874, L2936), and "MODEL VERSION HISTORY & REPOSITORY" (L3054).
  - On DOMContentLoaded it calls `initMap`, `fetchRunHistory`, `fetchAvailableModels`, `fetchAvailablePolygons`, and `pollTrainingStatus`, polls `/api/performance` every 4 s, and polls status every 3.5 s (L6453-6465). — [scripts/build_purged_ui.py](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py)
- **JS smell:** L6133 `steps.append ? steps.append({...}) : steps.push({...})` is a Python-ism in JS. It's harmless (arrays have no `.append`), but it's a sign of hand-authored JS inside a Python string with no linting. — [scripts/build_purged_ui.py#L6133](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py)
- External runtime dependencies: Google Fonts and unpkg Leaflet 1.9.4. CHANGELOG/commit `8eb40c6` mention an ESRI World Dark Gray basemap. — [scripts/build_purged_ui.py#L114-130](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py); git log

### Inferences
- Rebuilding the UI is heavy and side-effectful. It imports the whole server module, re-reads rasters for about 12 regions plus every catalogued run, and writes about 154 MB. With 537 MiB free, a rebuild plus one training archive could exhaust the disk.
- Because the bundle line is reused on failure, a broken loader may go unnoticed: the UI keeps showing stale data while the build "succeeds".
- Per AGENTS.md §6.1, `index.html`/`ui/index.html` must never be hand-edited. Even the CLI training script regenerates them, so agents should expect these files to change after any training run.
- A cheap guard is to extract the `<script>` and run `node --check` (as done here) in a test or a post-build step. That would have caught the bug fixed in `0c4d6ef`.

### Gaps
- I didn't render the page in a browser (no visual or runtime console check). A `node --check` pass doesn't rule out runtime errors or DOM-nesting bugs like the "blank homepage" one.
- I didn't time how long a rebuild takes. CHANGELOG v4.3.1 mentions "~60 seconds" of blocking for the static HTML rebuild ([CHANGELOG.md#L54-60](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md)).

## Q4. `dashboard_server.py`: API routes, subprocess training, and AGENTS.md §6 invariants (apiFetch, CORS)

### Takeaway
The server is a 2,997-line stdlib `ThreadingHTTPServer` (default `127.0.0.1:8765`). It has 14 GET routes and 9 POST routes, with no auth. `POST /api/train` and `POST /api/scan` launch `scripts/run_seven_pass_training.py` via `subprocess.Popen`, and `POST /api/runs/delete` runs `shutil.rmtree`. The AGENTS.md §6 invariants hold: CORS `*` is on every JSON/HTML/OPTIONS/HEAD response, and all 18 client fetches go through `apiFetch`. Two risks stand out: wide-open CORS combined with destructive POSTs, and an API key echoed by `/api/config`. A server instance was already running on 8765 (PID 5579, about 28 min uptime) and returned `GET /` 200 with 77,059,389 bytes.

### Cited Findings
- **Server core:** `ThreadingHTTPServer((args.host, args.port), DashboardHandler)`, with args `--port` (default 8765) and `--host` (default `127.0.0.1`). `log_message` is overridden to silence access logs. — [scripts/dashboard_server.py#L2982-2994, L2667-2668](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **GET routes** ([L2712-2861](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)):
  - `/` and `/index.html`: serves `ui/index.html` (77 MB read fully into memory via `read_bytes()` on each request) (L2714-2727)
  - `/api/region` and `/api/bundle?region|name=`: `_load_region_bundle(region)` (L2729-2734)
  - `/api/satellite_layer?region=&layer=`: `_render_satellite_layer` → base64 PNG + bounds (L2736-2748)
  - `/api/dual_domain_metrics`: reads `sift_dual_domain_metrics.json` from audits/, data/kalimantan/processed/, or outputs/seven_pass_kalimantan/ (L2750-2770)
  - `/api/config` and `/api/carto`: returns `CARTO_API_KEY` from the environment plus tile URLs with `?key=` appended (L2772-2792)
  - `/api/status` and `/api/train/status`: `_TRAIN_STATE` dict (L2794-2797)
  - `/api/performance`: `_get_system_performance()`, which shells out to `sysctl`, `vm_stat`, `pgrep -f run_kalimantan_50m_island_scan`, and `ps` (L2799-2802; L2536-2599)
  - `/api/model_introspect?region=&refresh=` (L2804-2810)
  - `/api/runs?region=`: `list_training_runs` (L2812-2817)
  - `/api/models`: `_list_available_models` (L2819-2822)
  - `/api/polygons`: `_list_available_polygons` (L2824-2827)
  - `/api/compare_runs?run_a=&run_b=` (L2829-2838)
  - `/api/run_bundle?run_id=&region=`: `ACTIVE`/`latest` maps to the region bundle, otherwise looks up the catalog `relative_path` (L2840-2860)
  - anything else: 404 JSON (L2861)
- **POST routes** ([L2863-2979](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)):
  - `/api/rerank`: `_recompute_targets_live(region, l_threshold, min_area_ha, weights w1..w4)` (L2868-2883)
  - `/api/validate_polygon`: normalize + `validate_polygon_and_resolution` (400 on error) (L2885-2894)
  - `/api/train` and `/api/scan`: `_trigger_background_training(params, scan_only=False/True)` (L2896-2904)
  - `/api/train/stop`: `_ACTIVE_TRAIN_PROC.terminate()` (L2906-2924)
  - `/api/runs/rename`: edits the catalog label (L2926-2944)
  - `/api/runs/delete`: removes the run from the catalog and `shutil.rmtree(run_dir)` if the path contains "run_history" and the run isn't active (L2946-2971)
  - `/api/clean_old_training {dry_run}`: `clean_old_training_data(keep_active_only=True)` (L2973-2977)
  - else 404
- **Subprocess training:** `_trigger_background_training` (L2214) returns `already_running` if `_TRAIN_STATE["running"]` (a single-job guard). It optionally rasterizes a polygon feature cube first, then builds `cmd = [sys.executable, scripts/run_seven_pass_training.py, --region, --domain, --rounds, --trees-per-round, --bags, --dae-epochs, --learning-rate, --max-depth, --l-threshold, --min-area-ha, --block-size-m, --run-label, (--data-dir, --out-dir, --base-model-dir, --scan-only|--reset-from-zero|--incremental, --disable-unet, --active-heads, --notes)]` and runs `subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, env=env)` in a worker thread. It parses `>>> [TELEMETRY] {json}` lines into `_TRAIN_STATE` (pass labels 1–7). — [scripts/dashboard_server.py#L2214-2240, L2391-2470](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **CORS invariant (AGENTS.md §6.3) is satisfied:** `_send_json` (used for all 200/400/404 JSON) sets `Access-Control-Allow-Origin: *` and `Access-Control-Allow-Private-Network: true` (L2670-2684). HTML GET (L2717-2721), HEAD (L2686-2702; note HEAD returns 200 for *any* path), and OPTIONS 204 (L2704-2710, Allow-Methods `GET, POST, OPTIONS, HEAD`, Allow-Headers `Content-Type, Authorization`) do the same. — [scripts/dashboard_server.py#L2670-2721](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **Gap in the CORS invariant:** `do_POST` runs `json.loads(self.rfile.read(length))` at L2866 with no try/except, so a malformed body raises before any response is written, and the client gets no CORS-bearing error response. — [scripts/dashboard_server.py#L2863-2866](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)
- **apiFetch invariant (AGENTS.md §6.2) is satisfied:** `getApiEndpoint(url)` rewrites relative paths to `http://127.0.0.1:8765` + path when `!window.location.protocol.startsWith("http")` or when the port is not 8765 (unless the host includes "ngrok"). `apiFetch(url, opts)` returns `fetch(getApiEndpoint(url), opts)`. The only raw `fetch(` in the builder is inside `apiFetch` itself (L3111), and there are 18 `apiFetch(` call sites (`/api/region`, `/api/runs`, `/api/satellite_layer`, `/api/models`, `/api/polygons`, `/api/validate_polygon`×2, train/scan endpoint, `/api/train/stop`, `/api/performance`, `/api/status`, `/api/runs/rename`, `/api/runs/delete`, `/api/model_introspect`, `/api/rerank`, run bundle). The protocol check is a redirect, not the forbidden early `return`. — [scripts/build_purged_ui.py#L3099-3112](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py); command: `grep -nE "[^i]fetch\(" | grep -v apiFetch`
- The UI calls neither `/api/compare_runs`, `/api/dual_domain_metrics`, `/api/config`, nor `/api/clean_old_training`, so those are server-only or dormant endpoints. — command: `grep -noE "apiFetch\(..."`
- **Live probe (read-only):** `lsof -i :8765` shows `Python 5579 ... TCP localhost:ultraseek-http (LISTEN)`, and `ps` shows `scripts/dashboard_server.py --port 8765` with 27:48 elapsed. `curl /api/status` returned `{"running": false, "mode": "idle", ..., "phase": "IDLE", ...}`, and `GET /` returned 200, 77,059,389 B in 0.052 s. — commands: `lsof`, `ps`, `curl`

### Inferences
- **Security sharp edge:** with `Access-Control-Allow-Origin: *`, `Allow-Private-Network: true`, and no auth or CSRF token, any web page open in the user's browser while the server runs could, in principle, POST to `http://127.0.0.1:8765/api/train`, `/api/runs/delete` (rmtree of archived runs), or `/api/clean_old_training`, and could read `/api/config`, including `CARTO_API_KEY`. `do_POST` parses the body as JSON regardless of Content-Type, so even a "simple" `text/plain` cross-origin POST (no preflight) would be accepted. Binding to 127.0.0.1 limits network exposure but not browser-borne requests. This follows from the code; I didn't test it.
- Serving the 77 MB HTML via `read_bytes()` per request is fine locally but memory-heavy under concurrent loads. Every page reload transfers 77 MB.
- AGENTS.md §6.3's instruction to kill stale PIDs and relaunch matters in practice: a server was already running (PID 5579), and module-level caches (`_SATELLITE_CACHE`, bundle cache) mean code changes aren't picked up without a restart.

### Gaps
- I didn't inspect `_load_region_bundle` (L740-1839, about 1,100 lines) or `clean_old_training_data` in detail. Exact deletion scope of `/api/clean_old_training` is not verified.
- I didn't verify whether `.env` actually sets `CARTO_API_KEY`. `.env` exists (mode 600) but wasn't read, to avoid handling secrets.

## Q5. `app/` and `design-system/`: is there a second (Next.js) frontend?

### Takeaway
`app/` + `ui/*.tsx` is an orphaned Next.js/React prototype from 2026-10-23/24. It has no `package.json`, `node_modules`, `tsconfig`, or Next config anywhere in the repo, it uses hard-coded mock data (`TARGET_DATASET`) rather than the API, and it can't be built as-is. The live product is the Python-generated single-file HTML. `design-system/` is a Markdown style guide from the same period.

### Cited Findings
- `app/page.tsx` (392 B) is `'use client'` and dynamically imports `../ui/GeospatialWorkspace` with `ssr: false`. `app/layout.tsx` (1,278 B) sets the title "GEOSPATIALS // Autonomous Mineral Intelligence Workspace" and loads Google Fonts. `app/globals.css` is byte-identical to `ui/globals.css` (3,375 B). — [app/page.tsx](file:///Volumes/Seagate%202/Geospatials/app/page.tsx); [app/layout.tsx](file:///Volumes/Seagate%202/Geospatials/app/layout.tsx); command: `cmp app/globals.css ui/globals.css`
- `ui/GeospatialWorkspace.tsx` (23,415 B) and `ui/components/{AsciiTomographicScanner,GeospatialMapViewport,StatisticalAnalyticsChart,TelemetryDrawer,TelemetryTable}.tsx`, `ui/hooks/useScrambledValue.ts`, and `ui/types/index.ts` import `react`, `lucide-react`, and `TARGET_DATASET` / `ASCII_GLYPH_RAMP` / `BRAILLE_FRAMES` from `./types`. There are no `fetch(` or `/api/` references. — command: `grep -hE "^import"`, `grep -nE "fetch\(|/api/"` over `ui/*.tsx`
- `find . -name package.json` (excluding .venv/.uv_cache) returned nothing. There's no `node_modules`, `tsconfig.json`, `next.config.*`, or `tailwind.config.*`, although `app/page.tsx` uses Tailwind classes (`min-h-screen w-full bg-[#090D16]`). — commands: `find`, `ls`
- Last commits touching these: `07db85b` (2026-10-24) "implement interactive geospatial dashboard..." and `d85274e` (2026-10-23). — command: `git log -- app design-system ui/GeospatialWorkspace.tsx ui/components`
- Commit `d009708` (2026-10-28) "remove ASCII tomography" removed it from the Python UI, yet `ui/components/AsciiTomographicScanner.tsx` remains. — git log; [ui/components/](file:///Volumes/Seagate%202/Geospatials/ui/components/)
- `design-system/geospatials/MASTER.md` (generated 2026-10-23 10:33:26; "Category: Analytics Dashboard", design dials Variance 7/10, Motion 6/10, Density 9/10) and `design-system/geospatials/pages/dashboard.md` (per-page overrides). The Python UI uses a similar font stack (Atkinson Hyperlegible, IBM Plex Mono, Newsreader...). — [design-system/geospatials/MASTER.md](file:///Volumes/Seagate%202/Geospatials/design-system/geospatials/MASTER.md); [scripts/build_purged_ui.py#L118-125](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py)

### Inferences
- The TSX files share the `ui/` directory with the canonical `ui/index.html` build artifact, which can mislead a new agent into editing React components that nothing serves. The Python builder is the single source of truth (AGENTS.md §6.1).
- `design-system/` is still useful as visual guidance for the Python UI, but nothing enforces it.

### Gaps
- No documentation explains whether the Next.js prototype was deliberately abandoned or is meant to be revived. I found no README or AGENTS mention of `app/`.

## Summary of sharp edges for a new agent

### Takeaway
The code and tests are basically sound, but the environment and repo state around them are fragile. These items all come from the cited findings above.

### Cited Findings
- The disk has 537 MiB free, which causes 5 test failures (4 unit + 1 integration) via `DiskSpaceError`. A UI rebuild writes about 154 MB, and archival requires ≥1 GB. — `df -h`; pytest output
- AGENTS.md's test count of 184 is stale; the real count is 226 collected. — [AGENTS.md#L116](file:///Volumes/Seagate%202/Geospatials/AGENTS.md); pytest output
- Most v4 code, the tests, and AGENTS.md/CHANGELOG.md/audits are untracked in git, and HEAD's `index.html` is 103 MB. — `git status`, `git cat-file -s`
- Dependency declarations are split and incomplete across pyproject.toml, requirements.txt, and requirements_sagemaker.txt; pystac-client, planetary-computer, and python-dotenv are undeclared; ruff and mypy are configured but not installed; there's no CI or pre-commit. — Q1 sources
- `build_purged_ui.py` uses CWD-relative paths, needs a prior `ui/index.html` containing an `EMBEDDED_BUNDLES` line, silently reuses stale bundles on loader errors, and runs automatically after CLI training runs. — [scripts/build_purged_ui.py#L12-22, L83-84, L6481-6485](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py); [scripts/run_seven_pass_training.py#L2973-2980](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)
- The server has no auth, CORS `*`, destructive POST endpoints, an API key echoed by `/api/config`, and unguarded `json.loads` in `do_POST`. — [scripts/dashboard_server.py#L2772-2792, L2863-2979](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)

### Inferences
- Suggested first steps for an agent (not performed, since this task was read-only): free disk space, then re-run the suite expecting 226/226. Add a `node --check` test for the generated bundle. Commit or track the untracked v4 modules. Consolidate dependencies into pyproject extras.

### Gaps
- I didn't check out a clean clone to confirm what breaks from tracked files alone.
