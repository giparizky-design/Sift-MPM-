# Stabilize Geospatials before Claude Code takes over

Geospatials is a working but fragile Mineral Prospectivity Mapping (MPM) system. Over 18 days, Google Antigravity agents turned a Yilgarn gold pilot into a Kalimantan engine with 4 metal heads (K=4) and 3 coal heads (K=3). The engine produces calibrated prospectivity maps, ranked drill-target polygons and geologist site cards, and you view them in a 77 MB single-file dashboard. **The code is broadly sound, but the repository around it is not safe to hand over yet.** The unit suite collects 226 tests and **222 pass; all 4 failures come from a full disk, not from bad logic** ([tests_ui_devenv.md L45-L57](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)). The urgent risks are operational:

- **About 11 days of work, including the whole v4 model layer and every governance document, has never been committed.**
- The data volume has **537 MiB free**.
- A CARTO API key is in git history.
- The commit at HEAD contains a 103 MB `index.html`, which is over GitHub's 100 MB per-file limit.
- Several "real" sensor channels in the ingestion script are seeded random noise.
- The dashboard server accepts unauthenticated, cross-origin requests that can start training and delete run folders.

Separately, the agent-written docs (AGENTS.md, CHANGELOG, the audits) claim "fully resolved and verified", but the code contradicts at least five of those claims. The conformal gate formula, buffered spatial CV, a label-lake integrity function, and the `insufficient_data` and PR-AUC-lift reporting are the material ones. Claude Code can load the existing `AGENTS.md` today with no changes, and every Antigravity artifact has a direct Claude Code equivalent. The order still matters: protect the uncommitted work and free disk space first, migrate the configuration second, and treat every inherited "verified" claim as a hypothesis until checked.

**How to read this report.** This report is a read-only synthesis of four research notes produced on 2026-11-08. Nothing in the repository was changed. Labels mean:

- **"Verified"**: a researcher observed it directly, by reading a file at a cited line or running a command whose output is recorded in the notes.
- **"Inference"**: the researcher's interpretation, or mine.
- **"Unverified"**: a gap the notes explicitly left open.

Citations point to repository files (with line numbers) where the notes give them. Facts that came from command output point to the line of the note that records the command.

---

## Eight operational risks outrank any code improvement

None of the following needs scientific judgment, and each one can destroy work or leak something before an improvement lands. The ranking reflects likely damage and urgency. Uncommitted work comes first because a single `git checkout .`, a failed disk, or an over-eager agent "cleanup" would wipe out 11 days of output with no recovery path.

| # | Risk | Evidence (verified unless marked) | Why it matters |
|---|---|---|---|
| 1 | **About 11 days of uncommitted work** | HEAD is `ad5476a` from 2026-10-28. There are only 32 commits, all between 10-22 and 10-28. `git status` shows **98 untracked, 67 modified and 43 deleted paths**. Untracked files include `AGENTS.md`, `CHANGELOG.md`, `audits/`, `src/schema/`, `src/models/{calibration,coal,conformal,fusion,metals}.py`, `src/evaluation/spatial_cv.py`, the SAR and polygon-rasterizer modules, `tests/integration/` and 12 unit-test files ([governance_and_migration.md L348-L366](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md); [tests_ui_devenv.md L28](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)). | The GitHub remote does not represent the running system. *Inference (not checked by checkout):* a clean clone would probably fail to import, because the tracked facade `src/models/pu_pipeline.py` re-exports untracked modules ([tests_ui_devenv.md L34](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)). |
| 2 | **Data volume is 100% full** | `df -h` shows 228 GiB total, **537 MiB available** ([tests_ui_devenv.md L26](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)). This already fails 4 unit tests and 1 integration test with `DiskSpaceError` from the 1 GB pre-flight check ([training_run_manager.py L45-L66](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py)). | Every UI rebuild writes about 154 MB (two 77 MB files), and run archival needs at least 1 GB, so training and archival cannot complete. *Inference:* committing the modified multi-GB binaries could also fail for lack of space. |
| 3 | **Secrets and sensitive identifiers in git** | The committed HEAD version of `config/carto.yaml` contains a non-empty `api_key` and `?key=` tile URLs. Only the uncommitted working copy blanks them ([scripts_data_outputs.md L139-L144](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)). The untracked `scripts/launch_sagemaker_training.py` hard-codes a SageMaker role ARN that contains the AWS account ID ([launch_sagemaker_training.py L36, L75-L78](file:///Volumes/Seagate%202/Geospatials/scripts/launch_sagemaker_training.py)). `docs/session/transcript.jsonl`, the 414-step Antigravity session including model "thinking", has been committed since the first commit, though a key-pattern scan found no matches ([governance_and_migration.md L109-L113](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)). `.env` is mode 600 and gitignored ([scripts_data_outputs.md L320](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)). | Blanking the CARTO key in a new commit does not remove it from history. The ARN is not in history yet, but a bulk "commit everything" would put it there. |
| 4 | **Files over GitHub limits, and git bloat** | `HEAD:index.html` is **103,484,719 bytes**, over GitHub's 100 MB hard limit. The remote is `github.com/giparizky-design/Geospatials`. `.git` is **1.59 GiB of loose objects with 0 packs** ([tests_ui_devenv.md L29](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)). 246 `.tif`, 8 `.joblib`, 8 `.parquet` and 8 `.npy` files are tracked because they were committed before `.gitignore` excluded them. The tracked working copies `feature_stack.joblib` (2.3 GB) and `dataset.parquet` (1.4 GB) are both modified ([scripts_data_outputs.md L163, L197-L205, L237-L238](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)). | *Unverified:* whether any push has already been rejected. Committing the modified binaries would add multi-GB blobs. |
| 5 | **Synthetic data presented as real** | In `harvest_kalimantan_multimodal.py`, Sentinel-2 NDVI and albedo are built from `default_rng(77)` noise, the thermal layer from `default_rng(88)`, and the "NOAA EMAG2v3" magnetic TMI is `gaussian_filter(rng.normal(0,45,…))` with `default_rng(555)`, which is then reduced to the pole ([harvest_kalimantan_multimodal.py L329-L345, L382, L563-L566](file:///Volumes/Seagate%202/Geospatials/scripts/harvest_kalimantan_multimodal.py)). Until v4.4.0, SAR HH/HV was DEM slope plus seeded noise ([v4.4 audit](file:///Volumes/Seagate%202/Geospatials/audits/2026-11-08_v4.4_sar_feature_dropout_audit.md)). | This directly violates the AGENTS.md §3 "Provenance Honesty" rule. *Inference:* derived features such as RTP and analytic signal of random TMI are noise the model can still fit spatially, so metal PR-AUCs are suspect. *Unverified:* which processed `.tif` channels actually came from these code paths. |
| 6 | **Open CORS with no authentication** | The server sends `Access-Control-Allow-Origin: *` and `Allow-Private-Network: true` on every response and has no auth. The POST routes include `/api/train`, `/api/scan`, `/api/runs/delete` (which calls `shutil.rmtree`) and `/api/clean_old_training`. `GET /api/config` returns `CARTO_API_KEY` ([dashboard_server.py L2670-L2721, L2772-L2792, L2863-L2979](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)). `do_POST` parses any body as JSON regardless of Content-Type, with no try/except (L2866). | *Inference, untested:* any web page open in the owner's browser can send a preflight-free `text/plain` POST to `127.0.0.1:8765` and start training or delete archived runs. AGENTS.md §6.3 *requires* CORS `*`, so fixing this needs a governance decision as well as a code change. |
| 7 | **Copyright and licensing exposure in agent tooling** | The `geology-harvester` skill tells agents to pull textbooks from LibGen and Anna's Archive mirrors ([SKILL.md L30-L33](file:///Volumes/Seagate%202/Geospatials/.agents/skills/geology-harvester/SKILL.md)). `agents/famous_literature_harvester.py` lists the same sources in its docstring ([famous_literature_harvester.py L1-L12](file:///Volumes/Seagate%202/Geospatials/agents/famous_literature_harvester.py)). | Porting this skill as-is would carry the instruction into Claude Code. *Unverified:* whether the Python code actually calls those endpoints. |
| 8 | **Unexplained deletions and mixed-provenance outputs** | Three rasters were deleted without a changelog or audit entry: `dist_plate_boundary_km.tif`, `dist_volcanic_arc_km.tif` and `volcanic_arc_halo.tif` ([governance_and_migration.md L328, L340](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)). The active East Kalimantan output folder was overwritten by a coal-only run with `n_cells` 117,196, against the documented 2.92 M ([seven_pass_summary.json](file:///Volumes/Seagate%202/Geospatials/outputs/seven_pass_east_kalimantan/seven_pass_summary.json)). The Seagate SSD that the 50 m scripts and `dashboard_server.py` expect is not mounted ([scripts_data_outputs.md L108](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)). | *Inference:* the deletions are probably intentional leakage or proxy cleanup, but only the owner can confirm before they are committed. Metal maps in the active folder probably come from earlier runs. |

These risks interact. Risks 1 and 2 must be handled together, because the safest protection for uncommitted work is an off-machine copy, and a full disk makes both local snapshots and large commits fragile. Risks 3 and 4 decide *how* the work gets committed: committing everything in bulk would freeze the CARTO key, the AWS account ID and more multi-GB binaries into a history that GitHub may already refuse. Risks 5 and 6 do not block migration, but they belong at the top of the improvement backlog. A new agent should know about them from its first session.

---

## A Yilgarn pilot became a Kalimantan engine in 18 days

The project started on 2026-10-20 as a Yilgarn Craton orogenic gold pilot. That framing survives in the stale `README.md` ("Status: Pre-data… Data download pending") and in the package metadata `mpm-yilgarn` v1.0.0 ([README.md L1-L8](file:///Volumes/Seagate%202/Geospatials/README.md); [pyproject.toml L6-L8](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)). The founding Antigravity session ran 414 steps between 10-20 and 10-21. It ended by noting that a blind-holdout PR-AUC of 1.000 was "an artifact of the synthetic setup" ([transcript.jsonl, last line](file:///Volumes/Seagate%202/Geospatials/docs/session/transcript.jsonl)).

The first audit (v1.0, 10-21) drove a pivot to Borneo for two reasons. Canopy blinds optical SWIR there, so the work shifted to SAR plus DEM. And **44% (22/50) of USGS MRDS points were island-centroid artifacts** ([INDEX.md L16-L35](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md)).

All 32 commits fall between 10-22 and 10-28. They cover:

- the 7-pass PU-learning pipeline and the OSM UI;
- ESDM GeoRIMA ingestion;
- island scans at 1.1 km, 500 m and 100 m;
- the 50 m mosaic;
- the 250 m K=4 engine;
- a burst of dashboard features.

After 10-28 the project stopped committing and kept working ([governance_and_migration.md L353-L362](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)).

The uncommitted period holds most of the substance, documented in 20 audits and about 45 changelog entries:

- **10-28, v2.0 physics audit:** found the Evans `plan_curvature` bug, a non-demeaned FFT taper, wrong equatorial reduction-to-pole (RTP), and a synthetic DEM-derived Bouguer gravity.
- **10-29, Antigravity "teamwork" run 1:** an orchestrated multi-agent run that stabilized the pipeline and reported "163/163 unit tests" and "VICTORY CONFIRMED" ([governance_and_migration.md L75-L79](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)).
- **10-31, v2.6 cheating/leakage audit:** showed that the Kutai coal maps "aligned with actual mines" because `geology_coal_contact_dist_km` was literally the distance to known concessions ([INDEX.md L26](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md)).
- **11-01, v2.8 clean retrain:** after remediation, Kutai scored spatial-CV ROC-AUC 0.8395 with a 2.85× PR-AUC lift at 4.7% prevalence.
- **11-07, teamwork run 2:** implemented the "v4.0 Production Upgrade" (math fixes D1–D10 and Bugs 1–10 from the v3.1 audit). It closed with "226/226" tests and another victory verdict ([governance_and_migration.md L80-L85](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)).

Run 2's verdict did not hold. Within hours, v4.0.1 fixed a 23-vs-28 column mismatch and an `UnboundLocalError`, and v4.3.1 fixed a `NameError` that crashed training launched from the UI. The gate's CLI checks had only been `--help` smoke tests ([CHANGELOG.md L54-L104](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md)).

On 11-08, the v4.4 audit found that the models had **0% SAR gain** and 59–77% gain from GEM fault distances. SAR had been missing, silently dropped by warm-start, and synthetic. The last recorded agent activity, between 21:14 and 21:22 that day, was v4.4.0 and v4.4.1:

- real ALOS-2 PALSAR-2 ingestion through Planetary Computer STAC;
- a `resolve_warm_start_feature_schema` policy and an `--on-schema-mismatch` flag;
- a radar overlay in the dashboard ([CHANGELOG.md L10-L52](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md)).

`HANDOFF.md` is still the v2.0 handoff from 10-28, and its "next steps" were since addressed ([HANDOFF.md L3-L7, L64-L78](file:///Volumes/Seagate%202/Geospatials/HANDOFF.md)).

The results history is sobering. Yilgarn reached a P-vs-RN PR-AUC of 0.8472. The leak-free Kelian corridor CV scored an out-of-fold PR-AUC of **0.0061 (ROC 0.4618)**, which is no better than chance. An East Kalimantan 10-round run peaked at round 2 (`pr_auc_fused` 0.1793) and over-boosted by round 10, and `porphyry_cu` had zero positives and defaulted to 0.50 ([HANDOFF.md L43-L49](file:///Volumes/Seagate%202/Geospatials/HANDOFF.md)). The currently active East Kalimantan run is coal-only with `pr_auc_fused` 0.0504, coal head PR-AUCs of 0.0167, 0.0187 and 0.1158, null metal heads, and only 3 ranked targets ([scripts_data_outputs.md L283-L297](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)).

*Inference:* the pattern is consistent. Strong results appeared when synthetic data or leakage was present, and weak ones appeared once it was removed. The open work therefore matters more than any new feature. That work is retraining with real SAR under `--on-schema-mismatch retrain`, then running the drop-column and permutation importance study under spatial CV that the v4.4 audit recommends.

---

## Nine subpackages feed one 2,985-line orchestrator

### The library is a toolbox; the pipeline lives in a script

`src/` holds **41 Python files and 20,211 lines** in 9 subpackages. AGENTS.md §2.1 lists only six of them, omitting `ingestion/`, `harmonization/` and `utils/`, which add about 2,140 lines.

| Subpackage | Lines | Role |
|---|---|---|
| `schema` | 275 | `canonical.py`: non-feature columns, forbidden raw-band patterns, head names, priors, `verify_no_raw_pixel_leakage`, `resolve_warm_start_feature_schema`. Has no `__init__.py`. |
| `features` | 5,152 | `FeatureStack` (C,H,W) container; FFT potential-field filters (`geophysics`); DEM morphometry, slip and dilation tendency (`structural`); ASTER and Sentinel-2 indices; `sar_palsar2`; coal and multimodal features; `polygon_rasterizer` |
| `labels` | 3,380 | Borneo K=4 and K=3 targets with hygiene filter; Yilgarn MINEDEX; two-step PU negatives, propensity, `OUTMOGCGate`; spatial blocks |
| `models` | 6,227 | `TwoStageK4Ensemble`, `TwoStageK3CoalEnsemble`, Elkan-Noto calibration, conformal, fusion, `DistrictDAEUNet`, retrain governance, run archival; `pu_pipeline` facade |
| `evaluation` | 919 | metrics and the buffered spatial-CV trunk |
| `outputs` | 2,119 | COG writers, target polygonization, site cards, model introspection for the UI |
| `ingestion` / `harmonization` / `utils` | 817 / 908 / 414 | ASTER and GA downloads, reprojection and cover masks, raster/vector I/O |

Source: [core_library_architecture.md L12-L19](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/core_library_architecture.md).

Imports run mostly one way: `schema`/`evaluation` → `labels` → `models` → `outputs`. The one inversion is `features/polygon_rasterizer.py` importing `borneo_labels` ([polygon_rasterizer.py L30-L39](file:///Volumes/Seagate%202/Geospatials/src/features/polygon_rasterizer.py)). Code quality looks good on the surface: only 2 of 338 functions lack return annotations, and there are no TODOs. The debt is structural, and comes in four forms.

**Legacy paths still run alongside v4 ones:**

| Legacy | v4 replacement |
|---|---|
| `MPMUNet` | `MPMDistrictDAEUNet` |
| `MPMXGBoost` | `_BaggedXGBoostHead` |
| `generate_target_polygons` | `polygonize_and_rank_targets` |
| unbuffered `spatial_block_kfold` | `BufferedSpatialBlockKFold` |

There are also two separate conformal implementations.

**Helpers are duplicated:** `_logit` and `_sigmoid` are defined three times, and the Hájek weights twice.

**Dead code remains:** `_hanning_taper_2d`, the `_ARTIFACT_*` centroid constants, and the ASPP `d4` branch.

**Four library files exceed 1,000 lines** ([core_library_architecture.md L325-L359](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/core_library_architecture.md)).

The end-to-end flow is wired in `scripts/run_seven_pass_training.py`, not in `src/`:

1. Load rasters into a `FeatureStack` and run the leakage check twice (L754, L766).
2. Build K=4/K=3 labels, the propensity model, exclusion masks and Hájek weights (L979, L1124).
3. Assign 5-fold spatial blocks using the **unbuffered** `spatial_block_kfold` (L880).
4. Fit the `TwoStageK4Ensemble` and the DAE-U-Net (L1216, L1281), plus their coal counterparts.
5. Fuse the models with precision weighting.
6. Calibrate with closed-boundary Elkan-Noto, gated by OUTMOGC. Fallbacks: ĉ = 0.35, and heads with no positives are forced to 0 (L1383-L1410).
7. Compute the conformal set and lower bound.
8. Run `polygonize_and_rank_targets` at L(x) ≥ 0.50 (L2096).
9. Write site-card JSON, COGs and GPKG (L2178).

Line numbers refer to [run_seven_pass_training.py](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py). Before writing new output, the script archives the previous run with symlink-safe `copytree` ([training_run_manager.py L213-L290](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py)).

*Inference:* the library's guarantees depend on how each script wires it. At least seven scripts chain polygonize → site cards on their own, so they can drift apart.

### Scripts, configuration and run modes

`scripts/` has 33 Python scripts. The live core is:

- `run_seven_pass_training.py`, with flags `--domain`, `--region`, `--rounds`, `--scan-only`, `--on-schema-mismatch`, `--reset`, and others ([L2626-L2780](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py));
- `dashboard_server.py`;
- `build_purged_ui.py`;
- the data builders `harvest_kalimantan_multimodal.py`, `build_east_kalimantan_250m.py`, `build_kutai_basin_100m.py` and `add_real_sar_channels.py`.

The orchestrator only accepts Kalimantan regions on a whitelist, so **Yilgarn cannot run through it**. Yilgarn uses the older `run_pilot.py`, driven by `config/pilot.yaml` ([run_seven_pass_training.py L2784-L2800](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)). Kalimantan has **no declarative config at all**. `borneo.yaml` and `logging.yaml` have no Python consumers, and `carto.yaml` has been superseded by environment variables. Reproducibility therefore depends on the CLI flags recorded in `seven_pass_summary.json` and the run catalog ([scripts_data_outputs.md L126-L149](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)).

About half of the scripts are frozen one-off experiments: the national Indonesia scans, the 50 m whole-island scan, the Sangatta hotspots, and three multi-round wrappers that `--rounds` has since made redundant. Untracked SageMaker and EC2 launchers default to `ml.m5.xlarge` in `ap-southeast-2`.

### Data, outputs and compute

On disk, `data/` is **13 GB** and `outputs/` is **11 GB**. The largest single items are:

- `feature_stack.joblib` (2.3 GB)
- `dataset.parquet` (1.4 GB)
- the Kutai 100 m cube (4.2 GB)
- each run's `models/` folder, up to 2.1 GB of DAE fold checkpoints

The environment adds `.venv` at 1.7 GB and `.uv_cache` at 1.1 GB ([scripts_data_outputs.md L157-L192](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)).

Each region writes to `outputs/seven_pass_<region>/`, with subfolders `maps/`, `targets/` and `models/` plus metrics JSON files. Older runs are archived to `outputs/run_history/` and indexed in `training_runs_catalog.json`.

Several catalog entries record `elkan_noto_c_hat` and `conformal_q_hat_90` at exactly 0.5, and some q̂ values exceed 1.0 (1.17–1.18). *Inference:* these look like fallback placeholders, which conflicts with the "no dummy 0.50" rule ([scripts_data_outputs.md L305](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/scripts_data_outputs.md)).

Training runs on CPU by default. The U-Net never auto-selects Apple MPS; it can be forced through `SIFT_DEVICE` ([unet_model.py L57-L69](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py)). Measured round times are 675–754 s at 250 m for East Kalimantan, so a 10-round run takes about 2 hours, and 2,957–3,085 s for 50 m hotspots ([ten_rounds_history.json](file:///Volumes/Seagate%202/Geospatials/outputs/seven_pass_east_kalimantan/ten_rounds_history.json)).

### A separate RAG product shares the repository

`mcp_server/geology_mcp_server.py` is a FastMCP server named `geology-rag-mcp` that exposes seven tools, including `query_geology_rag`. Together with `agents/`, it forms a geology-literature RAG system backed by LanceDB and SQLite. It shares the `.venv`, the `.env` and `data/` with the MPM pipeline, but nothing else ([geology_mcp_server.py L16-L19, L135-L670](file:///Volumes/Seagate%202/Geospatials/mcp_server/geology_mcp_server.py)).

`app/` plus `ui/*.tsx` is an orphaned Next.js prototype. It has no `package.json`, uses mock data, and cannot be built. It sits next to the real compiled `ui/index.html`, which makes it easy to mistake for live code ([tests_ui_devenv.md L167-L178](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

---

## Tests pass on logic, but nothing guards the environment or the 77 MB UI

### Real test results

The suite was run on 2026-11-08 and the results are verified:

- `tests/unit` collects **226 tests**. **222 passed and 4 failed in 199.8 s.**
- `tests/integration`: **17 passed, 1 failed.**
- `test_v4_math_and_bugs.py`: **13/13** passed, which matches AGENTS.md.

All five failures are the same `DiskSpaceError` raised by `check_disk_space(min_free_gb=1.0)`, reporting 0.53–0.60 GB free ([tests_ui_devenv.md L45-L62](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

*Inference:* with at least 1 GB free, the suite would very likely reach 226/226 and 18/18. A new agent must not "fix" these failures by lowering the threshold.

Coverage gaps are real:

- No test starts the HTTP server, hits a route, or checks CORS headers.
- No test touches `build_purged_ui.py` or the generated JavaScript.
- Six `src` modules have no direct test references: `sentinel2_indices`, both ingestion modules, `reproject`, `utils/io` and `xgboost_model`.
- About 30 scripts are untested ([tests_ui_devenv.md L66-L69](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

### The development environment does not reproduce from the repo

The `.venv` runs Python 3.14.3, created with Homebrew `venv`, not uv. It holds an editable install, and all heavy dependencies import: torch 2.14.1, xgboost 3.4.1, rasterio 1.5.2 and geopandas 1.2.0.

Dependencies are declared in three lists that disagree:

| File | What it covers |
|---|---|
| `pyproject.toml` | The geospatial/ML core |
| `requirements.txt` | Only the MCP/RAG stack |
| `requirements_sagemaker.txt` | A third subset, without torch |

Three runtime imports, `pystac_client`, `planetary_computer` and `dotenv`, are declared nowhere. ruff and mypy are configured but not installed. There is no lockfile, no CI and no pre-commit hook ([tests_ui_devenv.md L11-L24](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

*Inference:* a fresh clone plus `pip install -e .` would not produce a working environment.

### The dashboard is a self-referential build artifact

`build_purged_ui.py` is 6,490 lines of Python that hold the entire HTML/CSS/JS app as string literals. Its build works like this:

1. It reads the *existing* `ui/index.html` to find the `const EMBEDDED_BUNDLES =` line, and aborts if that line is missing.
2. It imports loaders from `dashboard_server.py` and embeds about 12 region bundles plus every catalogued run.
3. If any of that fails, it **silently reuses the stale bundle line**.
4. It writes two identical files of **77,059,389 bytes** each, using paths relative to the current directory.

Line 2996 alone is a 76.7 MB JSON blob ([build_purged_ui.py L12-L84, L6479-L6485](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py); [tests_ui_devenv.md L86-L94](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

Every CLI training run calls this build at the end, unless `MPM_SKIP_UI_REBUILD=1` is set ([run_seven_pass_training.py L2973-L2980](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)). The current inline script passes `node --check`, but nothing runs that check automatically. That gap matters because the two most recent UI fixes were a "blank homepage" and a "JS syntax error in generated bundle" ([tests_ui_devenv.md L95-L97](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

The UI, titled "SIFT // Spatial Inference & Fault Targeting", has three views: Home, Workspace (a Leaflet map with site cards, a ranked target portfolio, and conformal, PR-AUC, U-Net and XGBoost-trace tabs) and Training (launch, telemetry, run history).

### The server meets its own rules but is exposed

`dashboard_server.py` (2,997 lines) is a stdlib `ThreadingHTTPServer` on `127.0.0.1:8765`. It has 14 GET and 9 POST routes. On `/`, it reads the 77 MB HTML into memory for every request.

It satisfies both AGENTS.md §6 invariants:

- CORS `*` is set on JSON, HTML, OPTIONS and HEAD responses.
- All 18 client call sites go through `apiFetch`, whose protocol check redirects to `http://127.0.0.1:8765` instead of returning early ([build_purged_ui.py L3099-L3112](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py)).

There are two gaps. First, a malformed POST body raises an error before any CORS-bearing response is written. Second, HEAD returns 200 for any path ([dashboard_server.py L2686-L2702, L2863-L2866](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py)).

When the researchers looked, a live instance (PID 5579) was already serving on port 8765. The server keeps module-level caches, so code changes are not picked up until it restarts ([tests_ui_devenv.md L153-L158](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/tests_ui_devenv.md)).

---

## Project docs overstate what the code guarantees

The docs were written by agents that also approved their own work, so they read as confident certification. Spot checks against the code show that several of those claims are false, and a few of the gaps change what the outputs mean.

**The three most material gaps:**

- **Target-set gate.** Target sets are gated on a σ-penalized score, not on the point probability the docs describe.
- **Cross-validation.** The production CV has no spatial buffer, so the reported metrics may be inflated by spatial autocorrelation.
- **Thin heads.** Heads with too few validation positives are not flagged and can get placeholder values.

**Claims checked against the code:**

| Claim (source) | What the code actually does (source) | Verdict |
|---|---|---|
| Conformal set gate is "strictly evaluated on calibrated point probabilities P_k(s) ≥ 1 − q̂_k"; the lower bound uses t_{n−1,α}·σ ([AGENTS.md L91-L92](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | The gate is `p_cal − 0.5·σ ≥ 1 − q_set`. The nonconformity score is `1 − p + 0.5σ`. The lower bound uses a conformal studentized-residual quantile, and a legacy branch ties it back to q_set ([conformal.py L106-L127, L257, L280-L306](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py)) | **Contradicted.** The code matches the run-2 D1 spec; the docs are wrong. |
| "Unified 5-fold buffered spatial CV" with buffer B_m ([AGENTS.md §4.11](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | The buffered trunk exists, but the orchestrator, `train_borneo_pilot.py` and `run_pilot.py` all call the unbuffered `spatial_block_kfold`. No script calls the buffered version ([run_seven_pass_training.py L880](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py); [spatial_splits.py L125-L195](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py)) | **Contradicted in practice** |
| SHA-256 immutable positive label lake `verify_immutable_positive_lake` ([AGENTS.md §2.1](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | No such function exists anywhere in `src` ([core_library_architecture.md L301](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/core_library_architecture.md)) | **False** |
| Report PR-AUC lift over prevalence, and flag heads with fewer than 5 validation positives as `"insufficient_data"` ([AGENTS.md §3.3](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | No `.py` file contains `insufficient_data`. `pr_auc` is plain average precision. In-sample ĉ falls back to 0.50 and 0.35 ([metrics.py L46-L69](file:///Volumes/Seagate%202/Geospatials/src/evaluation/metrics.py); [metals.py L784-L785](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py)) | **Not implemented** |
| Never train on AI predictions (`FeedbackLoopError`) ([AGENTS.md §3.2](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | It is defined and raised only in `RetrainingPipeline`, which no script uses. The production path builds labels from vector sources ([retrain.py L38, L121-L165](file:///Volumes/Seagate%202/Geospatials/src/models/retrain.py)) | **Not on the production path** |
| Hájek weights w_i = (1/e)/Σ(1/e) ([AGENTS.md L95](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)) | The code normalizes by the *mean*, so positive weights average 1.0 rather than summing to 1 ([metals.py L70-L73](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py)) | Holds in substance; **the formula in the docs is wrong** |
| A "4-branch" ASPP scaled by `px_size_m` | The `d4` branch is computed but never used. At 250 m, all three dilations round to 1, which collapses the multi-scale receptive field ([unet_model.py L345-L384](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py)) | **Partly holds.** Only does real work at 50–100 m pixels. |
| Prior-corrected WoE fusion `logit(P_ens) + W⁺ − logit(π)` | Coal uses `logit(p_sub) + w·max(0, logit(p_opt) − logit(π_opt))`. Metals fusion has no prior term ([coal.py L111-L141](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py); [fusion.py L80-L128](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py)) | **Partly holds** |
| Island-centroid exclusion at (−1.9982, 113.9995) | The `src` constants are dead code. The real check is upstream in scripts, with different tolerances (±0.05° vs 5.5 km), and loaders default the flag to `False` ([borneo_labels.py L128-L143, L606-L608](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py)) | **Conditional.** The filter only works if the upstream flag was set. |
| Site cards emit `"not_available"` when a sensor layer is absent | Only when both values are `None`. Otherwise `sensor_source` defaults to `"EMIT_SWIR"` whichever sensor actually supplied the layer ([prospectivity_map.py L1121, L1465](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py)) | **Provenance risk** |
| `canonical.py` holds alias maps and propensity covariates | `CANONICAL_PROPENSITY_COLS` lives in `pu_negatives.py` L526. Alias resolution lives in `prospectivity_map.py` L555-L640 | **Misplaced in the docs** |

Checks that **hold**:

- the Evans `plan_curvature` correction;
- the demeaned FFT with MacLeod RTP, although the taper is now a reflect pad with a cosine ring, not a Hanning window;
- the coal softmax hierarchy with Σ ≤ 1;
- the logit(π_k) bias initialization;
- the restriction of propensity covariates to logistics variables ([core_library_architecture.md L270-L300](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/core_library_architecture.md)).

### Document drift

The documents disagree with each other as well as with the code:

| Document | What it says | What is actually true |
|---|---|---|
| AGENTS.md | Test count "184 passed"; "Last Updated v4.0.0"; module map claims 95 modules | 226 tests are collected. The latest release is v4.4.1. The module map omits `sar_palsar2`, `polygon_rasterizer`, three subpackages and the new API routes. There are now 102 `.py` files ([governance_and_migration.md L398-L405](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)) |
| `README.md` | Describes a pre-data Yilgarn pilot | The project is now a Kalimantan engine |
| `HANDOFF.md` | Frozen at v2.0 | Work continued through v4.4.1 |
| `PROJECT.md` | Reads like a project overview | It is the run-2 orchestrator's scope document |
| `CODEBASE_CORPUS.md` | 183 KB index with `#Lstart-Lend` line ranges | The line ranges drift with every edit |
| `audits/INDEX.md` | Ordered audit list | v4.4 is listed above v4.3, and there are no v4.1 or v4.2 entries. It cites `audits/pipeline_domain_head_audit_report.md`, which does not exist ([governance_and_migration.md L239-L243](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md)) |

Version numbering is also inconsistent. The audits use vX.Y and the changelog uses vX.Y.Z-slug, and the two series do not line up: changelog v2.9.12 corresponds to audit v2.8. The changelog skips from v1.5 to v1.8 and has two entries numbered v2.2.0. No git tags were observed, although `git tag -l` was not run.

*Inference:* treat every "RESOLVED", "100%" and "VICTORY CONFIRMED" statement as a claim to verify. The teamwork run-2 prompt itself said "Integrity mode: demo", and the gate agents wrote the adversarial tests that they then passed ([ORIGINAL_REQUEST.md L8-L9](file:///Volumes/Seagate%202/Geospatials/.agents/teamwork/ORIGINAL_REQUEST.md)).

---

## Antigravity artifacts map one-to-one to Claude Code

### Where things stand

There is no Claude Code configuration yet: no `CLAUDE.md`, `CLAUDE.local.md`, `.claude/` or `.mcp.json`. Claude Code v2.1.277 and later read `AGENTS.md` natively when no `CLAUDE.md` exists, and this session did load it ([Claude Code memory docs](https://code.claude.com/docs/en/memory)).

**One trap:** once a `CLAUDE.md` or `CLAUDE.local.md` exists, `AGENTS.md` stops loading unless that file imports it with `@AGENTS.md`. Creating a `CLAUDE.local.md` on its own would silently drop all project governance. Claude Code reads nothing under `.agents/`.

Antigravity also reads `AGENTS.md` as an always-on rule ([Antigravity Rules docs](https://antigravity.google/docs/rules)). That makes it the natural shared file if both tools stay in use.

`AGENTS.md` is 183 lines and 20,380 bytes. Its last 32 lines (L152-L183) are an **AWS guidance block injected by the AWS Agent Toolkit**, not project governance ([AGENTS.md L152-L183](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)). Claude Code guidance recommends keeping each CLAUDE.md under about 200 lines, and notes that `@` imports do not reduce context cost.

### Governance rules worth keeping

The two standing rules suit Claude Code well:

- **Versioned audits:** `audits/YYYY-MM-DD_vX.Y_<slug>.md`, each registered in `INDEX.md`.
- **Changelog:** a dated, versioned `CHANGELOG.md` entry prepended for every code, config or UI change ([AGENTS.md L8-L20](file:///Volumes/Seagate%202/Geospatials/AGENTS.md)).

So do the scientific guardrails in §3 and the UI rules in §6: never hand-edit `index.html`, always use `apiFetch`, keep CORS `*` (now contested by risk 6), and restart stale servers.

The difference is enforcement. Antigravity followed these rules as prose. Claude Code can enforce some of them deterministically with hooks, whereas CLAUDE.md content is only advisory ([Claude Code hooks guide](https://code.claude.com/docs/en/hooks-guide)).

### Migration mapping

| Antigravity artifact | What it does | Claude Code equivalent | Risk or caveat |
|---|---|---|---|
| `AGENTS.md` (root) | Always-on workspace rule | Read natively today. Durable setup: a short `CLAUDE.md` starting with `@AGENTS.md`, then Claude-specific notes ([memory docs](https://code.claude.com/docs/en/memory)) | Never create `CLAUDE.local.md` alone. Decide whether the AWS block (L152-L183) belongs in shared governance or in a personal `~/.claude/CLAUDE.md`. Correct or delete the false §4.4 formula. |
| `.agents/skills/geology-rag-expert/` | Model-triggered skill that grounds answers through `query_geology_rag` | `.claude/skills/geology-rag-expert/SKILL.md`; the `name` and `description` frontmatter already fit ([skills docs](https://code.claude.com/docs/en/skills)) | Works only once the MCP server is registered. The tool name becomes `mcp__geology-rag-mcp__query_geology_rag`. |
| `.agents/skills/geology-harvester/` plus an identical copy in `.antigravity/` | Literature-harvest procedure; protocol of never pasting keys in chat | `.claude/skills/geology-harvester/` with `disable-model-invocation: true` | **Remove the LibGen and Anna's Archive sources before porting.** Keep the credential protocol. Drop the duplicate copy. |
| 7 symlinks (`banner-design`, `brand`, `design`, `slides`, …) | Global `ui-ux-pro-max` Antigravity plugin skills | Optional copies (not symlinks) in `~/.claude/skills/` | Absolute links into `~/.gemini/...` break on any other machine. Check whether they are needed at all. |
| `.antigravity/mcp.json` → `geology-rag-mcp` | Launches the FastMCP RAG server over stdio | Project `.mcp.json` with the same `mcpServers` shape ([MCP docs](https://code.claude.com/docs/en/mcp)) | `./.venv/bin/python` assumes the current directory is the repo root. Project servers need one-time approval. Secrets stay in `.env` through dotenv. `lancedb`, `pymupdf4llm` and `mcp` must be installed in `.venv`, which is unverified. The Antigravity docs do not list `.antigravity/mcp.json` as a config path ([Antigravity MCP docs](https://antigravity.google/docs/mcp)). |
| `.agents/teamwork/**` (165 files, 2 runs) | Multi-agent run logs: orchestrator, workers, reviewers, challengers, auditors | No configuration needed. Roles could become `.claude/agents/*.md` subagents | `.agents/` is gitignored, so this history exists only on this machine. Read the verdicts as self-certification, not ground truth. |
| §1 changelog and INDEX rule | Prose rule | Keep in AGENTS.md. Optionally add a `Stop` hook reminder or a `/log-change` skill | Hooks are deterministic; CLAUDE.md is advisory. |
| §6 "never edit `index.html`" | Prose rule | `PreToolUse` hook on `Edit\|Write` that exits 2 for `index.html` and `ui/index.html` | Also deny `Read(./.env)`, and consider denying reads of the 77 MB bundle. |
| §6 UI and server rules | Apply only to UI and server files | `.claude/rules/ui-dashboard.md` with `paths:` frontmatter | Shrinks always-loaded context. |
| §2 module map and §4 resolution history | About 70 lines of rarely needed, partly false reference material | On-demand skill or a referenced doc, not imported | `CODEBASE_CORPUS.md` (183 KB) and `CHANGELOG.md` (110 KB) must be referenced, never imported. |
| Antigravity workflows | None exist in the repo | `.claude/skills/<name>/` | Candidates: run-tests, rebuild-ui, restart-dashboard, new-audit. |
| `docs/session/transcript.jsonl` | Committed founding trajectory | None | Contains model reasoning. Owner decides whether to archive it out of git. |

Source: [governance_and_migration.md L443-L464](file:///Volumes/Seagate%202/Geospatials/research_notes/Geospatials%20repo%20onboarding/governance_and_migration.md).

### What Claude Code must know from its first session

Some of these facts are verified and some are inferences drawn from them.

**Repository and environment**

- Run everything from the repo root with `.venv/bin/python` and `.venv/bin/pytest`. `build_purged_ui.py` uses paths relative to the current directory.
- `OMP_NUM_THREADS=1` is deliberate. It prevents XGBoost and PyTorch deadlocks.
- No training run, scan or UI rebuild should start until there is well over 1 GB of free disk.
- Never lower `min_free_gb` to make tests pass.
- Never run destructive git commands while the working tree holds the only copy of the v4 code.

**Training side effects**

- A CLI training run takes about 2 hours for 10 rounds.
- Unless `MPM_SKIP_UI_REBUILD=1` is set, it overwrites both 77 MB HTML files and archives the previous run.
- It refuses non-Kalimantan regions.

**Dashboard server**

- The server may already be running. Check `lsof -i :8765`.
- It caches modules, so restart it after code changes.

**Data provenance**

- Some ingestion channels are seeded random fields.
- The Seagate SSD paths are not mounted.
- The active output folders mix runs.

**Documentation**

- Treat the docs as claims to verify against the code.
- The README, HANDOFF.md, the test counts in AGENTS.md and the §4 formulas are known to be stale or wrong.

---

## A first-steps plan that gates improvements on safety

This analysis made no changes to the repository. The plan below is a readiness checklist for the owner to approve step by step. Several steps involve owner-only decisions: what to delete, whether to rewrite history, whether to rotate keys, and whether to change the CORS rule. A Claude Code session should not take those on by itself. Steps 1–4 are preconditions, and no improvement work should start before they are done.

| Priority | Step | Grounding | Owner decision? | Done when |
|---|---|---|---|---|
| **P0-1** | Make an off-machine copy of the working tree (excluding `.venv` and `.uv_cache`), with a note of `git status` | Risk 1: the v4 code exists only in the working tree | Yes: destination | A restorable copy exists outside this disk |
| **P0-2** | Free disk space to several GB above the 1 GB archival floor. Pause training, scans and UI rebuilds until then. | Risk 2: 537 MiB free. Rebuild writes about 154 MB; archival needs 1 GB or more | Yes: what to delete. Candidates by size include `.uv_cache` (1.1 GB, not gitignored), duplicate run `models/` (up to 2.1 GB each), and near-duplicate archived runs | `df -h` shows comfortable headroom |
| **P0-3** | Settle history hygiene **before** any commit or push: rotate the CARTO key in HEAD history; decide on the 103 MB `HEAD:index.html`; keep the AWS account ARN out of commits; decide the fate of `transcript.jsonl`; choose LFS, DVC or untracking for the tracked binaries | Risks 3–4 | Yes: rotation, history rewrite, binary strategy | A documented plan; no secrets in the next commit |
| **P0-4** | Confirm the three deleted rasters and the mixed-provenance active outputs; then commit the current state on a branch in logical groups (governance and audits; `src` and tests; UI bundle; data and output churn decided separately) | Risks 1, 8 | Yes: deletion intent, grouping | `git status` is clean or deliberately ignored; remote matches the working system |
| **P1-5** | Bootstrap Claude Code: a few-line `CLAUDE.md` importing `@AGENTS.md`; a `PreToolUse` hook protecting `index.html` and `ui/index.html`; deny rules for `.env`; project `.mcp.json` for `geology-rag-mcp`; port `geology-rag-expert`; port `geology-harvester` only after removing the shadow-library sources | Migration mapping | Yes: whether `.claude/` and `.mcp.json` are committed; where the AWS block lives | Claude Code loads the governance, the MCP tools resolve, and editing `index.html` is blocked |
| **P1-6** | Establish a verified baseline: rerun unit, integration and v4 suites; run `node --check` on the bundle's inline script; run `/doctor prompt-audit` (v2.1.283+) | Tests section; [memory docs](https://code.claude.com/docs/en/memory) | No | 226/226 and 18/18 recorded (expected, not yet observed) |
| **P1-7** | Reconcile docs with code: correct the AGENTS.md §4.4 conformal formula and the Hájek formula, remove `verify_immutable_positive_lake`, update test counts and the module map, refresh `HANDOFF.md` and `README.md`, fix `INDEX.md` ordering, pick one version scheme and tag | Docs section | Yes: version scheme | A new agent reading AGENTS.md gets no false guarantees |
| **P2-8** | Close the server exposure: an auth token or Origin check, a try/except around `do_POST`, and stop echoing the key from `/api/config` | Risk 6 | **Yes: requires amending AGENTS.md §6.3** | Cross-origin POSTs are rejected; CORS still works for `file://` mode |
| **P2-9** | Run a provenance audit of every processed channel against its code path; quarantine seeded-noise layers (Sentinel-2, thermal, EMAG2 TMI and derivatives) | Risk 5 | Yes: scientific acceptance | Every feature channel has real or `not_available` provenance |
| **P2-10** | Only then start scientific improvements: switch the orchestrator to `BufferedSpatialBlockKFold`; implement PR-AUC lift and `insufficient_data`; retrain with real SAR under `--on-schema-mismatch retrain`; run the v4.4 importance study; fix the ASPP behavior at 250 m and the `sensor_source` default; consolidate dependencies and add CI | Code-vs-docs table; v4.4 audit | Yes: priorities | Metrics come from buffered CV on provenance-clean features |

Some sequencing choices deserve a defense. Committing before freeing disk would be faster on paper. But git writes new objects for every modified multi-GB binary, so with half a gigabyte free the commit itself becomes a failure point. That is why the off-machine copy comes first. Some readers will want to bootstrap Claude Code (P1-5) on day one. That is safe in one narrow sense: `AGENTS.md` already loads. Any agent session before P0-4, though, is working on the only copy of the code. The hook that protects `index.html` does not protect `src/models/`.

---

## Conclusion

The inheritance problem here is not a lack of documentation. Geospatials has more governance prose than most research codebases its size. The problem is that the documentation was written by the agents being governed, and it records intent as accomplishment. The only test that actually failed in this study failed on disk space. The real defects are in places nobody tested: the CV the orchestrator actually calls, the noise generators inside the "real" ingestion script, and the server's trust model. For Claude Code, the main shift is from prose rules to enforced ones. Hooks, path-scoped rules and a smaller always-loaded context matter more than copying `AGENTS.md` across. So does a habit of reading the code before trusting the changelog.

There is also a scientific implication worth stating directly. Each time leakage or synthetic data was removed, the performance numbers fell, reaching chance level at Kelian. The most valuable first improvement is therefore an honest baseline, not a new model: buffered spatial CV on provenance-clean features with real SAR. Only against that baseline can the owner tell whether Kalimantan prospectivity is learnable from the data at hand, and that answer should decide where Claude Code's effort goes next.
