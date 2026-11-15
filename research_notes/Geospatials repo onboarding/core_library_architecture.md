# Core Python Library (`src/`) Architecture: Geospatials MPM

Scope: read-only study of `/Users/gip/Geospatials/src/` (41 `.py` files, 20,211 lines), carried out 2026-11-08. All citations point to files on disk as `file://` links with line anchors. "AGENTS" means `/Users/gip/Geospatials/AGENTS.md`. Line counts come from `wc -l`. The symbol inventory comes from a `grep` of `class`/`def` lines. The test count comes from `pytest --collect-only`. No training was run.

## Q1. What subpackages exist in `src/`, what does each do, how large is it, and how do they depend on each other?

### Takeaway
`src/` has 9 subpackages. AGENTS §2.1 lists only 6 of them. It leaves out `ingestion/`, `harmonization/` and `utils/`, which together hold about 2,140 lines. `models/` is the largest at about 6,200 lines. The import graph mostly runs in one direction (schema/evaluation → labels → models → outputs). There is one layering inversion: `features/polygon_rasterizer.py` imports from `labels/`.

### Cited Findings

**Size per subpackage (lines)**
- **schema**: 275. One file, `canonical.py`. There is no `__init__.py`, so it is an implicit namespace package ([src/schema/](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py)).
- **features**: 5,152 across 8 files: `coal_and_multimodal_features` 977, `geophysics` 921, `structural` 885, `polygon_rasterizer` 822, `aster_indices` 565, `feature_stack` 395, `sar_palsar2` 360, `sentinel2_indices` 226.
- **labels**: 3,380: `borneo_labels` 1,217, `pu_negatives` 969, `deposit_labels` 685, `spatial_splits` 508.
- **models**: 6,227: `unet_model` 1,631, `metals` 1,038, `coal` 817, `retrain` 767, `training_run_manager` 529, `xgboost_model` 437, `conformal` 347, `pu_pipeline` 324, `calibration` 209, `fusion` 128.
- **evaluation**: 919: `metrics` 540, `spatial_cv` 379.
- **outputs**: 2,119: `prospectivity_map` 1,491, `model_introspection` 628.
- **Not listed in AGENTS §2.1**: `ingestion` 817 (`aster_download` 463, `ga_download` 354), `harmonization` 908 (`reproject` 475, `cover_mask` 433), `utils` 414 (`io.py`).

**schema/canonical.py**
- Defines `NON_FEATURE_COLS`. There are 24 entries: coordinates and grid indices, label columns, 6 quarantined "circular-distance leakage" columns such as `dist_kelian_center_m`, and 3 quarantined polygon masks ([canonical.py:L18-L46](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L18)).
- `EXTENDED_NON_FEATURE_COLS` adds 9 more entries ([L51-L62](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L51)).
- Forbidden raw-band patterns come in two forms: substring patterns ([L68-L114](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L68)) and compiled regexes ([L117-L135](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L117)).
- Head names: `K4_HEAD_NAMES` = porphyry_cu, skarn_fe_cu, epithermal_au, laterite_fe_ni. `K3_COAL_HEAD_NAMES` = coal_k0_high_cv, coal_k1_med_cv, coal_k2_low_cv ([L140-L152](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L140)).
- Two sets of priors per domain:
  - Discovery priors used as ĉ floors: `DISCOVERY_PRIORS_K4`, `COAL_PRIORS_K3`.
  - Prevalence priors π_k: `METALS_PREVALENCE_PRIORS` from 0.005 to 0.015; `COAL_PREVALENCE_PRIORS` ([L158-L184](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L158)).
- Label constants: P = 1, RN = 0, U = −1 ([L189-L191](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L189)).
- `verify_no_raw_pixel_leakage` ([L194-L224](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L194)).
- `resolve_warm_start_feature_schema` handles policy `retrain` vs `prior` when a warm-start checkpoint's schema differs from the current dataset ([L229-L275](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L229)).
- Discrepancy: AGENTS says canonical.py holds "alias resolution maps, deposit type definitions, propensity covariates". It does not. `CANONICAL_PROPENSITY_COLS` lives in [src/labels/pu_negatives.py:L526](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L526), and feature-alias resolution lives inside `resolve_physical_layers` in [prospectivity_map.py:L555-L640](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L555).

**features/**
- `FeatureStack` dataclass holds a `(C,H,W)` float32 array, `feature_names`, a rasterio `profile`, and a `data_quality_mask` (True = invalid), with shape validation ([feature_stack.py:L54-L102](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L54)).
- Related helpers: `build_feature_stack` ([L103](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L103)), `flatten_to_dataframe` ([L259](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L259)), `reconstruct_map` ([L328](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L328)).
- `geophysics.py`:
  - FFT helpers `_pad_and_taper` ([L124](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L124)) and `reduced_to_pole` ([L178](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L178)).
  - Derivative filters: FVD, analytic signal, tilt, HGM, upward continuation ([L240-L427](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L240)).
  - AEM conductance and regolith inversion ([L428-L556](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L428)).
  - Bouguer slab, Moho, Parker-Oldenburg ([L557-L845](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L557)).
  - Merdith contrasts and host gates ([L642-L745](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L642)).
  - `load_and_compute_real_gravity_channels` ([L846](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L846)).
- `structural.py`:
  - Distance, lineament and intersection density ([L38-L207](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L38)).
  - DEM slope, aspect, TPI, plan_curvature, flow accumulation ([L208-L536](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L208)).
  - Stress resolution, slip/dilation tendency ([L537-L666](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L537)).
  - Cloude-Pottier and Freeman-Durden SAR decompositions ([L667-L750](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L667)).
  - `gravity_gradient_step`, D8 routing, stream-sediment deconvolution ([L751-L885](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L751)).
- `aster_indices.py`: Ninomiya-style ratios ([L73-L296](file:///Volumes/Seagate%202/Geospatials/src/features/aster_indices.py#L73)) and hyperspectral continuum-removed features ([L400-L565](file:///Volumes/Seagate%202/Geospatials/src/features/aster_indices.py#L400)). AGENTS names a function `extract_continuum_removed_absorptions`; the code calls it `continuum_removed_window` / `compute_hyperspectral_indices`.
- `sentinel2_indices.py`: NDVI, NDWI, clay, iron-oxide ([L54-L226](file:///Volumes/Seagate%202/Geospatials/src/features/sentinel2_indices.py#L54)).
- `sar_palsar2.py` (not in AGENTS): fetches PALSAR-2 γ0 through STAC, defines `SARDataUnavailableError`, and builds `SARProvenance` records ([L65-L360](file:///Volumes/Seagate%202/Geospatials/src/features/sar_palsar2.py#L65)).
- `coal_and_multimodal_features.py` (not in AGENTS):
  - Canopy weight ([L47](file:///Volumes/Seagate%202/Geospatials/src/features/coal_and_multimodal_features.py#L47)).
  - SAR bedding strike via a structure tensor ([L131](file:///Volumes/Seagate%202/Geospatials/src/features/coal_and_multimodal_features.py#L131)).
  - TWI/AMD ([L208](file:///Volumes/Seagate%202/Geospatials/src/features/coal_and_multimodal_features.py#L208)).
  - Litho/strat contacts ([L375](file:///Volumes/Seagate%202/Geospatials/src/features/coal_and_multimodal_features.py#L375)).
  - Gated optical ([L821](file:///Volumes/Seagate%202/Geospatials/src/features/coal_and_multimodal_features.py#L821)).
- `polygon_rasterizer.py` (not in AGENTS): polygon AOIs and `build_polygonal_feature_cube` ([L455](file:///Volumes/Seagate%202/Geospatials/src/features/polygon_rasterizer.py#L455)).

**labels/**
- `borneo_labels.py`:
  - Hygiene filter `filter_valid_ore_candidates` ([L56](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L56)).
  - Vector loaders ([L85](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L85)).
  - `build_k4_deposit_targets` ([L169](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L169)), `spatial_join_borneo_k4` ([L389](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L389)), `rasterize_borneo_k4_labels` ([L498](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L498)).
  - Coal rank labels: `build_k3_coal_rank_targets` ([L840](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L840)) and rim-anchored labels ([L1125](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L1125)).
  - `build_pooled_exploration_footprint` ([L1163](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L1163)).
- `deposit_labels.py` (Yilgarn): MINEDEX and WAMEX loaders, `rasterise_labels` (with a `buffer_p_m=500.0` default at L439), `label_summary` ([L67-L685](file:///Volumes/Seagate%202/Geospatials/src/labels/deposit_labels.py#L67)).
- `pu_negatives.py`:
  - Two-step PU: `step1_isolation_forest` ([L56](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L56)), `step2_select_reliable_negatives` ([L137](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L137)).
  - Propensity: `fit_exploration_propensity` ([L402](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L402)), `fit_pooled_exploration_propensity` ([L547](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L547)).
  - `compute_k4_exclusion_mask` ([L661](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L661)).
  - Asymmetric weights ([L772-L858](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L772)).
  - `OUTMOGCGate`: PCA(4, whitened) + 2-component GMM; points outside χ²₄,0.95 get Mahalanobis-decay Γ_closed ([L859-L969](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L859)).
- `spatial_splits.py`: `assign_spatial_blocks` ([L64](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L64)), the legacy unbuffered `spatial_block_kfold` ([L125](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L125)), camp integrity, blind holdout ([L201-L508](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L201)). It also re-exports the buffered CV from `evaluation.spatial_cv` ([L48-L53](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L48)).

**models/**
- `metals.py`:
  - `compute_hajek_asymmetric_head_weights` ([L30](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L30)), `compute_pa_intersection_weights` ([L80](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L80)), `compute_concentric_zonation_features` ([L162](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L162)).
  - `_BaggedXGBoostHead` ([L246](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L246)).
  - `TwoStageK4Ensemble` with fit, predict_proba, predict_std and SHAP ([L433-L1006](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L433)).
  - `fit_k4_independent_xgboost_heads` ([L1007](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L1007)).
- `coal.py`: `fuse_canopy_log_odds` ([L111](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L111)), `MacroBasinSequenceGate` ([L147](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L147)), `TwoStageK3CoalEnsemble` ([L248-L777](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L248)), `compute_hierarchical_coal_probabilities` ([L778](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L778)).
- `calibration.py`: `estimate_elkan_noto_c` ([L30](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L30)), `estimate_k4_discovery_priors` ([L95](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L95)), `calibrate_closed_boundary_elkan_noto` ([L144](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L144)), `calibrate_monotonic_logit_scaling` ([L176](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L176)).
- `conformal.py`: `ConformalQuantile` ([L19](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L19)), `weighted_conformal_quantile` ([L32](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L32)), lower bound and set gate ([L81-L127](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L81)), `compute_multi_mineral_conformal_bounds` ([L129](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L129)), `build_conformal_target_sets` ([L313](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L313)).
- `fusion.py`: PR-AUC ensemble weights and blending ([L27-L79](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py#L27)); `precision_weighted_log_odds_fusion` combines XGB and U-Net by inverse-variance log-odds ([L80-L128](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py#L80)).
- `unet_model.py`:
  - Legacy `ProspectivityUNet`/`MPMUNet` with an nnPU loss ([L222](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L222), [L765](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L765)).
  - `ASPPBottleneck` ([L330](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L330)).
  - `DistrictDAEUNet` ([L387](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L387)) and `nnPULoss` ([L598](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L598)).
  - `MPMDistrictDAEUNet` wrapper ([L1153-L1631](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L1153)).
- `xgboost_model.py`: legacy bagged `MPMXGBoost` ([L45](file:///Volumes/Seagate%202/Geospatials/src/models/xgboost_model.py#L45)).
- `retrain.py`: `FeedbackLoopError`, `MissingReviewerError`, `RetrainingPipeline` ([L38-L767](file:///Volumes/Seagate%202/Geospatials/src/models/retrain.py#L38)).
- `training_run_manager.py`: run archival and catalog ([L34-L529](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py#L34)).
- `pu_pipeline.py`: a facade, covered under Q5.

**evaluation/**
- `metrics.py`: pr_auc, roc_auc, recall_at_fpr, ndcg_at_k, spatial_cv_evaluate, compare_models, blind_holdout_report, plus a second conformal implementation (`spatial_conformal_calibrate` / `_bounds`) ([L46-L540](file:///Volumes/Seagate%202/Geospatials/src/evaluation/metrics.py#L46)).
- `spatial_cv.py`: `compute_physical_buffer_m` ([L29](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py#L29)), `spatial_buffered_block_kfold` ([L145](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py#L145)), `BufferedSpatialBlockKFold` ([L293](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py#L293)), `assemble_oof_prediction_cube` ([L337](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py#L337)).

**outputs/**
- `prospectivity_map.py`:
  - COG GeoTIFF writers ([L90-L296](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L90)).
  - Legacy `generate_target_polygons` ([L297](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L297)) and `export_targets_geopackage` ([L464](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L464)).
  - `save_conformal_bounds_geotiff` ([L525](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L525)), `resolve_physical_layers` ([L555](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L555)), `polygonize_and_rank_targets` ([L654](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L654)).
  - Skirrow-bucket SHAP partitioning ([L915-L1116](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L915)) and `emit_site_cards_json` ([L1117](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L1117)).
- `model_introspection.py` (not in AGENTS): XGB decision-path tracing and U-Net hook payloads for the UI ([L110-L628](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L110)).

**Not listed in AGENTS**
- `ingestion/`: ASTER search and download ([aster_download.py:L71-L463](file:///Volumes/Seagate%202/Geospatials/src/ingestion/aster_download.py#L71)), Geoscience Australia grids ([ga_download.py:L65-L354](file:///Volumes/Seagate%202/Geospatials/src/ingestion/ga_download.py#L65)).
- `harmonization/`: pilot grid and reprojection ([reproject.py:L68-L475](file:///Volumes/Seagate%202/Geospatials/src/harmonization/reproject.py#L68)), vegetation, water and cover masks plus AEM cover update ([cover_mask.py:L53-L433](file:///Volumes/Seagate%202/Geospatials/src/harmonization/cover_mask.py#L53)).
- `utils/io.py`: load/save raster and vector, `stack_rasters` ([L50-L414](file:///Volumes/Seagate%202/Geospatials/src/utils/io.py#L50)).

**Import graph (from `from src.` statements)**
- schema, evaluation and utils import nothing from `src`.
- ingestion and harmonization import only `utils`.
- labels imports `evaluation` (spatial_splits → spatial_cv).
- features imports `features`, `labels` and `schema`. The single `labels` import is [polygon_rasterizer.py:L30-L39](file:///Volumes/Seagate%202/Geospatials/src/features/polygon_rasterizer.py#L30), which pulls in `borneo_labels`.
- models imports `evaluation`, `labels`, `schema` and itself; for example `unet_model` imports `fusion` ([unet_model.py:L52](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L52)).
- outputs imports `models` and `schema`; `model_introspection` imports `DistrictDAEUNet` ([model_introspection.py:L118](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L118), [L263](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L263)).
- Nothing in `src` imports `scripts`.

**Dependencies**
- `pyproject.toml` still names the package `mpm-yilgarn` v1.0.0, "Orogenic Gold, Yilgarn Craton pilot". It requires rasterio, geopandas, shapely, xarray, numpy, scipy, scikit-image, scikit-learn, xgboost, shap, torch, pandas, pyarrow, joblib and requests ([pyproject.toml](file:///Volumes/Seagate%202/Geospatials/pyproject.toml)).
- `requirements.txt` covers only an MCP/RAG toolchain: mcp, httpx, pymupdf, lancedb, sentence-transformers, torch, pydantic, rich, typer ([requirements.txt](file:///Volumes/Seagate%202/Geospatials/requirements.txt)).

### Inferences
- `models` is the hub. `outputs` is downstream of it. `schema` is the leaf every other layer trusts.
- The features→labels edge is a layering smell. It is confined to `polygon_rasterizer`.
- `pyproject.toml` metadata predates the Kalimantan pivot, and `requirements.txt` does not describe the MPM library. The real environment is whatever is installed in `.venv`.
- CODEBASE_CORPUS.md counts files per subsystem (for example "src/models/: 11 files") and lists the ingestion, harmonization and utils dirs. That makes it a better map than AGENTS §2.1 ([CODEBASE_CORPUS.md:L1-L30](file:///Volumes/Seagate%202/Geospatials/CODEBASE_CORPUS.md)).

### Gaps
- `config/*.yaml` content and how `scripts/` load config were not studied; that was out of scope.
- `.venv` package versions were not checked.

## Q2. What is the end-to-end data flow, and what goes in and out?

### Takeaway
The library is a toolbox, not a pipeline. The end-to-end sequence lives in `scripts/run_seven_pass_training.py` (2,985 lines).

That script runs these stages in order:
1. Rasters/FeatureStack → leakage check.
2. K=4/K=3 labels + propensity + exclusion masks + Hájek weights.
3. Spatial block folds.
4. TwoStage XGB ensembles + DistrictDAE-UNet.
5. Precision-weighted fusion.
6. Closed-boundary Elkan-Noto calibration gated by OUTMOGC.
7. Conformal set and lower bound.
8. `polygonize_and_rank_targets` at L(x) ≥ 0.50.
9. Site-card JSON, COGs and GPKG.

### Cited Findings
- **Inputs.**
  - Harmonized rasters are loaded and stacked by [utils/io.py:L50-L414](file:///Volumes/Seagate%202/Geospatials/src/utils/io.py#L50) and [harmonization/reproject.py:L174-L475](file:///Volumes/Seagate%202/Geospatials/src/harmonization/reproject.py#L174).
  - The result is wrapped in `FeatureStack(array (C,H,W), feature_names, profile, data_quality_mask)` ([feature_stack.py:L54-L102](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L54)) and flattened to a per-cell DataFrame ([L259](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py#L259)).
- **Leakage check.** The orchestrator calls `verify_no_raw_pixel_leakage(feature_cols)` twice ([run_seven_pass_training.py:L754, L766](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L754)).
- **Labels.**
  - Borneo vector sources pass through `filter_valid_ore_candidates` inside `build_k4_deposit_targets` ([borneo_labels.py:L210-L212](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L210)) and are rasterized into K=4 label columns ([L498](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L498)).
  - Coal K=3 rank targets come from [L840](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L840).
- **PU and propensity.**
  - The orchestrator imports `fit_pooled_exploration_propensity`, `compute_k4_exclusion_mask` and `compute_asymmetric_head_weights` ([run_seven_pass_training.py:L47-L51](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L47)).
  - It calls the exclusion mask at [L979](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L979) and the head weights at [L1124](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1124).
  - Propensity covariates are restricted to `dist_roads`, `dist_rivers`, `dist_ports`, with a list of forbidden patterns ([pu_negatives.py:L526-L544](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L526)).
- **Spatial folds.** The orchestrator uses `assign_spatial_blocks` and the legacy `spatial_block_kfold(block_ids, n_folds=5, random_state=42)` ([run_seven_pass_training.py:L53, L880](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L880)). That function is a shuffled round-robin GroupKFold. It has no buffer exclusion ([spatial_splits.py:L125-L195](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L125)).
- **Models.**
  - `TwoStageK4Ensemble(...)` is built at [L1216](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1216). Warm start loads a prior ensemble via `.load` at [L602](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L602).
  - `MPMDistrictDAEUNet` is built at [L1281](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1281).
  - The coal analogues are at [L1486](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1486) and [L1546](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1546).
  - Inside `TwoStageK4Ensemble.fit`, the leakage check runs again ([metals.py:L559-L564](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L559)).
- **Calibration.**
  - Pass 4A estimates ĉ_k with `estimate_k4_discovery_priors` on validation-index cells, passing train/val block IDs for an overlap check ([run_seven_pass_training.py:L1383-L1390](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1383)).
  - If no validation positives exist, it falls back to the ensemble's in-sample `c_hat_`, or 0.35 ([L1391-L1396](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1391)).
  - It then applies `calibrate_closed_boundary_elkan_noto(p_raw, c_hat, gamma_closed=outmogc_gates[:,k], min_c=0.10)`. Heads with no positives are forced to 0 ([L1398-L1410](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1398)).
  - The formula is p_cal = Γ·min(1, p_raw / max(ĉ, min_c)) ([calibration.py:L144-L174](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L144)).
- **Coal hierarchy.** `compute_hierarchical_coal_probabilities` is imported and used by the orchestrator ([run_seven_pass_training.py](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py)). The formula is P_c = G_basin · P(coal_any) · softmax(z/T)_c ([coal.py:L778-L817](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L778)).
- **Conformal.** `compute_multi_mineral_conformal_bounds` is called for metals and coal. It returns per-head set gates, lower bounds and (q_set, q_res) dicts ([conformal.py:L129-L245](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L129)).
- **Outputs.**
  - `polygonize_and_rank_targets(calibrated_prob, lower_bound_L, upper_bound_U, dilation_tendency_Td, alteration_index, cover_depth_m, profile, l_threshold=0.50, closing_size=3, min_area_ha=1.0, weights=(0.40,0.25,0.20,0.15), ...)` ([prospectivity_map.py:L654-L670](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L654)). It is called at [run_seven_pass_training.py:L2096](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L2096) and L2286 (coal).
  - `emit_site_cards_json` follows at [L2178](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L2178) and L2369.
  - Float32 COG writers are at [prospectivity_map.py:L90-L554](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L90).
- **Archival.** `archive_existing_run(out_dir, region_name)` runs before a new run ([run_seven_pass_training.py:L587, L622](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L587)).
  - It copies `seven_pass_summary.json`, `blind_holdout.json`, `targets/`, `maps/`, and `models/` without `*.pt`, into `outputs/run_history/<region>/run_<ts>` ([training_run_manager.py:L213-L290](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py#L213)).
- **Other entry points that reuse the same outputs API:** `run_kalimantan_500m_and_100m_scans.py`, `run_kalimantan_50m_island_scan.py`, `run_viewport_7pass.py`, `train_unlabeled_25km_tile.py`, `run_sangatta_10rounds_and_hotspot_scan.py`, and `dashboard_server.py:L1927` (grep of `polygonize_and_rank_targets` / `emit_site_cards_json` in `scripts/`).

### Inferences
- Much of the orchestration logic lives in a script rather than in `src`. Calibration fallbacks, zeroing heads with no positives, and gate wiring are all in the 2,985-line script, so the library's own guarantees depend on how each script wires it.
- There are at least 7 scripts that independently chain polygonize → site cards. Divergence between them is a risk.

### Gaps
- The DAE pretraining, U-Net/XGB fusion call site and SHAP export inside the orchestrator were not traced line by line.
- `metrics.spatial_conformal_bounds` is a second conformal implementation. I did not confirm which scripts call it.

## Q3. Which scientific guardrails are enforced in code, where, and are they actually called?

### Takeaway
- **Implemented and wired in:** the leakage check and symlink-safe archival.
- **Implemented but conditional:** placer/centroid hygiene. It depends on an upstream `is_centroid_artifact` flag.
- **Implemented but never called by any pipeline:** `FeedbackLoopError`. Only the unused `RetrainingPipeline` and its unit test use it.
- **Not implemented anywhere in `src/` or `scripts/`:** the `insufficient_data` flag and PR-AUC lift over prevalence.

### Cited Findings

**`verify_no_raw_pixel_leakage`**
- Defined at [canonical.py:L194-L224](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L194). It raises `ValueError` on any substring or regex match.
- Called in the library at:
  - [metals.py:L559, L564](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L559) (fit)
  - [coal.py:L405, L410, L603, L608](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L405) (fit and predict)
  - [polygon_rasterizer.py:L767](file:///Volumes/Seagate%202/Geospatials/src/features/polygon_rasterizer.py#L767)
  - [model_introspection.py:L387](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L387)
- Called in scripts at [run_seven_pass_training.py:L754, L766](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L754), `train_borneo_pilot.py:L64`, `build_kutai_basin_100m.py:L190`, `harvest_kalimantan_multimodal.py:L859`, `run_viewport_7pass.py:L196`, `train_unlabeled_25km_tile.py:L270`, `run_kalimantan_500m_and_100m_scans.py:L275`, `run_sangatta_10rounds_and_hotspot_scan.py:L504`.
- Caveat 1: matching is substring-based on tokens such as `"b10"`, `"dn_"`, `"mrds"`, `"kelian"` ([canonical.py:L68-L114](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py#L68)). Any legitimate feature name containing one of those tokens would be rejected.
- Caveat 2: the guard checks names only, not values.

**`FeedbackLoopError`**
- Defined at [retrain.py:L38](file:///Volumes/Seagate%202/Geospatials/src/models/retrain.py#L38).
- Raised in `RetrainingPipeline.load_label_lake` when any label-lake parquet row has `label_source == "ai_prediction"` ([retrain.py:L121-L165](file:///Volumes/Seagate%202/Geospatials/src/models/retrain.py#L121)).
- A grep finds no reference to `RetrainingPipeline` or `src.models.retrain` anywhere in `scripts/`. Only `tests/unit/test_retrain.py` uses it.
- The 7-pass orchestrator builds labels from vector sources, not from the label lake. The guard is therefore not on the production training path.

**Placer and island-centroid exclusion**
- `filter_valid_ore_candidates` drops `commodity_label == "Emas Alluvial"` and rows where `is_centroid_artifact` is truthy ([borneo_labels.py:L56-L78](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L56)). It is applied to GEORIMA, iron and MRDS sources ([L210-L212](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L210)).
- The loaders default `is_centroid_artifact = False` when the column is missing ([L128-L143](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L128)).
- The constants `_ARTIFACT_LAT = -1.9982`, `_ARTIFACT_LON = 113.9995`, `_ARTIFACT_TOL_DEG = 0.01` ([L606-L608](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L606)) are never referenced in `src` (grep), so they are dead.
- The coordinate check actually happens upstream in scripts, with different tolerances:
  - `scripts/fetch_borneo_data.py:L69` uses ±0.05°.
  - `scripts/harvest_kalimantan_multimodal.py:L107-L108, L627, L794-L817` checks a "Gate 3" within 5.5 km.
- Alluvial sites stay inside the pooled exploration footprint used for propensity, by design ([borneo_labels.py:L330-L340](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L330)).

**`insufficient_data` and PR-AUC lift**
- A grep for `insufficient_data` across all `.py` files returns no matches. It appears only in `AGENTS.md` and `audits/2026-10-28_v1.3_sift_evaluation_and_provenance_audit.md`.
- `metrics.pr_auc` is plain `average_precision_score` with no lift and no minimum-positive check ([metrics.py:L46-L69](file:///Volumes/Seagate%202/Geospatials/src/evaluation/metrics.py#L46)). A grep of `metrics.py` finds no function or field that reports lift.
- Fold validation gates compute PR-AUC whenever at least one positive and one negative are present ([coal.py:L573-L574](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L573)).
- Related magic fallbacks:
  - `TwoStageK4Ensemble` sets `c_hat_[k] = 0.50` when a head has no positives ([metals.py:L784-L785](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L784)).
  - The orchestrator uses `else 0.5` for ĉ at [run_seven_pass_training.py:L375](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L375) and 0.35 at [L1393-L1394](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1393).

**Symlink-safe archival**
- All three `shutil.copytree` calls in `archive_existing_run` pass `symlinks=True, ignore_dangling_symlinks=True` ([training_run_manager.py:L273, L277, L282-L288](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py#L273)).
- It runs from the orchestrator ([L587, L622](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L587)) and from `run_sangatta_10rounds_and_hotspot_scan.py:L93`.
- A disk-space pre-flight check also runs ([training_run_manager.py:L249](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py#L249)).
- One `copytree` elsewhere lacks the flags: `scripts/move_to_geospatials_research.py:L72`. That copies a lancedb store, not run archives.

**Site-card provenance**
- `resolve_physical_layers` returns `None` for absent layers ([prospectivity_map.py:L555-L640](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L555)).
- The site card sets `"sensor_source": "not_available"` only when both the Al-OH label and the doublet value are `None`. Otherwise it emits the `sensor_source` parameter, which defaults to `"EMIT_SWIR"`, regardless of which sensor actually produced the layer ([L1121, L1465](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L1121)).

**Other guards that exist**
- Elkan-Noto ĉ estimation raises on train/validation block overlap ([calibration.py:L52-L61](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L52)).
- Zero-positive U-Net heads are masked from the loss ([unet_model.py:L700-L703](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L700)).

### Inferences
- Two of the five AGENTS guardrails, the feedback-loop guard and the `insufficient_data` flag, are not enforced on the production path. A new engineer should not assume either protection is active.
- The centroid guard is effectively "trust the upstream flag". Re-ingesting data without running the fetch or harvest scripts would silently disable it.
- `sensor_source` may assert EMIT even when ASTER-derived layers are used. This is a provenance-honesty risk; it should be checked against how scripts pass `sensor_source`.

### Gaps
- I did not check whether `dashboard_server.py` or the UI computes lift or an insufficient-data label client-side.
- I did not check how each script sets `sensor_source`.

## Q4. Does the code match the AGENTS.md §2 and §4 claims?

### Takeaway
I spot-checked 12 claims:
- **Hold:** 5 (plan curvature; demeaned FFT and MacLeod RTP; coal softmax hierarchy; logit(π_k) bias init; restricted propensity covariates).
- **Partly hold or worded inaccurately:** 4 (WoE fusion; Hájek normalization; the "4-branch" ASPP; 0-positive masking is fine but other claims around it are not).
- **Contradicted:** 3 (conformal gate "strictly on P_k"; unified buffered CV used by the main pipeline; `verify_immutable_positive_lake` exists).

### Cited Findings
1. **Evans plan_curvature: HOLDS.**
   - The code computes `-2.0*(D*H**2 + E*G**2 - F*G*H)/(G²+H²+ε)` with D = ½z_xx, E = ½z_yy, F = z_xy ([structural.py:L381-L398](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L381)). Flat cells are set to 0.
   - The docstring says edge and flat cells "may contain NaN", but the code sets them to 0 ([L373-L375](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L373)).
2. **Demeaned FFT with MacLeod RTP: HOLDS, with a changed implementation.**
   - Demeaning happens before padding ([geophysics.py:L147-L149](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L147)).
   - The taper is no longer a Hanning window. It is a reflect pad plus a cosine taper on the pad ring only ([L160-L170](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L160)). `_hanning_taper_2d` ([L61](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L61)) is now unreferenced (grep).
   - RTP flips the k_y sign so +k_y points north ([L211-L213](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L211)).
   - The pseudo-inclination is `max(|I|, 20°)`, applied only to the amplitude term; the filter is conj(p_phase)² / (|p_amp|²·|p_phase|²) ([L216-L229](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L216)). Its magnitude reduces to 1/|p_amp|², which matches the claim.
3. **Conformal set gate "strictly P_k(s) ≥ 1 − q̂_k": CONTRADICTED.**
   - The gate is `p_cal − 0.5·σ ≥ 1 − q_set` ([conformal.py:L106-L127](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L106)). The nonconformity score is `V = 1 − p + 0.5σ` ([L257, L268](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L257)).
   - The lower bound uses a conformal studentized-residual quantile, `q_res = quantile((1−p)/σ)`, not t_{n−1,α} ([L280-L291](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L280), [L81-L103](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L81)).
   - When `q_res_in` is `None`, a legacy branch couples the bound back to q_set: `p − 0.5σ − (1 − q_set)` ([L298-L306](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L298)).
4. **Prior-corrected WoE fusion: PARTLY HOLDS.**
   - Coal implements `l = logit(p_sub) + w_canopy · max(0, logit(p_opt) − logit(π_opt))` ([coal.py:L111-L141](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L111)). The AGENTS formula instead subtracts logit(π) from the total and has no canopy weight or max(0, ·).
   - The metals XGB/U-Net fusion is an inverse-variance average of log-odds with no prior term ([fusion.py:L80-L128](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py#L80)).
5. **Hájek weights: HOLDS in substance; the formula in AGENTS is wrong.**
   - The code normalizes by the mean of 1/e over P_k, so positive weights average 1.0 ([metals.py:L70-L73](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L70)). AGENTS writes division by the sum, which would make the weights sum to 1.
   - Unlabeled weights are the exclusion mask × 1.0 ([L76](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L76)).
   - The same logic is duplicated in `pu_negatives.compute_asymmetric_pu_weights` ([pu_negatives.py:L772-L830](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py#L772)), which is the version the orchestrator imports.
6. **ASPP "4-branch, scaled by px_size_m" and logit(π_k) bias init: bias HOLDS; ASPP only partly.**
   - Dilation = `max(1, round(r / (16·px_size_m)))` for radii [500, 1500, 3000, 6000] m ([unet_model.py:L345-L348](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L345)).
   - The `d4` value (6 km) is computed but never used. The branches are a 1×1 plus three dilated convs using d1–d3 ([L351-L384](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L351)).
   - At the default 250 m pixel size, all three dilations are 1 (500/4000 → 1, 1500/4000 → 1, 3000/4000 → 1), so the multi-scale receptive field collapses. This follows from the formula above.
   - The bias init to logit(π_k) holds ([L438-L447](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L438)).
   - Zero-positive head masking holds ([L700-L703](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L700)).
7. **Hierarchical coal softmax, Σ ≤ 1: HOLDS** ([coal.py:L806-L817](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L806)), and it is used by the orchestrator.
8. **Unified 5-fold buffered spatial CV: CONTRADICTED in practice.**
   - The buffered trunk exists ([spatial_cv.py:L29-L58, L145, L293](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py#L29)), with B_m = max(r*, 16·px) and a default r* of 6,000 m.
   - The main orchestrator calls the unbuffered `spatial_block_kfold` ([run_seven_pass_training.py:L880](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L880); [spatial_splits.py:L125](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py#L125)). So do `train_borneo_pilot.py` and `run_pilot.py`.
   - No script calls `BufferedSpatialBlockKFold` or `spatial_buffered_block_kfold` (grep across `src` and `scripts`).
9. **Propensity restricted to logistics covariates: HOLDS**, but the list lives in `pu_negatives.py:L526-L544`, not in canonical.py.
10. **"SHA-256 immutable positive label lake (`verify_immutable_positive_lake`)" in deposit_labels.py: FALSE.** No such function exists anywhere in `src`. The only SHA-256 use is the training-data hash in `retrain.py:L370-L398`. "500 m spatial deduplication" appears only as a `buffer_p_m=500.0` default on `rasterise_labels` ([deposit_labels.py:L435-L439](file:///Volumes/Seagate%202/Geospatials/src/labels/deposit_labels.py#L435)).
11. **"Report PR-AUC Lift and flag `insufficient_data`": NOT IMPLEMENTED** (see Q3).
12. **"Full unit test suite (184 passed)": STALE.** `pytest --collect-only` now collects 226 tests in `tests/unit`. They were collected, not run.

### Inferences
- AGENTS §4 is optimistic documentation. Items 3, 8 and 10 are material, because they change what the outputs mean: target sets are gated on a σ-penalized score, and CV metrics may be inflated by spatial autocorrelation without a buffer.
- The ASPP degeneracy at 250 m means the claimed "resolution-adaptive" receptive field only does real work at about 50–100 m pixels.

### Gaps
- §4 item 3 (real gravity replacing the DEM proxy) is in `scripts/run_indonesia_hd_upgrade.py`, which I did not open. `load_and_compute_real_gravity_channels` exists at [geophysics.py:L846](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L846).
- §4 items 10 (zonation monotone constraints) and 13 (chain-rule TreeSHAP) were not verified.

## Q5. Code-quality observations

### Takeaway
The code is heavily type-hinted: only 2 of 338 functions lack a return annotation, and there are no TODO or FIXME markers. The debt is structural rather than cosmetic:
- a back-compat facade;
- legacy model and CV paths running alongside the "v4" ones;
- duplicated helpers;
- dead constants;
- several files over 1,000 lines;
- orchestration logic that lives in 3,000-line scripts.

### Cited Findings
- **Facade.**
  - `pu_pipeline.py` re-exports calibration, coal, conformal, fusion, metals, PU-negatives and schema symbols, including private ones (`_logit`, `_sigmoid`, `_BaggedXGBoostHead`, `_FORBIDDEN_RAW_BAND_PATTERNS`) ([pu_pipeline.py:L1-L86](file:///Volumes/Seagate%202/Geospatials/src/models/pu_pipeline.py#L1)).
  - It also still holds legacy single-commodity code: `pu_settings`, `fit_pu_model` built on `MPMXGBoost`, `score_metrics`, `spatial_cv_pu` ([L95-L324](file:///Volumes/Seagate%202/Geospatials/src/models/pu_pipeline.py#L95)).
  - Scripts import through it, for example `train_borneo_pilot.py:L29`, `build_kutai_basin_100m.py:L25`, `run_sangatta_10rounds_and_hotspot_scan.py:L48`, `harvest_kalimantan_multimodal.py:L67`.
- **Legacy parallel paths.**
  - `ProspectivityUNet`/`MPMUNet` ([unet_model.py:L222, L765](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L222)) sit beside `DistrictDAEUNet`/`MPMDistrictDAEUNet`.
  - `MPMXGBoost` ([xgboost_model.py:L45](file:///Volumes/Seagate%202/Geospatials/src/models/xgboost_model.py#L45)) sits beside `_BaggedXGBoostHead`.
  - `generate_target_polygons` ([prospectivity_map.py:L297](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L297)) sits beside `polygonize_and_rank_targets`.
  - `spatial_block_kfold` sits beside `BufferedSpatialBlockKFold`.
  - There are two conformal implementations: `models/conformal.py` and `evaluation/metrics.py:L423-L540`.
- **Duplicated helpers.**
  - `_logit`/`_sigmoid` are defined three times: [calibration.py:L18-L27](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L18), [fusion.py:L17-L25](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py#L17), [coal.py:L101-L109](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L101).
  - `_safe_ratio` is defined twice: [aster_indices.py:L39](file:///Volumes/Seagate%202/Geospatials/src/features/aster_indices.py#L39), [sentinel2_indices.py:L27](file:///Volumes/Seagate%202/Geospatials/src/features/sentinel2_indices.py#L27).
  - `_downsample_to_8x8` and `downsample_8x8` sit in the same file ([model_introspection.py:L209, L271](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L209)). That file also imports `DistrictDAEUNet` both lazily and at module level ([L118, L263](file:///Volumes/Seagate%202/Geospatials/src/outputs/model_introspection.py#L118)).
  - Hájek weights are duplicated across `metals.py:L30` and `pu_negatives.py:L772`.
  - `fuse_bayesian_canopy_log_odds` is just an alias of `fuse_canopy_log_odds` ([coal.py:L144](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py#L144)).
- **Dead code.**
  - `_hanning_taper_2d` ([geophysics.py:L61](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L61)).
  - The `_ARTIFACT_*` constants ([borneo_labels.py:L606-L608](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L606)).
  - The unused ASPP `d4` ([unet_model.py:L348](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L348)).
  - `RetrainingPipeline`, which is only used by tests.
- **Large files** (over 1,000 lines): `unet_model.py` 1,631, `prospectivity_map.py` 1,491, `borneo_labels.py` 1,217, `metals.py` 1,038. `TwoStageK4Ensemble.fit` alone spans [metals.py:L539-L848](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L539), about 310 lines. In `scripts/`, `run_seven_pass_training.py` is 2,985 lines and `dashboard_server.py` is 2,997.
- **Type hints.** An AST scan of `src` found 338 functions; only 2 have no return annotation (one is `downsample_8x8(arr_2d)` at model_introspection.py:L271). Modules use `from __future__ import annotations`.
- **TODO/FIXME/XXX/HACK:** none in `src` (grep).
- **Paths.** There are no absolute `/Users/...` paths in `src` (grep). Paths are resolved relative to the repo:
  - `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` ([borneo_labels.py:L601](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L601)).
  - `RUN_HISTORY_DIR = OUTPUTS_DIR / "run_history"` ([training_run_manager.py:L30](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py#L30)).
  - Hard-coded Borneo bounding box ([borneo_labels.py:L602-L605](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py#L602)).
- **Magic numbers.**
  - Fallback ĉ values of 0.50 and 0.35 (see Q3).
  - Default conformal penalty of 0.5·σ ([conformal.py:L257](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py#L257)).
  - Fusion `default_sigma=0.10` ([fusion.py:L85](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py#L85)).
  - Coal π_opt defaults `{0:0.02, 1:0.04, 2:0.05}` hard-coded in `coal.py:L429`, separately from `COAL_PRIORS_K3` in canonical.py.
- **In-sample ĉ.** `TwoStageK4Ensemble` estimates `c_hat_` from training positives ([metals.py:L770-L785](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py#L770)). Elkan-Noto's e1 estimator assumes held-out positives, and the library's own `estimate_elkan_noto_c` documents that requirement ([calibration.py:L36-L40](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py#L36)). The orchestrator uses this in-sample value only as a fallback ([run_seven_pass_training.py:L1391-L1396](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L1391)).
- **Package hygiene.** `src/schema/` has no `__init__.py`, unlike the other subpackages, which have empty or one-line ones (file listing). `pyproject.toml` name and version are stale (Q1).

### Inferences
- The highest-value cleanups for a takeover are:
  1. Route the orchestrator through `BufferedSpatialBlockKFold`.
  2. Implement PR-AUC lift and `insufficient_data` in `evaluation/metrics.py`.
  3. Either wire `FeedbackLoopError` into the actual label build or correct AGENTS.
  4. Fix or document the conformal gate and the ASPP behaviour at 250 m.
  5. Consolidate the duplicated helpers into one module.
- `pu_pipeline.py` should stay until scripts import directly from the domain modules, because many scripts depend on it.

### Gaps
- No runtime profiling, and the test suite was not executed; I only collected tests.
- I did not measure coverage of legacy vs v4 paths.
- The `scripts/` layer, apart from the call sites cited above, is outside this researcher's scope.
