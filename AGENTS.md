# AGENTS.md — Geospatials Mineral Prospectivity Mapping (MPM) Agent Guide

**Last Updated:** `2026-11-07` (`v4.0.0-mpm-pipeline-upgrade`)  
**Primary Focus:** Kalimantan $K=4$ Deposit-Mechanics Discovery Engine (`EPSG:32750`), $K=3$ Kutai Coal Basin Engine (`EPSG:32750` / `EPSG:4326`), and Yilgarn Craton Orogenic Gold Baseline (`EPSG:7851`).

---

## 1. Mandatory Governance: Audits & Code Changelog

Every agent working in this repository MUST follow these two standing rules without exception:

1. **Versioned & Dated Audits ([`audits/`](file:///Volumes/Seagate%202/Geospatials/audits/)):**
   - Every technical, data, physics, or training audit MUST be saved in [`audits/`](file:///Volumes/Seagate%202/Geospatials/audits/) using the filename format `YYYY-MM-DD_vX.Y_<slug>.md`.
   - Every new audit MUST be added immediately to [`audits/INDEX.md`](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md) with its version number, ISO date, timestamp, clickable `file://` link, and summary of findings.
   - Read [`audits/INDEX.md`](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md) and the latest audits ([`audits/2026-11-07_v4.0_production_pipeline_upgrade_audit.md`](file:///Volumes/Seagate%202/Geospatials/audits/2026-11-07_v4.0_production_pipeline_upgrade_audit.md) and [`audits/2026-10-28_v2.0_full_codebase_physics_and_training_audit.md`](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-28_v2.0_full_codebase_physics_and_training_audit.md)) before modifying physics equations or training pipelines.

2. **Mandatory Code Changelog ([`CHANGELOG.md`](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md)):**
   - Whenever **any** code, script, test, UI bundle, or configuration file is created or modified, you MUST prepend a dated and versioned entry to [`CHANGELOG.md`](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md) documenting the timestamp, modified files, and exact changes made.

---

## 2. System Architecture & Module Map

> [!TIP]
> **Complete Script, Class & Function Corpus ([`CODEBASE_CORPUS.md`](file:///Volumes/Seagate%202/Geospatials/CODEBASE_CORPUS.md)):**
> Before searching blindly across files, consult [`CODEBASE_CORPUS.md`](file:///Volumes/Seagate%202/Geospatials/CODEBASE_CORPUS.md) (also archived at [`audits/2026-11-07_v3.2_complete_codebase_script_and_function_corpus.md`](file:///Volumes/Seagate%202/Geospatials/audits/2026-11-07_v3.2_complete_codebase_script_and_function_corpus.md)), which indexes all **95 Python modules/scripts, 40 classes, 148 methods, 491 top-level functions, and 24 nested helpers** with exact line ranges (`#Lstart-Lend`), parameter signatures, and clickable `file://` links.

### 2.1 Core Library (`src/`)
- **Canonical Schema & Column Registry (`src/schema/`):**
  - [`src/schema/canonical.py`](file:///Volumes/Seagate%202/Geospatials/src/schema/canonical.py): Single source of truth for canonical feature names, alias resolution maps, deposit type definitions, propensity covariates, and zero-raw-sensor pixel leakage guards (`verify_no_raw_pixel_leakage`).
- **Feature Engineering & Physics Engine (`src/features/`):**
  - [`src/features/geophysics.py`](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py): Demeaned FFT wavenumber potential-field filters (`reduced_to_pole` with MacLeod 1993 stabilization, `first_vertical_derivative`, `analytic_signal`, `tilt_derivative`, `upward_continue`), spatial `horizontal_gradient_magnitude`, `ternary_radiometrics`, AEM conductivity inversions, and 4 km Merdith regional contrast windows.
  - [`src/features/structural.py`](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py): Euclidean fault distances, line density, skeletonized fault intersections, DEM morphometry (`slope`, `aspect`, `topographic_position_index`, Evans 1980 `plan_curvature`), 2D Mohr-Coulomb fault stress resolution (`slip_tendency`, `dilation_tendency`), dual-pol SAR decomposition, and `gravity_gradient_step`.
  - [`src/features/aster_indices.py`](file:///Volumes/Seagate%202/Geospatials/src/features/aster_indices.py) & [`src/features/sentinel2_indices.py`](file:///Volumes/Seagate%202/Geospatials/src/features/sentinel2_indices.py): Ninomiya (2003) / Cudahy (2012) VNIR/SWIR/TIR mineral indices and continuum-removed SWIR absorption features (`extract_continuum_removed_absorptions`).
  - [`src/features/feature_stack.py`](file:///Volumes/Seagate%202/Geospatials/src/features/feature_stack.py): `FeatureStack` `(C, H, W)` raster container, `flatten_to_dataframe`, `reconstruct_map`, and `impute_features`.
- **Labels, PU Hygiene & Spatial Splits (`src/labels/`):**
  - [`src/labels/borneo_labels.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py): Kalimantan $K=4$ deposit mechanics (`porphyry_cu`, `skarn_fe_cu`, `epithermal_au`, `laterite_fe_ni`), alluvial placer & island-centroid (`-1.9982, 113.9995`) exclusion guards, OutMOGC spatial exclusion buffers, logistics exploration propensity, and Hájek asymmetric negative weights.
  - [`src/labels/deposit_labels.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/deposit_labels.py): Yilgarn MINEDEX gold deposit ingestion, SHA-256 immutable positive label lake (`verify_immutable_positive_lake`), and 500 m spatial deduplication.
  - [`src/labels/pu_negatives.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/pu_negatives.py): Two-step PU reliable negative selection (`step1_isolation_forest`, `step2_select_reliable_negatives`), SAR-PU propensity weighting, and barren lithology exclusion.
  - [`src/labels/spatial_splits.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py): Checkerboard block assignment (`assign_spatial_blocks`), connected-component deposit cluster integrity, and buffered spatial cross-validation.
- **Modular Model Subsystem (`src/models/`):**
  - [`src/models/metals.py`](file:///Volumes/Seagate%202/Geospatials/src/models/metals.py): Two-Stage $K=4$ bagged XGBoost ensemble (`TwoStageK4Ensemble`), concentric lateral alteration zonation features, Hájek self-normalized positive weights, and evidential P-A intersection weights.
  - [`src/models/coal.py`](file:///Volumes/Seagate%202/Geospatials/src/models/coal.py): Two-Stage $K=3$ Coal ensemble (`TwoStageK3CoalEnsemble`), `MacroBasinSequenceGate`, hierarchical coal presence, and conditional rank softmax.
  - [`src/models/calibration.py`](file:///Volumes/Seagate%202/Geospatials/src/models/calibration.py): Elkan-Noto posterior calibration, empirical discovery priors $\pi_k$, and closed-boundary monotonic logit scaling.
  - [`src/models/conformal.py`](file:///Volumes/Seagate%202/Geospatials/src/models/conformal.py): Weighted conformal quantile prediction set gate $\mathcal{C}_{1-\alpha}(s)$ and studentized conformal confidence bounds $[L(x), U(x)]$.
  - [`src/models/fusion.py`](file:///Volumes/Seagate%202/Geospatials/src/models/fusion.py): Precision-weighted Bayesian optical WoE log-odds fusion with prior correction and PR-AUC model ensembling.
  - [`src/models/pu_pipeline.py`](file:///Volumes/Seagate%202/Geospatials/src/models/pu_pipeline.py): Backward-compatible facade re-exporting calibration, conformal, metals, and fusion interfaces.
  - [`src/models/unet_model.py`](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py): `DistrictDAEUNet` and `MPMDistrictDAEUNet` with resolution-adaptive 4-branch ASPP bottleneck (`px_size_m`), output bias initialized to $\text{logit}(\pi_k)$, and 0-positive head loss masking.
  - [`src/models/xgboost_model.py`](file:///Volumes/Seagate%202/Geospatials/src/models/xgboost_model.py): Bagged `MPMXGBoost` ensemble with epistemic variance estimation.
  - [`src/models/retrain.py`](file:///Volumes/Seagate%202/Geospatials/src/models/retrain.py): Retraining governance (`FeedbackLoopError` guard against AI self-labeling).
  - [`src/models/training_run_manager.py`](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py): Automatic archival (`archive_existing_run`), master catalog, and run comparison.
- **Evaluation & GIS Deliverables (`src/evaluation/`, `src/outputs/`):**
  - [`src/evaluation/spatial_cv.py`](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py): Unified 5-fold buffered spatial cross-validation trunk (`spatial_buffered_block_kfold`, `BufferedSpatialBlockKFold`) enforcing physical buffer $B_m = \max(r^*_k, 16 \cdot \text{px\_size\_m})$.
  - [`src/evaluation/metrics.py`](file:///Volumes/Seagate%202/Geospatials/src/evaluation/metrics.py): PR-AUC, PR-AUC lift over prevalence, ROC-AUC, Recall@FPR, and NDCG@K.
  - [`src/outputs/prospectivity_map.py`](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py): Float32 COG GeoTIFF exporters, morphological target polygonization (`polygonize_and_rank_targets`), and Geologist Site Card JSON emission (`emit_site_cards_json`).

### 2.2 Key Execution Scripts (`scripts/`)
- [`scripts/run_seven_pass_training.py`](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py): Primary 7-Pass Kalimantan $K=4$ metals + $K=3$ coal training & inference orchestrator.
- [`scripts/run_10_rounds_training.py`](file:///Volumes/Seagate%202/Geospatials/scripts/run_10_rounds_training.py): Multi-round sequential training runner for East Kalimantan.
- [`scripts/build_east_kalimantan_250m.py`](file:///Volumes/Seagate%202/Geospatials/scripts/build_east_kalimantan_250m.py): Builds the $250\text{ m}$ ($2400\times 2000$, $2.92\text{M}$ valid cells) East Kalimantan feature cube and $K=4$ label rasters.
- [`scripts/run_borneo_features.py`](file:///Volumes/Seagate%202/Geospatials/scripts/run_borneo_features.py) & [`scripts/train_borneo_pilot.py`](file:///Volumes/Seagate%202/Geospatials/scripts/train_borneo_pilot.py): Leak-free 50 m Kelian corridor feature engineering and 5-fold spatial CV benchmark.
- [`scripts/dashboard_server.py`](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py) & [`scripts/build_purged_ui.py`](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py): Interactive Control Center HTTP server (`port 8765`) and UI bundle builder.

---

## 3. Core Scientific & Engineering Guardrails

1. **Zero Raw Sensor Pixel Leakage (`verify_no_raw_pixel_leakage`):**
   - Models must never ingest raw reflectance/radiance bands (`b1`–`b14`, `s2_b*`, `aster_b*`) or target-mine coordinate distances (`dist_kelian_center_m`). Only physically meaningful ratios, gradients, and geomechanical tensors are permitted.
2. **Strict Label Hygiene:**
   - Never train on unverified AI predictions (`FeedbackLoopError`).
   - Exclude USGS MRDS island-centroid coordinate artifacts (`(-1.9982, 113.9995)`) and alluvial placers (`"Emas Alluvial"`) from bedrock positive target sets $P_k$.
3. **Provenance Honesty:**
   - Never fabricate sensor metadata in site cards (`emit_site_cards_json`) when a physical raster layer (such as SWIR alteration, AEM conductance, or SAR $\alpha$) is absent from the feature stack—emit `null` and `"sensor_source": "not_available"`.
   - Report both raw PR-AUC and **PR-AUC Lift over Prevalence**, and flag any head with $<5$ held-out validation positives as `"insufficient_data"` rather than substituting a dummy `0.50` score.
4. **Symlink-Safe Run Archival (`archive_existing_run`):**
   - Whenever copying directories during model run archival ([`src/models/training_run_manager.py`](file:///Volumes/Seagate%202/Geospatials/src/models/training_run_manager.py)), always pass `symlinks=True, ignore_dangling_symlinks=True` to `shutil.copytree`. Never allow unmounted external drive symlinks or dangling cache pointers to crash the training pipeline.

---

## 4. Audit Resolutions & Mathematical Verification (v4.0.0-mpm-pipeline-upgrade)

All open discrepancies and pipeline audit findings from v2.0, v3.1, and the v4.0 architectural upgrade have been **fully resolved and verified**:

1. **Evans (1980) `plan_curvature` Formula Bug ([`src/features/structural.py`](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py)):**
   - **RESOLVED in v4.0.0:** Corrected formula to dimensionally consistent $-2.0 \cdot (D \cdot H^2 + E \cdot G^2 - F \cdot G \cdot H) / \text{denom}$.
2. **FFT `_pad_and_taper` Non-Demeaned Hanning Window & Equatorial RTP ([`src/features/geophysics.py`](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py)):**
   - **RESOLVED in v4.0.0:** Added mean subtraction before Hanning taper and Northward RTP sign correction with MacLeod (1993) pseudo-inclination stabilization ($\|I_{\text{amp}}\| \ge 20^\circ$) for equatorial Kalimantan.
3. **Synthetic DEM-Derived Bouguer Gravity ([`scripts/run_indonesia_hd_upgrade.py`](file:///Volumes/Seagate%202/Geospatials/scripts/run_indonesia_hd_upgrade.py)):**
   - **RESOLVED in v4.0.0:** Replaced $-0.1119 \cdot \text{dem}$ proxy with real satellite gravity (Sandwell & Smith / EGM2008 Free-Air minus Bouguer slab correction).
4. **Conformal Set Gate Decoupled from Studentized Bounds (D1, D9, Bug 1):**
   - **RESOLVED in v4.0.0:** Prediction set inclusion $\mathcal{C}_{1-\alpha}(s) = \{k : P_k(s) \ge \tau_k\}$ is strictly evaluated on calibrated point probabilities $P_k(s) \ge 1 - \hat{q}_k$, while lower bound $L_k(s) = P_k(s) - t_{n-1, \alpha} \cdot \sigma_k(s)$ is reserved for drill target polygonization.
5. **Prior-Corrected Bayesian Optical WoE Log-Odds Fusion (D2, Bug 5):**
   - **RESOLVED in v4.0.0:** Subtracted prior odds $\text{logit}(\pi_k)$ in evidential log-odds combination $\text{logit}(P_{\text{fused}}) = \text{logit}(P_{\text{ens}}) + W^+_{\text{swir}} - \text{logit}(\pi_k)$, preventing positive feedback amplification.
6. **Hájek Self-Normalized Positive Weights & Neutral Unlabeled Weights (D3, D4):**
   - **RESOLVED in v4.0.0:** Applied Hájek self-normalization $w_i = (1 / e(x_i)) / \sum_{j \in P} (1 / e(x_j))$ for positive samples and neutral unit weights $w_u = 1.0$ for unlabeled cells.
7. **Resolution-Adaptive ASPP Bottleneck & Vector Logit Bias Init (D4, D6, Bug 3, Bug 8):**
   - **RESOLVED in v4.0.0:** Output classification head biases initialized to $\text{logit}(\pi_k)$, 4-branch ASPP bottleneck dilations dynamically scaled by physical pixel size $\text{px\_size\_m}$, and 0-positive validation heads masked from loss.
8. **Hierarchical Coal Presence & Softmax Rank Probabilities (D5, Bug 4):**
   - **RESOLVED in v4.0.0:** Continuous `MacroBasinSequenceGate` scales coal rank probabilities $P(\text{Rank}_j \cap \text{Coal}) = P(\text{Coal}) \cdot \text{softmax}(z_j)$, guaranteeing $\sum_j P(\text{Rank}_j) \le 1.0$.
9. **Spatial Geometry in Physical Meters (D6, Bug 10):**
   - **RESOLVED in v4.0.0:** All morphological kernels, ASPP receptive fields, and distance transforms parameterized in physical meters rather than raw grid cell counts.
10. **Porphyry Cu Concentric Alteration Zonation (D7):**
    - **RESOLVED in v4.0.0:** Radial zoning features (`potassic_core_prox`, `phyllic_halo_prox`, `propylitic_margin_prox`) with monotonic directional constraints in XGBoost.
11. **Unified 5-Fold Buffered Spatial Cross-Validation (D8):**
    - **RESOLVED in v4.0.0:** Consolidated all spatial splitting into [`src/evaluation/spatial_cv.py`](file:///Volumes/Seagate%202/Geospatials/src/evaluation/spatial_cv.py) with physical buffer distance exclusion $B_m$.
12. **Causal Propensity Covariates Restriction (D10):**
    - **RESOLVED in v4.0.0:** Propensity models restricted strictly to non-geological accessibility and logistics infrastructure covariates (`CANONICAL_PROPENSITY_COLS`).
13. **Composite TreeSHAP & Site Card Physical Layer Fallbacks (Bug 6, Bug 7):**
    - **RESOLVED in v4.0.0:** Implemented chain-rule TreeSHAP decomposition for hierarchical coal models, authentic physical layer resolution from FeatureStack rasters, and 2nd–98th percentile clipping.

---

## 5. Development & Verification Commands

- **Run Full Unit Test Suite (`184 passed`):**
  ```bash
  .venv/bin/pytest tests/unit -q
  ```
- **Run Dedicated v4 Mathematical & Bug Verification Suite (`13 passed`):**
  ```bash
  .venv/bin/pytest tests/unit/test_v4_math_and_bugs.py -q
  ```
- **Run 7-Pass Training on East Kalimantan / Borneo:**
  ```bash
  .venv/bin/python scripts/run_seven_pass_training.py --region EAST_KALIMANTAN --reset
  ```
- **Start Interactive Dashboard Server (`http://127.0.0.1:8765`):**
  ```bash
  .venv/bin/python scripts/dashboard_server.py --port 8765
  ```

---

## 6. Dashboard UI Architecture & Server IPC Guardrails

1. **Single Source of Truth for Web UI ([`scripts/build_purged_ui.py`](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py)):**
   - Never manually edit [`ui/index.html`](file:///Volumes/Seagate%202/Geospatials/ui/index.html) or [`index.html`](file:///Volumes/Seagate%202/Geospatials/index.html) directly. Both files are strictly compiled artifacts generated by [`scripts/build_purged_ui.py`](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py).
   - Any UI layout, styling, JavaScript runtime, or parameter modification MUST be implemented inside [`scripts/build_purged_ui.py`](file:///Volumes/Seagate%202/Geospatials/scripts/build_purged_ui.py) and compiled by executing `.venv/bin/python scripts/build_purged_ui.py`.

2. **Resilient Dual-Mode API Transport (`apiFetch`):**
   - The frontend is designed to run both when served over HTTP (`http://localhost:8765`) and as a standalone file on disk (`file:///...`).
   - All client-side fetch calls in the JavaScript runtime MUST use the `apiFetch(url, options)` transport wrapper instead of raw `fetch("/api/...")`.
   - `apiFetch` automatically resolves relative API paths to `http://localhost:8765` whenever `window.location.protocol === "file:"` or when running on a non-8765 origin.
   - Never introduce hard protocol checks like `if (!window.location.protocol.startsWith("http")) return;` that silently disable server polling and control dispatch in `file://` mode.

3. **Server CORS Invariants & Process Lifecycle:**
   - [`scripts/dashboard_server.py`](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py) MUST send `Access-Control-Allow-Origin: *` on all responses (including 200, 204 `OPTIONS` preflight, 400, and 404 error responses).
   - Before testing new server routes or modifying API endpoints, check for stale server processes (`lsof -i :8765`), terminate obsolete PIDs, and relaunch the daemon process.

4. **API Access Control (v4.16, audit v4.14 `B01`/`B02`):**
   - Every POST route requires the `X-MPM-Token` header and `Content-Type: application/json`. A missing or wrong token gets 403, a wrong media type 415, and a body that is not a JSON object 400. The check runs in `DashboardHandler._read_authorized_json_body` before any route. New POST routes must not bypass it.
   - The token is `$MPM_DASHBOARD_TOKEN` if set, else the 0600 file `~/.config/geospatials/dashboard_token`, created on first start. Never log or commit its value.
   - The server injects the token into the `<meta name="mpm-api-token">` placeholder only when serving `/` to a request with no `Origin` header, a non-cross-site `Sec-Fetch-Site`, and a loopback `Host`. In `file://` mode the UI asks for the token once and keeps it in `localStorage`. `apiFetch` adds the header and content type to every non-GET call, so all POSTs must go through it.
   - `Access-Control-Allow-Origin: *` stays on every response (item 3). The token, not CORS, is what keeps other web pages from driving the server.
   - `_resolve_base_model_dir` returns only directories that resolve inside `outputs/`, because the orchestrator `joblib.load`s from that directory.
   - The server binds loopback only. A non-loopback `--host` needs `--allow-remote`. Do not run it on `0.0.0.0` (the EC2 bootstrap no longer does); use `ssh -L 8765:127.0.0.1:8765` instead.

---

<!-- BEGIN AWS Agent Toolkit rules -->
# AWS Guidance for the new AWS experience

This user has signed up for the new AWS experience. This experience lets you sign into AWS using a social provider and requires the following additional context.

Where this guidance conflicts with the project's own instructions, the project's instructions take precedence.

## Context

### Terminology:

- Say "project" instead of "account" — a project contains an AWS account and settings for sharing with other collaborators
- Say "team member" instead of "IAM user" — users are invited by email, not created or federated in IAM
- Say "AWS Settings" when referring to management tasks at [settings.aws.com](https://settings.aws.com/) (project management, billing, team members, spend limits). Users view their actual AWS resources in the AWS Management Console.
- Say "selected Region" when referring to the user's Region — not "home Region"
- The user has a managed IAM experience. This includes a managed service control policies (SCP) and resource control policies (RCP) that govern the use of AWS. They will still need to use IAM to create policies to let services work with each other. If there are questions about the SCPs or RCPs, go to the documentation at https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html

### Constraints:

- All projects share a single AWS Region determined by the user's contact address. Resources cannot be created in other Regions
- When developing:
  - MUST create all Regional resources in the project's assigned Region
  - You CAN create AWS WAF and Cloudwatch Logs resources in us-east-1 when there are global resources (like a global WAF instance) that require a connection to dependencies in us-east-1.
  - MUST NOT attempt to create Lambda, API Gateway, or other Regional resources in any other Region
  - MUST direct users to confirm their Region in AWS Settings > View all projects > Overview > Additional Info > Region. If the user cannot confirm their Region, check in ~/.aws/config
  - MUST NOT use Lambda@Edge — excluded from both Lambda and CloudFront
  - MUST NOT use CloudFormation StackSets — no multi-account or multi-Region deployments
  - MUST NOT attempt cross-Region actions — no cross-Region replication for DynamoDB/S3/RDS, no multi-Region KMS keys
  - MUST NOT use Route 53 cross-Region routing — geolocation, latency-based, and failover routing policies are not available
- Help level: MEDIUM
<!-- END AWS Agent Toolkit rules -->
