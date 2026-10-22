# Session Handoff: MPM Yilgarn Orogenic Gold Pilot & Borneo Transfer Region

Paste this file (or point the new agent at it) to resume the project on another account.

## Goal
Evidence-based mineral prospectivity mapping (MPM) at 50 m grid resolution.
- **Primary Pilot:** Orogenic Gold, Eastern Goldfields, Yilgarn Craton, WA (`EPSG:7851`).
- **Secondary Evaluation / Transfer-Learning Region:** Epithermal Au-Ag, Porphyry Cu-Au, Iron & Coal, Borneo (`EPSG:32750`).
- **Framing:** Positive-Unlabeled (PU) learning (P = verified economic deposits, RN = verified barren drillholes, U = unexplored background).
- **Core Guardrails:**
  - Dimensionless prospectivity scores and epistemic uncertainty only; no unverified claims of predicting grade, tonnage, or depth from surface/regional geophysics.
  - Leakage-resistant spatial block cross-validation with cluster integrity enforcement and blind holdouts.
  - Strict human review governance on retraining; AI predictions can never become training ground truth (`FeedbackLoopError` guard).

---

## State & Executed Deliverables (Third Session)

### 1. Primary Pilot: Yilgarn Craton (WA) — Version `v2_real`
Trained, spatially evaluated, and exported on authentic government data:
* **Feature Cube (`data/processed/feature_stack.joblib`):**
  - 16 feature layers: authentic 30 m Copernicus DEM slope/aspect/TPI, Euclidean distance to 19 official GSWA faults (`dist_faults_m`), distance to greenstone contacts, 6 geophysics derivatives (TDR, AS, 1VD, gravity HGM, K, Th/U), and 5 spectral alteration indices with vegetation mask.
* **Ground-Truth Labels (`data/processed/dataset.parquet`):**
  - 40,000 cells (10 km x 10 km) in EPSG:7851.
  - 2,520 verified positive deposit cells ($P=1$) from 435 MINEDEX gold mines/deposits.
  - 1,036 verified deep exploration drillhole test cells ($RN=0$, depth $\ge 50$ m outside 250 m deposit halo).
  - 36,444 background cells ($U=-1$).
* **Model Ensemble (`outputs/models/v2_real/xgboost_ensemble.joblib`):**
  - PU Step 1: Isolation Forest scored 29,812 unlabeled cells, mining 2,982 reliable negatives.
  - PU Step 2: 10-bagging XGBoost ensemble trained on $(P \cup RN)$.
* **Leakage-Resistant Spatial Evaluation (`outputs/reports/`):**
  - **Spatial Block Cross-Validation (5 Folds):** PR-AUC = **0.847 ± 0.195** (vs prevalence baseline of 0.529), Recall@FPR10 = **0.631**.
  - **Blind District Holdout:** PR-AUC = **0.837** (500 m buffer ring excluded from training).
* **GIS Deliverables Produced (`outputs/`):**
  - Prospectivity Map: `outputs/maps/prospectivity_v2_real.tif` (Float32 COG)
  - Epistemic Uncertainty Map: `outputs/maps/uncertainty_v2_real.tif` (Float32 COG)
  - SHAP Explanations Map: `outputs/maps/shap_top5_v2_real.tif` (Top 5 features: `mag_as`, `dist_faults_m`, `dem_slope`, `dem_tpi`, `mag_1vd`)
  - Target Polygons: `outputs/targets/targets_v2_real.gpkg` (5 high-priority vector drill targets)

### 2. Secondary Region: Borneo (Kelian Epithermal Au-Ag & Mineral Systems)
Trained, evaluated, and exported on authentic Borneo geodata:
* **Feature Cube (`data/borneo/processed/feature_stack.joblib`):**
  - 193,600 cells (22 km x 22 km, 440 x 440 grid) in UTM Zone 50S (`EPSG:32750`).
  - 9 feature layers: authentic 30 m Copernicus DEM (elev, slope, aspect, TPI, plan curvature), multiscale lineament distance, distance to Kelian central diatreme breccia complex, and proximities to the 79 official ESDM Iron deposits and 850 Coal concessions.
* **Ground-Truth Labels (`data/borneo/processed/dataset.parquet`):**
  - 312 verified Kelian epithermal gold deposit cells ($P=1$) vs 193,288 background units ($U=-1$).
* **Model Ensemble (`outputs/borneo/models/xgboost_ensemble_borneo.joblib`):**
  - PU Step 1: Isolation Forest scored 193,288 unlabeled cells, mining 19,329 reliable negatives.
  - PU Step 2: 10-bagging XGBoost ensemble trained on $(P \cup RN)$.
* **Evaluation (`outputs/borneo/reports/spatial_metrics_borneo.csv`):**
  - Kelian Prospectivity Ranking: PR-AUC = **0.948** (vs prevalence baseline of 0.030), ROC-AUC = **0.998**, Recall@FPR10 = **1.000**.
* **GIS Deliverables Produced (`outputs/borneo/`):**
  - Prospectivity Map: `outputs/borneo/maps/prospectivity_kelian.tif` (Float32 COG)
  - Epistemic Uncertainty Map: `outputs/borneo/maps/uncertainty_kelian.tif` (Float32 COG)
  - Target Polygons: `outputs/borneo/targets/targets_kelian.gpkg` (Vector target delineating the hydrothermal diatreme breccia complex)

### 3. Kalimantan Iron & Coal Datasets (`data/borneo/`)
* **79 Iron Deposits:** `data/borneo/kalimantan_iron_deposits.gpkg` & `.csv` (57 primary skarn/vein, 21 lateritic Fe-Ni on Meratus ophiolite, 1 placer sand; 35 operating mines).
* **850 Coal Concessions:** `data/borneo/kalimantan_coal_deposits.gpkg` & `.csv` (71.1 Gt resources, 23.0 Gt reserves across Kutai, Barito, and Tarakan basins; 846 operating mines).

### 4. Code & Test Suite Status
* **56/56 unit tests passing** via `.venv/bin/pytest tests/unit`.
