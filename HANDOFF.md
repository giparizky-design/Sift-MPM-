# Session Handoff: Geospatials MPM & Kalimantan K=4 Discovery Engine

**Handoff Version:** `v2.0`  
**Date:** `2026-10-28` (`2026-10-28T23:50:26+08:00`)  
**Agent Guide:** Read [`AGENTS.md`](file:///Volumes/Seagate%202/Geospatials/AGENTS.md) first.  
**Audit Registry:** Read [`audits/INDEX.md`](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md) (`v1.0` through `v2.0`).  
**Code Changelog:** Read [`CHANGELOG.md`](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md) (must be updated on every code change).

---

## 1. Project Overview & Current Architecture

This repository implements an evidence-based **Positive-Unlabeled (PU) Mineral Prospectivity Mapping (MPM)** and **Conformal Risk Ranking** platform specialized for **Kalimantan / Borneo ($K=4$ deposit mechanics)** and benchmarked on the **Yilgarn Craton, Western Australia**.

### Core 7-Pass Kalimantan $K=4$ Architecture
1. **Pass 1 — Label Hygiene & OutMOGC Exclusion Masks ([`src/labels/borneo_labels.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py)):**
   - $K=4$ deposit classes:
     - `k=0`: `porphyry_cu` (Porphyry Cu-Au)
     - `k=1`: `skarn_fe_cu` (Contact Metasomatic Fe-Cu Skarn)
     - `k=2`: `epithermal_au` (Neogene Low/Intermediate-Sulfidation Epithermal Au-Ag)
     - `k=3`: `laterite_fe_ni` (Supergene Ultramafic Ophiolite Lateritic Fe-Ni)
   - Excludes alluvial placers (`"Emas Alluvial"`) and colonial island-centroid coordinates (`-1.9982, 113.9995`) from bedrock positives $P_k$, while retaining explored footprints for logistics propensity modeling $e(\mathbf{s})$.
2. **Pass 2 — Spatial Block Assignment & Logistics Propensity ([`src/labels/spatial_splits.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/spatial_splits.py)):**
   - Assigns spatial checkerboard blocks with connected-component deposit cluster integrity and computes asymmetric negative sample weights $w_-(\mathbf{s}) = 0.25 + 0.75 e(\mathbf{s})$.
3. **Pass 3 — Multi-Scale $K=4$ Ensemble Training ([`src/models/pu_pipeline.py`](file:///Volumes/Seagate%202/Geospatials/src/models/pu_pipeline.py), [`src/models/unet_model.py`](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py)):**
   - **Tabular Branch ([`TwoStageK4Ensemble`](file:///Volumes/Seagate%202/Geospatials/src/models/pu_pipeline.py#L697-L958)):** Stage 1 fits bagged XGBoost heads for Skarn ($k=1$), Epithermal ($k=2$), and Laterite ($k=3$) with P–A intersection evidential feature weighting; Stage 2 feeds out-of-fold Skarn + Epithermal probabilities as telescoping features into the Porphyry head ($k=0$).
   - **Spatial Context Branch ([`MPMDistrictDAEUNet`](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L1011-L1411)):** 250 m decimated ($8\times 8\text{ km}$ patches) Denoising Autoencoder + $K=4$ U-Net trained with `asymmetric_pu_focal_loss`.
4. **Pass 4 — Closed-Boundary Elkan-Noto Posterior Calibration:**
   - Estimates discovery prior $\hat{c}_k$ per head and gates calibrated posteriors by OutMOGC petrophysical boundary envelopes (`OUTMOGCGate`).
5. **Pass 5 — Non-Exchangeable Spatial Conformal Prediction ([`src/evaluation/metrics.py`](file:///Volumes/Seagate%202/Geospatials/src/evaluation/metrics.py)):**
   - Computes 90% lower/upper risk bounds $[L_k(\mathbf{s}), U_k(\mathbf{s})]$, multi-mineral conformal sets $\mathcal{C}_{0.90}(\mathbf{s})$, and `telescoped_system_flag`.
6. **Pass 6 — Morphological Target Polygonization & Ranking ([`src/outputs/prospectivity_map.py`](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py)):**
   - Extracts high-confidence target polygons where $\max_k L_k(\mathbf{s}) \ge L_{\text{thresh}}$.
7. **Pass 7 — Skirrow 4-Bucket Mineral Systems TreeSHAP & Site Cards:**
   - Partitions TreeSHAP explanations into `1_energy_and_metal_sources`, `2_crustal_conduits`, `3_physicochemical_traps`, and `4_alteration_and_anomaly`.

---

## 2. Summary of Datasets & Past Training Runs (`outputs/`)

All 29 historical training runs are cataloged in [`outputs/run_history/training_runs_catalog.json`](file:///Volumes/Seagate%202/Geospatials/outputs/run_history/training_runs_catalog.json) and analyzed in detail in [`audits/2026-10-28_v2.0_full_codebase_physics_and_training_audit.md`](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-28_v2.0_full_codebase_physics_and_training_audit.md):

| Phase / Dataset | Grid & Resolution | Cells / Positives | Key Metrics & Audit Findings |
| :--- | :---: | :---: | :--- |
| **Yilgarn Pilot (`v2_real` & `RUN_0001`)** | $200\times 200$ ($50\text{ m}$, `EPSG:7851`) | $40{,}000$ cells / $1{,}596$ $P$, $1{,}036$ $RN$ | `P vs. RN` 5-fold PR-AUC = **`0.8472`**, ROC-AUC = **`0.8003`**; 7-Pass `P vs. U` Fused PR-AUC = **`0.2465`**. Uses real GSWA magnetics, gravity, radiometrics, and ASTER SWIR. |
| **Borneo Kelian Clean 5-Fold CV (`train_borneo_pilot.py`)** | $440\times 440$ ($50\text{ m}$, `EPSG:32750`) | $193{,}600$ cells / $1{,}382$ $P$ | Pure Copernicus 30m DEM + geomechanics (12 features). Achieves $1.75\times\text{--}1.99\times$ PR-AUC lift ($\text{ROC-AUC } 0.71\text{--}0.75$) in Folds 1 & 3, overall OOF PR-AUC = `0.0061` ($\text{ROC-AUC } 0.4618$). |
| **Kalimantan 500m Island Scan (`RUN_0009`)** | $1800\times 2200$ ($550\text{ m}$, `EPSG:4326`) | $2{,}163{,}587$ cells / $3{,}558$ $P$ | Single-head 7-Pass scan across all 5 Kalimantan provinces: `pr_auc_xgb=0.3965`, `pr_auc_unet=0.5644`, `pr_auc_fused=0.5138`. |
| **Borneo $K=4$ Multi-Head (`RUN_0014`–`RUN_0016`)** | $440\times 440$ ($50\text{ m}$ / $500\text{ m}$) | $193{,}600$ cells / $2{,}567$ $P$ | `epithermal_au` head PR-AUC = `0.0554`; other 3 heads had 0 validation positives in the 22 km Kelian box and defaulted to dummy `0.50` (`pr_auc_fused=0.3888`). |
| **East Kalimantan 250m 10-Round Run (`RUN_0017`–`RUN_0026`)** | $2400\times 2000$ ($250\text{ m}$, `EPSG:4326`) | $2{,}920{,}344$ cells / $1{,}490$ $P$ | Peaked in Round 2 (`pr_auc_fused=0.1793`, active heads mean `0.0725`: `skarn=0.0659`, `epithermal=0.0837`, `laterite=0.0678`) before over-boosting to `0.1451` (active heads mean `0.0268`) in Round 10. Head 0 (`porphyry_cu`) had 0 positives inside `114.5°E–119.5°E` and defaulted to `0.50`. |

---

## 3. Audit Trail (`audits/`)

All audits are versioned and dated in [`audits/`](file:///Volumes/Seagate%202/Geospatials/audits/) (indexed in [`audits/INDEX.md`](file:///Volumes/Seagate%202/Geospatials/audits/INDEX.md)):
- **`v1.0` (`2026-10-21`):** [2026-10-21_v1.0_borneo_geospatial_data_audit.md](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-21_v1.0_borneo_geospatial_data_audit.md)
- **`v1.1` (`2026-10-26`):** [2026-10-26_v1.1_yilgarn_data_feasibility_audit.md](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-26_v1.1_yilgarn_data_feasibility_audit.md)
- **`v1.2` (`2026-10-27`):** [2026-10-27_v1.2_k4_engine_guardrails_and_10round_audit.md](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-27_v1.2_k4_engine_guardrails_and_10round_audit.md)
- **`v1.3` (`2026-10-28`):** [2026-10-28_v1.3_sift_evaluation_and_provenance_audit.md](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-28_v1.3_sift_evaluation_and_provenance_audit.md)
- **`v2.0` (`2026-10-28`):** [2026-10-28_v2.0_full_codebase_physics_and_training_audit.md](file:///Volumes/Seagate%202/Geospatials/audits/2026-10-28_v2.0_full_codebase_physics_and_training_audit.md)

---

## 4. Prioritized Next Steps for the Next Session

1. **Fix Physics Engine Equations (`src/features/`):**
   - Fix Evans (1980) `plan_curvature` formula in [`src/features/structural.py:406`](file:///Volumes/Seagate%202/Geospatials/src/features/structural.py#L406) (`D * H**2 + E * G**2 - F * G * H`).
   - Fix `_pad_and_taper` in [`src/features/geophysics.py:110-143`](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L110-L143) so grids are demeaned before padding and cosine-tapered only in the pad margin (`1.0` across interior pixels).
   - Fix `reduced_to_pole` in [`src/features/geophysics.py:150-226`](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L150-L226) (`theta = np.arctan2(kx, -ky)` + MacLeod low-latitude stabilization for $\|I\| < 20^\circ$).
   - Fix zero-crossing discontinuity in `aggregate_merdith_4km_window` ([`src/features/geophysics.py:718-723`](file:///Volumes/Seagate%202/Geospatials/src/features/geophysics.py#L718-L723)).
2. **Fix Training, Calibration & Evaluation Pipeline (`src/models/`, `scripts/`):**
   - Expand `build_k4_deposit_targets` ([`src/labels/borneo_labels.py`](file:///Volumes/Seagate%202/Geospatials/src/labels/borneo_labels.py)) to map GeoRIMA/MRDS commodities across East Kalimantan and omit 0-positive heads from macro-averaged PR-AUC (instead of substituting `0.50`).
   - Fix `decimate_spatial_grid` on `block_ids` in [`src/models/unet_model.py:1224`](file:///Volumes/Seagate%202/Geospatials/src/models/unet_model.py#L1224) to use nearest-neighbor (`order=0`).
   - Fix `conformal_q_hat_90` logging in [`scripts/run_seven_pass_training.py:707`](file:///Volumes/Seagate%202/Geospatials/scripts/run_seven_pass_training.py#L707) (`q_hat_k4_list` instead of `c_hat_k4`).
   - Fit `OUTMOGCGate` and exclusion masks strictly on `train_idx`, and compute true 5-fold out-of-fold XGBoost scores for `spatial_oof_metrics.json`.
   - Remove fabricated sensor fallbacks in [`emit_site_cards_json`](file:///Volumes/Seagate%202/Geospatials/src/outputs/prospectivity_map.py#L964-L1035) and make `/api/rerank` read-only in [`scripts/dashboard_server.py`](file:///Volumes/Seagate%202/Geospatials/scripts/dashboard_server.py).
3. **Test Suite Verification:**
   - Currently **108 / 108 unit tests pass** (`.venv/bin/pytest tests/unit -q`). Add unit tests locking in the physics and leakage fixes above, and log all changes in [`CHANGELOG.md`](file:///Volumes/Seagate%202/Geospatials/CHANGELOG.md).
