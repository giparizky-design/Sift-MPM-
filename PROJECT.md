# Project: Multi-Domain MPM Pipeline v4.0 Production Upgrade

## Architecture
- **Canonical Schema & Common Definitions**: `src/schema/canonical.py` serves as single source of truth for `NON_FEATURE_COLS`, `_FORBIDDEN_RAW_BAND_PATTERNS`, commodity head names, priors, and labels.
- **Mathematical Physics & Calibration Models**: Modular decomposition in `src/models/` (`metals.py`, `coal.py`, `calibration.py`, `conformal.py`, `fusion.py`, `unet_model.py`) preserved with zero breakage via facade module `src/models/pu_pipeline.py`.
- **Spatial Geometry & Label Hygiene**: Physical metric parameterization (`px_size_m`), causal pre-mining propensity covariates, and Hájek self-normalized positive/unlabeled weights in `src/labels/`.
- **Spatial Validation Trunk**: Unified 5-fold buffered cross-validation (`src/evaluation/spatial_cv.py`) enforcing buffer zone $B_m = \max(r^*_k, 16 \cdot \text{px\_size\_m})$.
- **7-Pass Orchestration & Inference**: Full-fold negative evaluation in Pass 7A, post-U-Net hierarchical coal softmax in Pass 4B, unified conformal bounds in Pass 5 and chunked inference, physical layer fallback reconstruction, and dynamic summary telemetry in `scripts/`.
- **Repository Governance**: 100% backward compatibility, full regression suite passing, updated `CODEBASE_CORPUS.md`, `AGENTS.md`, `audits/INDEX.md`, and `CHANGELOG.md`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | D1: Decoupled Conformal Bounds | Decouple 90% set gate from studentized lower bound using propensity quantiles | M1 | R1-D1 |
| 2 | D2: Prior-Corrected Optical WoE | `w_opt,c^+(s) = max(0, logit(p_opt,c(s)) - logit(pi_opt,c))` in TwoStageK3CoalEnsemble | M1 | R1-D2 |
| 3 | D3: Eliminate SAR/SCAR Double-Correction | Hájek self-normalized positive weights (mean 1.0 over P_k), w_unlab = 1.0 outside exclusion, monotonic logit scaling | M1 | R1-D3 |
| 4 | D4: Hájek nnPULoss & Bias Init | Fold-level mean normalization, vector-valued pi_p, output bias init logit(pi_k) | M1 | R1-D4 |
| 5 | D5: Coal Rank Softmax & Post-U-Net Gate | Move MacroBasinSequenceGate post-U-Net; replace Noisy-OR with hierarchical rank softmax | M1, M3 | R1-D5 |
| 6 | D6: Spatial Geometry in Physical Meters | Parameterize coal rim halos, canopy NDVI footprints, ASPP dilations by px_size_m | M2 | R1-D6 |
| 7 | D7: Porphyry Cu Meta-Features | Align Stage-2 tree parameters (depth=6, n_bag=10), concentric multi-ring zonation, monotonic constraints | M1 | R1-D7 |
| 8 | D8: Unified Buffered Spatial CV | 5-fold buffered CV trunk with physical buffer B_m = max(r*_k, 16*px_size_m), strict in-fold fitting | M2 | R1-D8 |
| 9 | D9: Importance-Weighted Conformal | Likelihood ratio weights w_conf(x_i) = 1/clip(e(x_i), 0.05, 1.0) under exploration shift | M1 | R1-D9 |
| 10 | D10: Causal Propensity Covariates | Restrict to pre-mining regional infrastructure (roads, rivers, ports), exclude slope & haul roads | M2 | R1-D10 |
| 11 | Bug 1: Conformal Inference Unification | Unify conformal set gates and studentized lower bounds across pu_pipeline and chunked inference | M1, M3 | R2-Bug1 |
| 12 | Bug 2: Remove Pass 7A Negative Truncation | Evaluate full validation folds without [:min(600, len(u_va_f))] truncation in Pass 7A | M3 | R2-Bug2 |
| 13 | Bug 3: Purge Synthetic Coal Seeding | Remove coal_pos_all[c_idx::3] seeding; mask zero-positive heads from U-Net loss updates | M1, M3 | R2-Bug3 |
| 14 | Bug 4: Physical Gate Features & Subsets | Replace dead binary masks with continuous physical features; enforce SUBCANOPY_COAL_FEATURES | M1, M2 | R2-Bug4 |
| 15 | Bug 5: Coal Validation Gating & Warm-Start | Pass w_canopy=w_canopy_val and evaluate prior_ensemble in TwoStageK3CoalEnsemble.fit() | M1 | R2-Bug5 |
| 16 | Bug 6: Coal TreeSHAP & Gate Telemetry | Implement composite chain-rule TreeSHAP in TwoStageK3CoalEnsemble; wire outmogc & basin gates to site cards | M1, M3 | R2-Bug6 |
| 17 | Bug 7: Physical Layer Fallbacks | Reconstruct physical layers from dataframe by exact name when stack is None; 2nd-98th percentile clip | M3 | R2-Bug7 |
| 18 | Bug 8: U-Net Propensity & ASPP Bottleneck | Pass propensity weights into U-Net; 4-branch ASPP bottleneck; precision-weighted log-odds fusion | M1 | R2-Bug8 |
| 19 | Bug 9: Canonical Schema Module | Centralize NON_FEATURE_COLS & _FORBIDDEN_RAW_BAND_PATTERNS into src/schema/canonical.py; support dae_*_fold_*.pt | M1, M3 | R2-Bug9 |
| 20 | Bug 10: Dynamic Summary Telemetry | Export dynamic weights, accurate physical resolutions, live PR-AUCs in summary JSON | M3 | R2-Bug10 |
| 21 | R3: Repository Restructuring | Modular layout with facade module in src/models/pu_pipeline.py for 100% backward compatibility | M4 | R3 |
| 22 | R4: Verification, Corpus & Governance | 0 regression unit tests, new v4.0 tests, CODEBASE_CORPUS.md, AGENTS.md, audits/, CHANGELOG.md | M4, M5 | R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Mathematical Core & Model Discrepancies | D1, D2, D3, D4, D5 (softmax), D7, D9, Bug 1 (models), Bug 3 (U-Net mask), Bug 5, Bug 6 (TreeSHAP), Bug 8, Bug 9 (schema) | none | DONE |
| M2 | Spatial Geometry, Propensity, Exclusion & Spatial Splits | D6, D8, D10, Bug 4 (physical gate features & sub-canopy features) | M1 | DONE |
| M3 | 7-Pass Training Orchestration, Inference, Telemetry & Fallbacks | Bug 1 (chunked inference), Bug 2 (Pass 7A full negs), Bug 3 (remove seeding loop), Bug 4 (Pass 4B gate timing), Bug 6 (site card gates), Bug 7 (fallbacks), Bug 9 (import schema & checkpoint search), Bug 10 (dynamic telemetry) | M1, M2 | DONE |
| M4 | Repository Restructuring, Verification Suite & Governance | R3 restructuring & facade modules, new v4.0 unit tests, full test suite verification, CODEBASE_CORPUS.md, AGENTS.md, audits/, CHANGELOG.md | M1, M2, M3 | DONE |
| M5 | Final Gate Verification & Forensic Audit | Independent reviews (Reviewers 1 & 2), adversarial tests (Challengers 1 & 2), forensic audit (Auditor 1) | M1, M2, M3, M4 | DONE |

## Interface Contracts
### `src/schema/canonical.py` ↔ `src/models/` & `scripts/`
- `NON_FEATURE_COLS`: Tuple[str, ...] containing 24 canonical leakage/quarantined columns.
- `_FORBIDDEN_RAW_BAND_PATTERNS`: Tuple[Pattern, ...] regex patterns for raw sensor bands and coordinate distances.
- `K4_HEAD_NAMES`: Tuple[str, ...] = ("porphyry_cu", "skarn_fe_cu", "epithermal_au", "laterite_fe_ni").
- `K3_COAL_HEAD_NAMES`: Tuple[str, ...] = ("coal_k0_high_cv", "coal_k1_med_cv", "coal_k2_low_cv").
- `DISCOVERY_PRIORS_K4`: Dict[str, float] = {"porphyry_cu": 0.05, "skarn_fe_cu": 0.03, "epithermal_au": 0.08, "laterite_fe_ni": 0.02}.
- `COAL_PRIORS_K3`: Dict[str, float] = {"coal_k0_high_cv": 0.02, "coal_k1_med_cv": 0.04, "coal_k2_low_cv": 0.05}.

### `src/models/conformal.py` ↔ `scripts/run_seven_pass_training.py`
- `compute_multi_mineral_conformal_bounds(probs, epistemic, y_k, ...)`:
  - Returns `Tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, float]]`:
    - `p_set_gates`: Boolean mask array where $p_{\text{cal}} - 0.5\sigma \ge 1 - \hat{q}_{0.90, w}$.
    - `p_low90`: Conformal studentized lower bounds clipped to $[0, 1]$: $\text{clip}(p_{\text{cal}} - \hat{q}_{\text{res}, 90, w} \cdot \sigma, 0, 1)$.
    - `q_set_dict`: Dict of propensity-weighted set quantiles per head.
    - `q_res_dict`: Dict of studentized residual quantiles per head.

### `src/models/coal.py` ↔ `scripts/run_seven_pass_training.py`
- `TwoStageK3CoalEnsemble`:
  - `fit(X, y_k3, w_tones, w_canopy, w_canopy_val, X_val, y_val, ...)`
  - `predict_proba(X, w_canopy)`: returns optical-fused calibrated probabilities with prior correction.
  - `compute_head_shap(head_idx, X_sample, w_canopy)`: returns composite chain-rule TreeSHAP matrix.
- `compute_hierarchical_coal_probabilities(p_cal_coal_any, rank_logits, basin_gate, temperature=1.0)`:
  - Evaluates $P_{\text{coal\_total}}(s) = G_{\text{basin}}(s) \cdot P_{\text{cal}}(\text{coal\_any} \mid x_s)$ and applies temperature-scaled softmax across rank logits.

### `src/evaluation/spatial_cv.py` ↔ `scripts/run_seven_pass_training.py`
- `spatial_buffered_block_kfold(gdf_or_coords, n_splits=5, block_size_m=5000.0, buffer_m=1000.0, ...)`:
  - Yields `(train_idx, val_idx)` ensuring no training pixel is within `buffer_m` of validation block boundaries.

## Code Layout & File Boundaries
- `src/schema/canonical.py`: Owned by M1 (centralized schemas, constants, commodity priors).
- `src/models/pu_pipeline.py`: Owned by M1 (facade module re-exporting submodules).
- `src/models/metals.py`, `coal.py`, `calibration.py`, `conformal.py`, `fusion.py`: Owned by M1.
- `src/models/unet_model.py`: Owned by M1 (Hájek weights, vector priors, ASPP bottleneck, bias init).
- `src/labels/spatial_splits.py`, `borneo_labels.py`, `pu_negatives.py`: Owned by M2.
- `src/evaluation/spatial_cv.py`: Owned by M2 (buffered spatial CV).
- `scripts/run_seven_pass_training.py`: Owned by M3 (pipeline orchestration, Passes 1-7, inference).
- `scripts/train_unlabeled_25km_tile.py`: Owned by M3 (physical fallbacks, dynamic telemetry).
- `src/outputs/prospectivity_map.py`: Owned by M3 (site cards, physical gates).
- `src/outputs/model_introspection.py`: Owned by M3 (schema import, checkpoint patterns).
- `CHANGELOG.md`, `CODEBASE_CORPUS.md`, `AGENTS.md`, `audits/`: Owned by M4.
- `tests/unit/test_v4_math_and_bugs.py`: Owned by M4 (new verification test suite).
