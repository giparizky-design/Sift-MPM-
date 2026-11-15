# Mineral Prospectivity Mapping 


## System Overview

This repository implements an evidence-based **Mineral Prospectivity Mapping (MPM)** system for **Orogenic Gold** mineralisation. It operates on a 50 × 50 m prediction grid using a **Positive-Unlabeled (PU) learning** framework.


## Critical Caveats

> **This system outputs a dimensionless prospectivity score (0–1) representing the similarity of a 50 m cell's geophysical-spectral signature to known orogenic gold deposits. It does NOT predict:**
> - Whether ore definitely exists
> - Grade (g/t Au), tonnage (t), or depth (m)
> - The presence of every hidden deposit (recall cannot be measured on truly unexplored ground)

---

## Architecture

```
src/
├── ingestion/          # Data downloaders (ASTER/LP DAAC, GA open portal)
├── harmonization/      # Reprojection → 50 m MGA51 grid, cover masks
├── features/           # ASTER indices, Sentinel-2, geophysics, structural, DEM
├── labels/             # Deposit labels (MINEDEX), PU negative mining, spatial CV splits
├── models/             # XGBoost baseline + U-Net challenger + retraining pipeline
├── evaluation/         # PR-AUC, spatial block CV, blind holdout, model comparison
└── outputs/            # COG GeoTIFF exports, SHAP maps, target polygon GeoPackage
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure your Earthdata token for ASTER downloads
export EARTHDATA_TOKEN="<your-token>"  # https://urs.earthdata.nasa.gov/

# 3. Download GA data manually (see src/ingestion/ga_download.py for URLs)
#    Place aeromagnetics TMI/RTP in: data/raw/magnetics/
#    Place radiometrics in:          data/raw/radiometrics/
#    Place gravity in:               data/raw/gravity/
#    Download WAMEX drillholes from: https://wamex.dmirs.wa.gov.au

# 4. Run harmonization stage
python scripts/run_pilot.py --config config/pilot.yaml --stage harmonize

# 5. Run full pipeline
python scripts/run_pilot.py --config config/pilot.yaml --stage all --version-tag v1
```

---

## Data Sources

| Dataset | Provider | Resolution | Licence | Notes |
|---|---|---|---|---|
| ASTER L2 (AST_07XT) | NASA LP DAAC | 15–90 m | Open (US Gov) | SWIR only pre-April 2008 |
| Sentinel-2 L2A | ESA Copernicus | 10–60 m | CC BY 3.0 IGO | Dry-season composite |
| Aeromagnetics (TMI/RTP) | Geoscience Australia | 80 m grid | CC BY 4.0 | Eastern Goldfields merged |
| Radiometrics (K/U/Th) | Geoscience Australia | 100 m grid | CC BY 4.0 | ~45 cm penetration depth |
| Gravity (Bouguer) | Geoscience Australia | 400 m grid | CC BY 4.0 | Upsampled to 50 m with flag |
| AW3D30 DEM | JAXA | 30 m | Free (registration) | |
| GSWA Geology 1:500k | GSWA | 1:500 000 | CC BY 4.0 | Generalised boundaries |
| MINEDEX deposits | GSWA/DMIRS | Point (variable) | CC BY 4.0 | Precision ±10–500 m |
| WAMEX drillholes | DMIRS WA | Point (collar) | CC BY-NC 4.0 | Post-1990: ≤100 m precision |

---

## Modeling Approach

### Label Framing (PU Learning)

| Class | Symbol | Criteria |
|---|---|---|
| Positive | **P** | MINEDEX record ≥100 koz contained Au, drill-verified, coordinate confirmed on geological map |
| Reliable Negative | **RN** | ≥3 RC/diamond holes intersecting greenstone basement ≥50 m depth, best assay <0.05 g/t Au, no prospect within 2 km |
| Unlabeled | **U** | Everything else — treated as unknown, not barren |

### Evaluation (Leakage-Resistant)

- **Spatial Block CV:** 50 × 50 km blocks; mining camps never split across folds
- **Blind Holdout:** Laverton Belt withheld from all training/tuning
- **Primary Metric:** Precision-Recall AUC (PR-AUC) — not ROC-AUC (misleading under imbalance)

### Model Selection Rule

| Condition | Decision |
|---|---|
| U-Net spatial-CV PR-AUC > XGBoost + 0.05 AND p < 0.05 (Wilcoxon) | Deploy U-Net |
| Otherwise | Retain XGBoost baseline (simpler, interpretable) |

---

## Outputs

| File | Format | Description |
|---|---|---|
| `prospectivity_v{n}.tif` | Float32 COG GeoTIFF | 0–1 score per 50 m cell |
| `uncertainty_v{n}.tif` | Float32 COG GeoTIFF | Ensemble variance (epistemic) |
| `provenance_v{n}.tif` | Uint8 multi-band COG | Data quality flags per layer |
| `shap_top5_v{n}.tif` | Float32 multi-band COG | SHAP value for top 5 features |
| `targets_v{n}.gpkg` | GeoPackage | High-score polygons for field targeting |

---

## Field Validation & Retraining

```
Field campaign →  Assay results → Geologist review (licensed reviewer ID required)
                                         ↓
                              Immutable label lake (parquet)
                                         ↓
                           Bi-annual controlled retraining
                          (model v2 evaluated on v1's targets)
```


---


