# Mineral Prospectivity Data Feasibility and Audit Report

- **Primary Pilot:** Eastern Goldfields, Yilgarn Craton, Western Australia (`yilgarn_orogenic_gold_v1`)
- **Secondary Evaluation Region:** Borneo (Central Kalimantan Arc & East Malaysia, `borneo_epithermal_porphyry_v1`)
- **Target Commodities:** Gold (Au), Copper (Cu), Antimony (Sb)
- **Target Prediction Grid:** 50 m × 50 m cells
- **Target Coordinate Reference Systems:**
  - Yilgarn Pilot: `EPSG:7851` (GDA2020 / MGA Zone 51)
  - Borneo Pilot: `EPSG:32649` (WGS 84 / UTM Zone 49N)
- **Audit Date:** 2026-10-21

---

## 1. Primary Pilot: Authentic Western Australian Datasets Acquired

All foundational datasets for the Eastern Goldfields pilot AOI (`[350000, 6590000, 360000, 6600000]` EPSG:7851) have been retrieved from official government open-access APIs and public cloud buckets:

| Layer Name | Source & API Protocol | Format & CRS | Records / Shape | Native Res | Validation Verdict |
|---|---|---|---|---|---|
| **MINEDEX Deposits** | GSWA / DMIRS SLIP REST (Layer 0) | GeoPackage (`EPSG:7851`) | 480 gold mines/deposits | 10–50 m | **Verified Positive Labels (P).** Filtered out infrastructure (mills/plants). Includes Sir John, Great Boulder, Federal, Fortunatus. |
| **WAMEX Drillholes** | GSWA / DMIRS SLIP REST (Layer 28) | GeoPackage (`EPSG:7851`) | 5,000 drillhole collars | Precise collar coords | **Verified Drillhole Network.** Target commodities: Au, Au-Ni-Zn, Base Metals. Spatial anchors for Reliable Negatives (RN). |
| **Bedrock Geology** | GSWA 1:500k Bedrock Geology (Layer 4) | GeoPackage (`EPSG:7851`) | 12 polygons | 1:500 000 | **Verified.** Eastern Goldfields Superterrane greenstones, Kurrawang Formation, komatiites, felsic volcanics. |
| **Linear Faults/Shears** | GSWA Linear Structures (Layer 1) | GeoPackage (`EPSG:7851`) | 19 fault line features | 1:500 000 | **Verified Structural Trap Layer.** 340.9 km cumulative fault trace length for Euclidean distance calculation. |
| **Copernicus DEM** | ESA / AWS S3 Open Data COG | GeoTIFF (`EPSG:7851`) | 200 × 200 cells | 30 m | **Verified Elevation Grid.** Reprojected to 50 m grid. Elevation spans 114.7 m to 463.1 m (mean 366.0 m). |

---

## 2. Secondary Region Audit: Borneo (Kalimantan, Sarawak, Sabah)

### 2.1 Environmental & Sensor Feasibility
* **The Equatorial Canopy Mask:** Unlike arid Western Australia where ASTER/Sentinel-2 SWIR band ratios (Al-OH 2.20 µm, Fe-oxide 0.86 µm, silica 8.6 µm) map surface alteration directly, Borneo's dense multi-tier rainforest canopy completely obscures bedrock spectral reflectance.
* **Sensor Hierarchy Solution:**
  1. **Primary Structural Sensor:** JAXA ALOS-2 PALSAR-2 L-band SAR (~24 cm wavelength). Penetrates leaf canopy to ground level; delivers fault/shear roughness and bedrock lineaments.
  2. **Topographic Geomorphology:** Copernicus GLO-30 DEM (30 m). Extracts slope, aspect, TPI, ridge curvature, and caldera ring boundaries.
  3. **Deep Geophysics:** NOAA EMAG2v3 global magnetics (2 arc-minutes ~ 3.7 km). Flags regional volcano-plutonic arc sutures.

### 2.2 Ground Truth Audit & Centroid Clustering Detection
* Programmatic audit of the USGS Mineral Resources Data System (MRDS) across Borneo (`[-4.5, 108.5, 7.5, 119.5]`) retrieved **50 records**.
* **Data Quality Caveat Uncovered:** **22 out of 50 records (44%) are placed at the Kalimantan island centroid (`lat: -1.9982°, lon: 113.9995°`)**. Database compilers assigned this default centroid when colonial literature lacked precise coordinates.
* **Genuine Geolocated Ground Truth (28 deposits):**
  - **Kelian Gold Deposit** (`-0.02063°, 115.45167°`): Giant Neogene low-sulfidation carbonate-base metal epithermal Au deposit (~240 t Au).
  - **Mamut Porphyry Cu-Au** (`6.01770°, 116.64942°`): Sabah, Malaysia (~100 Mt @ 0.48% Cu, 0.5 g/t Au).
  - **Bau Gold Mine** (`2.59430°, 113.30325°`): Carlin-like sediment-hosted Au-Sb in Jurassic-Cretaceous limestone/shale.
  - **Cempaka-Danau Seran & Riam Kanan** (`-3.40752°, 114.89474°`): Diamond, Au, and PGE alluvial paleochannels.
  - **Busang Prospect** (`1.25147°, 116.23273°`): Hydrothermal alteration zone, East Kalimantan.
  - **Sebuku Island** (`-3.49807°, 116.35943°`): Ophiolitic lateritic Ni-Fe-Cr.
  - **West Kalimantan / Sarawak Bauxite** (`0.95150°, 108.94960°` & `1.75142°, 109.83291°`): Lateritic bauxite on weathered felsic intrusions.

---

## 3. Storage & Artifacts Checklist

All datasets have been acquired and staged locally:
- Authentic WA Pilot Data: `data/raw_real/`
- Scraped Borneo Exploration Data: `data/borneo/`
- Secondary Region Configuration: `config/borneo.yaml`
- Ingestion Orchestration Script: `scripts/fetch_real_pilot_data.py`
- Borneo Scraper Script: `scripts/fetch_borneo_data.py`
- Comprehensive Audit Document: `docs/borneo_data_audit.md`
