# Data Directory Architecture & Provenance

This directory stores raw geospatial, geophysical, geological, and label datasets, along with interim harmonized rasters and processed feature cubes.

## Directory Layout

```
data/
├── raw_real/               # Authentic, live datasets acquired via open public APIs
│   ├── minedex_deposits.gpkg # 480 verified MINEDEX gold mines/deposits (DMIRS SLIP Layer 0)
│   ├── wamex_drillholes.gpkg # 5,000 open-file exploration drillholes (DMIRS SLIP Layer 28)
│   ├── geology.gpkg        # 12 GSWA 1:500k interpreted bedrock geology polygons (Layer 4)
│   ├── faults.gpkg         # 19 GSWA linear faults/shear structures (Layer 1, 340 km traces)
│   ├── dem/dem.tif         # 30 m Copernicus GLO-30 DEM reprojected to 50 m grid in EPSG:7851
│   ├── geology/            # Geology & faults packaged for pipeline
│   └── labels/             # Formatted MINEDEX & WAMEX subdirectories
│
├── borneo/                 # Scraped open geospatial data for Borneo transfer pilot
│   ├── borneo_mrds_occurrences.gpkg # USGS MRDS mineral occurrences (lat -4.5 to 7.5, lon 108.5 to 119.5)
│   ├── borneo_mrds_occurrences.csv  # 50 records categorized into 28 genuine vs 22 centroid artifacts
│   ├── kalimantan_iron_deposits.gpkg # 79 official Iron (Fe) deposits from ESDM (skarn, laterite, sand)
│   ├── kalimantan_iron_deposits.csv  # Full tabular attributes: resource MT, status, deposit type
│   ├── kalimantan_coal_deposits.gpkg # 850 official Coal deposits/mines from ESDM (71.1 Gt resources)
│   └── kalimantan_coal_deposits.csv  # Full tabular attributes: coal rank, reserves MT, operator status
│
├── raw/                    # Staged raw benchmark/pilot datasets
├── interim/                # 50 m harmonized and reprojected raster layers
├── processed/              # Feature stacks (joblib), labels (npy), and ML datasets (parquet)
└── external/               # Regional auxiliary data
```

## Data Provenance & Open Access Protocols

### 1. Western Australia (Eastern Goldfields Pilot)
* **MINEDEX Deposits (Layer 0):** `https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Industry_and_Mining/MapServer/0`
  * Filtered for `target_com LIKE '%GOLD%'` and verified mine/deposit stages.
* **WAMEX Drillholes (Layer 28):** `https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Industry_and_Mining/MapServer/28`
  * Real collars with hole IDs, depths, drill types (RC/Diamond/RAB), and operators.
* **GSWA 1:500 000 Geology & Faults:** `https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Geology_and_Soils_Map/MapServer`
  * Layer 4: Bedrock geology polygons.
  * Layer 1: Linear structures (faults and shears).
* **Copernicus DEM (GLO-30):** Direct AWS Open Data S3 COG streaming (`s3://copernicus-dem-30m`).

### 2. Borneo Region (Kalimantan, Sabah, Sarawak)
* **ESDM Badan Geologi Iron Registry (`kalimantan_iron_deposits.gpkg`):**
  * Queried official ESDM REST layer: `https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Sumber_Daya_dan_Cadangan_Mineral_Logam/MapServer/0`.
  * 79 official Iron (Fe) sites: 57 Besi Primer (skarns/veins in West/Central Kalimantan), 21 Besi Laterit (ophiolitic laterites in South Kalimantan / Sebuku), 1 Pasir Besi.
* **ESDM Badan Geologi Coal Registry (`kalimantan_coal_deposits.gpkg`):**
  * Queried official ESDM REST layer: `https://geoportal.esdm.go.id/gis4/rest/services/BGD_TU/Potensi_Batubara/MapServer/0`.
  * 850 official Coal deposits and active concessions across Kutai, Barito, and Tarakan basins (71.1 Billion Tonnes resources, 23.0 Billion Tonnes reserves, 846 operating mines).
* **USGS MRDS Mineral Occurrences:** Queried via WFS 1.1.0 (`https://mrdata.usgs.gov/services/wfs/mrds`).
  * Programmatic detection of island centroid clustering artifact at `(-1.9982°, 113.9995°)`.
  * Genuine ground truth: Kelian Au-Ag, Mamut Cu-Au, Bau Au-Sb, Busang, Sebuku Ni-Fe-Cr, Singkawang & Sematan Bauxite.
* **NOAA EMAG2v3 Magnetics:** Global 2-arc-minute (~3.7 km) magnetic anomaly grid (`https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_UpCont_DataTiff.tif`).
* **Canopy Strategy:** Optical SWIR alteration is masked by equatorial rainforest; SAR L-band (ALOS PALSAR) and DEM geomorphology prioritized.
