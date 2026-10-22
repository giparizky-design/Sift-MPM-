# Borneo Mineral Prospectivity: Geospatial Data Audit & Transfer-Learning Feasibility

**Document Version:** 1.0  
**Scope:** Island of Borneo — Kalimantan (Indonesia), Sarawak & Sabah (Malaysia), Brunei Darussalam.  
**Target Systems:** Epithermal Au-Ag (Central Kalimantan), Porphyry Cu-Au (Sabah), Sediment-Hosted Au-Sb (Bau, Sarawak), and Lateritic Bauxite (West Kalimantan / Sarawak).  
**Primary Predictor Grid:** 50 × 50 m (scientific evaluation of native resolutions below).

---

## 1. Executive Summary & The Tropical Remote Sensing Dilemma

Deploying machine learning mineral prospectivity mapping (MPM) to the equatorial tropical environment of Borneo presents fundamentally different physics and data constraints compared to arid, bedrock-exposed terrains like the Western Australian Yilgarn Craton.

In the Yilgarn Craton, thin soil cover and sparse semi-arid vegetation allow spaceborne Shortwave Infrared (SWIR) sensors (ASTER, Sentinel-2) to map hydroxyl-bearing clay alterations (kaolinite, illite, alunite) directly from surface reflectance. In Borneo, **dense primary rainforest canopy, thick peat/humus topsoils, and pervasive equatorial cloud cover render surface optical SWIR reflection virtually blind to bedrock hydrothermal alteration**. 

Therefore, any scientifically sound prospectivity architecture for Borneo must pivot from surface spectral mineralogy to **deep-penetrating radar geomorphology (L-band SAR), high-resolution digital terrain modeling, and subsurface geophysics**.

```
                   YILGARN CRATON (Arid)          BORNEO (Equatorial Tropical)
Surface Cover:     Sparse scrub / thin regolith   Dense multi-tier rainforest canopy & peat
Optical Alteration:EXCELLENT (ASTER SWIR 15-30m)  BLINDED (masked by canopy chlorophyll)
Primary Sensation: SWIR + Radiometrics + Mag     L-band SAR + DEM Geomorphology + Deep Mag
Target Geology:    Archean Orogenic Lode Gold     Neogene Epithermal Au-Ag & Porphyry Cu-Au
```

---

## 2. Geological & Mineral Systems Framework

Borneo occupies a critical tectonic position at the junction of the Sundaland continental core, the active Pacific-Eurasian subduction complex, and the South China Sea marginal basins.

### 2.1 Major Mineral Systems in Borneo

1. **Neogene Epithermal Gold-Silver Systems (Central Kalimantan Arc):**
   * *Examples:* **Kelian** (giant low-sulfidation carbonate-base metal epithermal Au deposit, ~240 t Au, associated with Miocene andesite-dacite intrusives), **Mt. Muro** (epithermal quartz-vein Au-Ag), **Masupa Ria**, **Mirah**, **Busang**.
   * *Critical Mineral System Components:*
     * *Source:* Subduction-related calc-alkaline magmatism (Miocene Sintang Intrusive Suite).
     * *Transport / Architecture:* Deep crustal trans-tensional fault corridors, regional strike-slip transfer faults, and caldera ring fractures.
     * *Trap / Deposition:* Permeable pyroclastic horizons, intrusive breccia pipes, and boiling zones driven by hydrostatic decompression.

2. **Porphyry Copper-Gold Systems (Sabah, East Malaysia):**
   * *Examples:* **Mamut** (Mt. Kinabalu batholith flank, Sabah; porphyry Cu-Au with adamellite-granodiorite porphyry stock, mined ~100 Mt @ 0.48% Cu, 0.5 g/t Au).
   * *Critical Components:* High-level polyphase porphyry intrusions emplaced into ophiolitic and turbiditic country rocks.

3. **Sediment-Hosted / Intrusion-Related Gold-Antimony (Bau District, Southwest Sarawak):**
   * *Examples:* **Bau Goldfield** (Carlin-style sediment-hosted Au and stibnite veins hosted in Late Jurassic to Early Cretaceous Bau Limestone and Pedawan Formation shale, associated with Miocene dacite dykes).
   * *Critical Components:* Calcareous-shale structural traps, fault intersections along the Tai Parit fault.

4. **Lateritic Bauxite Belts (West Kalimantan & Southwest Sarawak):**
   * *Examples:* **Tayan**, **Mempawah**, **Kendawangan** (West Kalimantan) and **Sematan / Lundu** (Sarawak).
   * *Critical Components:* Deep chemical weathering of Cretaceous aluminous granitoids, basic-intermediate volcanics, and syenites under tropical humid paleoclimates.

---

## 3. Data Audit: Open & Accessible Geospatial Layers for Borneo

### 3.1 Digital Elevation Models & Structural Geomorphology

Because vegetation masks surface rock spectra, topographic derivatives are essential to extract regional fault trends, caldera rings, volcanic necks, and intrusive domes.

| Layer | Provider | Native Resolution | Coverage / Access | Capabilities & Limitations in Borneo |
|---|---|---|---|---|
| **Copernicus DEM (GLO-30)** | ESA / AWS Open Data | 30 m (1 arc-sec) | 100% Borneo; Open COG via `s3://copernicus-dem-30m` | Digital Surface Model (DSM). Captures canopy top in rainforest, but excellent at resolving regional structural lineaments, drainage diversions, and volcanic crater rims. |
| **ALOS AW3D30** | JAXA / OpenTopography | 30 m | 100% Borneo; Free registration | High vertical accuracy. Slightly more sensitive to cloud/snow voids than Copernicus DEM. |
| **NASADEM / SRTM v3** | NASA LP DAAC | 30 m | 100% Borneo (-4.5° to 7.5° N) | C-band radar interferometry (2000). Penetrated partial canopy, but has void fills in rugged terrain (Kinabalu range). |

### 3.2 Synthetic Aperture Radar (SAR) — Canopy Penetration

SAR is the primary earth observation tool capable of imaging structural fabrics through tropical weather and vegetation.

| Sensor / Product | Wavelength / Band | Penetration Depth | Spatial Resolution | Prospectivity Utility in Borneo |
|---|---|---|---|---|
| **ALOS-2 PALSAR-2 (JAXA)** | L-band (~24 cm) | Penetrates forest canopy to ground surface / branches | 12.5 m – 25 m mosaic | **Optimal sensor for Borneo.** L-band microwaves pass through leaf canopy, reflecting off ground topography, rock surfaces, and soil moisture boundaries. Exposes faults, fractures, and lithological contacts hidden under jungle. |
| **Sentinel-1 GRD (Copernicus)** | C-band (~5.6 cm) | Canopy top to outer branches | 10 m (VV/VH dual-pol) | All-weather, cloud-free revisit (12 days). Maps surface roughness, soil moisture saturation, and floodplains. Useful for regolith/alluvial mapping. |

### 3.3 Satellite Spectral Reflection (Optical & Thermal)

| Sensor | Spectral Bands | Resolution | Borneo Feasibility Verdict |
|---|---|---|---|
| **ASTER (TIR)** | Bands 10–14 (8.2–11.3 µm) | 90 m | **Moderate utility.** Thermal infrared can detect regional silica enrichment (quartz stockworks) and large-scale thermal inertia differences, even through partial vegetation. Pre-2008 scenes have high cloud contamination. |
| **ASTER (SWIR)** | Bands 4–9 (1.6–2.4 µm) | 30 m | **Severely limited.** SWIR absorption features (2.2 µm Al-OH, 2.35 µm Mg-OH/carbonate) are overwhelmed by dense green canopy reflectance. Valid only over open-pit mines, active quarries, or deforestation scars. |
| **Sentinel-2 L2A** | B2–B12 (VNIR/SWIR) | 10–20 m | **Limited to cleared ground.** Cloud-free dry season compositing (August–September) enables NDVI masking to restrict spectral alteration analysis strictly to bare soil / exposed bedrock pixels. |

### 3.4 Geophysics: Magnetics & Gravity

| Dataset | Provider | Native Resolution | Coverage | Notes & Limitations |
|---|---|---|---|---|
| **EMAG2v3 (Earth Magnetic Anomaly Grid v3)** | NOAA NCEI / CGMW | 2 arc-minutes (~3.7 km) | 100% Borneo (marine and onshore) | Grid resolution is 3.7 km. While suitable for crustal-scale suture mapping (e.g. Lupar Line, Boyan Suture), it cannot resolve 50 m camp-scale intrusive stocks without regional airborne surveys. |
| **WGM2012 / BGI Gravity** | Bureau Gravimétrique International | 2 arc-minutes (~3.7 km) | Complete | Free-air and Bouguer anomaly; maps major sedimentary basins (Kutei Basin, Barito Basin) and crustal basement highs. |
| **Airborne Geophysics** | Indonesia Badan Geologi / JMG Malaysia | 200 m – 1 km line spacing | Patchy / Proprietary | High-resolution aeromagnetic/radiometric surveys exist for selected mining belts, but are not in open global public domain. Requires institutional partnership or data licensing with ESDM / JMG. |

---

## 4. Ground Truth & Deposit Labels Audit

A programmatic audit of the USGS Mineral Resources Data System (MRDS) across the Borneo bounding box (`[-4.5, 108.5, 7.5, 119.5]`) retrieved **50 recorded mineral deposits and occurrences**.

### Critical Audit Finding: Centroid Clustering Artifacts

Our spatial parsing uncovered that **22 out of the 50 records (44%) are placed at the geographic centroid of Kalimantan (`lat: -1.99820, lon: 113.99949`)**.
* In historical compilations, when the precise mine coordinates were unspecified in colonial Dutch/British literature, compilers assigned the generic island centroid.
* In some instances, historical tin mines located on Bangka/Belitung islands were misattributed to the Borneo centroid.
* **Strict Rule:** Any automated MPM ingestion must filter out coordinates clustering within 5 km of `(-1.9982, 113.9995)` to prevent massive false-positive label contamination at the island centroid.

### Verified Genuine Ground-Truth Deposits in Borneo

Filtering out centroid artifacts leaves **28 genuine, high-precision deposit locations**:

1. **Kelian Gold Deposit** (`-0.02063°, 115.45167°`): Epithermal Au-Ag, East Kalimantan.
2. **Mamut Porphyry Cu-Au** (`6.01770°, 116.64942°`): Porphyry Cu-Au, Mt. Kinabalu, Sabah.
3. **Bau Gold Mine** (`2.59430°, 113.30325°`): Carlin-like sediment-hosted Au-Sb, Sarawak.
4. **Cempaka-Danau Seran Paleochannels** (`-3.49807°, 114.74946°`): Diamond & alluvial Au, South Kalimantan.
5. **Riam Kanan River** (`-3.40752°, 114.89474°`): Diamond, Au, PGE alluvial system, South Kalimantan.
6. **Busang Prospect** (`1.25147°, 116.23273°`): Hydrothermal alteration zone / prospect, East Kalimantan.
7. **Sebuku Island** (`-3.49807°, 116.35943°`): Lateritic Ni-Fe-Cr deposit on obducted ophiolite.
8. **Singkawang & Sematan Bauxite** (`0.95150°, 108.94960°` & `1.75142°, 109.83291°`): Lateritic bauxite on weathered felsic intrusions.

---

## 5. Recommended Architecture for Borneo Pilot / Transfer Learning

To validate transfer learning from Western Australia to Borneo:
1. **Pilot AOI Selection:** Scope down to the **Central Kalimantan Epithermal Gold Belt (Kelian–Mt. Muro Corridor)** or the **Bau Gold District (Sarawak)**.
2. **Feature Stack Adaptations:**
   * Drop optical SWIR indices as primary features; replace with vegetation-conditioned masks.
   * Add **ALOS PALSAR-2 L-band backscatter (HH, HV, HH/HV ratio)** for structural radar roughness and canopy-penetrated fault delineation.
   * Add **Copernicus DEM morphometric derivatives** (topographic position index, multiscale lineament density, ridge/valley curvature).
   * Up-sample regional **EMAG2v3 magnetics** using explicit uncertainty tracking (coarse 3.7 km native resolution flagged in provenance mask).
3. **Validation Strategy:** Spatial block holdout isolating the Kelian deposit block from surrounding prospects to test out-of-camp generalization.

---

## 6. Kalimantan Iron & Coal Systems: Datasets & Spatial Audit

In addition to gold, porphyry, and bauxite, Kalimantan is globally significant for coal and holds substantial primary and lateritic iron resources. The official registry from the Indonesian Ministry of Energy and Mineral Resources (ESDM / Badan Geologi) was acquired via the `https://geoportal.esdm.go.id/gis4` REST service.

### 6.1 Iron (Fe) Deposit Systems in Kalimantan (79 Sites)

* **Source File:** [`data/borneo/kalimantan_iron_deposits.gpkg`](file:///Users/gip/Geospatials/data/borneo/kalimantan_iron_deposits.gpkg) & [`.csv`](file:///Users/gip/Geospatials/data/borneo/kalimantan_iron_deposits.csv)
* **Commodity Breakdown:**
  * **Besi Primer (57 sites):** Contact metasomatic skarn and hydrothermal magnetite-hematite deposits associated with Cretaceous granitic plutons intruding Paleozoic-Mesozoic carbonates and volcanic-sedimentary strata in West Kalimantan (Sambas, Sanggau, Ketapang, Bengkayang) and Central Kalimantan.
  * **Besi Laterit (21 sites):** Supergene lateritic iron / nickel-iron caps formed by tropical humid weathering of obducted Cretaceous Meratus ultramafic ophiolites in South Kalimantan (Sebuku Island, Pulau Laut, Tanah Bumbu, Kotabaru).
  * **Pasir Besi (1 site):** Coastal titaniferous magnetite placer sand along southern beaches.
* **Exploration Status:**
  * 35 Operasi Produksi (operating mines / concessions)
  * 26 Eksplorasi / Eksplorasi Rinci (detailed exploration)
  * 17 Prospeksi / Survei Tinjau (prospecting and reconnaissance)
  * 1 Studi Kelayakan (feasibility study)

### 6.2 Coal (Batubara) Basins & Concessions in Kalimantan (850 Sites)

* **Source File:** [`data/borneo/kalimantan_coal_deposits.gpkg`](file:///Users/gip/Geospatials/data/borneo/kalimantan_coal_deposits.gpkg) & [`.csv`](file:///Users/gip/Geospatials/data/borneo/kalimantan_coal_deposits.csv)
* **Total Resource Base Recorded:**
  * **Total Coal Resources:** 71,076.5 Million Tonnes (71.1 Billion Tonnes)
  * **Total Coal Reserves:** 23,042.7 Million Tonnes (23.0 Billion Tonnes)
  * **Operating Sites:** 846 of 850 sites are active operating concessions (`Operasi Produksi`).
* **Calorific / Rank Classification:**
  * **Kalori Rendah (342 sites):** Low CV / Lignite to sub-bituminous (e.g. Warukin Formation in Barito Basin, Wara pit).
  * **Kalori Tinggi (319 sites):** High CV / Bituminous to semi-anthracite (e.g. Tanjung Formation Eocene seams and tectonic-compressed zones in Kutai Basin).
  * **Kalori Sedang (189 sites):** Medium CV / Sub-bituminous (e.g. Balikpapan Formation, Tutupan pit).
* **Major Basins & Operations:**
  * **Kutai Basin (East Kalimantan):** Kaltim Prima Coal (KPC Sangatta), Berau Coal (Lati, Binungan), Bayan Resources (Tabang), Mahakam River corridor (Samarinda/Tenggarong).
  * **Barito Basin (South & Central Kalimantan):** PT Adaro Indonesia (giant Tutupan, Wara, Paringin pits in Tabalong/Balangan), Arutmin (Satui, Senakin), Kideco Jaya Agung (Pasir Basin).
  * **Tarakan Basin (North Kalimantan):** Plio-Pleistocene Sajau and Tarakan formations.
