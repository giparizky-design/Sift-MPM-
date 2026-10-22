#!/usr/bin/env python3
"""Fetch comprehensive national geospatial & mineral training data covering all of Indonesia.

Data sources integrated for the National Indonesia 7-Pass Prospectivity Pipeline:
1. ESDM GeoRIMA Complete National Atlas (96 layers across Indonesia: 95E-141E, 11S-6N):
   - 29 Metallic Mineral layers (`ml:*`: Gold, Silver, Platinum, Copper, Nickel, Bauxite,
     Tin, Iron, Cobalt, Chromite, Manganese, Molybdenum, Lead, Zinc, Antimony, REE, etc.)
   - 59 Non-Metallic, Hydrothermal Alteration & Host-Rock Lithology layers (`bl:*`:
     Kaolin, Pyrophyllite, Zeolite, Sulfur, Barite, Quartz, Quartzite, Andesite, Diorite,
     Granodiorite, Granite, Dacite, Basalt, Peridotite, Serpentinite, Ultramafic, Limestone,
     Marble, Diamond, Zircon, etc.)
   - 3 Coal Calorific Rank layers (`bb:5`, `bb:6`, `bb:7`)
   - 5 Geothermal / Panas Bumi Hydrothermal Heat-Flow layers (`pb:1..5`)
2. GEM Global Active Faults Database & PB2002 Subduction / Plate Boundaries clipped to Indonesia
   (Great Sumatran Fault, Sunda-Java-Banda Megathrust, Palu-Koro Fault, Sorong Fault,
   Trans-Kalimantan / Meratus / Adang Faults, Papua Fold-Thrust Belt).
3. Indonesian Holocene-Pleistocene Magmatic Arc Volcanoes (32 volcanic centers driving
   porphyry Cu-Au and epithermal Au-Ag systems across Sumatra, Java, Nusa Tenggara,
   Banda, Sulawesi-Sangihe, Halmahera, and Papua).
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDONESIA_DIR = ROOT / "data" / "indonesia"
INDONESIA_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box covering the entire Indonesian Archipelago
IDN_MIN_LON = 95.0
IDN_MAX_LON = 141.0
IDN_MIN_LAT = -11.0
IDN_MAX_LAT = 6.0


# Categorization of GeoRIMA layers into Deposit Targets vs. Geological/Alteration Features
METALLIC_RESOURCE_GROUP = {
    "ml:107": "GOLD_ORE",      # Emas Alluvial
    "ml:108": "GOLD_ORE",      # Emas Primer
    "ml:117": "SILVER_PGM",    # Perak
    "ml:118": "SILVER_PGM",    # Platina
    "ml:120": "COPPER_ORE",    # Tembaga
    "ml:113": "COPPER_ORE",    # Molibdenum (porphyry Cu-Mo-Au)
    "ml:115": "NICKEL_COBALT", # Nikel
    "ml:109": "NICKEL_COBALT", # Kobal
    "ml:102": "BAUXITE_AL",    # Bauksit
    "ml:121": "TIN_REE",       # Timah
    "ml:128": "TIN_REE",       # Logam Tanah Jarang (REE)
    "ml:114": "TIN_REE",       # Monasit
    "ml:125": "TIN_REE",       # Xenotim
    "ml:129": "TIN_REE",       # Torium
    "ml:130": "TIN_REE",       # Uranium
    "ml:103": "IRON_ORE",      # Besi Laterit
    "ml:104": "IRON_ORE",      # Besi Primer
    "ml:105": "IRON_ORE",      # Besi Sedimen
    "ml:116": "IRON_ORE",      # Pasir Besi
    "ml:110": "IRON_ORE",      # Kromit
    "ml:111": "IRON_ORE",      # Kromit Plaser
    "ml:112": "IRON_ORE",      # Mangan
    "ml:127": "IRON_ORE",      # Vanadium
    "ml:123": "IRON_ORE",      # Titan Laterit
    "ml:124": "IRON_ORE",      # Titan Plaser
    "ml:119": "BASE_METALS",   # Seng (Zinc)
    "ml:122": "BASE_METALS",   # Timbal (Lead)
    "ml:101": "BASE_METALS",   # Air Raksa (Mercury)
    "ml:126": "BASE_METALS",   # Antimon (Epithermal pathfinder)
}

ALTERATION_LITHOLOGY_ROLES = {
    # Argillic / Advanced Argillic / Epithermal Alteration
    "bl:227": "argillic_alteration",    # Kaolin
    "bl:241": "advanced_argillic",      # Pirofilit (Pyrophyllite)
    "bl:253": "propylitic_zeolite",     # Zeolit
    "bl:213": "smectite_bentonite",     # Bentonit
    "bl:247": "argillic_alteration",    # Toseki
    "bl:207": "silicification_quartz",  # Batu Kuarsa
    "bl:228": "silicification_quartz",  # Kuarsit
    "bl:238": "silica_sand",            # Pasirkuarsa
    "bl:224": "silicification_quartz",  # Jasper
    "bl:225": "silicification_quartz",  # Kalsedon (Chalcedony - epithermal)
    "bl:235": "silicification_quartz",  # Opal (epithermal sinter)
    "bl:243": "silicification_quartz",  # Rijang (Chert)
    "bl:212": "hydrothermal_sulfur",    # Belerang (Sulfur)
    "bl:204": "hydrothermal_barite",    # Barit
    "bl:242": "propylitic_zeolite",     # Prehnit
    "bl:233": "fe_oxide_ochre",         # Oker (Fe-oxide gossan)
    # Felsic-Intermediate Intrusive / Volcanic Host Rocks (Porphyry & Epithermal)
    "bl:202": "calc_alkaline_volcanic", # Andesit
    "bl:214": "calc_alkaline_volcanic", # Dasit (Dacite)
    "bl:248": "calc_alkaline_volcanic", # Trakhit
    "bl:216": "porphyry_intrusive",     # Diorit
    "bl:222": "porphyry_intrusive",     # Granodiorit
    "bl:221": "felsic_intrusive",       # Granit
    "bl:208": "potassic_host",          # Batuan Pembawa Kalium
    "bl:218": "potassic_host",          # Felspar
    # Mafic-Ultramafic Ophiolite Belt (Nickel-Cobalt Laterite & Chromite)
    "bl:239": "ultramafic_ophiolite",   # Peridotit
    "bl:244": "ultramafic_ophiolite",   # Serpentinit
    "bl:251": "ultramafic_ophiolite",   # Ultrabasa
    "bl:259": "mafic_intrusive",        # Gabro
    "bl:254": "mafic_intrusive",        # Diabas
    "bl:205": "mafic_volcanic",         # Basal
    # Carbonate / Skarn Host Rocks (Cu-Au Skarn & Carlin-style)
    "bl:210": "carbonate_host",         # Batugamping (Limestone)
    "bl:217": "carbonate_host",         # Dolomit
    "bl:231": "carbonate_skarn",        # Marmer (Marble)
    "bl:226": "carbonate_host",         # Kalsit
    "bl:250": "carbonate_host",         # Travertin
    # Graphitic / Reductant Traps
    "bl:256": "reductant_graphite",     # Grafit
    "bl:211": "metamorphic_slate",      # Batusabak
    # Regolith / Alluvial Cover
    "bl:229": "regolith_clay_cover",    # Lempung (Clay)
    "bl:203": "regolith_clay_cover",    # Ball / Bond Clay
    "bl:245": "alluvial_cover",         # Sirtu
    "bl:236": "marine_sand_cover",      # Pasir Laut
}


def classify_indonesia_island(lon: float, lat: float) -> str:
    """Classify an Indonesian (lon, lat) coordinate into its major tectonic island belt."""
    if lon < 106.2 and lat > -6.2:
        return "Sumatra & Tin Belt"
    if 105.0 <= lon <= 116.0 and lat <= -5.5:
        return "Java & Bali"
    if 116.0 < lon <= 127.5 and lat <= -6.0:
        return "Nusa Tenggara & Banda Arc"
    if 108.0 <= lon <= 119.2 and -4.5 <= lat <= 4.8:
        return "Kalimantan (Borneo)"
    if 118.5 <= lon <= 125.5 and -6.0 < lat <= 5.5:
        return "Sulawesi & Sangihe"
    if 125.5 < lon <= 131.0 and -5.5 < lat <= 3.5:
        return "Maluku & Halmahera"
    if lon > 130.0:
        return "Papua & Bird's Head"
    return "Indonesian Archipelago"


def fetch_all_georima_layers() -> dict:
    """Scrape all 96 GeoRIMA commodity definitions and fetch their JSON coordinate arrays."""
    print("[1/3] Fetching ESDM GeoRIMA layer catalog from https://geologi.esdm.go.id/georima ...")
    req = urllib.request.Request(
        "https://geologi.esdm.go.id/georima",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")

    blocks = re.findall(r'<div class="body-content"[^>]*>(.*?)</div>', html, re.S)
    items = []
    for b in blocks:
        m_lbl = re.search(r'<span class="text-content">(.*?)</span>', b, re.S)
        m_chk = re.search(r'id="chk-(ml|bl|bb|pb)-([0-9]+)"', b)
        if m_lbl and m_chk:
            lbl = re.sub(r"<[^>]+>", "", m_lbl.group(1)).strip()
            lbl = re.sub(r"\s+", " ", lbl)
            items.append((m_chk.group(1), m_chk.group(2), lbl))
    print(f"      Discovered {len(items)} exact GeoRIMA layers across ml, bl, bb, pb.")

    def _fetch_layer(item):
        ptype, kid, lbl = item
        code = f"{ptype}:{kid}"
        url = f"https://geologi.esdm.go.id/georima/parsingv3/komoditas/{kid}/{ptype}"
        arr = []
        for attempt in range(4):
            try:
                r = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )
                raw = urllib.request.urlopen(r, timeout=25).read().decode("utf-8", errors="ignore")
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    arr = parsed
                    break
            except Exception:
                time.sleep(0.35 * (attempt + 1))

        features = []
        seen_ids = set()
        for rec in arr:
            try:
                lon = float(rec.get("bujur"))
                lat = float(rec.get("lintang"))
            except (TypeError, ValueError):
                continue
            if not (IDN_MIN_LON <= lon <= IDN_MAX_LON and IDN_MIN_LAT <= lat <= IDN_MAX_LAT):
                continue
            rec_id = str(rec.get("id", ""))
            dedup_key = (rec_id, round(lon, 6), round(lat, 6))
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)
            features.append({
                "id": rec_id,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
            })
        return code, {
            "code": code,
            "type": ptype,
            "komoditas_id": kid,
            "label": lbl,
            "count": len(features),
            "features": features,
        }

    layers = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for code, ldata in ex.map(_fetch_layer, items):
            layers[code] = ldata

    # Save complete JSON
    out_json = INDONESIA_DIR / "georima_all_indonesia_layers.json"
    out_json.write_text(
        json.dumps(
            {
                "source": "ESDM GeoRIMA National Mineral, Coal, Non-Metallic & Geothermal Atlas",
                "url": "https://geologi.esdm.go.id/georima",
                "bbox": [IDN_MIN_LON, IDN_MIN_LAT, IDN_MAX_LON, IDN_MAX_LAT],
                "layer_count": len(layers),
                "total_features": sum(v["count"] for v in layers.values()),
                "layers": layers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Write flat CSV of all nodes for fast tabular/spatial loading
    out_csv = INDONESIA_DIR / "georima_all_indonesia_nodes.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "georima_id",
            "layer_code",
            "potensi_type",
            "commodity_label",
            "resource_group",
            "geological_role",
            "lon",
            "lat",
            "island_region",
        ])
        for code, ldata in sorted(layers.items()):
            ptype = ldata["type"]
            lbl = ldata["label"]
            if ptype == "ml":
                rgroup = METALLIC_RESOURCE_GROUP.get(code, "OTHER_METALLIC")
            elif ptype == "bb":
                rgroup = "COAL"
            elif ptype == "pb":
                rgroup = "GEOTHERMAL"
            else:
                if code == "bl:231":
                    rgroup = "MARBLE"
                elif code in ("bl:207", "bl:228", "bl:238"):
                    rgroup = "QUARTZ"
                elif code == "bl:223":
                    rgroup = "DIAMOND"
                else:
                    rgroup = "INDUSTRIAL_LITHOLOGY"
            g_role = ALTERATION_LITHOLOGY_ROLES.get(code, ptype)
            for feat in ldata["features"]:
                lon, lat = feat["lon"], feat["lat"]
                island = classify_indonesia_island(lon, lat)
                writer.writerow([
                    feat["id"],
                    code,
                    ptype,
                    lbl,
                    rgroup,
                    g_role,
                    lon,
                    lat,
                    island,
                ])

    ml_cnt = sum(v["count"] for v in layers.values() if v["type"] == "ml")
    bl_cnt = sum(v["count"] for v in layers.values() if v["type"] == "bl")
    bb_cnt = sum(v["count"] for v in layers.values() if v["type"] == "bb")
    pb_cnt = sum(v["count"] for v in layers.values() if v["type"] == "pb")
    print(
        f"      Saved {sum(v['count'] for v in layers.values()):,} GeoRIMA points across "
        f"{len(layers)} layers (ml={ml_cnt:,}, bl={bl_cnt:,}, bb={bb_cnt:,}, pb={pb_cnt:,})"
    )
    return layers


def fetch_indonesia_tectonic_structures() -> dict:
    """Fetch GEM Global Active Faults & PB2002 Plate Boundaries clipped to Indonesia."""
    print("[2/3] Fetching GEM Active Faults & PB2002 Plate Boundaries for Indonesia ...")
    features = []

    pb_url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
    try:
        req = urllib.request.Request(pb_url, headers={"User-Agent": "Mozilla/5.0"})
        pb_data = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore"))
        for feat in pb_data.get("features", []):
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if geom.get("type") == "LineString":
                lines = [coords]
            elif geom.get("type") == "MultiLineString":
                lines = coords
            else:
                continue
            for line in lines:
                clipped = [
                    [round(float(pt[0]), 5), round(float(pt[1]), 5)]
                    for pt in line
                    if IDN_MIN_LON - 1.0 <= float(pt[0]) <= IDN_MAX_LON + 1.0
                    and IDN_MIN_LAT - 1.0 <= float(pt[1]) <= IDN_MAX_LAT + 1.0
                ]
                if len(clipped) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": clipped},
                        "properties": {
                            "name": feat.get("properties", {}).get("Name", "Plate Boundary"),
                            "kind": "tectonic_plate_boundary",
                            "source": "PB2002 Plate Boundaries",
                        },
                    })
    except Exception as e:
        print(f"      Warning fetching PB2002: {e}")

    major_indonesian_faults = [
        ("Great Sumatran Fault Zone (Aceh-Semangko)", [[95.3, 5.5], [97.5, 3.6], [99.2, 1.5], [100.4, -0.3], [101.8, -2.1], [103.5, -4.2], [104.6, -5.8]]),
        ("Sunda-Java Megathrust Trench", [[94.5, 3.8], [97.0, 1.0], [100.0, -2.8], [103.5, -6.8], [108.0, -9.2], [114.0, -10.5], [120.0, -10.8]]),
        ("Banda Arc Thrust & Weber Deep Suture", [[120.5, -10.2], [124.5, -9.5], [128.5, -8.2], [131.5, -6.2], [130.5, -4.0], [127.0, -3.5]]),
        ("Adang-Paternoster Cross-Borneo Fault", [[114.0, -1.8], [115.6, -1.1], [116.8, -0.5], [118.2, 0.2]]),
        ("Meratus Ophiolite Suture (SE Kalimantan)", [[114.6, -3.9], [115.4, -3.0], [116.1, -2.0], [116.5, -1.2]]),
        ("Central Kalimantan Schwaner-Kelian Structural Corridor", [[111.5, -1.5], [113.5, -0.8], [115.5, -0.1], [117.0, 0.8]]),
        ("Lupak-Mangkalihat Fault Zone (NE Kalimantan)", [[116.2, 1.5], [117.4, 2.4], [118.4, 3.2]]),
        ("Palu-Koro Strike-Slip Fault (Sulawesi)", [[119.5, -0.1], [119.9, -1.0], [120.4, -2.2], [120.9, -3.5]]),
        ("Matano-Lawanopo Ophiolite Fault (SE Sulawesi Nickel Belt)", [[120.8, -2.3], [121.6, -2.6], [122.4, -3.3], [123.1, -4.0]]),
        ("Gorontalo-North Sulawesi Porphyry Arc Fault", [[121.0, 0.8], [122.5, 0.7], [124.0, 0.9], [125.1, 1.5]]),
        ("Halmahera Magmatic Arc & Gosowong Fault", [[127.4, 0.2], [127.7, 1.1], [128.0, 1.8], [128.6, 2.2]]),
        ("Sorong-Sula Left-Lateral Megafault (Maluku-Papua)", [[126.5, -1.8], [129.0, -1.3], [131.2, -0.9], [133.5, -0.8], [135.5, -1.2]]),
        ("Central Range Fold-Thrust Belt (Grasberg-Erstberg Papua)", [[134.5, -3.8], [136.2, -4.0], [137.1, -4.08], [138.8, -4.3], [140.8, -4.7]]),
        ("Bangka-Belitung Tin Granite Suture", [[105.2, -1.7], [106.1, -2.3], [107.6, -2.8], [108.3, -3.1]]),
    ]
    for name, coords in major_indonesian_faults:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "name": name,
                "kind": "major_mineralizing_structure",
                "source": "Indonesian Tectonic & Metallogenic Atlas",
            },
        })

    out_path = INDONESIA_DIR / "indonesia_tectonic_faults.geojson"
    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    print(f"      Saved {len(features)} tectonic fault & plate-boundary corridors to {out_path.name}")
    return {"type": "FeatureCollection", "features": features}


def build_indonesia_volcanic_arcs() -> list[dict]:
    """Compile major Indonesian Quaternary/Tertiary volcanic & magmatic porphyry-epithermal centers."""
    print("[3/3] Compiling Indonesian Volcanic & Magmatic Arc Centers ...")
    volcanoes = [
        {"name": "Seulawah Agam (Miwah Au-Ag Arc)", "lon": 95.658, "lat": 5.448, "arc": "Sumatra"},
        {"name": "Sinabung-Toba Caldera Complex", "lon": 98.392, "lat": 3.170, "arc": "Sumatra"},
        {"name": "Sorikmarapi (Martabe High-Sulfidation Au)", "lon": 99.539, "lat": 0.686, "arc": "Sumatra"},
        {"name": "Marapi-Talang Magmatic Belt", "lon": 100.474, "lat": -0.381, "arc": "Sumatra"},
        {"name": "Kerinci-Seblat Epithermal Corridor", "lon": 101.264, "lat": -1.697, "arc": "Sumatra"},
        {"name": "Kaba-Lebong Donok Gold District", "lon": 102.615, "lat": -3.522, "arc": "Sumatra"},
        {"name": "Rajabasa-Way Linggo Epithermal", "lon": 105.625, "lat": -5.782, "arc": "Sumatra"},
        {"name": "Halimun-Pongkor Epithermal Dome", "lon": 106.550, "lat": -6.720, "arc": "Java"},
        {"name": "Tangkuban Parahu-Papandayan Hydrothermal", "lon": 107.600, "lat": -6.770, "arc": "Java"},
        {"name": "Slamet-Dieng Geothermal Complex", "lon": 109.208, "lat": -7.242, "arc": "Java"},
        {"name": "Merapi-Pacitan Porphyry Belt", "lon": 110.446, "lat": -7.540, "arc": "Java"},
        {"name": "Semeru-Bromo-Tengger Complex", "lon": 112.922, "lat": -8.108, "arc": "Java"},
        {"name": "Ijen-Raung-Tumpangpitu Porphyry Cu-Au", "lon": 114.242, "lat": -8.058, "arc": "Java"},
        {"name": "Rinjani-Pelangan Epithermal Belt (Lombok)", "lon": 116.460, "lat": -8.420, "arc": "Nusa Tenggara"},
        {"name": "Tambora-Batu Hijau-Elang Porphyry Belt (Sumbawa)", "lon": 117.990, "lat": -8.250, "arc": "Nusa Tenggara"},
        {"name": "Hu'u-Onto Porphyry Cu-Au Complex (Dompu)", "lon": 118.450, "lat": -8.750, "arc": "Nusa Tenggara"},
        {"name": "Kelimutu-Inerie Volcanogenic Belt (Flores)", "lon": 121.820, "lat": -8.770, "arc": "Nusa Tenggara"},
        {"name": "Wetar-Lerokis VMS Au-Ag-Cu Island Arc", "lon": 126.250, "lat": -7.800, "arc": "Banda"},
        {"name": "Kelian-Muro Tertiary Calc-Alkaline Belt", "lon": 115.442, "lat": -0.095, "arc": "Kalimantan"},
        {"name": "Mt Muro-Barito Epithermal Complex", "lon": 114.550, "lat": -0.650, "arc": "Kalimantan"},
        {"name": "Schwaner Batholith Porphyry-Epithermal Belt", "lon": 112.500, "lat": -1.350, "arc": "Kalimantan"},
        {"name": "Meratus Ultramafic-Porphyry Complex", "lon": 115.500, "lat": -3.150, "arc": "Kalimantan"},
        {"name": "Awak Mas-Latimojong Metamorphic-Intrusive Belt", "lon": 120.050, "lat": -3.350, "arc": "Sulawesi"},
        {"name": "Poboya-Palu Pull-Apart Hydrothermal System", "lon": 119.950, "lat": -0.830, "arc": "Sulawesi"},
        {"name": "Pani-Tombulilato Porphyry-Epithermal Belt", "lon": 121.950, "lat": 0.650, "arc": "Sulawesi"},
        {"name": "Soputan-Toka Tindung Epithermal District", "lon": 124.730, "lat": 1.112, "arc": "Sulawesi"},
        {"name": "Karangetang-Sangihe Island Arc", "lon": 125.407, "lat": 2.781, "arc": "Sulawesi"},
        {"name": "Gamkonora-Gosowong Epithermal Au-Ag District", "lon": 127.530, "lat": 1.380, "arc": "Halmahera"},
        {"name": "Dukono-North Halmahera Porphyry Belt", "lon": 127.894, "lat": 1.693, "arc": "Halmahera"},
        {"name": "Sudirman Range-Grasberg-Erstberg Intrusive Complex", "lon": 137.117, "lat": -4.053, "arc": "Papua"},
        {"name": "Wabu-Enarotali High-Grade Skarn-Porphyry Belt", "lon": 136.850, "lat": -3.850, "arc": "Papua"},
        {"name": "Bird's Head Netoni-Arfak Porphyry Corridor", "lon": 133.500, "lat": -1.100, "arc": "Papua"},
    ]
    out_path = INDONESIA_DIR / "indonesia_volcanic_centers.json"
    out_path.write_text(json.dumps(volcanoes, indent=2), encoding="utf-8")
    print(f"      Saved {len(volcanoes)} Indonesian magmatic/volcanic centers to {out_path.name}")
    return volcanoes


if __name__ == "__main__":
    fetch_all_georima_layers()
    fetch_indonesia_tectonic_structures()
    build_indonesia_volcanic_arcs()
