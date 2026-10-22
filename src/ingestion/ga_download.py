"""
src/ingestion/ga_download.py
==============================
Geoscience Australia (GA) open-data download and loading helpers for the
MPM predictor-cube ingestion pipeline.

Product catalogue
-----------------
All products listed below are publicly available under the
**Creative Commons Attribution 4.0 International Licence (CC BY 4.0)**
unless otherwise noted.

Web portals
-----------
- **ELVIS (Elevation Information System)**: https://elevation.fsdf.org.au
  National LiDAR and DEM tiles.
- **eCat (GA's electronic catalogue)**: https://ecat.ga.gov.au
  Aeromagnetics, gravity, radiometrics, and geology grids.
- **NCI THREDDS (NCI/GA shared)**: https://thredds.nci.org.au

Aeromagnetics products
----------------------
- **TMI (Total Magnetic Intensity)**: Gridded at 80 m resolution.
  eCat ID 71725.  Suitable for lineament mapping.
- **RTP (Reduction to Pole)**: Pole-reduced version of the TMI.
  eCat ID 71726.  RTP is preferred for structural interpretation because
  it removes the asymmetric anomaly shape caused by magnetic inclination;
  use TMI only if RTP is unavailable for the target area.

Gravity products
----------------
- **Bouguer Anomaly**: eCat ID 73697.
- **Free-Air Anomaly**: eCat ID 73696.

Radiometrics
------------
- **K (potassium)**: eCat ID 71375.
- **Th (thorium)**: eCat ID 71375.
- **U (uranium)**: eCat ID 71375.
- **Dose Rate (total count)**: eCat ID 71375.

References
----------
- Milligan, P.R. & Connard, G.G. (1997). *Guide to Using Magnetic Anomaly
  Data*. AGSO Record 1997/17.
- Blakely, R.J. (1995). *Potential Theory in Gravity and Magnetic
  Applications*. Cambridge University Press.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.utils.io import load_raster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known GA product registry
# ---------------------------------------------------------------------------

def list_ga_products() -> dict[str, dict[str, Any]]:
    """Return a registry of known Geoscience Australia grid products.

    Each entry documents the download URL (or portal), native resolution,
    file format, and licence so that the pipeline is self-documenting and
    auditable.

    Returns:
        Dict mapping product name (``str``) to a metadata dict with keys:

        - ``"url"`` (``str``): Primary download URL or portal page.
        - ``"ecat_id"`` (``str | None``): GA eCat record ID if applicable.
        - ``"format"`` (``str``): Native file format.
        - ``"resolution_m"`` (``int``): Native pixel size in metres.
        - ``"licence"`` (``str``): Data licence.
        - ``"notes"`` (``str``): Geoscientific usage notes.

    Example:
        >>> products = list_ga_products()
        >>> products["tmi"]["resolution_m"]
        80
    """
    return {
        "tmi": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71725",
            "ecat_id": "71725",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 80,
            "licence": "CC BY 4.0",
            "notes": (
                "Total Magnetic Intensity (TMI) — national airborne survey mosaic. "
                "Use for lineament and structural mapping. Anomaly shape is inclination-dependent; "
                "prefer RTP where available."
            ),
        },
        "rtp": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71726",
            "ecat_id": "71726",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 80,
            "licence": "CC BY 4.0",
            "notes": (
                "Reduction-to-Pole (RTP) aeromagnetics. "
                "RTP is preferred for structural interpretation because it removes the "
                "asymmetric anomaly shape caused by the Earth's magnetic inclination at "
                "non-polar latitudes, placing anomalies directly over their causative bodies."
            ),
        },
        "bouguer_gravity": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/73697",
            "ecat_id": "73697",
            "format": "GeoTIFF",
            "resolution_m": 400,
            "licence": "CC BY 4.0",
            "notes": (
                "Complete Bouguer Anomaly (CBA). Terrain-corrected; suitable for "
                "mapping crustal-scale density variations and identifying buried "
                "intrusions / sedimentary basins."
            ),
        },
        "free_air_gravity": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/73696",
            "ecat_id": "73696",
            "format": "GeoTIFF",
            "resolution_m": 400,
            "licence": "CC BY 4.0",
            "notes": "Free-Air Anomaly. Not terrain-corrected; use CBA in preference.",
        },
        "radiometrics_k": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71375",
            "ecat_id": "71375",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 100,
            "licence": "CC BY 4.0",
            "notes": (
                "Potassium (K%) radiometric grid. High K values indicate felsic intrusions, "
                "K-alteration haloes, or evaporite sequences — all relevant to base-metal and "
                "gold prospectivity."
            ),
        },
        "radiometrics_th": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71375",
            "ecat_id": "71375",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 100,
            "licence": "CC BY 4.0",
            "notes": (
                "Thorium (Th ppm) radiometric grid. Th behaves immobility during weathering; "
                "the Th channel most faithfully reflects bedrock composition through regolith."
            ),
        },
        "radiometrics_u": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71375",
            "ecat_id": "71375",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 100,
            "licence": "CC BY 4.0",
            "notes": (
                "Uranium (U ppm) radiometric grid. Mobile in oxidising conditions; "
                "elevated U can indicate redox fronts relevant to uranium and IOCG deposits."
            ),
        },
        "radiometrics_dose": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/71375",
            "ecat_id": "71375",
            "format": "GeoTIFF / ERMapper ERS",
            "resolution_m": 100,
            "licence": "CC BY 4.0",
            "notes": "Total dose rate (nGy/h). Combines K, Th, U signal; useful as a lithological proxy.",
        },
        "dem_1sec": {
            "url": "https://elevation.fsdf.org.au",
            "ecat_id": None,
            "format": "GeoTIFF (SRTM-derived)",
            "resolution_m": 30,
            "licence": "CC BY 4.0",
            "notes": (
                "1-second (≈30 m) SRTM-derived DEM from the ELVIS portal. "
                "Use for terrain analysis: slope, aspect, topographic wetness index, "
                "and flow accumulation for the water/playa mask."
            ),
        },
        "surface_geology": {
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/74619",
            "ecat_id": "74619",
            "format": "GeoPackage / Shapefile",
            "resolution_m": 0,   # vector product
            "licence": "CC BY 4.0",
            "notes": (
                "1:1 000 000 Surface Geology of Australia. "
                "Rasterise the 'ROCKTYPE1' attribute for use as a categorical predictor."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Tile download (intentional NotImplementedError stub)
# ---------------------------------------------------------------------------

def download_ga_grid(
    product_name: str,
    aoi_bounds_geo: tuple[float, float, float, float],
    dest_dir: str | os.PathLike,
) -> None:
    """Download GA grid tiles for a given product and AOI.

    .. important::
        **This function is intentionally not implemented.**

        Geoscience Australia's **ELVIS** elevation portal
        (https://elevation.fsdf.org.au) and the **eCat** catalogue
        (https://ecat.ga.gov.au) do **not** expose a stable, documented
        REST API suitable for automated tile queries or bounding-box
        sub-setting.  As of the time of writing (2024), both portals
        require manual interaction (click-to-select tiles on a web map,
        then email or browser download).

        Attempting to scrape or reverse-engineer these portals violates
        GA's terms of use and would produce brittle, unmaintainable code.

        **Manual download instructions**:

        1. Navigate to https://ecat.ga.gov.au and search for the product
           by name (see :func:`list_ga_products` for eCat IDs).
        2. On the product page, click **"Add to cart"** or
           **"Download"** and follow the prompts.
        3. For ELVIS DEM tiles, go to https://elevation.fsdf.org.au,
           draw your AOI on the map, select the DEM product, and submit
           your email for download notification.
        4. Place downloaded files in ``dest_dir`` and update your
           pipeline config to point to them.

    Args:
        product_name: Key from :func:`list_ga_products`.
        aoi_bounds_geo: AOI bounding box ``(west, south, east, north)``
                        in the product's native CRS (usually WGS84 geographic).
        dest_dir: Intended local destination directory (ignored — files must
                  be downloaded manually).

    Raises:
        NotImplementedError: Always, with instructions for manual download.
    """
    products = list_ga_products()
    if product_name not in products:
        known = ", ".join(sorted(products.keys()))
        raise KeyError(
            f"Unknown GA product '{product_name}'. Known products: {known}"
        )

    product_info = products[product_name]
    manual_url = product_info["url"]

    raise NotImplementedError(
        f"Automated download of '{product_name}' is not supported.\n\n"
        "Geoscience Australia's ELVIS and eCat portals do not expose a public "
        "REST API for programmatic tile sub-setting.\n\n"
        f"Please download the data manually from:\n  {manual_url}\n\n"
        f"AOI (west, south, east, north): {aoi_bounds_geo}\n"
        f"Place the downloaded file(s) in: {dest_dir}\n\n"
        "After downloading, use load_ga_aeromagnetics() or src.utils.io.load_raster() "
        "to read the files into the pipeline."
    )


# ---------------------------------------------------------------------------
# Aeromagnetics loader
# ---------------------------------------------------------------------------

def load_ga_aeromagnetics(
    tmi_path: str | os.PathLike,
    rtp_path: str | os.PathLike | None,
) -> dict[str, tuple[Any, dict]]:
    """Load GA TMI and/or RTP aeromagnetic GeoTIFFs from local paths.

    .. note::
        **RTP is preferred for structural interpretation.**
        The Reduction-to-Pole (RTP) transformation removes the asymmetric
        anomaly shape introduced by the Earth's magnetic inclination at
        non-polar latitudes, positioning each anomaly directly above its
        causative body.  This greatly simplifies lineament tracing and
        depth estimation.

        Use TMI only if RTP is unavailable for the target area (e.g. in
        remote or poorly surveyed regions where only the raw TMI grid has
        been released by GA).

    Args:
        tmi_path: Path to the Total Magnetic Intensity (TMI) GeoTIFF.
                  Must exist.
        rtp_path: Path to the Reduction-to-Pole (RTP) GeoTIFF, or ``None``
                  if not available.  When ``None``, only TMI is loaded and
                  a warning is emitted.

    Returns:
        Dict with up to two entries:

        - ``"tmi"`` → ``(array, profile)`` where *array* is a
          ``(1, H, W)`` float32 ndarray and *profile* is the rasterio
          profile.
        - ``"rtp"`` → ``(array, profile)`` — only present if *rtp_path*
          was provided.

    Raises:
        FileNotFoundError: If *tmi_path* does not exist.

    Example:
        >>> mag_layers = load_ga_aeromagnetics(
        ...     tmi_path="data/ga/tmi_80m.tif",
        ...     rtp_path="data/ga/rtp_80m.tif",
        ... )
        >>> rtp_array, rtp_profile = mag_layers["rtp"]
        >>> print(rtp_array.shape)
        (1, 5000, 6000)
    """
    result: dict[str, tuple[Any, dict]] = {}

    # Load TMI (required)
    tmi_path = Path(tmi_path)
    if not tmi_path.exists():
        raise FileNotFoundError(
            f"TMI raster not found: {tmi_path}. "
            "Download from https://ecat.ga.gov.au (eCat ID 71725) and place at the specified path."
        )
    tmi_data, tmi_profile = load_raster(tmi_path)
    result["tmi"] = (tmi_data, tmi_profile)
    logger.info("Loaded TMI: %s  shape=%s", tmi_path.name, tmi_data.shape)

    # Load RTP (optional but strongly preferred)
    if rtp_path is None:
        logger.warning(
            "RTP path not provided. Using TMI only. "
            "Structural interpretation will be less accurate because TMI anomalies "
            "are offset from their causative bodies at Australian magnetic latitudes. "
            "Download RTP from https://ecat.ga.gov.au (eCat ID 71726) if available."
        )
    else:
        rtp_path = Path(rtp_path)
        if not rtp_path.exists():
            logger.warning(
                "RTP file not found at '%s'. Falling back to TMI only. "
                "Download RTP from https://ecat.ga.gov.au (eCat ID 71726).",
                rtp_path,
            )
        else:
            rtp_data, rtp_profile = load_raster(rtp_path)
            result["rtp"] = (rtp_data, rtp_profile)
            logger.info("Loaded RTP: %s  shape=%s", rtp_path.name, rtp_data.shape)

    return result
