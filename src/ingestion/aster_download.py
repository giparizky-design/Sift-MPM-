"""
src/ingestion/aster_download.py
================================
ASTER Level-2 surface-reflectance download helper for the MPM pipeline.

NOTE: ASTER SWIR detector failed April 2008.  Only pre-2008 scenes contain
valid SWIR (bands 4-9).  This function filters accordingly.

Background
----------
The Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER)
is a multispectral imager on the Terra satellite (launched 1999).  Its 6
Short-Wave Infrared (SWIR) bands (2.0–2.5 µm, 30 m resolution) are the
primary remote-sensing tool for mapping hydrothermal alteration minerals
(kaolinite, alunite, muscovite, chlorite, carbonate, epidote) in arid and
semi-arid terrains.

**Critical instrument note**: The ASTER SWIR detector experienced a
temperature excursion event in April 2008 that permanently degraded all
six SWIR channels.  Post-April 2008 SWIR data are dominated by detector
noise and are UNUSABLE for mineral mapping.  This module enforces a hard
cut-off of 2008-04-01 on all scene searches.

Product used
------------
``AST_07XT`` — ASTER Surface Reflectance VNIR and SWIR (on-demand, 30 m)
distributed through the NASA LP DAAC Earthdata system.
Requires a free NASA Earthdata account and a Bearer token.

Authentication
--------------
Generate a Bearer token at:
https://urs.earthdata.nasa.gov/profile → Generate Token

References
----------
- Rowan, L.C. & Mars, J.C. (2003). Lithologic mapping in the Mountain Pass,
  California area using Advanced Spaceborne Thermal Emission and Reflection
  Radiometer (ASTER) data. *Remote Sensing of Environment*, 84(3), 350-366.
- USGS ASTER SWIR Notice: https://lpdaac.usgs.gov/news/discontinuation-of-aster-swir/
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NASA CMR granule search endpoint
_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
# ASTER on-demand Surface Reflectance collection short name
_COLLECTION_SHORT_NAME = "AST_07XT"
# ASTER SWIR hard cut-off (detector failure)
_SWIR_CUTOFF_DATE = date(2008, 4, 1)
# Maximum concurrent download workers (LP DAAC rate-limit guidance)
_MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# Scene search
# ---------------------------------------------------------------------------

def search_aster_scenes(
    aoi_wgs84_bounds: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_cover_max: int = 10,
) -> list[dict[str, Any]]:
    """Query the NASA CMR API for ASTER AST_07XT granules over an AOI.

    Uses the NASA Common Metadata Repository (CMR) granule search REST API.
    Results are paged automatically (100 granules per request) until all
    matching granules have been retrieved.

    Args:
        aoi_wgs84_bounds: AOI bounding box in WGS84 geographic coordinates
                          as ``(west, south, east, north)`` in decimal degrees.
        date_start: Acquisition start date in ``"YYYY-MM-DD"`` format.
        date_end: Acquisition end date in ``"YYYY-MM-DD"`` format (inclusive).
        cloud_cover_max: Maximum acceptable cloud cover percentage (0–100).
                         Defaults to ``10``.

    Returns:
        List of granule metadata dicts as returned by CMR.  Each dict
        contains at minimum:

        - ``"producer_granule_id"``: ASTER scene ID (e.g. ``"AST_07XT_00301012000_20150402.hdf"``).
        - ``"time_start"``: ISO-8601 acquisition start time.
        - ``"links"``: List of download / browse link dicts, each with
          ``"href"`` and ``"rel"`` keys.
        - ``"cloud_cover"``: Cloud cover percentage (float or ``None``).

    Raises:
        requests.HTTPError: For non-401 HTTP errors.
        PermissionError: If the CMR returns 401 Unauthorized (missing / expired
                         Earthdata credentials — note that CMR search itself is
                         unauthenticated but LP DAAC download requires a token).
        ValueError: If *aoi_wgs84_bounds* or date strings are malformed.

    Example:
        >>> granules = search_aster_scenes(
        ...     aoi_wgs84_bounds=(114.0, -30.0, 122.0, -22.0),
        ...     date_start="2000-01-01",
        ...     date_end="2008-03-31",
        ...     cloud_cover_max=10,
        ... )
        >>> print(len(granules), "scenes found")
    """
    west, south, east, north = aoi_wgs84_bounds
    if not (-180 <= west < east <= 180):
        raise ValueError(
            f"Invalid longitude bounds: west={west}, east={east}. "
            "Ensure west < east and both in [-180, 180]."
        )
    if not (-90 <= south < north <= 90):
        raise ValueError(
            f"Invalid latitude bounds: south={south}, north={north}. "
            "Ensure south < north and both in [-90, 90]."
        )

    # CMR expects bounding box as "west,south,east,north"
    bounding_box = f"{west},{south},{east},{north}"
    temporal = f"{date_start}T00:00:00Z,{date_end}T23:59:59Z"

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_granules: list[dict[str, Any]] = []
    page = 1
    page_size = 100

    logger.info(
        "Searching CMR: collection=%s  bbox=%s  temporal=%s  cloud_cover≤%d%%",
        _COLLECTION_SHORT_NAME,
        bounding_box,
        temporal,
        cloud_cover_max,
    )

    while True:
        params: dict[str, Any] = {
            "short_name": _COLLECTION_SHORT_NAME,
            "bounding_box": bounding_box,
            "temporal": temporal,
            "cloud_cover[]": f"0,{cloud_cover_max}",
            "page_size": page_size,
            "page_num": page,
        }

        try:
            response = session.get(_CMR_URL, params=params, timeout=60)
        except requests.ConnectionError as exc:
            raise requests.ConnectionError(
                f"Failed to reach NASA CMR ({_CMR_URL}). Check network connectivity."
            ) from exc

        if response.status_code == 401:
            raise PermissionError(
                "NASA CMR returned HTTP 401 Unauthorized. "
                "This usually means your Earthdata session has expired. "
                "Generate a new Bearer token at: "
                "https://urs.earthdata.nasa.gov/profile → Generate Token"
            )

        response.raise_for_status()

        data = response.json()
        granules: list[dict] = data.get("feed", {}).get("entry", [])

        if not granules:
            break

        all_granules.extend(granules)
        logger.debug("Page %d: retrieved %d granules (total so far: %d)", page, len(granules), len(all_granules))

        # CMR signals last page when fewer results than page_size returned
        if len(granules) < page_size:
            break
        page += 1

    logger.info("CMR search complete: %d granules found.", len(all_granules))
    return all_granules


# ---------------------------------------------------------------------------
# Individual granule download
# ---------------------------------------------------------------------------

def download_granule(
    granule_url: str,
    dest_dir: str | os.PathLike,
    earthdata_token: str,
) -> Path:
    """Download a single ASTER granule from the LP DAAC with streaming.

    Streams the file in 8 MB chunks and shows a tqdm progress bar.
    If the destination file already exists with a non-zero size it is skipped
    (idempotent — safe to re-run after partial downloads as long as the
    incomplete file has been removed).

    Args:
        granule_url: Direct HTTPS download URL from the granule's ``links``
                     list (``rel = "http://esipfed.org/ns/fedsearch/1.1/data#"``
                     or similar).
        dest_dir: Local directory in which to save the file.
        earthdata_token: NASA Earthdata Bearer token string (without the
                         ``"Bearer "`` prefix — this function adds it).

    Returns:
        ``pathlib.Path`` to the downloaded file.

    Raises:
        requests.HTTPError: For non-2xx responses.
        PermissionError: If the server returns 401 (token expired or invalid).

    Example:
        >>> path = download_granule(
        ...     "https://e4ftl01.cr.usgs.gov/.../AST_07XT_00301012000.hdf",
        ...     dest_dir="data/raw/aster",
        ...     earthdata_token="eyJ0...",
        ... )
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = granule_url.split("/")[-1].split("?")[0]
    dest_file = dest_dir / filename

    if dest_file.exists() and dest_file.stat().st_size > 0:
        logger.debug("Skip (already exists): %s", dest_file)
        return dest_file

    headers = {"Authorization": f"Bearer {earthdata_token}"}
    chunk_size = 8 * 1024 * 1024  # 8 MB

    with requests.get(granule_url, headers=headers, stream=True, timeout=300) as resp:
        if resp.status_code == 401:
            raise PermissionError(
                "LP DAAC returned HTTP 401 Unauthorized while downloading "
                f"'{filename}'. Your Earthdata Bearer token may have expired. "
                "Generate a new one at: https://urs.earthdata.nasa.gov/profile → Generate Token"
            )
        resp.raise_for_status()

        total_bytes = int(resp.headers.get("Content-Length", 0))

        with (
            open(dest_file, "wb") as fp,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=False,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fp.write(chunk)
                pbar.update(len(chunk))

    logger.info("Downloaded: %s  (%.1f MB)", dest_file, dest_file.stat().st_size / 1e6)
    return dest_file


# ---------------------------------------------------------------------------
# SWIR validity filter
# ---------------------------------------------------------------------------

def filter_swir_valid(granule_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only granules acquired before the ASTER SWIR detector failure.

    The ASTER SWIR detector (bands 4–9, 2.0–2.5 µm) failed permanently in
    April 2008.  Any data acquired on or after 2008-04-01 are dominated by
    detector noise and are unsuitable for mineral alteration mapping.

    Args:
        granule_list: List of granule metadata dicts from
                      :func:`search_aster_scenes`.  Each dict must have a
                      ``"time_start"`` key in ISO-8601 format.

    Returns:
        Filtered list containing only granules with ``time_start`` before
        ``2008-04-01``.

    Example:
        >>> valid = filter_swir_valid(granules)
        >>> print(len(valid), "SWIR-valid scenes")
    """
    valid: list[dict[str, Any]] = []
    skipped = 0

    for granule in granule_list:
        time_start_str: str = granule.get("time_start", "")
        try:
            # ISO-8601: "2007-11-15T01:22:00.000Z" → date(2007, 11, 15)
            acq_date = date.fromisoformat(time_start_str[:10])
        except (ValueError, TypeError):
            logger.warning(
                "Cannot parse time_start='%s' for granule '%s'; excluding.",
                time_start_str,
                granule.get("producer_granule_id", "unknown"),
            )
            skipped += 1
            continue

        if acq_date < _SWIR_CUTOFF_DATE:
            valid.append(granule)
        else:
            skipped += 1

    logger.info(
        "filter_swir_valid: %d valid (pre-2008-04-01), %d excluded.",
        len(valid),
        skipped,
    )
    return valid


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def batch_download(
    aoi_bounds: tuple[float, float, float, float],
    dest_dir: str | os.PathLike,
    earthdata_token: str,
    date_start: str = "2000-01-01",
    date_end: str = "2008-03-31",
    cloud_cover_max: int = 10,
    max_workers: int = _MAX_WORKERS,
) -> list[Path]:
    """Orchestrate search → SWIR filter → download for an AOI.

    Searches the CMR catalogue, removes post-SWIR-failure scenes, then
    downloads all remaining granules using a thread pool (max 4 workers to
    avoid overwhelming LP DAAC rate limits).

    Args:
        aoi_bounds: AOI bounding box in WGS84 as ``(west, south, east, north)``
                    decimal degrees.
        dest_dir: Local destination directory for downloaded files.
        earthdata_token: NASA Earthdata Bearer token.
        date_start: Search start date (``"YYYY-MM-DD"``).  Defaults to
                    ``"2000-01-01"`` (ASTER launch + commissioning).
        date_end: Search end date (``"YYYY-MM-DD"``).  Defaults to
                  ``"2008-03-31"`` (day before SWIR failure).  Changing this
                  beyond ``"2008-03-31"`` is allowed but the SWIR filter will
                  silently drop the additional scenes.
        cloud_cover_max: Maximum cloud cover percentage.  Defaults to ``10``.
        max_workers: Maximum concurrent download threads.  LP DAAC recommends
                     no more than 4 concurrent connections.  Defaults to ``4``.

    Returns:
        List of ``pathlib.Path`` objects pointing to successfully downloaded
        files (including pre-existing files that were skipped).

    Raises:
        PermissionError: If the Earthdata token is invalid.
        requests.HTTPError: For other HTTP errors during search or download.

    Example:
        >>> paths = batch_download(
        ...     aoi_bounds=(114.0, -30.0, 122.0, -22.0),
        ...     dest_dir="data/raw/aster",
        ...     earthdata_token=os.environ["EARTHDATA_TOKEN"],
        ... )
        >>> print(f"Downloaded {len(paths)} scenes")
    """
    max_workers = min(max(1, max_workers), _MAX_WORKERS)

    # Step 1: Search CMR
    granules = search_aster_scenes(
        aoi_wgs84_bounds=aoi_bounds,
        date_start=date_start,
        date_end=date_end,
        cloud_cover_max=cloud_cover_max,
    )

    if not granules:
        logger.warning("No ASTER scenes found for the specified AOI and date range.")
        return []

    # Step 2: Filter to SWIR-valid scenes
    valid_granules = filter_swir_valid(granules)

    if not valid_granules:
        logger.warning(
            "No SWIR-valid scenes remain after filtering. "
            "All scenes found were acquired on or after 2008-04-01."
        )
        return []

    # Step 3: Extract download URLs from granule metadata
    # LP DAAC data links use rel ending in "/data#"
    download_tasks: list[tuple[str, str]] = []  # (url, granule_id)
    for granule in valid_granules:
        links: list[dict] = granule.get("links", [])
        granule_id: str = granule.get("producer_granule_id", "unknown")
        data_urls = [
            lnk["href"]
            for lnk in links
            if lnk.get("href", "").startswith("https://")
            and "data#" in lnk.get("rel", "")
        ]
        if not data_urls:
            # Fallback: try any https link containing the granule ID
            data_urls = [
                lnk["href"]
                for lnk in links
                if lnk.get("href", "").startswith("https://")
                and granule_id.split(".")[0] in lnk.get("href", "")
            ]
        if data_urls:
            download_tasks.append((data_urls[0], granule_id))
        else:
            logger.warning("No download URL found for granule: %s", granule_id)

    logger.info(
        "Starting batch download: %d scenes  workers=%d  dest=%s",
        len(download_tasks),
        max_workers,
        dest_dir,
    )

    downloaded_paths: list[Path] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_granule, url, dest_dir, earthdata_token): (url, gid)
            for url, gid in download_tasks
        }
        for future in as_completed(future_to_url):
            url, gid = future_to_url[future]
            try:
                path = future.result()
                downloaded_paths.append(path)
            except Exception as exc:
                logger.error("Failed to download %s: %s", gid, exc)
                failed.append(gid)

    if failed:
        logger.warning(
            "%d granule(s) failed to download: %s",
            len(failed),
            ", ".join(failed),
        )

    logger.info(
        "batch_download complete: %d succeeded, %d failed.",
        len(downloaded_paths),
        len(failed),
    )
    return downloaded_paths
