"""Feature Stack Assembly for Mineral Prospectivity Mapping.

This module assembles all preprocessed spatial feature layers (ASTER alteration
indices, Sentinel-2 spectral features, geophysical grids, structural proximity
maps, and terrain attributes) into a single C×H×W array suitable for
machine-learning models.

Design decisions:
  - **C×H×W layout** (channels-first) is used throughout to be consistent
    with PyTorch conventions (for U-Net ingestion) while remaining usable
    with scikit-learn after flattening via :func:`flatten_to_dataframe`.
  - **Data quality masking**: a per-pixel boolean mask marks cells as
    unreliable (e.g. sensor gaps, vegetation-masked ASTER pixels, water
    bodies, out-of-extent areas). Masked cells are excluded from model
    training and their predictions are written as nodata in the output map.
  - **GeoTIFF I/O**: all features are read from pre-processed single-band
    GeoTIFF files. The CRS, transform, and dimensions are taken from the
    first feature file and validated against all subsequent files.

Typical workflow::

    config = {
        "features": {
            "aloh_content": "processed/aster_aloh.tif",
            "ndvi":         "processed/s2_ndvi.tif",
            ...
        }
    }
    stack = build_feature_stack(config, processed_dir="/data/mpm/")
    df    = flatten_to_dataframe(stack, label_array=deposit_raster)
    # … train model …
    pred  = model.predict(df.drop(columns=["label","row","col"]))
    mpm   = reconstruct_map(pred, df["row"].values, df["col"].values,
                            H=stack.array.shape[1], W=stack.array.shape[2])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureStack:
    """Container for the assembled multi-layer feature raster.

    Attributes:
        array: Float32 feature array of shape (C, H, W) where C is the
               number of feature channels, H is the number of rows (northings),
               and W is the number of columns (eastings). NaN encodes missing
               values within the spatial extent.
        feature_names: Ordered list of length C giving a human-readable name
                       for each channel. Matches the channel axis of ``array``.
        profile: Rasterio dataset profile dictionary (CRS, transform, count,
                 dtype, nodata, height, width, driver). Needed for writing
                 output rasters with correct georeference.
        data_quality_mask: Boolean array of shape (H, W). ``True`` indicates
                           that the cell is **unreliable or invalid** (sensor
                           gap, permanent cloud, water body, vegetation mask).
                           Masked cells are excluded from model training and
                           inference. ``False`` = usable cell.
    """

    array: np.ndarray                                    # (C, H, W) float32
    feature_names: list[str]
    profile: dict
    data_quality_mask: np.ndarray = field(default=None)  # (H, W) bool

    def __post_init__(self) -> None:
        """Validate consistency between array shape and metadata."""
        if self.array.ndim != 3:
            raise ValueError(
                f"array must be 3-D (C, H, W), got shape {self.array.shape}"
            )
        C, H, W = self.array.shape
        if len(self.feature_names) != C:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) does not "
                f"match channel count ({C})"
            )
        if self.data_quality_mask is not None:
            if self.data_quality_mask.shape != (H, W):
                raise ValueError(
                    f"data_quality_mask shape {self.data_quality_mask.shape} "
                    f"does not match spatial extent ({H}, {W})"
                )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_feature_stack(
    config: dict,
    processed_dir: str,
) -> FeatureStack:
    """Read preprocessed feature GeoTIFFs and assemble them into a FeatureStack.

    Reads each feature layer listed in ``config["features"]`` from the
    ``processed_dir`` directory, validates spatial consistency (same CRS,
    dimensions, and transform), and stacks them channel-wise. A combined
    data quality mask is derived from the union of all per-band nodata/NaN
    pixels plus any explicit mask provided under ``config["quality_mask"]``.

    Config schema::

        {
            "features": {
                "feature_name_1": "relative/path/to/layer1.tif",
                "feature_name_2": "relative/path/to/layer2.tif",
                ...
            },
            "quality_mask": "relative/path/to/quality_mask.tif",  # optional
            "nodata": -9999  # optional, fallback nodata value if not in file
        }

    Args:
        config: Configuration dictionary. Must contain at minimum the key
                ``"features"`` mapping feature name strings to relative
                GeoTIFF file paths.
        processed_dir: Absolute path to the directory that contains the
                       processed feature GeoTIFF files. All paths in
                       ``config["features"]`` are resolved relative to this
                       directory.

    Returns:
        A populated :class:`FeatureStack` with:
            - ``array`` shape (C, H, W), dtype float32.
            - ``feature_names`` matching the ordered keys of
              ``config["features"]``.
            - ``profile`` from the first feature file.
            - ``data_quality_mask`` (H, W) bool — True = bad/masked pixel.

    Raises:
        KeyError: If ``config`` is missing the required ``"features"`` key.
        FileNotFoundError: If any listed GeoTIFF path does not exist.
        ValueError: If any feature raster has different dimensions or CRS
                    from the first raster.
        RasterioIOError: If a GeoTIFF cannot be opened by rasterio.
    """
    if "features" not in config:
        raise KeyError("config must contain a 'features' key")

    feature_dict: dict[str, str] = config["features"]
    fallback_nodata: float = config.get("nodata", -9999.0)

    feature_names = list(feature_dict.keys())
    paths = [
        os.path.join(processed_dir, rel_path)
        for rel_path in feature_dict.values()
    ]

    # Validate existence
    for name, path in zip(feature_names, paths):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature '{name}': file not found at {path}"
            )

    # Read first file to establish reference grid
    reference_profile: dict | None = None
    ref_height: int = 0
    ref_width: int = 0
    ref_crs = None
    ref_transform = None

    layers: list[np.ndarray] = []

    for name, path in zip(feature_names, paths):
        try:
            with rasterio.open(path) as src:
                profile = src.profile
                data = src.read(1).astype(np.float32)  # first band, float32
                nodata = src.nodata if src.nodata is not None else fallback_nodata

                # Establish or validate reference grid
                if reference_profile is None:
                    reference_profile = dict(profile)
                    ref_height = src.height
                    ref_width = src.width
                    ref_crs = src.crs
                    ref_transform = src.transform
                else:
                    if src.height != ref_height or src.width != ref_width:
                        raise ValueError(
                            f"Feature '{name}' has dimensions "
                            f"({src.height}, {src.width}) but expected "
                            f"({ref_height}, {ref_width})"
                        )
                    if src.crs != ref_crs:
                        raise ValueError(
                            f"Feature '{name}' CRS {src.crs} does not match "
                            f"reference CRS {ref_crs}"
                        )

                # Mask nodata as NaN
                data[data == nodata] = np.nan

        except RasterioIOError as exc:
            raise RasterioIOError(
                f"Cannot open feature '{name}' at {path}: {exc}"
            ) from exc

        layers.append(data)

    # Stack into (C, H, W) array
    array = np.stack(layers, axis=0)  # (C, H, W), float32

    # Build data quality mask: True where ANY channel has NaN
    dq_mask = np.any(np.isnan(array), axis=0)  # (H, W)

    # Optionally merge with an explicit quality mask GeoTIFF
    quality_mask_path = config.get("quality_mask")
    if quality_mask_path:
        full_qm_path = os.path.join(processed_dir, quality_mask_path)
        if os.path.exists(full_qm_path):
            with rasterio.open(full_qm_path) as qm_src:
                qm_data = qm_src.read(1)
                explicit_mask = qm_data.astype(bool)
            dq_mask = dq_mask | explicit_mask
        else:
            import warnings
            warnings.warn(
                f"quality_mask path not found: {full_qm_path}. "
                "Skipping explicit quality mask."
            )

    # Update profile for single-band output (for compatibility with
    # reconstruct_map → write_raster workflows)
    out_profile = dict(reference_profile)
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
    )

    return FeatureStack(
        array=array,
        feature_names=feature_names,
        profile=out_profile,
        data_quality_mask=dq_mask,
    )


# ---------------------------------------------------------------------------
# Flattening and spatial reconstruction
# ---------------------------------------------------------------------------

def flatten_to_dataframe(
    stack: FeatureStack,
    label_array: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Convert a C×H×W FeatureStack to a (N_valid × C) pandas DataFrame.

    Only pixels where ``stack.data_quality_mask`` is False (i.e. usable cells)
    are included in the output. This removes masked cells from model training
    without requiring the model to handle NaN. The original spatial position
    (row, col) of each pixel is preserved as columns so that predictions can
    be projected back onto the map using :func:`reconstruct_map`.

    Args:
        stack: Assembled :class:`FeatureStack` with shape (C, H, W).
        label_array: Optional 2-D integer or float array of shape (H, W)
                     containing training labels (e.g. 1 = known deposit,
                     0 = pseudo-absence). If provided, a ``"label"`` column
                     is appended. Cells with label nodata (NaN or −9999) are
                     not automatically excluded — filter the DataFrame
                     downstream if needed.

    Returns:
        DataFrame with columns:
            - ``"row"`` (int): Row index in the original H×W grid.
            - ``"col"`` (int): Column index in the original H×W grid.
            - One column per feature channel (named by ``stack.feature_names``).
            - ``"label"`` (optional): label values at valid pixels.
        Row order follows raster scan order (top-left to bottom-right).

    Raises:
        ValueError: If ``label_array`` has a shape incompatible with
                    the spatial dimensions of ``stack``.
    """
    C, H, W = stack.array.shape

    if label_array is not None:
        label_array = np.asarray(label_array)
        if label_array.shape != (H, W):
            raise ValueError(
                f"label_array shape {label_array.shape} does not match "
                f"spatial extent ({H}, {W})"
            )

    # Valid pixel mask: False in data_quality_mask means usable
    if stack.data_quality_mask is not None:
        valid_mask = ~stack.data_quality_mask  # True = include
    else:
        valid_mask = np.ones((H, W), dtype=bool)

    # Row and column indices for valid cells
    row_idx, col_idx = np.where(valid_mask)  # both shape (N_valid,)

    # Extract feature values: stack.array has shape (C, H, W)
    # Index with (C, N_valid) → transpose to (N_valid, C)
    feature_values = stack.array[:, row_idx, col_idx].T  # (N_valid, C)

    data: dict[str, np.ndarray] = {
        "row": row_idx.astype(np.int32),
        "col": col_idx.astype(np.int32),
    }
    for i, name in enumerate(stack.feature_names):
        data[name] = feature_values[:, i]

    if label_array is not None:
        data["label"] = label_array[row_idx, col_idx]

    return pd.DataFrame(data)


def reconstruct_map(
    predictions: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    H: int,
    W: int,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Reconstruct a 2-D raster map from 1-D model predictions.

    Inverse operation of :func:`flatten_to_dataframe`. Places predicted
    values (probability, class label, or regression output) back into the
    correct spatial cells of an H×W grid. Cells without predictions are
    filled with the nodata sentinel value.

    Args:
        predictions: 1-D float array of model predictions, length N_valid.
                     Must be in the same order as the rows of the DataFrame
                     produced by :func:`flatten_to_dataframe`.
        row_idx: 1-D integer array of row indices for each prediction.
                 Obtained from ``df["row"].values``.
        col_idx: 1-D integer array of column indices for each prediction.
                 Obtained from ``df["col"].values``.
        H: Height (number of rows) of the output raster.
        W: Width (number of columns) of the output raster.
        nodata: Sentinel value written to cells without a prediction.
                Defaults to −9999.0.

    Returns:
        Float32 array of shape (H, W). Predicted cells contain model output;
        all other cells contain ``nodata``.

    Raises:
        ValueError: If ``predictions``, ``row_idx``, and ``col_idx`` have
                    different lengths, or if any index falls outside [0, H)
                    or [0, W).
    """
    predictions = np.asarray(predictions, dtype=np.float32).ravel()
    row_idx = np.asarray(row_idx, dtype=np.int32).ravel()
    col_idx = np.asarray(col_idx, dtype=np.int32).ravel()

    N = len(predictions)
    if len(row_idx) != N or len(col_idx) != N:
        raise ValueError(
            f"predictions ({N}), row_idx ({len(row_idx)}), and "
            f"col_idx ({len(col_idx)}) must all have the same length."
        )

    # Bounds check
    if row_idx.size > 0:
        if row_idx.min() < 0 or row_idx.max() >= H:
            raise ValueError(
                f"row_idx values must be in [0, {H}), "
                f"got range [{row_idx.min()}, {row_idx.max()}]"
            )
        if col_idx.min() < 0 or col_idx.max() >= W:
            raise ValueError(
                f"col_idx values must be in [0, {W}), "
                f"got range [{col_idx.min()}, {col_idx.max()}]"
            )

    # Initialise output with nodata
    output = np.full((H, W), fill_value=nodata, dtype=np.float32)

    # Place predictions at the correct spatial locations
    output[row_idx, col_idx] = predictions

    return output
