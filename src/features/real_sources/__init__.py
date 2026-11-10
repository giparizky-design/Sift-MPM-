"""Real-data source fetchers. Nothing in this package generates values: a source that cannot
be retrieved raises :class:`RealDataUnavailableError` and the layer is omitted."""

from .core import (
    DERIVED,
    MEASURED,
    GridSpec,
    LayerProvenance,
    ProvenanceManifest,
    RealDataUnavailableError,
)

__all__ = [
    "DERIVED",
    "MEASURED",
    "GridSpec",
    "LayerProvenance",
    "ProvenanceManifest",
    "RealDataUnavailableError",
]
