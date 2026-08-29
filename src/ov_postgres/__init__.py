"""PostgreSQL + pgvector backend for OpenViking's vector store.

Point the vectordb backend at the adapter's import path::

    "vectordb": {
      "backend": "ov_postgres.adapter.PgVectorCollectionAdapter",
      "custom_params": {"dsn": "postgresql://localhost/openviking"}
    }
"""

from .adapter import PgVectorCollectionAdapter
from .collection import PgVectorCollection
from .filters import FilterCompiler, UnsupportedFilterError
from .schema import CollectionSchema, FieldSpec

__all__ = [
    "CollectionSchema",
    "FieldSpec",
    "FilterCompiler",
    "PgVectorCollection",
    "PgVectorCollectionAdapter",
    "UnsupportedFilterError",
]

try:  # populated by hatch-vcs at build time
    from ._version import __version__
except ImportError:  # editable install or source checkout
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("ov-postgres")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
