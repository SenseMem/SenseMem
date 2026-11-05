"""Local persistence primitives used by the embedded SenseMem runtime."""

from sensemem.infrastructure.local._lock import DataDirectoryInUseError
from sensemem.infrastructure.local.assets import (
    AssetStore,
    AssetStoreError,
    AssetTooLargeError,
)
from sensemem.infrastructure.local.store import (
    IndexCandidate,
    IndexDocument,
    IndexOperation,
    LocalStore,
    LocalStoreClosedError,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
    UnsupportedSchemaError,
)

__all__ = [
    "AssetStore",
    "AssetStoreError",
    "AssetTooLargeError",
    "DataDirectoryInUseError",
    "IndexCandidate",
    "IndexDocument",
    "IndexOperation",
    "LocalStore",
    "LocalStoreClosedError",
    "StoredAsset",
    "StoredEmbedding",
    "StoredMemory",
    "StoredOperation",
    "UnsupportedSchemaError",
]
