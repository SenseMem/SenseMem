"""SQLite source of truth for the local SenseMem runtime.

`LocalStore` is the one entry point; the row values and errors here are what its families
accept and return. The family modules are classes over the connection-level functions in the
private modules, and product code outside this package imports none of them directly.
"""

from sensemem.infrastructure.local.store._codec import datetime_text, optional_datetime_text
from sensemem.infrastructure.local.store._identity import CONSENT_PREDICATE, canonical_subject
from sensemem.infrastructure.local.store._naming_projection import NamingProjectionFactory
from sensemem.infrastructure.local.store._recall import (
    RECALL_MAX_ROWS,
    RECALL_MAX_TERM_CHARS,
    RELATED_EDGES,
)
from sensemem.infrastructure.local.store._schema import SCHEMA_VERSION
from sensemem.infrastructure.local.store.errors import (
    LocalStoreCapacityError,
    LocalStoreClosedError,
    ProtectedMemoryError,
    StaleOperationError,
    StoredDataError,
    UnsupportedSchemaError,
)
from sensemem.infrastructure.local.store.local_store import LocalStore
from sensemem.infrastructure.local.store.rows import (
    IdentityLink,
    IndexCandidate,
    IndexDocument,
    IndexOperation,
    RecallDigest,
    RecallRead,
    RelatedRead,
    SpeechPlan,
    StoredAsset,
    StoredCandidate,
    StoredEmbedding,
    StoredEvidenceClauseChange,
    StoredMemory,
    StoredOperation,
    StoredQueryFailure,
    StoredTextSelector,
    StoredTextSpanPiece,
    validate_asset_name,
)

__all__ = [
    "CONSENT_PREDICATE",
    "RECALL_MAX_ROWS",
    "RECALL_MAX_TERM_CHARS",
    "RELATED_EDGES",
    "SCHEMA_VERSION",
    "IdentityLink",
    "IndexCandidate",
    "IndexDocument",
    "IndexOperation",
    "LocalStore",
    "LocalStoreCapacityError",
    "LocalStoreClosedError",
    "NamingProjectionFactory",
    "ProtectedMemoryError",
    "RecallDigest",
    "RecallRead",
    "RelatedRead",
    "SpeechPlan",
    "StaleOperationError",
    "StoredAsset",
    "StoredCandidate",
    "StoredDataError",
    "StoredEmbedding",
    "StoredEvidenceClauseChange",
    "StoredMemory",
    "StoredOperation",
    "StoredQueryFailure",
    "StoredTextSelector",
    "StoredTextSpanPiece",
    "UnsupportedSchemaError",
    "canonical_subject",
    "datetime_text",
    "optional_datetime_text",
    "validate_asset_name",
]
