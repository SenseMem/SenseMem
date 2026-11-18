"""Host-owned descriptions of the SDK's bundled agent tools."""

PERCEIVE_MEMORY = (
    "Recognize speakers and faces in an existing memory_id supplied by the caller or earlier results. "
    "kinds selects speech and/or faces; null uses configured plugins. Returns source-local "
    "speech and face observations with identity IDs. Recognition may be cached; it does not "
    "confirm a person's name or bind face and voice identities."
)
GET_IDENTITY = (
    "Set identity_id to a face or speaker ID returned by recognition to read its profile. Returns "
    "the supported name, relationship, confirmation status and evidence IDs, or null for "
    "an unknown identity. An unnamed recognized person has a profile with name=null. "
    "Evidence IDs identify sources; their contents are not supplied by this lookup."
)
SEARCH_MEMORIES = (
    "Find stored evidence using a focused text query and optional limit from 1 through 100. "
    "Returns ranked hits with memory_id, content, score and asset_ids. Empty or bounded hits "
    "do not establish absence or a complete set. Asset IDs identify media; their bytes are "
    "not attached. Use compile_context when you need structured context for a task."
)
COMPILE_CONTEXT = (
    "Compile bounded context for a text goal when a task needs related facts and their "
    "provenance. Returns rendered context, source_ids, excerpt_source_ids, "
    "projection_source_ids and omitted counts. Preserve the context's conflict, unknown, "
    "partial-source and omitted-media warnings; the bundle does not establish a complete set."
)
TRANSCRIBE = (
    "Transcribe retained original audio or video for an existing memory_id supplied by the caller "
    "or earlier results. Returns per-asset transcripts with the model and space that produced them, "
    "or an empty transcripts list when no asset matches the backend's supported modalities. Results "
    "are temporary and do not write a stored transcript or confirm speaker identity."
)
