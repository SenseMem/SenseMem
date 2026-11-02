# Plugin architecture

## Decision

MindBridge uses a stable memory kernel with explicit, typed computation plugins. It does not use a
global registry, arbitrary pipeline hooks, automatic package installation, or a second execution
plane.

This keeps the product rich without making the API sprawling: applications define their own input
and capture forms at the edge, while replaceable model plugins implement narrow capabilities behind
the same memory operations.

The kernel owns memory identity and semantics, capability routing, validation, SQLite and media-CAS
durability, the index outbox, SQLite hydration, and final context construction. A plugin may perform
inference, but it cannot own or bypass those rules.

## Implemented composition

The supported plugins are the narrow backend protocols inventoried in
[architecture](architecture.md#model-boundary). That page owns the list; this page owns
the rules for adding to it. The declarative entry point constructs bundled implementations without
exposing runtime details:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "google/embeddinggemma-2",
            "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        },
        "speech": {"provider": "funasr"},
        "settings": {"index_speech": True},
    }
) as memory:
    memory.add("Remember this")
```

`Memory.from_config` validates a closed catalog of bundled providers, constructs owned adapters, and
then delegates through `Memory.from_plugins` to the normal constructor. Plugin bundles contain
values; protocols and declarations are validated once at assembly, before storage opens.
Custom and third-party
implementations remain explicit object injection. All paths share capability validation, durable
identity checks, storage, and execution. Omitting an optional plugin adds no model call or optional
dependency. Reusing one object for several protocols is supported by the object API, and the bundled
OpenAI adapter does exactly that.

Declarative configuration reaches a slot when there is a bundled implementation for its closed
provider catalog to name; [configuration](configuration.md#declarative-configuration) owns the
current list of accepted keys. A protocol with no bundled implementation deliberately has no key,
and is composed by passing an object to `Memory`, `Memory.from_plugins`, or `MemoryPlugins` — see
[the Python SDK reference](api/python-sdk.md#backend-protocols) for the signatures.

Composition is fixed for one `Memory` lifetime. Changing an adapter requires closing and reopening
the instance; changing the embedder also requires a compatible durable space or a new directory.
There is no mid-operation hot swap.
Routing and `Memory.capabilities` share the immutable assembly snapshot; reading the public
document never probes the plugin again. An unsupported Sentence Transformers capability returns
`False`; an exception during discovery fails assembly rather than silently removing a modality.

`GenerationRouter` composes several `GenerationBackend` objects in the answerer slot. It chooses
against the question and grounded evidence together, preferring complete native support and then
the narrowest capability set; caller order resolves ties. Text, text-plus-image and text-plus-video
can therefore use different models without application dispatch. Declarative `generation.routes`
constructs the same composition. Model rejection remains an explicit error.
The kernel probes the optional `GenerationRoutingBackend` protocol, so evidence wrappers can
preserve routing without exposing their concrete implementation to the answer plane.

`EmbeddingRouter` composes specialized encoders only when they declare the same vector space and
dimension. `EmbeddingInputSupport.embedding_routes` declares exact native combinations and
`max_assets_per_modality` limits through immutable `EmbeddingRoute` values;
when an aggregate is unsupported, the kernel uses supported atomic keys. Distinct vector spaces
cannot share this index. These compositions own each distinct nested resource once.

All consumers share source-backed native/derived routing. A caption or transcript can supply text
to embedding, queries, generation and formation while original assets remain durable. Required
conversion is automatic when a matching plugin exists; `query_vision` separately enables optional
supplemental retrieval routes. A transcript describes speech, not unseen video content.
Routing checks each attachment separately. Existing source-labelled captions and transcripts remain
usable after reopening without their producer. Removing a speech plugin preserves the stored
recipe and cached identity evidence; reattaching an incompatible recipe still fails explicitly.
Embedding model names are diagnostic: compatibility depends on the declared space and dimension,
so composing or reordering compatible encoders does not require re-ingestion.
Partial descriptions do not cover other assets, and an empty transcript supplies no fallback.
Lexical queries use available speech-to-text preparation without calling the query embedder.

## Kernel and plugin boundary

Current computation plugins own model and runtime details such as device, precision, batching,
thresholds intrinsic to that model, or a caller-supplied remote SDK client. Potential OCR, emotion,
visual-grounding and semantic-extraction capabilities belong on this side only after a
concrete implementation proves a narrow contract.

Validation has one owner per boundary. Assembly validates configuration and backend declarations;
typed input values validate scope and spatial constraints; model output validation precedes any
durable effect. Embedding adapters decode vectors, and the kernel owns L2 normalization. The
local store keeps database constraints and persisted-data/FP32 checks, without revalidating
configured recognition thresholds or typed retrieval scopes on each call.

Adapters classify provider errors and own their request retry budgets. The kernel makes one
vision call per batch, propagates failures of required modality conversion, and may skip failed
optional enhancement when another usable representation exists. Storage translation handles
concrete I/O, database, capacity and persisted-data errors; internal programming failures remain
visible to the outer error boundary.

`RerankingBackend` is an optional computation plugin with explicit `reranking_capabilities`.
It receives a query and ordered documents as `ModelInput` values and returns an equally sized
tuple of finite relevance scores. The retrieval plane owns
SQLite hydration and filtering, eligibility, exact source-window selection, candidate identity,
and final ordering. It automatically includes supported text and original media, protects candidate
files with temporary leases and preserves captioned originals. Inputs with no supported content
are unscored; other candidates retain their positions. A reranker cannot introduce records, change source IDs, bypass the relevance or ambiguity
gate, or rewrite stored content. Invalid results and model failures are explicit errors rather
than silent fallback. Request-local scores appear separately as `RetrievalCandidateTrace.reranking_score`;
the existing bounded score keeps its meaning.

SQLite, the media CAS, the durable outbox, and Zvec are not public plugins. They have one supported
implementation, and abstracting them now would add indirection without a second product use case.
Metadata remains application data, never an execution, isolation, or authorization boundary.

A plugin whose output the kernel persists must declare that output's *space* — `embedding_space`,
`transcription_space`, `face_space`, `formation_space`, `vision_space`. The kernel caches derived
work per asset and space, so the space is what stops two recipes from sharing one result; the
plugin owns it because only the plugin knows what its recipe is. It is not the model name: a
describer that edits its prompt, or a transcriber that changes its keyword list, produces different
output from the same model and must return a different space. Declaring a stable space over a
changed recipe is the one failure the kernel cannot detect, because a stale derived value is
indistinguishable from a fresh one once it is inside a searchable document.

Request-local reranking scores are not persisted derivations and need no `*_space`. The
`reranking_model` property identifies the loaded model for diagnostics. Changing a reranker
requires reopening the instance but does not change stored embeddings or require re-ingestion.

`VisionDescriptionBackend.describe` returns plain observation text. Text beginning with `Fact:`
has no identity side effect. The optional `VisionAnalysisBackend.describe_analysis` returns typed
`VisualDescription(text, facts, speaker_names)` values. `SpeakerName(speaker_label, name)` is an
explicit proposal about the source-local diarised label. The identity plane validates those
proposals, preserves existing names, and owns auditable, reversible `IDENTIFY` writes and all
cross-modal merges. The bundled OpenAI adapter translates its provider output into these values.
The kernel uses the `typed-v1` description-cache format; old cache rows are recomputed.

`EvidenceGeneration` composes any generation backend with independent `MediaSelectionBackend`
and `SourceMetadataBackend` implementations. The kernel owns bounded selection and evidence
preservation; `OpenAIModels.select_media` supplies only the diagnostic computation.
`GazetteerEnricher` implements local source metadata lookup. `GenerationMediaBackend` optionally
reports which original assets fit a provider request before enrichment. Metadata is untrusted
request context and never becomes a durable fact.

The optional `server` and `mcp` dependencies are packaging boundaries, not plugins. REST, MCP, and
the product CLI remain thin transports over the application-composed `Memory`.

`Memory.add_stream` is also not a plugin boundary. It repeats the existing durable `add` operation
over caller-segmented observations. `AsyncOmniPrefetch` is a thin lifecycle helper over the same
`AsyncMemory.search`; it neither performs perception nor owns a second execution plane. Capture,
ASR partials, frame selection, and finality stay with the application.

## Admission rule

A new public capability protocol is admitted only together with all of:

1. A concrete implementation in `src/`, not a test double.
2. A reachable caller on a product path a user invokes, not only on an adjacent or optional one.
3. Declared accepted modalities and typed output.
4. Provenance, configuration, resource lifecycle, concurrency behavior, privacy boundary, and
   failure mapping.
5. Tests that exercise the capability through the public product path.

Face analysis is the worked example: `FaceBackend` arrived with a local OpenCV implementation, an
end-to-end identity use case, and product-path tests. A capability that satisfies the type system
but nothing else is exported surface area with no user, and it costs the same to maintain as one
that works.

`VisionDescriptionBackend` meets the rule through `OpenAIModels.describe`, the optional
[`vision` configuration](configuration.md#visual-descriptions), and the `add`/`add_many` product path.

`RerankingBackend` meets it through `Qwen3VLReranker` and `SentenceTransformersReranker`, the optional
[`reranking` configuration](configuration.md#semantic-reranking), and the shared retrieval path
used by `search`, `ask` and `compile`. The plugin supplies aligned relevance scores; the kernel owns
source identity, filtering and result ordering. Both adapters pin checkpoint revisions, disable
remote code and close with their owning `Memory`. Their dependencies and runtime controls live
in the configuration guide. Product-path tests establish integration; task-specific quality
improvements require separate evaluation.

## Embodied integration boundary

MindBridge may improve embodied memory representation, retrieval, provenance, temporal validity,
and failure diagnosis without becoming a robot AgentOS. Planner, skill-runner, verifier,
edge/cloud routing, robot-control, benchmark harness, and model-training contracts remain outside
the product boundary. Procedural memory is evidence and is never executed as code.

New entity, relation, or spatial projections must remain additive to authoritative records and earn
their complexity through MindBridge's own measured retrieval results. A paper's architecture or
benchmark score is not by itself a reason to replace the flat durable representation.

Interaction memory uses the existing semantic, episodic, and procedural roles. Any
emotion, trait, or response-policy inference stays outside the kernel or behind a future concrete
typed analysis plugin, on the phased path in [affective memory](affective-memory.md). Derived
records retain evidence provenance and use ordinary `Memory` writes; plugins may not write around
durability or reinterpret metadata as routing, isolation, or trust.

## Trade-offs and revisit triggers

Typed declarative construction makes bundled adapters concise; explicit construction keeps custom
dependencies, lifecycle, privacy, and hardware selection visible. Both require reopening `Memory`
to change composition.

Add package entry-point discovery only when an independently distributed plugin requires it. Add a
new public capability only with a concrete adapter plus lifecycle, failure, privacy, provenance, and
product-path tests. Add richer evidence or relationship storage only when the executable embodied
loop demonstrates that existing records cannot represent the required state without loss.
