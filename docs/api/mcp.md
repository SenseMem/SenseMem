# MCP API

Give an agent durable memory through thirteen typed tools. The host selects the models and tool
authority; the agent can store observations, retrieve evidence, compile context, and ask grounded
questions through the same SenseMem kernel.

Jump to [authority switches](#withholding-a-group-of-tools), [startup](#start-the-adapter),
[tools](#tools), [results](#result-objects), [errors](#validation-and-errors),
or [input limits](#input-limits).

## Surface

The optional MCP adapter exposes thirteen typed tools over one injected synchronous `Memory`, or
fewer when the host withholds a group of them.
It validates tool input, calls the matching SDK operation, and returns structured public values.
It does not own storage, provider selection, or the injected memory. Finalized media arrives
through ordinary content parts; live audio and vision packet ingestion, and `StreamEvent`
reduction, stay Python-only because a tool call is a finite request.

Tools cover ingestion, retrieval, grounded answers, context compilation, perception, identity
administration, and feedback. The host chooses which groups an agent may call.

Naming, unlinking, and erasure tools exercise host-granted identity authority. Naming and unlinking
are logged and reversible; erasure is logged and irreversible. `get_identity` reads the current
profile.

### Withholding a group of tools

Three keyword switches decide which authority an agent holds. Each defaults to `True`; setting
one to `False` withholds its group by never registering it, so calling one of its tools fails as an
unknown tool rather than as a refusal. The server instructions name the missing groups, because
an agent should not discover a boundary by failing.

| Switch | Withholds | Why a host would |
| --- | --- | --- |
| `identity_operations` | `register_identity`, `get_identity`, `unlink_identity`, `forget_identity` | Naming and erasing a person is host authority |
| `embodied_operations` | `perceive_memory` | `perceive_memory` also commits the corroborated cross-modal identity merge, so leaving it registered keeps identity *binding* on the wire even with the identity tools withheld. `ask_memory` reaches the same merge through its own face recognition, so this switch also passes `link_identities=embodied_operations` into every `ask_memory` call |
| `write_operations` | `add_memory`, `delete_memory`, `reinforce_memories` | The host wants to withhold explicit record writes and feedback |

```python
build_mcp_server(
    memory,
    identity_operations=False,
    embodied_operations=False,
    write_operations=False,
)
```

With all three switches `False`, five tools remain: `search_memories`, `ask_memory`,
`compile_context`, `get_memory`, and `list_memories`.

Recall can persist query/recognition caches and bounded failure signals. Set
`reinforce_on_answer=False` on the owner to disable automatic ranking feedback. With
`embodied_operations=False`, `ask_memory` passes `link_identities=False`: recognition can identify
someone in evidence, while cross-modal merges are withheld.

## Start the adapter

Install the MCP extra together with the extras required by the chosen backends, then host the
server in the application that owns `Memory`:

```bash
uv add "sensemem[local,mcp]"
```

```python
from sensemem import Memory
from sensemem.api.mcp import build_mcp_server

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "google/embeddinggemma-2",
        },
    }
) as memory:
    build_mcp_server(memory).run("stdio")
```

```text
build_mcp_server(
    memory: Memory,
    *,
    identity_operations: bool = True,
    embodied_operations: bool = True,
    write_operations: bool = True,
    tracer: Tracer | None = None,
    deliberate_every: float | None = None,
) -> MCPServer[None]
```

`deliberate_every` is the opt-in slow loop described in
[configuration](../configuration/lifecycle.md#running-the-loop-inside-a-server-process). It runs in the
server lifespan and is disabled by default. No tool exposes consolidation directly.

`build_mcp_server` publishes an assembly-time `memory.capabilities` snapshot in its
`instructions`. A connecting agent learns the model and modality declarations without a tool call.
The instructions recommend `compile_context` for task-ready context and list the tool groups
the host has withheld. They also mark returned memories, procedures, names, metadata, transcripts
and visible text as untrusted evidence. Such content cannot authorize writes, identity changes,
or policy changes. The caller's request and host policy supply that authority. Relevant conflicts
and unknowns remain part of the context; empty or bounded retrieval does not prove absence.

The document's `operations` match reachable capabilities: `perceive` requires
`embodied_operations=True`; write-path formation and visual description require
`write_operations=True`; transcription remains available for spoken queries. Control-plane
operations stay with the owner. Rebuild the server after changing the owner's composition.

## Lifecycle and ownership

Keep the injected owner in a `with Memory(...)` or `with Memory.from_config(...)` block around
`server.run(...)`, as in the [deployment example](../deployment.md#mcp-owner). Withholding tool
groups narrows what the agent can call; it does not provide client authentication or filesystem
isolation. Use separate physical directories for independent memory domains.

`build_mcp_server` borrows `memory`; it neither opens nor closes it. The host must keep the owner
alive for the server lifetime and close it during shutdown. Do not run another `Memory`, REST, or
MCP owner against the same physical `data_dir`. Composition and process ownership are defined in
[architecture](../architecture.md) and [configuration](../configuration.md).

SenseMem adds no authentication to any MCP transport. Stdio inherits local process permissions;
an SSE or streamable-HTTP host must add authentication, authorization, TLS, request limits, and
rate limits. MCP error `subject` is unredacted and may contain an owner-local path. See
[deployment](../deployment.md) for process and transport boundaries.

## Contract

### Content input

`content`, `query`, and `question` use the same strict `input_text`, `input_image`, and `input_file`
union as [REST content input](rest.md#content-input): a non-blank string or 1 through 16 ordered
parts. Source fields are mutually exclusive. Data URLs must contain base64 media; `file_data`
requires a concrete image, video, or audio MIME type.

Remote URLs, local paths, `file:` URLs, unknown nested fields, and `input_image.detail` are
rejected. The MCP-specific media bounds are listed below.

### Tools

| Tool | Arguments and defaults | Structured result | Annotation |
| --- | --- | --- | --- |
| `add_memory` | required `content`; `occurred_at=None`; `occurred_end=None`; `metadata=None`; `memory_type="semantic"`; `context=None` | `MemoryResult` | write, idempotent |
| `search_memories` | required `query`; `limit=10`; `memory_type=None`; `reference_at=None`; `occurred_from=None`; `occurred_until=None`; `scope=None`; `explain=false` | `{"hits":[SearchHitResult,...],"trace":null}` | write, not idempotent |
| `ask_memory` | required `question`; `limit=5`; `memory_type=None`; `reference_at=None`; `scope=None`; `answer_policy="strict"` | `AnswerResponse` | write, not idempotent |
| `compile_context` | required `goal`; `budget=None`; `reference_at=None`; `scope=None`; `allow_partial_sources=false` | `ContextBundleResult` | write, not idempotent |
| `get_memory` | required `memory_id` | `MemoryResult` | read-only |
| `list_memories` | `limit=100`; `cursor=None` | `PageResult` | read-only |
| `delete_memory` | required `memory_id` | `{"deleted":bool}` | destructive, idempotent |
| `reinforce_memories` | required `memory_ids`, 1 through 100 | `{"reinforced":int}` | write, not idempotent |
| `perceive_memory` | required `memory_id`; `kinds=null` or an array containing `"speech"` and/or `"faces"` | `{"speech":[SpeakerSegment,...],"faces":[FaceObservation,...],"coverage":[PerceptionCoverage,...]}` | write, not idempotent |
| `register_identity` | required `identity_id`, `name`; `relationship=None` | `{"registered":true}` | idempotent write |
| `get_identity` | required `identity_id` | `{"identity":IdentityProfile\|null}` | read-only |
| `unlink_identity` | required `alias_id` | `{"restored_identity_id":str\|null}` | destructive, idempotent |
| `forget_identity` | required `identity_id` | `{"erasure":IdentityErasure}` | destructive, not idempotent |

All timestamps must be timezone-aware. An event end requires a start and must be later than it.

MCP tool calls have one finite response and therefore expose completion latency, not TTFT. The
adapter emits a `sensemem.mcp.request` server span with the method, tool name, error status and
total server time. Use Python `ask_stream` or REST `POST /v1/answers/stream` when first-token
delivery is part of the performance contract.
Search event bounds are a half-open overlap filter; two bounds require
`occurred_until > occurred_from`. `memory_type` is `semantic`, `episodic`, or `procedural`.
Pagination cursors are opaque and must be passed back unchanged.

`context` carries typed observation basis, source ID, confidence, validity, optional spatial pose,
and optional symbolic `place_id`. `scope.valid_at` selects world validity and `scope.known_at`
selects the transaction version known then; `scope.place_id` matches the stored label exactly;
`scope.identity_id` scopes to one person -- the memories that identity is the subject of, plus the
memories whose media shows their face or carries their speech, with a merged alias resolving to
the surviving identity; `scope.near` and `scope.radius_m` must appear together, and their frame ID
and observer/subject anchor must match the stored spatial context. SQLite reapplies every filter
after candidate retrieval.

`add_memory` is content-addressed. `delete_memory` reports whether a record existed. Search,
answer, compile, and perception are not marked read-only because their SDK path
persists lazy transcript caches, identity evidence, and -- when a recall comes back with nothing,
an answer abstains, or a bundle carries no evidence -- the bounded recall-failure signal the
control plane's `QUERY_FAILURE` trigger reads; they are also not advertised as idempotent. Every
tool has `open_world_hint=false`. `ask_memory` requires an answerer in the
injected memory; without one it returns `model_error/backend_not_configured`. With the default
`reinforce_on_answer=True`, a non-abstained result also reinforces backend-declared hits; an
abstained result records the query failure without reinforcement. Its `answer_policy`
is `strict` or `best_effort`: the default refuses when the evidence is thin, while `best_effort`
asks the model for the most likely answer, choosing an offered option for multiple choice,
and still reports `abstained`, so an agent that must produce an answer keeps the confidence
signal without losing the answer. Under `best_effort` a question that grounds nothing spends a
generation call too, where the default returns without one.

`compile_context` returns a structured evidence bundle with provenance, conflicts, and unknowns.
It runs without answer generation and creates no memory record. Like search, it can cache spoken
queries and record empty-recall signals. Use `ask_memory` for one grounded answer.

Its optional `budget` defaults to 16,000 characters and 24 items. The transport uses
`freshness_seconds` for the SDK's `freshness` timedelta. It must be positive, finite, and
representable as a `timedelta`. `max_chars` accepts 1–65,536; `max_items` accepts 1–100;
`max_media_items` is non-negative or `null`. `max_latency_ms` is a deadline checked between
stages. See [context compilation](../context-compilation.md) for selection and closure.

`allow_partial_sources=true` enables verified raw-text excerpts when their full parent does not
fit. The default `false` uses full records and leaves `excerpts` empty. Non-boolean values fail
validation.

The embodied and identity tools follow the SDK operation they dispatch to:

- `perceive_memory` recognizes one stored memory's applicable assets. `coverage` records each
  kind/asset analysis or its skip reason, so an empty detection result is distinct from skipped
  analysis. A null reason means completed or cached analysis. Results retain separate
  typed speech and face collections. Omitting `kinds` selects configured plugins automatically;
  selecting a kind explicitly requires that backend when relevant media exists. Recognition
  caches identity evidence and may apply a corroborated cross-modal merge.
- `register_identity` names any recognized face or voice identity. It replaces an existing name
  and preserves `relationship` when omitted. An unknown ID returns `identity_not_found`.
- `get_identity` follows merge aliases and returns a profile for a recognized identity, including
  one without a confirmed name. An unknown ID returns `identity: null`.
- `unlink_identity` reverses one face-and-voice merge and returns the restored ID, or `null` when
  no record names which modality was contributed. It resets that pair's accumulated evidence and
  does not suppress the pair: a voice and face that keep co-occurring are merged again.
- `forget_identity` erases a person -- biometric templates, aliases, and the indexed name -- and
  returns counts of what was destroyed as the audit record. Memories and media survive and a
  transcript keeps its words with the speaker attribution dropped. It cannot be undone, and the
  second call reports `identity_not_found` because the person is already gone, which is why it is
  the one tool advertised as destructive and not idempotent.
- `reinforce_memories` records cumulative positive feedback: each call raises `access_count` for
  the named memories and moves the ranker's reinforcement factor, so it is published with
  `idempotent_hint=false` and a lost response must not be retried blindly. Duplicate IDs count
  once and unknown IDs are skipped, so `reinforced` below the number sent means the rest do not
  exist.

Every tool and every tool argument carries a published description. Read those before guessing at
coordinate frames, units, or media sources; they are part of the tool schema, not a separate guide.

### Retrieval trace

`search_memories` with `explain=true` routes to `Memory.search(..., explain=True)` and adds a `trace`
object beside the unchanged `hits`. `trace.candidates` lists every candidate that was considered
with its effective score components (`dense_relevance`, `dense_confidence`, `lexical_relevance`,
`lexical_rerank_bonus`, `lexical_match`, `gate_relevance`, `base_relevance`,
`reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `reranking_score`, `rank`) and, when it
did not become a hit, a `rejected_by` value of `stale_index`, `occurrence_range`, `missing_memory`,
`memory_type`, `minimum_relevance`, `ambiguity`, or `limit`. `trace.candidate_limit` is the final
candidate depth requested per native route. `trace.exhaustive` is true when every route returned
fewer candidates than its bound; it does not make approximate search exact.
`trace.ambiguous` says whether the result was suppressed for being too close to call. Without
`explain`, `trace` is `null` and no extra work is done.

A `SearchHitResult.score` is `final_score`, but the relevance gate compares `gate_relevance`, which
is a different quantity. Tuning a floor against the returned `score` therefore compares the wrong
two numbers. The floor itself, `minimum_relevance`, and `ambiguity_margin` are fixed when the owner
constructs `Memory`; no tool argument can widen them for one call, so an empty result is answered by
reading `trace` and changing the query, filters, or the owner's configuration.

### Result objects

Successful calls populate MCP `structuredContent`:

| Object | Fields |
| --- | --- |
| `AssetResult` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResult` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context`, `place_id`, `forgotten_at` |
| `SearchHitResult` | all memory fields plus `score` |
| `SearchResult` | `hits`, `trace`; `trace` is `null` unless `explain=true` |
| `ReinforceResult` | `reinforced` |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResult` | `items`, `next_cursor` |
| `ContextBudgetResult` | `max_chars`, `max_items`, `max_media_items` or `null`, `memory_types` or `null`, `min_confidence`, `freshness_seconds`, `max_latency_ms`, `selected_proofs` |
| `ContextConflictResult` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextUnknownResult` | `kind`, `detail` |
| `ContextBundleResult` | `goal`, `reference_at`, `budget`, the `SearchHitResult` arrays `relationships`, `scene`, `episodes`, `facts`, `procedures`, `traits`, the `AffectCueResult` array `affect`, the mixed `actors` array of `SearchHitResult`, `NamedActorResult`, and `ProvisionalActorResult`, plus `conflicts`, `unknowns`, `occurred_from`, `occurred_until`, `frames`, `places`, `omitted`, `chars`, `elapsed_ms`, `deadline_exceeded`, separately typed `excerpts`, `proofs`, `rendered` |
| `ContextExcerptResult` | `source_memory_id`, structured `matched_index_id`, partial `content`, digest-bound `selector`, score, source timestamps, `memory_type`, `context`, `place_id`; the rendered line warns that omitted text may qualify it |
| `TextSpanSelectorResult` | parent and embedding-input digests, recipe version, and ordered exact `TextSpanPieceResult` values with role, half-open code-point offsets, source text, and piece digest |
| `ProvisionalActorResult` | `identity_id`, `memory_ids` |
| `NamedActorResult` | `identity_id`, `name`, `memory_ids`, `naming_assertion_id` |
| `AffectCueResult` | every `SearchHitResult` field plus `event_ids`: the events formed from the same observations the cue cites in its own `context.evidence_ids`, which is co-occurrence inside one capture and not an attributed cause |
| `SpeakerSegment` | `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `FaceObservation` | `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `IdentityProfile` | `identity_id`, `name`, `relationship`, `confirmed`, `evidence_ids` |
| `IdentityErasure` | `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`, `face_observations`, `speech_segments` |

The compile tool accepts `budget.selected_proofs=false`. Opting in delivers the same complete
`ContextProof` witnesses as the [REST context response](rest.md#contract), in an additive
`proofs` array. Each witness has `anchor_id` and complete `nodes` with `memory_id`, `confidence`,
`sources`, and nullable `capture_id`. The full evidence audit union stays on every hit. Certificate
text and its records are budgeted together; `scope.known_at` retains full provenance. Other
deliveries return `proofs=[]`. The MCP surface still contains thirteen tools.

Perception, identity, and erasure results publish the SDK values returned by `Memory.perceive`,
`Memory.identity`, and `Memory.forget_identity`, field for field:
`bounding_box` is `[x, y, width, height]` normalized within the frame. `identity_score` is the
match score when available; `speaker_name` is `null` until a visible naming assertion names the
speaker. `observed_at_ms` is the
offset within a video. A `speaker_id` or `identity_id` from the perception tool is accepted by
`get_identity`, `register_identity`, `unlink_identity`, and
`forget_identity`.

The other fields have the same meanings as the [REST response objects](rest.md#response-objects).
Successful result objects never serialize filesystem paths; error `subject` is the exception
described above.

## Errors and limits

### Validation and errors

Only the documented names and their exact top-level arguments are accepted. An unknown
tool or top-level argument returns `validation_error/unknown_field`; schema and SDK input failures return
`validation_error/input_invalid`. Unknown values are not echoed.

Failed tool calls set `isError` and their text content is exactly the same JSON envelope as
[REST](rest.md#error-envelope), with nothing before or after it. One `JSON.parse` of the text
succeeds for every failure, including `memory_not_found`, `model_error`, and `storage_error`:

```json
{
  "code": "validation_error",
  "reason": "unknown_field",
  "retryable": false,
  "stage": null,
  "subject": null,
  "message": "tool arguments contain unknown fields",
  "trace_id": "trace_0123456789abcdef0123456789abcdef",
  "issues": [
    {
      "location": ["arguments", "run_id"],
      "message": "Extra inputs are not permitted",
      "type": "extra_forbidden"
    }
  ]
}
```

Stable SDK codes are `sensemem_error`, `validation_error`, `memory_not_found`,
`identity_not_found`, `model_error`, `model_output_truncated`,
`storage_error`, and `index_unavailable`; unexpected failures use `internal_error`. Reasons and
retryability use the [shared vocabulary](rest.md#codes-and-reasons).

An explicit model refusal is `model_error/model_refused` with `retryable=false`; the provider's
refusal text is withheld. Temporary upstream failures use `provider_unavailable` with
`retryable=true`, while exhausted credits and spend/usage limits use permanent `quota_exhausted`.
The client must still bound retries rather than interpreting `retryable` as a retry loop.

Unlike unauthenticated REST, the MCP adapter retains `subject` for every SDK code to support owner
diagnostics. That is safe only for trusted clients; a network host must protect or redact the error
envelope at its outer boundary. Provider exceptions, credentials, and unexpected implementation
details are still never serialized.

### Operations without a tool

The remaining Python operations use the SDK, local CLI, or the REST routes listed below.

| Operation | Why, and what to call instead |
| --- | --- |
| `add_stream` | Call `add_memory` for each completed observation; use SDK streams for a live source. |
| `ask_stream` | Use `ask_memory` for a completed answer or SDK/REST streaming for incremental delivery. |
| `add_many` | Use the SDK or `POST /v1/memories/batch` for an atomic source batch. Use `add_memory` for individual observations. |
| `reindex` | Run index rebuilds through the SDK or local CLI in the owning process. |
| `optimize` | Operator-scheduled index maintenance through the SDK or local CLI. |
| `capture`, `settle`, `pending_captures` | Schedule deferred enrichment through the SDK or REST; `add_memory` returns searchable records. |
| `record_consent`, `consent` | The host collects the subject's statement and uses the SDK or consent REST route with `identity_operations=True`. |
| `export`, `apply_retention` | Use the SDK or gated REST routes for host-authorized subject export and physical retention. |
| `consolidation_candidates`, `consolidate`, `consolidate_pending`, `apply`, `record_outcome`, `forget`, `rollback`, `operations` | Run control-plane operations through the SDK or local CLI, where the owner schedules inference and audits or reverses changes. |

`Memory.capabilities` has no tool either: it is a property rather than an operation, published
here as the server `instructions` and by REST in [`GET /healthz`](rest.md#endpoints). The tools
that need a backend say so in their description and return `model_error` when it is missing.

`list_memories` supports the same default page size and opaque cursor contract as `Memory.list`.

### Input limits

Tool calls share the directory's 32 request slots with REST, before framework worker dispatch.
Overload returns the stable tool error envelope with `reason="capacity_exceeded"`,
`stage="admission"`, and `retryable=true`. Process-wide budgets apply across independent
directories; see [runtime limits](../operations.md#concurrency-and-overload). The serving host
must also bound HTTP connections and MCP sessions.

| Bound | MCP value |
| --- | --- |
| Content parts | 1 through 16 |
| One data-URL source string | 8,192 characters |
| One decoded `file_data` value | 8 MiB |
| Normalized text, including combined text parts | 65,536 characters |
| Search, answer, or page `limit` | 1 through 100 |
| Context `budget.max_chars` | 1 through 65,536 |
| Context `budget.max_items` | 1 through 100 |
| Context `budget.max_media_items` | 0 or more, or `null` |
| Serialized metadata | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |

MCP has no aggregate framing budget, but each inline media value is bounded before model or storage
work. It has no local-path input, remote fetch, large-file upload tool, capture-stream tool,
coordinate-frame transform, logical scope, or separate authentication policy; the MCP host owns
transport access control. Configured model backends may
impose a smaller aggregate budget, including the
[OpenAI inline limits](python-adapters.md).
