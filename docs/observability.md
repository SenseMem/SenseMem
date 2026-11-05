# Observability

For lifecycle and recovery, see [Operations](operations.md).

## Enable tracing

SenseMem depends on `opentelemetry-api`; without an SDK, spans are no-ops. Install the existing
observability extra and configure a provider before constructing `Memory`, or inject a tracer:

```bash
uv add "sensemem[local,observability]"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from sensemem import SentenceTransformersEmbedder, Memory

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

with Memory(
    "./data",
    embedder=SentenceTransformersEmbedder(
        model_id="google/embeddinggemma-2",
    ),
) as memory:
    memory.add("Remember this")
    memory.search("Remember")

provider.shutdown()
```

## Span hierarchy

Every span sets `sensemem.span.kind` to `operation`, `stage`, `model`, or `transport`.

| Level | Useful span names |
| --- | --- |
| Operation | `sensemem.add`, `.add_many`, `.search`, `.ask`, `.delete`, `.reindex`, `.optimize` |
| Stage | `sensemem.content.prepare`, `.retrieve`, `.storage.*`, `.index.*`, `.retrieval.score_completion`, `.retrieval.rank`, `.identity.*` |
| Model | `sensemem.model.embedding`, `.transcription`, `.face`, `.generation`, `.formation`, `.vision` |
| Transport | `sensemem.http.request`, `sensemem.mcp.request` |

Each `add_stream()` item creates an ordinary `sensemem.add` span. `search(..., explain=True)` and
speculative prefetch use `sensemem.search`. `AsyncMemory` preserves tracing context across its
worker thread. Both buffered `AsyncMemory.ask()` and `AsyncMemory.ask_stream()` record their
initial executor queue delay.

Inside `sensemem.ask`, `sensemem.retrieve` measures the complete prerequisite-to-ranked-hits
leg: question content preparation, reference-time and temporal parsing, scope normalization,
required query speech work, query embedding, index lookup, and ranking. Its existing
`sensemem.content.prepare`, model, `sensemem.index.search`, and `sensemem.retrieval.rank`
children remain available for attribution.

Hybrid searches emit `sensemem.retrieval.score_completion` only when a parent admitted by the
lexical route was absent from the bounded ANN result. The stage covers reading that parent's
persisted document vectors from SQLite and exact local cosine scoring; it makes no model request.

`sensemem.index.sync` retains one parent span and separates its durable stages without changing
the SQLite-before-Zvec-before-ack order:

| Child stage | Work measured |
| --- | --- |
| `sensemem.index.sync.sqlite.read` | Read and hydrate the durable SQLite outbox batch. |
| `sensemem.index.sync.zvec.apply` | Delete or upsert the derived Zvec documents. |
| `sensemem.index.sync.zvec.flush` | Flush the applied Zvec changes; present only in the drain that reaches the flush bound, in `optimize()` and `reindex()`, and at `close()`. |
| `sensemem.index.sync.zvec.optimize` | Run the conditional Zvec optimization check after a flush. |
| `sensemem.index.sync.sqlite.ack` | Acknowledge the flushed outbox rows in SQLite. |

## Capture timing

Capture acknowledgement, settle duration, and time to settled are three different numbers and
are measured separately. `sensemem.capture` and `sensemem.settle` are distinct operation
spans, and the settle span carries the capture-to-settled interval its batch closed:

| Attribute | Meaning |
| --- | --- |
| `sensemem.capture.records_settled` | Records this call made searchable. |
| `sensemem.capture.records_failed` | Records that failed and kept their queue row. |
| `sensemem.capture.max_time_to_searchable_ms` | Longest capture-to-settled interval this call closed, over every row settled including `add()` recoveries. Absent when nothing settled. |

## Model usage and provenance

Model spans include request model, module, batch size, modalities, and
`sensemem.model.request_count`. Provider-reported usage is recorded without estimation:

`sensemem.model.request_count` counts logical adapter calls issued by SenseMem. Retries hidden
inside a third-party SDK transport remain included in latency, but cannot be counted separately;
token totals and completeness cover observable logical responses, not an earlier hidden attempt for
which the provider returned no usage.

| Attribute | Meaning |
| --- | --- |
| `sensemem.token_usage.expected_request_count` | Requests expected to report token usage. |
| `sensemem.token_usage.reported_request_count` | Requests that supplied usage. |
| `sensemem.token_usage.complete` | Every expected request supplied a usable total. |
| `sensemem.token_usage.total_tokens` | Exact total when complete; otherwise only an exact reported lower bound. |
| `sensemem.token_usage.input_tokens.complete` | Every expected request supplied an input-token component. |
| `sensemem.token_usage.output_tokens.complete` | Every expected request supplied an output-token component. |
| `sensemem.token_usage.cached_input_tokens.complete` | Every expected request supplied a cached-input component. |
| `sensemem.token_usage.reasoning_output_tokens.complete` | Every expected request supplied a reasoning-output component. |
| `sensemem.token_usage.input_tokens.<modality>` | Exact input tokens for text, image, video, audio, or unattributed input. |
| `sensemem.token_usage.output_tokens.<modality>` | Exact output tokens for the same modality set. |
| `gen_ai.usage.cache_read.input_tokens` | Provider-reported cached input tokens. |
| `gen_ai.usage.reasoning.output_tokens` | Provider-reported reasoning output tokens. |
| `sensemem.token_usage.audio_seconds` | Provider- or runtime-reported processed audio duration. |

Operation spans roll up descendant model usage. Local backends report request counts and available
audio duration but do not invent token counts. Cached-input and reasoning-output tokens participate
in the same model-to-operation rollup. Embedding model spans set `sensemem.embedding.task` to
`retrieval.document` or `retrieval.query`, so ingest and retrieval inference can be separated.

OpenAI-compatible model spans record `gen_ai.response.model` and
`gen_ai.openai.response.system_fingerprint` only when the response supplies a non-empty value.
Streaming inspects the final usage chunk too. `sensemem.model.response_models` and
`sensemem.model.response_system_fingerprints` retain all distinct values when one span covers
several responses; the scalar field is `mixed` instead of silently assigning the whole span to the
last response. These are response provenance, not a copy of the request or response body.

Provider request IDs are recorded as `gen_ai.request.id` for a single request and collected in
`sensemem.model.request_ids` when one model span covers multiple requests. They are taken from
the SDK response or provider error, including failure paths; an absent ID is left absent rather
than invented. A local trace ID identifies the SenseMem operation, while the provider request
ID identifies the upstream call for support. Keep both in protected operator telemetry.

## First output and transport timing

The first-output clocks deliberately use different scopes and units:

| Attribute | Unit | Boundary |
| --- | --- | --- |
| `sensemem.operation.time_to_first_token_ms` | Milliseconds | Start of the complete `sensemem.ask` operation through its first non-empty visible text yield, including retrieval and generation setup. Absent if no visible text is yielded. |
| `sensemem.model.time_to_first_token` | Seconds | Generation-model request through its first non-empty text delta. This preserves the generation-only model baseline and is absent when no model token is produced. |
| `gen_ai.response.time_to_first_chunk` | Seconds | OpenAI-compatible streaming request through the first provider chunk, even when that chunk carries no visible text. |
| `sensemem.async.queue_time_ms` | Milliseconds | Initial buffered or streaming `AsyncMemory.ask*` executor queue delay. It is also included in the operation TTFT, while remaining available for decomposition. |

Transport spans measure server-side delivery separately from model and operation timing:

| Transport | Attributes and boundary |
| --- | --- |
| REST | `sensemem.http.request` records the low-cardinality `http.route` template, `sensemem.transport.response_mode`, `sensemem.transport.server_time_to_headers_ms`, `sensemem.transport.server_time_to_first_body_byte_ms` when a non-empty body is sent, and `sensemem.transport.server_total_ms`. All three times are milliseconds from server request handling; they are not client-observed network latency. |
| MCP | `sensemem.mcp.request` records buffered `sensemem.transport.server_total_ms` in milliseconds. MCP does not stream tool results, so it does not claim TTFT, first-chunk, or first-body timing. |

Providers may also record `gen_ai.response.finish_reasons` on generation model spans.

## Degradation and recognition

Watch these degradation and recognition attributes:

- `sensemem.grounding.media_diagnosis.valid`, `.selected`, and `.truncated` describe on-demand
  selection; `.error_reason` is present when the failure has a classified reason. Invalid or
  truncated diagnosis adds no media.
- `sensemem.embedding.elided_parts` counts retrieval keys rejected because media exceeded the
  embedding model's inline limit.
- `sensemem.embedding.video_sampled_inputs` counts video inputs embedded as four ordered stills
  because the model's prompt exceeded its context.
- `sensemem.grounding.media_elided_hits` and `sensemem.grounding.dropped_hits` count evidence
  removed from OpenAI grounding requests.
- `sensemem.recall.shape`, `sensemem.recall.ops`, `sensemem.recall.exhaustive_rows`,
  `sensemem.recall.complete`, `sensemem.recall.replan`, and `sensemem.recall.fallback`
  describe one `sensemem.recall` stage, present only when `recall_planning` is on. Every
  planning failure resolves to the same fallback plan, so `fallback` is the only signal that the
  planner did not decide that question, and `complete` is what licenses a count over the set.
- Experimental `recall_query_expansion` adds a `sensemem.recall.ground` stage.
  `sensemem.recall.query_count`, `query_failures`, and `query_limited` are cumulative per ask;
  use the final round rather than summing them. Other `sensemem.recall.query_*` counts describe
  that round: `candidates` is the distinct returned query hits, `new_candidates` excludes the
  original ranked pool, `selected` counts novel hits offered after per-query row caps,
  `grounded` counts actual additions, `budget_dropped` counts selected hits rejected by character
  or media caps, and `omitted` counts all novel query hits left out. A zero `new_candidates`
  means no expansion beyond the original pool, not proof of corpus completeness. These counts
  precede provider-side media packing and cannot establish correct reader use.
- `sensemem.formation.refused_proposals` counts the proposals the kernel refused to ground,
  totalled over the whole operation span and present only when formation ran; the model adapter's
  `sensemem.formation.dropped_proposals` counts the ones it could not read at all.
- `sensemem.identity.faces` and `sensemem.identity.speakers` report observations, distinct
  identities, existing matches, creations, and cache status for each analyzed asset.

## Trace content

Ordinary spans omit memory text, media bytes, paths, metadata, request and response bodies, and
provider exception details. Grounding diagnostics can record memory and asset IDs; treat exported
traces as private operational data. Apart from the non-content response model, fingerprint, and provider request IDs
above, a failed span
receives only generic error status. Use `search(..., explain=True)` for one retrieval investigation and
the structured error envelope for failures.
