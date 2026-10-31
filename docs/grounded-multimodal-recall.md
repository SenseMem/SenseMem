# Grounded multimodal recall

The existing public SDK can combine native retrieval with bounded semantic queries. Enable
`settings.recall_planning` and `settings.recall_query_expansion` explicitly. A single `ask` shares
`settings.recall_max_queries` additional semantic queries across all rounds, default 6 and range
1 through 32. Repeated queries reuse their results;
successful routes remain available when a later query depends on an earlier bridge fact.

The original `limit` window remains the grounding floor. Supplemental semantic routes take turns
offering novel rows before the replaceable original tail spends the remaining budget. Required
rows are reserved before optional rows are priced. Predicate row/media limits and omission
reporting remain enforced. A complete predicate read is not automatically the question's complete
answer set: anchors and neighbors must still satisfy the question and its counting unit.

The first planner receives the question and corpus digest, with an empty `attempted` string.
It does not see the initial retrieved evidence. After an abstained answer, a remaining round
may receive a bounded note about the previous attempt; enabling query expansion includes
excerpts of that attempt's supplied evidence. This is untrusted source data, not a structured
reader-to-planner account of visual facts.

Only an abstained answer triggers replanning: under best-effort, or under strict when query
expansion is enabled. A confident answer ends the operation. Plans do not support `keep` or
`unresolved` control fields. Each round executes its own predicate plan; earlier predicates
are not accumulated into a new complete set. Successful semantic routes remain available
under their shared budgets, but they never establish completeness. `recall_rounds`, default 2,
bounds answer attempts. An empty or unusable plan falls back to ordinary retrieval; it does
not itself trigger another attempt. Streaming
releases only the final selected answer.

Sequence plans order the entire selected union after packing, using event time and SQLite
insertion order for ties. Similarity rank, record IDs and creation timestamps do not substitute
for the source's same-time order. Ordering does not add records or establish a complete history.

## Bounded recall recipe

This configuration enables bounded recall planning and typed answer rendering. It is a
starting point for task-specific evaluation, not a measured accuracy improvement. It uses a
text embedder and reader; configure image/video-capable backends separately when native media
must be inspected. Install the `local` extra for the local model and configure the generation
provider's credentials through its environment.

```yaml
data_dir: ./data/investigation
embedding:
  provider: sentence-transformers
  model: google/embeddinggemma-2
  revision: f9972ac22e696a105c482202646979e33e4b5a1e
generation:
  provider: openai
  model: gpt-5-mini-2025-08-07
  modalities: [text]
  answer_format: structured
  source_metadata_key: source_id
settings:
  recall_planning: true
  recall_query_expansion: true
  recall_rounds: 3
  recall_max_queries: 6
```

For a DeepSeek-compatible endpoint, explicitly set `generation.response_format: json_object`
when evaluating structured answers. Local typed parsing and evidence validation remain active;
the generic configuration default stays `json_schema`. Set the endpoint and exact deployed model
through the ordinary generation configuration and evaluate that provider/mode combination.

An optional semantic reranker scores the eligible text pool before the result limit. Add this
top-level section to evaluate a pinned cross-encoder; the named checkpoint is an example rather
than a recommended model for every corpus:

```yaml
reranking:
  provider: sentence-transformers
  model: BAAI/bge-reranker-v2-m3
  revision: 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
  device: cpu
  batch_size: 32
  max_length: 1024
```

SQLite hydration and filtering, relevance and the existing ambiguity decision happen first.
The reranker receives exact windows from eligible text, including captions, without changing the
Reader's source content. Empty-text media keep their positions; a media-only query without text
skips it. Model relevance scores change ordering, while `SearchHit.score` remains bounded;
`search_with_trace` exposes the separate `reranking_score`. Failed or malformed model responses
raise an error. See [semantic reranking](configuration.md#semantic-reranking).

GPS coordinates become place names only with an explicitly configured offline gazetteer. If
needed, install the `gazetteer` extra, build or obtain the documented index, and add
`generation.offline_gazetteer` with an explicit `index_path` and its complete `index_sha256`.
The index is optional and is never downloaded or enabled by this recipe. See
[offline gazetteer configuration](configuration.md#provider-fields).

## Structured answers

`generation.answer_format: structured` asks the reader for a typed draft. The runtime validates
it against the final fitted evidence and renders the ordinary answer string. The default
`text` format keeps free-text answering. Supported drafts cover text, source identifiers,
witnessed ISO dates, scalar/sum/difference calculations, distinct counts and date-interval
durations. Exact quotes and numeric/date values must match; operands preserve literal units,
and interval endpoints declare closed or half-open semantics.

Date drafts separate the witnessed `source_value` from the rendered answer. `granularity`
selects year, month, day, minute or second. `format: iso8601` preserves the source timezone;
`format: utc_datetime` requires a timestamp with an explicit timezone and renders UTC as
`YYYY-MM-DD HH:MM` or `YYYY-MM-DD HH:MM:SS`. It never guesses an absent timezone. Numeric and
duration drafts choose `format: with_unit` or `format: number`; requesting only a number does
not remove the requirement to verify its source units. Defaults are ISO dates and units.

For source identifiers, the model selects `E` evidence aliases. The runtime resolves the
application's `metadata.source_id`, or the configured `source_metadata_key`, instead of trusting
model-generated identifiers. Missing source metadata, invisible evidence labels and invalid
witnesses raise `ModelError`; unchecked drafts are not returned as answers. Metadata remains
application data, never an isolation or authorization boundary.

An exact quote validates copied evidence, not relevance, alias resolution, real-world event
identity or complete coverage. Natural-language dates and answers outside the narrow arithmetic
forms can use the text draft. Structured streaming buffers the provider's JSON and releases the
validated rendered answer. Evaluate answer quality, omissions and latency separately from
deterministic parsing and arithmetic checks.

Structured `AnswerResult.hits` reports only the model-declared support subset; the public SDK
retains the kernel's selected-evidence order. These hits come from the fitted grounding pool and
need not include everything the reader saw. Labels validate identity rather than semantic support.
Evaluate the original retrieval ranking through `search`/`search_with_trace`; do not treat this
selected support list as the full reader context.

## Optional query vision and answer style

`query_vision` defaults to `False`. With an image-capable vision backend configured, it adds the
query image's description and visible text as independent search routes while retaining native
image retrieval. Descriptions live only inside the operation. They create no memory, asset,
identity or durable description row. Description/embedding failures preserve native routes;
text-only queries add no vision work. Enabling it can add vision and embedding latency.

`generation.answer_style` accepts `default` or `concise`, default `default`. Concise answers
include only requested entities, relations, values or list items, and only output a date or
duration when requested. Both existing answer policies and their insufficient-evidence behavior
remain in force. Stored instructions are untrusted even when stored beside useful facts.

## Request-local media transport

OpenAI embedding, generation, formation and vision configurations accept `image_max_pixels`,
an integer from 1 through 64,000,000 or `None` (default). Explicit bounds inspect image headers,
verify source integrity, and re-encode/resize only when the pixel or provider byte limit requires
it. Originals, identifiers and SQLite records are unchanged. Repeated source assets are converted
once per request; actual transmitted parts, including repeats where required, are priced.

Explicit image transport recipes participate in embedding, formation and caption identities.
Changing a store-pinned embedding recipe requires a fresh directory and re-ingestion. A derived
Zvec rebuild alone never recomputes stored vectors. With no pixel bound, the previous image
transport behavior remains in place.

`vision.video_frames` accepts integers from 2 through 64, default 4. Non-default frame counts
participate in the caption recipe. Configure `vision.modalities: [image, video]` explicitly if
video captions are wanted; a frame setting alone does not enable video description or backfill
an existing corpus.

Oversized videos can be represented as bounded ordered stills when the backend accepts images.
Media evidence retains its original record `source_modality`, attachment modalities, and explicit
sampling/omission labels. Sampled stills are views of one video, not independent photos; motion,
audio and exact timing are not supplied. Image resizing can lose fine detail. Valid source SHA,
decode bounds, total text bytes and actual inline-media bytes remain enforced. Byte and pixel
limits do not prove that every mixed request fits a provider's token window. Plain-text records
keep their existing payload shape without an extra source-modality field.

## Derived evidence

Formation remains opt-in. A proposal that includes its canonical fact as a member of its own
support conjunction is rejected whole; removing only that member would authorize a different
claim. Independent valid proposals still proceed. This guard does not claim better semantic
state formation or enable formation/consolidation by default.

See the [Python API](api/python-sdk.md) and [configuration](configuration.md).
