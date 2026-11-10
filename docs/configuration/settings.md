# Runtime settings

Control how SenseMem retrieves, packs evidence, ranks memories, and matches identities through
the `settings` mapping. Each setting below lists its default and effect.

For direct construction, pass `config=MemoryConfig(...)` to `Memory` or `Memory.from_plugins`.
Tracing uses the separate `tracer=` argument. [Configuration](../configuration.md) covers model slots.

## Retrieval and grounding

| Field | Default | Meaning |
| --- | --- | --- |
| `retrieval_mode` | `hybrid` | Dense+lexical, dense only, or lexical only |
| `index_quantization` | `none` | none/fp16/unrotated int8/rabitq; rebuilds from stored vectors on change |
| `minimum_relevance` | `0.10` | Query-relevance gate before reinforcement/decay |
| `ambiguity_margin` | `0.01` | Reject unresolved top-two ties at limit 1 |
| `evidence_budget_chars` | `24000` | Soft answer-grounding expansion allowance; `None` disables ranked-tail expansion |
| `query_vision` | `False` | Temporary query description/OCR routes using the vision slot |
| `index_speech` | `True` | Optional persisted speech-identity analysis when using a SpeechBackend |

Changing retrieval mode on reopen needs no reindex/re-embedding; use separate directories for
simultaneous comparisons. Lexical skips query embedding, while speech/vision query preparation
can still run and ingestion still stores embeddings.

Thresholds/margins accept [0,1]. Zero relevance or ambiguity disables that gate; zero identity
similarity is the most permissive match threshold. Relevance scales differ by mode: dense cosine,
native normalized full-text relevance, or hybrid fusion. Calibrate changed thresholds per mode.

`minimum_relevance` includes observation confidence and temporal boost, excluding reinforcement
and decay. Thus an eligible hit's final score can fall below the gate. `search(explain=True)`
shows `gate_relevance`, `retention_factor`, and `temporal_factor`. Natural temporal preference
is additive; explicit event filters provide hard exclusion.

`index_speech` reuses FunASR analysis without another call; it does nothing with no speech backend
or a plain transcriber such as OpenAI speech. Disabling it suppresses optional add-time matching,
not required transcript fallback.

## Evidence budgets

The ordinary answer window is the first `limit` hits, with a best-hit modality floor. The soft
budget can add ranked tail records, pricing each image/audio/video asset as 2000/4000/12000
characters. Mandatory raw records survive an undersized allowance; derived assertions require
their complete eligible evidence closure to fit. Shared records count once.

`None` removes ranked-tail expansion but source closure and enabled planning/linked expansion
can still add records. Lowering limit alone may not reduce the final prompt because tail packing
refills it. The [answer API](../api/python-memory.md#answer-grounding) owns source closure and
returned-hit semantics.

Character prices omit system/question text, provenance, labels, message framing, and actual media
tokenization. OpenAI answer text has a separate 4 MiB safety limit plus inline-media/video bounds.
`generation.max_tokens` limits output, not input. Provider context overflow raises permanent
`context_length_exceeded` (HTTP 413) without evidence-dropping retries. Exact token preflight
requires the endpoint's own counting protocol. [Request-byte fitting](providers.md#media-selection-and-byte-limits)
and the hard [compile budget](../context-compilation.md#budget) are separate controls.

## Recall planning

| Field | Default | Meaning |
| --- | --- | --- |
| `recall_planning` | `False` | Ask the answerer for a bounded recall program |
| `recall_set_budget_chars` | `30000` | Supplemental matched/semantic allowance after mandatory evidence |
| `recall_set_max_rows` | `60` | Cap matched set/sequence/entity rows, retaining the read's earliest rows |
| `recall_rounds` | `2` | Answer-attempt ceiling; 1 disables follow-up |
| `recall_query_expansion` | `False` | Enable distinct planner semantic searches; requires planning |
| `recall_max_queries` | `6` | Additional searches across steps/rounds, 1–32 |

An answerer without `plan_recall` cannot plan: opening warns once and asks retain ordinary search.
The first planner receives question and corpus digest with `attempted=""`, without initial hits.
After abstention, a later plan can receive a bounded previous-attempt note and, with expansion,
evidence excerpts as untrusted data. Plans have no keep/unresolved controls.

A best-effort abstention can replan; strict abstention can replan when query expansion is enabled.
A non-abstained answer stops. Empty/unusable plans fall back without requesting another round.
Replaceable answers buffer instead of leaking intermediate streamed text.

### Matched reads and completeness

Matched sets supplement the mandatory ranked window under the lower of set/evidence character
allowances. They add at most `recall_set_max_rows` and twice limit media rows; the mandatory
ranked window is outside that media cap. Omitted matched rows remain explicitly not shown.
A sequence presents final evidence in corpus time order, with SQLite insertion order for ties;
that is not proof of complete event history.

Visibility and valid/known time filter match/window reads before row limits. Metric distance
filters during hydration and preserves truncation accounting. Failed reads provide no match
count and are incomplete.

A predicate matching more than one fifth of active records, or more than four times limit in a
small corpus, contributes no matched rows. The reader is told to retain ranked evidence.
Each structural edge applies this selectivity rule independently.

`entity(name)` intersects a caller's identity scope before its limit. Same-name identities need
matching `scope.identity_id` to disambiguate; otherwise that entity step is unread and ranking
remains. Unknown names fall back to literal matching; unknown scoped identities match nothing.

### Semantic query expansion

Without expansion, similar(query,k) reads the original ranking and initiates no new search.
With it, distinct trimmed queries cache 100-hit rankings. Original-query deepening and failed
searches consume the same query ceiling; cached reads do not. Successful routes persist across
internal rounds at their greatest requested depth, in discovery order, without creating another
budget. New asks reset cache and ceiling.

All rewrites preserve caller scope, type, reference time, inferred temporal intent, and query
media. Routes offer one novel record each in turn, de-duplicate IDs, and add at most limit novel
rows per query. Their combined allowance is the lower set/evidence budget when both exist.
Semantic additions precede the optional original ranked tail; linked rows follow it. If none fits,
ordinary grounding is restored. Supplemental semantic media stop at twice limit.

Each new plan executes its own predicates; cached routes never form a cumulative complete set.
Packing cannot readmit matched rows dropped by their own caps. Evaluate support, answer quality,
and input cost before claiming an expansion benefit.

## Evidence expansion

| Field | Default | Meaning |
| --- | --- | --- |
| `evidence_expansion` | `False` | Follow exact query-media context and existing structural links |
| `evidence_expansion_max_rows` | `24` | Combined cap on exact-source context and linked additions |

No extra model call is made. Original query media matching a scoped-ranked source by exact asset
ID and SHA can add that source's immediate corpus neighbors, for at most limit occurrences.
Duplicate attachments de-duplicate. Similar-looking different bytes do not qualify. These rows
retain scope/type and have priority before semantic additions and ranked tail.

After those and the ranked tail, existing capture/person/place/lineage/evidence edges can add
records within remaining characters, the combined row cap, and twice limit supplemental media.
Each edge is selective independently; a broad identity edge cannot suppress a selective capture edge.
Capture lookup currently scans source IDs; the other edges are indexed.

Mandatory windows remain protected. With no character budget, expansion still observes row/media
caps. When the ranked tail fills the budget, structural additions have no room. Matched rows
omitted by their own bounds cannot reenter. Readers see these as linked records, never a complete
predicate result or licensed total. Report counts of actual additions, not merely linked candidates.

## Formation, lifecycle, and ranking

| Field | Default | Meaning |
| --- | --- | --- |
| `formation_history_max_rows` | `0` | Read-only history per target; maximum 128 |
| `formation_history_budget_chars` | `6000` | Positive whole-record content allowance |
| `independent_evidence` | `False` | Store-pinned experimental AND/OR support projection |
| `decay_half_life_days` | `None` | Positive query-time decay half-life |
| `reinforce_on_answer` | `True` | Rank reinforcement of backend-declared non-abstained hits |
| `memory_budget_records` | `None` | Active-record budget creating PRESSURE work; never automatic deletion |
| `query_failure_window_seconds` | `3600.0` | Window for repeated near-equal empty recalls |
| `query_failure_history` | `512` | Bounded retained empty-query signal rows |
| `retention` | Empty | Internal policy value; declarative files use top-level retention |

See [formation history and retention](lifecycle.md), [independent evidence](../typed-memory.md#experimental-independent-evidence),
[decay](../memory-types-time-and-decay.md#decay-and-reinforcement), and
[deliberation triggers](../api/python-control.md#consolidation_candidates).

## Identity thresholds

| Field | Default |
| --- | --- |
| `speaker_similarity` | `0.78` |
| `speaker_margin` | `0.05` |
| `face_similarity` | `0.363` |
| `face_margin` | `0.05` |
| `identity_link_min_assets` | `2` |

Calibrate these matching thresholds for the intended devices and population. A high threshold
fragments people; a low one can falsely merge them. Matching uses the best score
from up to 20 exemplars per identity, so pairwise calibration alone is insufficient. Margins reject
ambiguous competitors and cannot replace calibration, especially with one enrolled identity.

Use separate labelled enrollment/evaluation samples from the intended microphones, cameras,
environment, and population. Measure false matches/rejections at the deployed bank size and
recheck changed recipes or conditions. Identity scores are similarity signals; use host
authentication and authorization for access control.
