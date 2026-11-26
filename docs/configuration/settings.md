# Runtime settings

Control how SenseMem retrieves, packs evidence, ranks memories, and matches identities through the `settings` mapping. Each setting below lists its default and effect.

For direct construction, pass `settings=MemorySettings(...)` to `Memory` or `Memory.from_plugins`. Tracing uses the separate `tracer=` argument. [Configuration](../configuration.md) covers model slots.

## Retrieval and grounding

| Field | Default | Meaning |
| --- | --- | --- |
| `retrieval_mode` | `hybrid` | Dense+lexical, dense only, or lexical only |
| `index_quantization` | `none` | `none`, `fp16`, unrotated `int8`, or `rabitq`; rebuilds from stored vectors on change |
| `minimum_relevance` | `0.10` | Query-relevance gate before reinforcement/decay |
| `ambiguity_margin` | `0.01` | Minimum base margin for top-one retrieval; a reranker can resolve a close pair |
| `evidence_budget_chars` | `24000` | Soft answer-grounding expansion allowance; `None` disables ranked-tail expansion |
| `query_vision` | `False` | Temporary query description/OCR routes using the vision slot |
| `index_speech` | `True` | Optional persisted speech-identity analysis when using a `SpeechBackend` |

Changing `retrieval_mode` on reopen needs no reindex or re-embedding; use separate directories for simultaneous comparisons. `lexical` skips query embedding while speech/vision query preparation can still run and ingestion continues to store embeddings.

Thresholds and margins accept `[0, 1]`. Zero disables `minimum_relevance` or `ambiguity_margin`. `minimum_relevance` compares against `gate_relevance` (including observation confidence and temporal boost, excluding reinforcement and decay); inspect `search(..., explain=True)` to tune it.

`index_speech` reuses `SpeechBackend` analysis without another call and does nothing with a plain transcriber. Disabling it suppresses optional add-time speaker matching while preserving required transcript fallback.

## Evidence budgets

The ordinary answer window protects the first `limit` hits (with a best-hit modality floor). `evidence_budget_chars` can append ranked tail records, pricing each image/audio/video asset at 2000/4000/12000 characters. Mandatory raw records survive an undersized budget, while derived assertions require their complete eligible evidence closure to fit. Setting `evidence_budget_chars=None` disables ranked-tail expansion; source closure and enabled planning/linked expansion can still add records. See [answer grounding](../api/python-memory.md#answer-grounding).

Character prices are packing estimates, not provider token counts. OpenAI answer text also enforces a 4 MiB safety ceiling plus inline-media bounds; provider context overflow raises permanent `context_length_exceeded` (HTTP 413). See [request-byte fitting](providers.md#media-selection-and-byte-limits) and [compile budgets](../context-compilation.md#budget).

## Recall planning

| Field | Default | Meaning |
| --- | --- | --- |
| `recall_planning` | `False` | Ask the answerer for a bounded recall program for answering or context compilation |
| `recall_set_budget_chars` | `30000` | Supplemental matched/semantic allowance after mandatory evidence |
| `recall_set_max_rows` | `60` | Cap matched set/sequence/entity rows, retaining earliest rows |
| `recall_rounds` | `2` | Answer-attempt ceiling (`1` disables follow-up) |
| `recall_query_expansion` | `False` | Enable distinct planner semantic searches (requires `recall_planning=True`) |
| `recall_max_queries` | `6` | Additional searches across steps/rounds (`1`–`32`) |

An answerer without `plan_recall` warns once at open and falls back to ordinary retrieval. Either answer policy can replan after abstention when an earlier plan was usable and attempts remain; a non-abstained answer or fallback ends the operation, and streaming delivers only the selected answer.

### Matched reads and completeness

Matched sets supplement the mandatory ranked window under the lower of `recall_set_budget_chars` and `evidence_budget_chars`, capped at `recall_set_max_rows` and `2 * limit` supplemental media rows. Visibility and valid/known time filter match/window reads before row limits; predicates matching more than 20% of active records (or `4 * limit` in a small corpus) contribute no matched rows so ranking remains intact. `entity(name)` intersects caller identity scope before its limit and requires `scope.identity_id` to disambiguate same-name identities.

### Semantic query expansion

With `recall_query_expansion=True`, distinct planner queries cache 100-hit rankings across rounds within the same `ask()` under the shared `recall_max_queries` ceiling, preserving caller scope, type, reference time, inferred temporal intent, and query media. Routes contribute novel hits in round-robin order (up to `limit` per query) ahead of the ranked tail, bounded by `2 * limit` supplemental media rows.

## Evidence expansion

| Field | Default | Meaning |
| --- | --- | --- |
| `evidence_expansion` | `False` | Follow exact query-media context and existing structural links |
| `evidence_expansion_max_rows` | `24` | Combined cap on exact-source context and linked additions |

No extra model call is made. Query media matching a scoped-ranked source by exact asset ID and SHA can add immediate corpus neighbors, followed after the ranked tail by selective capture, person, place, lineage, and evidence edges within remaining character, row, and `2 * limit` supplemental media caps.

## Formation, lifecycle, and ranking

| Field | Default | Meaning |
| --- | --- | --- |
| `formation_history_max_rows` | `8` | Read-only history per target (`0` disables; max `128`) |
| `formation_history_budget_chars` | `6000` | Positive whole-record content allowance |
| `independent_evidence` | `False` | Store-pinned experimental AND/OR support projection |
| `decay_half_life_days` | `None` | Positive query-time decay half-life |
| `reinforce_on_answer` | `False` | Opt-in ranking update for backend-declared non-abstained hits |
| `memory_budget_records` | `None` | Active-record budget creating `PRESSURE` work (never automatic deletion) |
| `query_failure_window_seconds` | `3600.0` | Window for repeated near-equal empty recalls |
| `query_failure_history` | `512` | Bounded retained empty-query signal rows |
| `retention` | Empty | Direct-SDK policy value; declarative files use top-level `retention` |

See [formation and retention](lifecycle.md), [independent evidence](../typed-memory.md#experimental-independent-evidence), [decay](../memory-types-time-and-decay.md#decay-and-reinforcement), and [deliberation triggers](../api/python-control.md#consolidation_candidates).

## Identity thresholds

| Field | Default |
| --- | --- |
| `speaker_similarity` | `0.78` |
| `speaker_margin` | `0.05` |
| `face_similarity` | `0.363` |
| `face_margin` | `0.05` |

Calibrate matching thresholds for your target devices, environment, and bank size (matching compares against up to 20 exemplars per identity). Margins reject ambiguous competitors. Identity scores are similarity signals, not host authentication or access control.
