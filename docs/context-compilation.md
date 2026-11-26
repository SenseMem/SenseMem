# Context compilation

Give your agent structured memory that fits its context budget. `memory.compile(goal)` selects evidence and returns a `ContextBundle` organized into actors, facts, events, procedures, affect, and traits, with conflicts and missing-context notes.

Use the bundle in your own prompt or workflow. Compilation needs no generation model; `ask()` adds grounded answer generation when you want SenseMem to produce the answer.

## Compile a first bundle

Install the [local embedding extra](configuration.md#1-install-only-the-surfaces-you-use). This example needs no generation model:

```python
from sensemem import ContextBudget, Memory

config = {
    "data_dir": "./data/compile-example",
    "embedding": {
        "provider": "sentence-transformers",
        "model": "google/embeddinggemma-2",
    },
}

with Memory.from_config(config) as memory:
    memory.add("The spare key is in the blue toolbox.")
    bundle = memory.compile(
        "Where is the spare key?",
        budget=ContextBudget(max_chars=2000, max_items=8),
    )
    print(bundle.render())
    print(bundle.unknowns)
```

`bundle.hits` holds complete records; `bundle.render()` produces prompt text. Keep `unknowns` and conflict notes with that text so missing evidence remains visible. `AsyncMemory` provides the same operation. REST uses `POST /v1/context`, MCP uses `compile_context`, and the CLI uses `sensemem compile`.

Both `render()` and `compact().text` mark memory content, including procedures, as untrusted evidence. Pass these presentations as task context in a user or tool message, keeping host instructions separate.

## Contract

```text
compile(
    goal: ContentInput,
    *,
    budget: ContextBudget | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
    allow_partial_sources: bool = False,
    supplements: Sequence[ContextSupplement] = (),
) -> ContextBundle
```

Compilation starts with one retrieval round at candidate depth `max(100, 3 * budget.max_items)`, tripled when `min_confidence` or `freshness` is set. Each requested memory type gets its own index route; selecting every type uses the unfiltered route. Scope, relative time, and authoritative SQLite visibility match search.

Compilation preserves conflicts for the host to resolve and never generates an answer. Spoken-query preparation may cache a transcript, and an empty bundle records a bounded query-failure signal; neither creates a memory record. A full candidate window that still loses evidence reports `candidates_exhausted`.

### Temporary context from an application or agent

Pass completed tool results or agent observations as `ContextSupplement` values. They are request-local evidence with an explicit source, not memory records or instructions. Compilation does not embed, form, or persist them:

```python
from sensemem import ContextBudget, ContextSupplement

bundle = memory.compile(
    "What should I bring?",
    budget=ContextBudget(max_chars=4000, max_items=8),
    supplements=(ContextSupplement("Rain is forecast this afternoon.", source="weather-tool"),),
)
print(bundle.render())
```

An async supplier can use [`AsyncContextSource`](api/python-streams.md#asynccontextsource) while the pipeline continues, then pass its completed supplements to `AsyncMemory.compile()`.

Supplements take character and item slots in submission order before stored evidence is packed. Oversized supplements are skipped so smaller later ones can fit. `observed_at` and `expires_at` must be timezone-aware; supplements expired at `reference_at`, observed in the future, or outside the freshness window are excluded. `bundle.supplements` holds admitted values, and `omitted_supplements` plus `unknowns` explain exclusions. REST's `supplements` array and MCP's `compile_context` accept the same fields.

### Goal-directed recall

With `recall_planning=True` and a planner-capable answerer, compilation executes one bounded recall program for the goal. `recall_query_expansion=True` permits focused semantic queries under the same `recall_max_queries`, `recall_set_budget_chars`, and `recall_set_max_rows` limits used by answering, and `evidence_expansion=True` offers linked evidence. There are no answer-generation or replanning rounds.

Planned reads precede the original ranking for selection while still passing source closure, scope, consent, filters, and final budgets. An unavailable or invalid planner retains original evidence and reports `stage_skipped`; empty or truncated planned reads report `evidence_unavailable` or `candidates_exhausted`.

## Budget

| Field | Default | Meaning |
| --- | --- | --- |
| `max_chars` | `16000` | Ceiling on priced grounding, including text and media weights |
| `max_items` | `24` | Complete memories, excerpts, and admitted temporary supplements |
| `max_media_items` | `None` | Delivered native-media cap; `0` permits text-only source projections |
| `memory_types` | `None` | Exact type filter pushed into retrieval |
| `min_confidence` | `0.0` | Typed confidence threshold; untyped observations count as `1.0` |
| `freshness` | `None` | Positive `timedelta` relative to event end, event start, then creation time |
| `max_latency_ms` | `None` | Positive deadline for skipping optional enrichment |
| `selected_proofs` | `False` | Deliver selected current-knowledge witnesses instead of the full audit union |

Character/item/deadline bounds are positive integers, `max_media_items` is non-negative, and `min_confidence` is in `[0, 1]`. `reference_at` supplies the clock for relative time and freshness.

### Character accounting

| Grounding component | Charge |
| --- | --- |
| Full memory line | Rendered line plus newline, including ID, confidence, and validity |
| Excerpt line | Rendered line plus newline, including warning, source ID, and offsets |
| Text-only projection | Complete rendered parts, source ID, role labels, and omitted-media declaration; no native-media weight |
| Header and section headings | Rendered characters; the budget line is priced at its widest |
| Image / audio / video asset | 2000 / 4000 / 12000 characters |
| Asset with unresolved modality | 4000 characters |
| Named/provisional actor and selected-proof lines | Rendered characters and their headings |
| Temporary supplements | Rendered source, content, timestamps, and their section heading |

These are conservative packing weights, not provider token counts. The header is mandatory; a ceiling below its minimum raises `ValidationError`. Without diagnostic appendices, `len(bundle.render()) <= bundle.chars <= max_chars`; conflict/unknown blocks and the `Omitted:` trailer sit outside the ceiling so thin bundles still explain omissions.

### Text-only sources for media

The compiler tries complete media records first. When a raw observation has text parts but its native media exceeds the character or `max_media_items` cap, the compiler can deliver a `ContextProjection` in `bundle.projections` under `Text-only sources` without requiring `allow_partial_sources`.

A projection retains `source_memory_id`, complete ordered `parts` with original/derived role labels and omitted-media placeholders, path-free `omitted_assets`, timestamps, context, and `coverage="text_only"`. Each projection spends one item slot and its full rendered text cost without truncation. `budget_excluded` reports omitted native media.

This fallback applies only to an observation whose required closure is itself alone; it cannot replace a source required by a derived assertion, discharge a selected proof, or hide a required conflict. Full hits, exact raw excerpts, and projections of one source are mutually exclusive.

### Deadline

The compiler checks `max_latency_ms` after section assembly, before optional conflict reporting and affect/event enrichment. It does not cancel running work; skipped stages produce `stage_skipped` unknowns, and `elapsed_ms` and `deadline_exceeded` report timing.

## Sections

Typed kind selects the section first; generic records fall back to their memory type.

| Section | Records |
| --- | --- |
| `actors` | Entities and recognized named/provisional identities |
| `relationships` | Relations |
| `scene` | States |
| `facts` | Other semantic records |
| `episodes` | Episodic records except affect |
| `procedures` | Procedural records, including response policies |
| `affect` | Affect cues |
| `traits` | Traits |

For independent flat candidates, rank fills the top half of `max_items`; remaining slots give one place to each missed section when all spanned sections fit, otherwise rank controls the whole selection. Derived closures or conflicts require strictly ranked atomic groups. The bundle also reports included event bounds, sorted spatial `frames` and symbolic `places`, complete `hits` in rank order, and `excerpts` in selection order.

## Evidence-closed selection

A derived assertion is admitted with its complete recursive `context.evidence_ids` closure. Every source must satisfy scope, type, confidence, freshness, active/known-time, and consent rules; missing, filtered, cyclic, or traversal-bounded support withholds the assertion and reports `evidence_unavailable`. Closures spend item, character, and media budgets atomically with shared records counted once. `ask()` and `ask_stream()` use the same full closure under their separate soft evidence allowance; see [answer grounding](api/python-memory.md#answer-grounding).

### Selected delivery certificates

`ContextBudget(selected_proofs=True)` enables current-knowledge `bundle.proofs`, delivering acyclic `ContextProof` witnesses for complete AND assessments and raw captures instead of the full audit union. `bundle.proof_scoring` reports the instance's evidence policy across SDK, REST, MCP, and rendered text:

| Rule | Count and score |
| --- | --- |
| `capture_groups` (default instance) | After excluding eligible host-confirmed naming conditions, singleton assessment members use their stored `source_group_id`; each group contributes its maximum score and groups combine by noisy-OR. Joint assessments remain one alternative with their maximum score. |
| `independent_pair` (`independent_evidence=True`) | Complete path footprints establish independence; confidence is the maximum single-path or independent-pair score, with each path capped by its lowest node confidence. |

`ContextProofNode.source_group_id` is a SQLite-produced grouping declaration separate from `capture_id`. For model inferences bound to a stable identity, a matching host-confirmed naming condition remains a required delivered premise without counting as an experience observation. `ContextProof.structurally_independent_of()` compares structural footprints without typed context.

Stored `evidence_ids` keep the full audit union. Per anchor, proof search reads at most 256 ancestry nodes and 4096 clause-member rows, retains up to 64 witnesses, and limits visits/joins to 4096. Historical `scope.known_at` retains full-closure delivery.

![Complete support preserves AND premises, combines count and confidence witnesses, and charges shared sources once.](assets/tech-report/support-compilation.png)

## Exact raw excerpts

`allow_partial_sources=True` permits one exact dense-matched text projection when a pure-text raw observation cannot fit whole. Only raw text parts with stored selectors qualify; aggregate keys, lexical-only matches, media, and derived records do not.

An excerpt appears in `bundle.excerpts`, never `hits`, carrying the parent ID, matched index ID, and digest-bound `TextSpanSelector` with half-open Unicode code-point offsets. Hydration verifies stored digests against the authoritative parent before admitting the excerpt. Excerpts cannot satisfy evidence clauses or reinforcement citations, and a later derived closure replaces an excerpt only if the full record closure fits atomically.

### Compact presentation

`bundle.compact()` returns a frozen `ContextPresentation` with compact text and request-local symbols (`mN` for memories, `iN` for identities) preserving stable IDs, namespaces, structural roles, and coverage (`full`, `partial`, `text_only`, `reference_only`). `resolve(symbol, namespace=...)` decodes an alias; `resolve_citation(symbol)` accepts full, partial, and text-only memory citations and rejects reference-only or identity aliases. Compaction follows selection and leaves budgets, closure, `render()`, and `hits` unchanged.

## Named and provisional actors

Included memories resolve their semantic identity or asset-level speech/face identity edge without requiring a naming assertion to rank. Current naming uses the same valid/known-time view; present consent withdrawal still suppresses actor rendering.

| Entry | Fields |
| --- | --- |
| `NamedActor` | `identity_id`, `name`, supporting included `memory_ids`, `naming_assertion_id` |
| `ProvisionalActor` | `identity_id` and supporting included `memory_ids`; no name |

Actor entries consume no item slot and spend remaining character budget after ranked hits (named before provisional); omitted lines report `budget_excluded`.

## Affect cues

`bundle.affect` holds `AffectCue` entries (hit fields plus up to eight co-occurring `event_ids` sharing at least one source observation under the same visibility/scope bounds). **Shared observation means co-occurrence, not causation.** Co-occurring event IDs spend no extra item/character budget and may be skipped after `max_latency_ms`.

### Attribution delivery

Typed `affect_target` and `affect_elicitor` relations enter `relationships` as separate claims. Before admitting a ranked attribution, the compiler reads all active scoped affirmations and denials for that exact affect/role/object group from SQLite and admits the group with its required cue, object, and source evidence atomically. Truncated lookups, unavailable peer evidence, consent/filter exclusions, budget limits, or a passed deadline withhold the attribution and report the applicable unknown while leaving independently deliverable cues in `affect`.

## Conflicts

Functional lineages (`STATE` and user-stated `TRAIT`) have one standing value. For an eligible ranked functional claim, SQLite reads complete its lineage with all eligible competing values under the same scope, identity, place, spatial frame, valid/known time, closure, and consent rules. All distinct representatives and their sources must fit atomically; otherwise every affirmative member is withheld and `evidence_unavailable` is reported so filters or budgets cannot hide known competition.

## Unknowns and rendering

| Unknown kind | Meaning |
| --- | --- |
| `budget_excluded` | A bound excluded candidates or actor lines |
| `evidence_unavailable` | Required support or conflict completion could not be delivered |
| `section_empty` | A requested type leaves a relevant section empty |
| `scope_empty` | Supplied scope matched nothing |
| `modality_unsupported` | Goal media cannot be searched by this composition |
| `stage_skipped` | Deadline skipped optional enrichment |
| `consent_withheld` | Current policy withheld actors; their source memories may remain eligible |
| `candidates_exhausted` | The full candidate window still lost evidence |

`render()` deterministically emits the goal, reading guide, reference time, budget, non-empty sections in order, conflict/unknown blocks, and an omission trailer. Discover configured operations through `memory.capabilities`; see [Memory capabilities](api/python-values.md#memory-capabilities).
