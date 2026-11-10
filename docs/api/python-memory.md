# Python memory operations

Write searchable records with `add()`, defer inference with `capture()`, and choose
`search()`, `compile()`, or `ask()` to bring evidence back to a task. This reference defines
the arguments, results, and lifecycle of each `Memory` operation.

See [construction and content](python-sdk.md), [identity and data lifecycle](python-identity.md),
[control operations](python-control.md), and [async streams](python-streams.md) for related APIs.

| Task | Methods |
| --- | --- |
| Write searchable records | [add](#add), [add_many](#add_many), [add_stream](#add_stream) |
| Defer model work | [capture](#capture), [settle](#settle), [pending_captures](#pending_captures) |
| Recall or answer | [search](#search), [ask](#ask), [ask_stream](#ask_stream), [compile](#compile) |
| Read or maintain | [get](#get), [list](#list), [reinforce](#reinforce), [delete](#delete), [reindex](#reindex), [optimize](#optimize) |

## add

```text
add(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
) -> MemoryRecord
```

Adds one content-addressed record, searchable on return. Repeating completed canonical input
is idempotent; pending capture/formation work can still run on retry. Original media remains stored.

## add_many

```text
add_many(
    contents: Sequence[ContentInput],
    *,
    occurred_at: Sequence[datetime | None] | None = None,
    occurred_end: Sequence[datetime | None] | None = None,
    metadata: Sequence[Mapping[str, object] | None] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: Sequence[ObservationContext | None] | None = None,
) -> tuple[MemoryRecord, ...]
```

Uses one model batch and SQLite transaction. Per-item optional sequences must match content count.
Empty input returns `()`. At most 1024 inputs are accepted before model/storage work.
Use streams for larger sets.

## add_stream

```text
add_stream(
    contents: Iterable[ContentInput | StreamInput],
    *,
    capture: bool = False,
) -> Iterator[MemoryRecord]
```

Requests one item at a time and yields each durable/searchable record before pulling another.
A later failure preserves earlier writes and identifies `contents[N]` in error subject.
With `capture=True`, new records are durable/readable but await settlement; existing readiness stays.

## capture

```text
capture(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
) -> MemoryRecord
```

Returns after SQLite commit and before model work. New records are available to get/list and
invisible to search/ask/compile until settled. Existing searchable records retain vectors/readiness.
IDs match add for the same input. The same embedder-capability boundary rejects unusable input.

## settle

```text
settle(
    *,
    limit: int = 100,
    max_attempts: int = 3,
    memory_ids: Sequence[str] | None = None,
) -> int
```

Attempts deferred speech/transcription, embedding, index apply, and formation in enqueue order.
It appends derived text without replacing caller text. Successful rows leave the queue; failures
retain stage/reason/attempts while other rows can finish. The first failure is raised after the
batch, with its memory ID as subject.

Rows reaching `max_attempts` are skipped. Explicit `memory_ids` ignore that ceiling, skip unqueued IDs,
and let the host retry or prioritize records. Concurrent callers claim each record once;
disjoint work uses bounded workers. Plain text can embed in a batch. Index failures retain queue
and outbox; retry applies committed vectors without re-embedding.

Add/add-many recover a queued record regardless of attempts. Opening, searching, answering,
compiling, and closing never settle implicitly; the host owns scheduling.

## pending_captures

```text
pending_captures(
    *,
    limit: int = 100,
    memory_ids: Sequence[str] | None = None,
) -> tuple[PendingCapture, ...]
```

Returns up to limit oldest pending rows. `PendingCapture` includes `memory_id`, `enqueued_at`,
attempts, `last_error`, and awaiting: enrichment means no searchable vectors; formation means the
record is searchable and owes typed formation. A short-lived formation row under add is expected.

Use `memory_ids` to inspect specific rows. Absence means settled or never stored; get distinguishes
the two. See [the backlog runbook](../operations.md#capture-backlog-before-a-snapshot).

## search

```text
search(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    scope: RetrievalScope | None = None,
    explain: bool = False,
) -> SearchResult
```

Returns `SearchResult` with ranked hits and optional trace. Explain preserves hits and reports
IDs, score components, ranks, and rejections, excluding query/content/metadata/media/vectors/paths.

`reference_at` is timezone-aware; omission uses the resolved query clock or current UTC. Explicit
occurrence bounds are half-open event-overlap filters, exclude missing event times, and require
an increasing interval. Natural-language dates are a soft preference; coarse clock phrases need
not resolve a window.

Scope supplies valid/known time, identity, symbolic place, or matching-frame metric distance.
Near/radius must be paired and radius is non-negative. SQLite reapplies authoritative filters.
See [time and scope](../memory-types-time-and-decay.md).

Lexical-only candidates can receive exact persisted-vector scoring. Quantized modes rescore all
visible candidates from FP32 parts/query keys before relevance gating and ranking, using a wider
candidate pool. This cannot make ANN discovery exhaustive. Trace `index_ids` identify candidates
that admitted a parent, not every rescored part or route. Exact-score ties preserve an existing
match when possible; compiler excerpts follow the winning dense part.

## ask

```text
ask(
    question: ContentInput,
    *,
    limit: int = 5,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
    link_identities: bool = True,
    answer_policy: AnswerPolicy = "strict",
) -> AnswerResult
```

Requires an answerer. The resolved question clock is appended after routing whenever text reaches
the answerer, including transcribed speech. Returned hits describe delivered/declared evidence,
not the original retrieval ranking.

### Answer policy

Strict abstains on insufficient evidence with a readable fallback and reason. Best-effort requests
the most likely supported answer or offered choice, marking low confidence. Its terminal result
can therefore contain a useful answer and `abstained=True` together. With no hits, best-effort still
calls the model and reports `no_evidence`. Recognized marker text is removed from the terminal answer.

`link_identities=True` permits corroborated voice/face merges during answer-time recognition, logged
and reversible. `False` still recognizes faces but commits no merge or new link.

### Answer grounding

The ranked limit window is protected, with its best-hit modality floor. A modality absent from
the first limit hits can contribute its best hit in the last slot. `evidence_budget_chars` widens
this window with ranked records, pricing image/audio/video at 2000/4000/12000 characters.
`None` disables that tail, while optional planning/linked reads and source closure can still add.

Derived assertions require their complete recursive `evidence_ids` closure under the same scope
and memory-type filter. Missing, filtered, cyclic, or bounded support withholds the assertion.
Shared records count once. Mandatory raw observations retain the soft-budget guarantee;
derived assertions inside the initial window may still be withheld. With no character budget,
closure can add beyond limit. Selected-proof delivery belongs only to compile.

Media selection, transport limits, or fallback withholding a required supplied source or its routed
media also withhold the dependent assertion. Raw evidence can remain. Assets stay leased through
completion/failure, including concurrent deletion. Predicate-completeness notes reflect closure
packing; later transport omission explicitly marks the delivered set incomplete.

Default text answers report the fitted grounding pool; structured answers report only declared
support, intersected with grounded hits and retaining kernel order. Labels do not establish semantic
support. Use search to evaluate ranking.

When any selected hit has typed context, all selected hits receive E labels. Raw-only text answers
retain their unlabelled format; structured answers always label fitted evidence. Validity,
recording, and occurrence times remain distinct.

Direct adapters can label omitted cited sources S without claiming delivery. Beyond 64 unique
omitted support IDs, bounded counts/pairwise overlaps replace individual labels; these cannot
determine higher-order unions. `supporting_record_count` counts unique `evidence_ids`, not captures
or independent corroboration. Source origin and supersession are separate relations unless also
cited. Omitted media is counted by modality beside retained text.

Non-abstained results reinforce backend-declared canonical hits when `reinforce_on_answer=True`.
Abstention records query failure without reinforcement. Set the option false for independent
evaluation questions. Declared use is not verified support.

Planning and packing controls live in [runtime settings](../configuration/settings.md).

## ask_stream

```text
ask_stream(
    question: ContentInput,
    *,
    limit: int = 5,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
    link_identities: bool = True,
    answer_policy: AnswerPolicy = "strict",
) -> Generator[AnswerChunk, None, AnswerResult]
```

Runs the same retrieval/grounding/reinforcement path and yields `AnswerChunk` text deltas followed
by one terminal result chunk. Chunks carry text or result, never both. With text output, provider
deltas can include the abstention marker; the terminal result carries the cleaned answer.

```python
answer = None
for chunk in memory.ask_stream("Where is the red toolbox?"):
    if chunk.result is None:
        print(chunk.text, end="", flush=True)
    else:
        answer = chunk.result
```

Validation happens at call time. Non-streaming backends produce one complete delta. Consume to
the terminal chunk or close an abandoned generator. A caller's pause between chunks remains inside
the model span; reasoning before visible text still contributes first-token latency.

Only one answer is delivered. Replaceable planned rounds buffer until selected; structured drafts
buffer until validation/rendering. First planning gets no initial evidence; later abstained rounds
may receive a bounded attempt note/excerpts. Best-effort or strict-with-expansion can replan within
`recall_rounds`, while non-abstention ends the operation. Predicates do not accumulate complete sets.

`AsyncMemory.ask_stream` returns an async generator. One reusable worker executes its entire stream;
eight are available per `Memory` across wrappers/REST. Overflow raises retryable `capacity_exceeded`
on first pull. Terminal consumption returns capacity; aclose cleans up an abandoned stream.

## compile

```text
compile(
    goal: ContentInput,
    *,
    budget: ContextBudget | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
    allow_partial_sources: bool = False,
) -> ContextBundle
```

Retrieves once and returns `ContextBundle` without generation or conflict resolution.
It may cache a spoken-query transcript and record an empty-query signal. The complete contract,
budget prices, source closure, selected proofs, excerpts, compact presentation, actors, and
unknowns are owned by [Context compilation](../context-compilation.md).

## get

```text
get(memory_id: str) -> MemoryRecord
```

Returns the canonical record, including hidden/retired context. Unknown IDs raise
`MemoryNotFoundError`.

## list

```text
list(*, limit: int = 100, cursor: str | None = None) -> Page
```

Returns a `Page` of records using an opaque keyset cursor. Read `next_cursor` to continue.

## reinforce

```text
reinforce(memory_ids: Sequence[str]) -> int
```

Records explicit positive ranking feedback, de-duplicates IDs, skips missing records, and returns
the existing distinct count updated. It writes no reversible operation row and differs from
`MemoryIntent.REINFORCE`, which adds source evidence. See
[decay and reinforcement](../memory-types-time-and-decay.md#decay-and-reinforcement).

## delete

```text
delete(memory_id: str) -> bool
```

Physically removes a record from get/list/retrieval and returns whether it existed.
Repeated deletion returns `False`. It does not provide secure erasure; related text can remain in
other records, operation payloads, or backups.

| Removed | Retained |
| --- | --- |
| Record, typed/version/evidence/clause history, formation tracking, capture queue | Append-only operation rows, potentially including proposal text/rationale |
| Embeddings and derived vectors through the durable outbox | Independent records and their vectors |
| Record/asset links and orphaned descriptors/blobs | Media still referenced by another record |
| Analyses/observations/transcripts keyed on a removed asset | Assets kept by remaining records |
| Unnamed, unlinked identities with no remaining observations or registry claims | Named/merged identities; use `forget_identity` to erase a person |
| Derived records losing every complete grounded support clause | Claims with surviving alternative clauses, which may still paraphrase the deleted source |

Deleting one clause member retires that entire conjunction and rebuilds the audit union.
Physical deletion creates no rollback operation. Source removal recomputes dependent confidence,
visibility, and validity rather than deleting unrelated source records.

## reindex

```text
reindex() -> int
```

Rebuilds Zvec from authoritative SQLite embeddings without calling the embedder. Returns the
number of memories rebuilt. See [index maintenance](../operations.md#index-maintenance-and-repair).

## optimize

```text
optimize() -> None
```

Merges and flushes staged derived-index vectors. It does not repair media or convert an embedding
space. Use the [maintenance runbook](../operations.md#index-maintenance-and-repair).
