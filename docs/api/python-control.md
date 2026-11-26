# Python control operations

Consolidate evidence, correct claims, forget reversibly, and audit or undo memory changes. These methods run in the owning host process, where the application controls inference budgets and administrative authority.

Start with [memory management](../memory-management.md). See [typed evidence](../typed-memory.md) for support semantics and [operation values](python-values.md#control-operations) for fields. REST and MCP leave these operations with the owner.

| Task | Methods |
| --- | --- |
| Discover/run model work | [consolidation_candidates](#consolidation_candidates), [consolidate](#consolidate), [consolidate_pending](#consolidate_pending) |
| Apply host changes | [apply](#apply), [forget](#forget) |
| Review/undo/assess | [operations](#operations), [rollback](#rollback), [record_outcome](#record_outcome) |

## consolidation_candidates

```text
consolidation_candidates(
    *,
    limit: int = 32,
    idle: bool = False,
) -> tuple[ConsolidationCandidate, ...]
```

Derives due work from committed state without a timer, queue, or consolidation backend. Rows interleave triggers; `QUERY_FAILURE` rows may embed and search to locate nearby context.

| Trigger | Candidate meaning | `evidence_count` |
| --- | --- | --- |
| `EVIDENCE` | New independent support for a derived record (target then sources in `memory_ids`) | New links |
| `CONTRADICTION` | Different current visible functional values in one lineage | Distinct values |
| `FEEDBACK` | New explicit/automatic answer reinforcement | Confirmations |
| `QUERY_FAILURE` | At least two near-equal empty recalls in the configured window | Failures |
| `PRESSURE` | Least useful records while active count exceeds an explicit budget | Excess records |
| `IDLE` | Host `idle=True`; never-weighed derived lineage, oldest first | 1 |

Search, abstaining ask, and empty compile record bounded query-failure signals within `query_failure_history`. A pass marks evidence weighed even if it applies nothing; new signals or rollback make work due again.

## consolidate

```text
consolidate(
    *,
    evidence_ids: Sequence[str] | None = None,
    query: ContentInput | None = None,
    limit: int = 32,
    trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    query_failure: QueryFailure | None = None,
) -> ConsolidationReport
```

Requires a consolidator. Selects evidence from explicit `evidence_ids`, otherwise active search for `query`, otherwise the newest `limit` active records. Explicit IDs may name hidden or retired assertions as target-only records; forgotten records are never shown. A supplied `query_failure` requires `trigger=MemoryTrigger.QUERY_FAILURE` (forwarded automatically by `consolidate_pending()`).

Each proposal rechecks targets and premises in its own SQLite transaction and either commits with one log row or is rejected with a reason in `report.rejected`.

- Model corrections require standing, shown counterevidence disjoint from targets (`counterevidence_required` otherwise). Host-authored `apply()` corrections may omit counterevidence.
- Standalone model `FORGET` operations require `trigger=MemoryTrigger.PRESSURE` and an explicit `MemorySettings.memory_budget_records`, and cannot forget below that budget.
- Models cannot consolidate a derived claim together with only its own sources (`dependent_evidence`).

## consolidate_pending

```text
consolidate_pending(
    *,
    limit: int = 32,
    max_rounds: int = 4,
    idle: bool = False,
) -> ConsolidationReport
```

Runs due candidates under each row's trigger for at most `max_rounds`, stopping on a zero-yield pass and calling no backend when no work is due.

## apply

```text
apply(operation: MemoryOperation) -> MemoryOperationRecord
```

Applies one host operation without calling a consolidation backend. A refusal raises `ValidationError` with the kernel rejection reason.

| Intent | Required checks | Effect and rollback |
| --- | --- | --- |
| `REINFORCE` | One unretired derived target; shown, existing, unlinked evidence excluding target | Add independent support; rollback retires those links |
| `CONSOLIDATE` | Valid proposal, standing shown sources, targets within cited set, supported modality/frame | Add joint-supported record, forget named details, apply supersession; rollback reverses changes |
| `CORRECT` | Shown existing derived targets (model proposals cite counterevidence) | Retire current versions; rollback adds inverse versions |
| `FORGET` | Shown existing, not-forgotten targets (model proposals require pressure + budget) | Set `forgotten_at`; rollback clears it |
| `IDENTIFY` | Existing identity present in at least one standing cited speech/face observation | Commit naming assertion and refresh projections; rollback restores predecessor |
| `MERGE` | Explicit host confirmation of complementary voice/face identities (never from backend) | Bind under host-selected identity; rollback splits and records negative pair |
| `CONSENT` | Host statement only (never from backend) | Record consent assertion; rollback restores predecessor |

### Eligibility and rejection

Duplicate standing operations are rejected. Multi-target eligibility is all-or-nothing (`target_not_shown`, `unknown_target`, `not_derived`, `already_forgotten`, `stale`). Generic `REINFORCE`/`CORRECT`/`FORGET` refuse protected `naming_assertion` and `consent_assertion` targets. Passes reject `inconsistent_batch` if one operation retires support needed by another, and `target_is_evidence` on self-citation.

### Lineage and identity effects

Qualified state/user-stated-trait supersession records exact version pairs in `superseded` for rollback. Lineages key on kind, subject identity, predicate, and spatial frame; reviewed scalar predicate aliases (such as `preferred_drink` and `favorite_beverage`) share one canonical claim key while preserving original spelling. Consolidation `target_ids` set `forgotten_at` while retaining evidence links and appear in `forgotten_ids`.

Identity operations (`MERGE`, unlink's `CORRECT`, erasure's `FORGET`) use `IdentityChange` rather than memory `target_ids`. See [identity and consent](python-identity.md).

### Replay

Replaying an operation onto a fresh store with matching sources and recipe reproduces operation keys and content-addressed derived IDs (event-derived IDs also require matching source `occurred_at`).

## forget

```text
forget(memory_ids: Sequence[str]) -> MemoryOperationRecord | None
```

Host-authorized cognitive forgetting. Recall omits the record while `get()`, `list()`, and `forgotten_at` retain audit state. Unknown IDs raise `MemoryNotFoundError`; empty input, already-forgotten targets, or protected naming assertions return `None` without changes. Only `rollback()` restores recall; physical deletion uses `delete()`.

## rollback

```text
rollback(operation_id: int) -> bool
```

Returns `False` for unknown or already-reversed operations, identity erasure, or operations with standing dependents (reverse dependent lineages and namings newest first). Restores exact clause predecessors with inverse transaction intervals so historical `known_at` reads remain intact.

## operations

```text
operations(*, limit: int = 100) -> tuple[MemoryOperationRecord, ...]
```

Lists the append-only operation log newest first. See [operation values](python-values.md#control-operations).

## record_outcome

```text
record_outcome(operation_id: int, outcome: MemoryOutcome, *, note: str | None = None) -> bool
```

Records or replaces a `CONFIRMED` or `REFUTED` assessment and note for measurement without changing policy or rolling back the operation. Unknown operation IDs return `False`.
