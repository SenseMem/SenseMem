# Python control operations

Consolidate evidence, correct claims, forget reversibly, and audit or undo memory changes.
These methods run in the owning host process, where the application controls inference budgets
and administrative authority.

Start with [memory management](../memory-management.md). See [typed evidence](../typed-memory.md)
for support semantics and [operation values](python-values.md#control-operations) for fields.
REST and MCP leave these operations with the owner.

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

Derives due work from committed state without a timer, queue, or consolidation backend.
Rows interleave triggers; small limits can still favor earlier trigger groups and guarantee no
bounded waiting time. Query-failure rows cost a query embedding/search; a limit bounds output,
not scan/history aggregation cost.

| Trigger | Candidate meaning | `evidence_count` |
| --- | --- | --- |
| `EVIDENCE` | New independent support for a derived record; target then sources in `memory_ids` | New links |
| `CONTRADICTION` | Different current visible functional values in one lineage | Distinct values |
| `FEEDBACK` | New explicit/automatic answer reinforcement | Confirmations |
| `QUERY_FAILURE` | At least two near-equal empty recalls in the configured window | Failures |
| `PRESSURE` | Least useful records while active count exceeds an explicit budget | Excess records |
| `IDLE` | Host `idle=True`; never-weighed derived lineage, oldest first | 1 |

Search, abstaining ask, and empty compile record bounded query-failure signals, compared with
case/accent/punctuation/whitespace normalization. The owner's query text stays within its memory
domain and `query_failure_history` bound.

A pass marks its evidence weighed even if it applies nothing. Persistent-signal candidates return
only after newer support/feedback/failure/conflict; pressure and idle are weighed-once without a
signal timestamp. Rolled-back work becomes due again. Eligibility filtering precedes per-trigger
limits for evidence/contradiction/feedback/pressure/idle, allowing a finite unchanged backlog to
advance. Query failure has its own bounded search window. Continuous arrivals can still delay work.

Candidates use retained history/operation marks, so a poll can scan settled records even when its
result is short. Removing history would lose audit/rollback/weighed information.

## consolidate

```text
consolidate(
    *,
    evidence_ids: Sequence[str] | None = None,
    query: ContentInput | None = None,
    limit: int = 32,
    trigger: MemoryTrigger = MemoryTrigger.MANUAL,
) -> ConsolidationReport
```

Requires a consolidator. Select evidence by explicit IDs, otherwise active search for query,
otherwise newest limit active records. Hidden/forgotten records are not shown. Explicit host IDs
also bound eligible hidden targets; query/default windows never widen to hidden targets.

Evidence gather and apply phases use the formation lock; the model round trip releases it.
Identity commits use that same lock. Every proposal rechecks targets/premises in its own
transaction; ineligible changes during reasoning refuse it as stale.

Each proposal either commits fully with one log row or is rejected with a reason. A pass is not
atomic across proposals: earlier commits remain if later proposals fail. The report carries
operations, rejected pairs, weighed evidence count, rounds, skipped candidates, and `model_calls`;
applied is the operation count. `weighed=0` means no backend was reached.

One consolidation's cited set is a joint support clause; separate operations add alternatives.
Evidence linkage is model-declared provenance, not semantic proof. Intent-specific rules appear
under apply below.

Model-proposed corrections must cite standing, shown counterevidence disjoint from their targets;
the citations remain in the operation log and idempotency key. Missing citations are rejected as
`counterevidence_required`. The apply transaction rechecks those premises. Host-authored
`apply()` corrections may omit counterevidence because the caller supplies the authority.

Standalone model `FORGET` operations require `trigger=MemoryTrigger.PRESSURE` and an explicit
`MemoryConfig.memory_budget_records`. The SQLite transaction refuses a batch that would leave
fewer unforgotten records than that budget, including when another operation already relieved
the pressure; it reports `stale` without partial changes. Other model forgetting is
`unauthorized`. Manual `forget()` and host-authored `apply()` retain their explicit authority.

Consolidation inputs expose shown source indices and omitted-source counts. A model cannot
consolidate a derived claim together with only its own sources as if they were independent
duplicates; that proposal is `dependent_evidence`. This check preserves the sources for later
correction or genuine new evidence. Host-authored `apply()` may explicitly re-express them.

## consolidate_pending

```text
consolidate_pending(
    *,
    limit: int = 32,
    max_rounds: int = 4,
    idle: bool = False,
) -> ConsolidationReport
```

Runs due candidates with each row's trigger, for at most `max_rounds`. `idle=True` admits the host's
approved idle window. It stops on a zero-yield pass without spinning; no due candidates means no
backend call. Candidate scheduling arguments belong here, while explicit trigger belongs to
consolidate. Returns the same `ConsolidationReport`.

## apply

```text
apply(operation: MemoryOperation) -> MemoryOperationRecord
```

Applies one host operation without calling a consolidation backend. It can still embed a derived
record or changed identity projection. Its window is the host's named targets and evidence.
A refusal raises `ValidationError` with the kernel rejection reason.

| Intent | Required checks | Effect and rollback |
| --- | --- | --- |
| `REINFORCE` | One unretired derived target; shown, existing, unlinked evidence excluding the target | Add independent support; rollback retires those links |
| `CONSOLIDATE` | Valid proposal, standing shown sources, targets within its own cited set, supported affect modality/spatial frame | Add joint-supported record, forget named details, apply qualified supersession; rollback reverses those changes |
| `CORRECT` | Shown existing derived targets; model proposals cite shown counterevidence excluding the target | Retire current versions; rollback adds inverse versions |
| `FORGET` | Shown existing, not-forgotten targets; model proposals require host-configured record pressure | Set `forgotten_at` within remaining record budget; rollback clears it |
| `IDENTIFY` | Existing identity present in at least one standing cited speech/face observation | Commit naming assertion and repaint name/transcript projections; rollback restores predecessor |
| `MERGE` | Never accepted from a backend: unauthorized | Kernel corroboration only; rollback splits |
| `CONSENT` | Never accepted from a backend: unauthorized | Host statement only; rollback restores predecessor |

### Eligibility and rejection

Operation keys hash canonical operation JSON plus recipe. An already-applied, standing key is
duplicate. Named targets outside the window are `target_not_shown`. Multi-target eligibility is
all-or-nothing; failures include `unknown_target`, `not_derived`, and `already_forgotten`.
Transaction rechecks return stale without writes. Hidden unretired reinforcement targets can
still receive eligible support; hidden/retired cited premises cannot seed a new operation.

`REINFORCE`/`CORRECT`/`FORGET` refuse bound `naming_assertion` and `consent_assertion` targets.
`IDENTIFY` additionally rejects `unknown_identity` and `identity_not_in_evidence`. Model names carry
`MODEL_INFERENCE` and require two independent groups; `register_identity` is the immediate
`USER_STATEMENT` host path.

Model corrections without counterevidence are `counterevidence_required`; unshown premises are
`evidence_not_shown`, and self-citation is `target_is_evidence`. Host corrections can omit
counterevidence. Model forgetting requires a `PRESSURE` trigger and explicit record budget;
the commit rechecks remaining pressure instead of forgetting below that budget. Host-authored
forgetting retains its explicit authority.

A pass rejects `inconsistent_batch` when one operation retires support another accepted operation
uses. Self-supporting consolidation or reinforcement is `target_is_evidence`. The kernel never
repairs a malformed clause by dropping just the bad source.

### Lineage and identity effects

Qualified state/user-stated-trait supersession can reach outside the shown window by the kernel's
own rule; exact version pairs are recorded in superseded for rollback. Kind, subject identity,
predicate, and spatial frame separate lineages. Registered scalar aliases such as `preferred_drink`,
`favorite_beverage`, and 喜欢的饮料 share a key; original predicate spelling remains. Unknown
predicates/compound values keep literal keys. Relations and inferred traits accumulate.

Consolidation `target_ids` set `forgotten_at` in the same transaction, retain evidence links, and
appear as `forgotten_ids` on that operation. This differs from generic `FORGET` and physical deletion.

Identity IDs use `IdentityChange` rather than memory `target_ids`: `MERGE`, unlink's `CORRECT`, and
erasure's `FORGET` carry identity/moved IDs. A row names memories or one identity, never both.
Unlink rollback re-merges; identity erasure logs IDs/counts only and cannot reverse.
[Identity/consent](python-identity.md) owns those authority boundaries.

### Replay

Replay onto a fresh store with the same sources/recipe reproduces operation keys and
content-addressed derived IDs. `applied_at` is new. Event-derived IDs reproduce only with matching
source `occurred_at`. A changed derivation recipe intentionally produces different IDs.

## forget

```text
forget(memory_ids: Sequence[str]) -> MemoryOperationRecord | None
```

Host-authorized cognitive forgetting needs no backend/window. Recall omits the record while
get/list/`forgotten_at` retain audit state. Unknown IDs raise `MemoryNotFoundError`. Empty input,
already-forgotten targets, protected naming assertions, moved targets, or duplicate operations
return `None` without changes. All IDs apply or none. Only rollback restores recall; re-add returns
the same forgotten record. Physical delete instead returns `False` for missing IDs and cannot reverse.

## rollback

```text
rollback(operation_id: int) -> bool
```

Returns `False` for unknown/already-reversed operations, identity erasure, or changes a later
standing operation depends on. Reverse dependent lineages and namings newest first.
Clause rollback restores an exact predecessor with an inverse transaction interval, preserving
unrelated alternatives and refusing to overwrite later clause versions. Historical `known_at`
continues to expose the support/confidence that stood then. Physical deletion is not an intent.

## operations

```text
operations(*, limit: int = 100) -> tuple[MemoryOperationRecord, ...]
```

Lists the append-only log newest first, including intent, affected IDs, provenance, superseded
versions, rollback state, and optional outcome. Returned values are defined in
[the value reference](python-values.md#control-operations).

## record_outcome

```text
record_outcome(operation_id: int, outcome: MemoryOutcome, *, note: str | None = None) -> bool
```

Records or replaces a later `CONFIRMED`/`REFUTED` assessment and note. Unknown operations return `False`.
`None` outcome means unassessed, not confirmed. The kernel never uses these labels for decisions;
recording `REFUTED` does not reverse anything. Use rollback for a change in behavior and retain
denominators when calculating slow-loop quality.
