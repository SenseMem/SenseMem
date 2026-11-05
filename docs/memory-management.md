# Memory management

The host schedules deliberate changes to stored memory. A consolidation backend proposes
operations; the kernel validates and logs each accepted operation. Ordinary writes and retrieval
need no consolidation backend.

## Run a deliberate pass

Configure a [consolidation backend](configuration/lifecycle.md#agentic-memory-management).
With an open `memory`:

```python
from sensemem import MemoryTrigger

report = memory.consolidate(
    query="What do we know about Ana?",
    trigger=MemoryTrigger.MANUAL,
)
for record in report.operations:
    print(record.operation_id, record.operation.intent, record.changed_ids)
for operation, reason in report.rejected:
    print(operation.intent, reason)
```

Evidence comes from explicit `evidence_ids`, otherwise a query, otherwise the newest active records.
Hidden and forgotten records are not shown. The host may explicitly name a hidden target for
reinforcement or correction; model proposals cannot widen that window.

Each proposal commits in its own transaction. A pass can therefore contain both applied and rejected
operations. Multi-target proposals are all-or-nothing, and changed targets/sources are rechecked
at commit and refused as stale. Duplicate operation keys do not apply twice.

## Choose an operation

| Intent or call | Purpose |
| --- | --- |
| `CONSOLIDATE` | Add a derived claim with the cited set as one joint support clause |
| `MemoryIntent.REINFORCE` | Add independent evidence to a derived claim; logged and reversible |
| `CORRECT` | Retire a bad derived version while preserving history |
| `FORGET` / `forget()` | Exclude records from recall while retaining them for audit |
| `IDENTIFY` | Propose an evidence-grounded name for an observed identity |
| `register_identity()` / `record_consent()` | Record host-authorized naming or consent |
| `reinforce()` | Update ranking utility; no evidence linkage or reversible log row |

A consolidation forgets its sources only when it explicitly names them in `target_ids`; source
links remain, and `forgotten_ids` records the reversible effect. State replacement follows
[authority and evidence rules](typed-memory.md#state-replacement). Relations can hold simultaneous
values; user-stated traits represent a current preference.

Model proposals cannot authorize consent or identity merges, and cannot alter protected naming
or consent assertions through generic operations. Physical deletion is never a proposal intent.
Model corrections require shown counterevidence; model forgetting requires an explicit host
record budget and pressure trigger. Direct host changes retain their separate authority.
The [control API](api/python-control.md) owns exact checks and rejection reasons.

## Schedule due work

```python
report = memory.consolidate_pending(idle=is_charging_and_idle)
```

The host declares idleness. The loop derives candidates from committed signals, processes them
under their own trigger, and repeats up to `max_rounds` (default 4).

| Trigger | Signal |
| --- | --- |
| `EVIDENCE` | New independent evidence not yet weighed |
| `CONTRADICTION` | Conflicting visible functional values |
| `FEEDBACK` | New explicit or automatic answer reinforcement |
| `QUERY_FAILURE` | Repeated near-equal empty recalls within the configured window |
| `PRESSURE` | Active records exceed an explicit `memory_budget_records` |
| `IDLE` | Host-approved consideration of never-weighed derived lineages |

A pass marks evidence weighed even when every proposal is refused. New signals make it due again;
rollback can reopen work. Pressure/idle candidates are weighed once without a new event.
Candidate limits bound returned rows, not scan cost or waiting time. Query-failure polling can
embed and search, so schedule it within the inference budget. No timer or worker runs implicitly.

## Review, measure, and undo

`operations()` lists newest-first history. `record_outcome()` records a later confirmed/refuted
assessment for measurement and never changes policy. `rollback(operation_id)` reverses an
eligible operation; dependent operations and lineage changes must reverse newest first.
Re-adding forgotten content does not restore it.

Physical deletion and identity erasure cannot be recovered by rollback. Use
[the local consolidation benchmark](benchmarks/local.md#measure-consolidation-quality) to measure
precision, contradiction recovery, false retirement, and rollback on a fixed fixture. It does not
establish real-world model quality.
