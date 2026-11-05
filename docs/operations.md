# Operations

Operate the process that owns the local data directory. SenseMem has no separate database
service, queue, or background owner. Use [deployment](deployment.md) for topology and
[troubleshooting](troubleshooting.md) for error diagnosis.

| Task | Procedure |
| --- | --- |
| Start, stop, or check liveness | [Lifecycle](#start-and-stop), [health](#health) |
| Handle load or capacity errors | [Concurrency](#concurrency-and-overload), [capacity](#capacity) |
| Preserve or recover data | [Backup](#backup), [restore](#restore) |
| Repair derived search state | [Index maintenance](#index-maintenance-and-repair) |
| Change models or runtime | [Representation changes](#changing-the-runtime-or-stored-representations) |
| Inspect timing, usage, or quality | [Observability](observability.md), [model quality](model-quality.md) |

## Start and stop

Open one owner with the store's schema, model spaces, and index recipe. Startup validates
compatibility, opens/creates Zvec, and drains durable index work. It does not settle captures.

Use a context manager or call `Memory.close()`. Close rejects new work, waits for active calls,
releases leases/resources, and releases the directory lock. The lock file may remain; its
operating-system lock determines ownership.

`close(timeout_seconds=30.0)` bounds the full caller wait, including backend cleanup and flush.
Timeout raises `StorageError(reason="timeout", stage="close")`; cleanup continues and retains
ownership. Call close again to wait for that cleanup. Never delete a lock file to bypass ownership.
Configure finite provider timeouts because running synchronous calls cannot be forcibly cancelled.

Before deployment/composition changes, use [doctor](api/cli.md#doctor-and-explain) to check configuration
and loaders without writing memory. Remote doctor checks REST liveness.

## Concurrency and overload

Fixed directory/process budgets bound operations, model work, async submission, stream workers,
index routes, SQLite connections, input materialization, and captures. See
[runtime limits](runtime-limits.md) for the complete table and worker semantics.

Overload raises retryable `StorageError(reason="capacity_exceeded", stage="admission")`.
REST returns 503 with `Retry-After: 1`; MCP preserves reason/retryability. Bound producers and
retry with bounded backoff. Health checks stay outside admission.

Close abandoned streams; cancelling a waiter retains capacity until its synchronous work stops.
These limits do not cap process RSS, GPU memory, provider quotas, or framework queues. Measure
successful throughput, tail latency, and rejection rates with
[the local SDK load test](benchmarks/local.md#measure-local-sdk-concurrency) before sizing producers.

## Health

```bash
curl --fail http://127.0.0.1:8000/healthz
```

The unauthenticated endpoint proves process liveness and reports configured capabilities narrowed
to REST operations. It calls no model, inspects no outbox, and performs no retrieval.
Run end-to-end canaries in a separate directory.

## Backup

### Capture backlog before a snapshot

A backup preserves queued work without making it searchable. Inspect through the existing owner:

```python
for item in memory.pending_captures(limit=100):
    print(item.memory_id, item.awaiting, item.attempts, item.last_error)
```

Enrichment rows lack vectors; formation rows are searchable and owe only typed formation.
The page is bounded; a full page may indicate more backlog. Track enqueue age, attempts, and stage.
Call settle only during the host's approved model-work window. Failed batches can still settle
other rows. A record at the default three-attempt ceiling remains queued; fix its cause and raise
the ceiling or name it explicitly with `settle(memory_ids=[...])`.
Neither startup nor shutdown drains captures automatically.

### Stopped-owner copy

1. Close the owner and verify no process holds its directory lock.
2. Copy the complete directory, including any SQLite side files.
3. Record the SenseMem version and model/index configuration.
4. Restore the copy to a temporary path and verify known text/media records.
5. Restart the original owner.

`state.sqlite3` and `assets/` are required. Zvec can be omitted if restore capacity/time permits
rebuilding it. Media is not recoverable from descriptors or transcripts. A copied lock file does
not make a snapshot busy. For online backups, use a SQLite-aware snapshot mechanism rather than
copying a running main database file alone.

## Restore

Restore into a new empty path while the target service is stopped:

1. Restore the complete snapshot, or SQLite plus assets with Zvec absent.
2. Apply the service account's ownership and restrictive permissions.
3. Open one owner with the recorded runtime and embedding/transcription/face/index configuration.
4. Verify get/list results and representative text/media searches.
5. Route traffic only after verification.

Missing Zvec rebuilds from SQLite embeddings without re-embedding historical content.
Unsupported schema or space/dimension/recipe mismatches fail on open. Use the original runtime to
read source data when moving to an incompatible representation, then replay into a fresh directory.

## Index maintenance and repair

Use the live owner for normal maintenance:

```python
count = memory.reindex()
memory.optimize()
print(f"reindexed {count} memories")
```

Reindex replaces derived state from SQLite FP32 vectors, replays concurrent writes, and merges
remaining segments. Optimize drains/acknowledges pending index work, merges staged vectors, and
flushes. Neither repairs media or converts embedding spaces.

Startup/add/delete/search drain the index outbox. Applied rows are searchable immediately;
acknowledgement follows flush at 256 applied rows, on deletion, during optimize/reindex, and at
close. Automatic optimization is bounded. A flushed session can merge multiple segments at close;
read-only sessions do not. Allow shutdown time for flush/compaction.

If Zvec cannot open or is corrupt:

1. Stop the owner and retain a tested backup.
2. Move only `zvec/` aside; retain SQLite and assets.
3. Start one owner with its original configuration and wait for rebuilding.
4. Verify searches before removing the old index.

Never edit the outbox, replace the index, or move authoritative files while an owner is live.
RocksDB log files are journals; do not remove them with a filename glob.

### Space and quantization

In-place optimization normally merges without replacing the collection. Descriptor pressure or
accumulated idmaps can still require compaction and room for a second index. Interrupted compaction
may leave `.zvec.compact-*` debris or a missing index; the next open cleans/rebuilds derived state.

Supported quantization is none/fp16/int8/rabitq. Changing it within the current recipe rebuilds from
stored vectors without model calls. INT8 is unrotated; RABITQ needs dimensions 64–4095 and native
support. FP32 vectors remain authoritative, so smaller total disk use is not guaranteed.
Quantized candidates get exact local rescoring, which cannot recover undiscovered ANN neighbors.

Applications can set Zvec process resources once, before constructing memory:

```python
import zvec

zvec.init(query_threads=4, optimize_threads=2, memory_limit_mb=2048)
```

SenseMem does not call this initializer. Keep unquantized vectors unless measured capacity,
latency, and recall justify a lossy mode.

## Capacity

Record pressure and physical retention are separate: `memory_budget_records` creates a
deliberation signal, while declared retention acts only when the host calls it:

```python
report = memory.apply_retention(dry_run=True)
print(report.deleted, report.asset_ids, report.capture_memory_ids)
```

Review direct/cascading IDs before applying. Physical deletion is irreversible; cognitive
forgetting is reversible. See [retention configuration](configuration/lifecycle.md#retention-policy)
and [data-subject operations](api/python-identity.md#data-subject-rights).

Monitor total SQLite/media/index bytes and free inodes; space for a second index; file descriptors;
operation/model latency; stable errors; capture backlog; backup age and tested recovery time.
Original media, canonical FP32 vectors, and composite/text embedding keys remain storage costs.

## Changing the runtime or stored representations

1. Create and restore-test a backup.
2. Stop the owner.
3. Change code/configuration together.
4. Reopen with compatible identities or replay sources into a new directory.
5. Verify records, media, retrieval, and telemetry before traffic.

Only SQLite schema 21 is accepted. Startup validates tables, constraints, provenance, and policy
markers; it performs no migration or automatic re-embedding. Initial face setup is allowed only
without existing face-analysis/space markers and pins both face identities together.
Operation logs require current complete payloads; incompatible logs raise StorageError.

| Change | Required action |
| --- | --- |
| Embedding space/dimension | Replay into a fresh directory |
| Persisted transcription/face space | Keep a compatible backend or replay |
| Current-recipe quantization | Rebuild from stored FP32 vectors |
| Missing Zvec | Reopen and rebuild from SQLite |
| Evidence projection policy | Fresh directory with the intended policy |
| SQLite schema | Read sources with the original runtime, then replay |

Do not bypass checks by editing schema metadata or the outbox. Rebuilds preserve semantic/evidence
history and selectors; they do not recover physical deletion. Run the [model quality gate](model-quality.md)
for model/prompt changes and use [observability](observability.md) to verify the deployed path.
