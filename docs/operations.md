# Operations

Keep memory available and recoverable with explicit lifecycle, backup, and index-maintenance procedures. Run these operations through the process that owns the data directory. [Deployment](deployment.md) covers topology; [troubleshooting](troubleshooting.md) covers errors.

| Task | Procedure |
| --- | --- |
| Start, stop, or check liveness | [Lifecycle](#start-and-stop), [health](#health) |
| Handle load or capacity errors | [Concurrency](#concurrency-and-overload), [capacity](#capacity) |
| Preserve or recover data | [Backup](#backup), [restore](#restore) |
| Repair derived search state | [Index maintenance](#index-maintenance-and-repair) |
| Change models or runtime | [Representation changes](#changing-the-runtime-or-stored-representations) |
| Inspect timing, usage, or quality | [Observability](observability.md), [model quality](model-quality.md) |

## Start and stop

Open one owner with the store's compatible model spaces and index recipe. Startup validates compatibility, opens or creates Zvec, and drains durable index work; it does not settle captures.

Use a context manager or call `Memory.close()`. Close rejects new work, waits for active calls, releases leases and resources, and releases the directory lock. The lock file may remain on disk; its operating-system lock determines ownership.

`close(timeout_seconds=30.0)` bounds the full caller wait, including backend cleanup and flush. Timeout raises `StorageError(reason="timeout", stage="close")` while cleanup continues and retains ownership; call close again to wait for completion. Never delete `.sensemem.lock` to bypass ownership. Configure finite provider timeouts because running synchronous calls cannot be forcibly cancelled.

Before deployment changes, use [doctor](api/cli.md#doctor-and-explain) to check configuration and loaders without writing memory. Remote doctor checks REST liveness.

## Concurrency and overload

Fixed directory and process budgets bound operations, model work, async submissions, stream workers, index routes, SQLite connections, input materialization, and captures; see [runtime limits](runtime-limits.md).

Overload raises retryable `StorageError(reason="capacity_exceeded", stage="admission")`. REST returns 503 with `Retry-After: 1`; MCP preserves reason and retryability. Bound producers and retry with bounded backoff. Health checks stay outside admission.

Close abandoned streams; cancelling an async waiter retains capacity until synchronous work stops. Measure throughput, tail latency, and rejections with [the local SDK load test](benchmarks/local.md#measure-local-sdk-concurrency) before sizing producers.

## Health

```bash
curl --fail http://127.0.0.1:8000/healthz
```

The unauthenticated endpoint proves process liveness and reports configured capabilities narrowed to REST operations. It calls no model, inspects no outbox, and performs no retrieval. Run end-to-end canaries in a separate directory.

## Backup

### Capture backlog before a snapshot

A backup preserves queued work without making it searchable. Inspect pending items through the existing owner:

```python
cursor = None
while page := memory.pending_captures(limit=100, cursor=cursor):
    for item in page:
        print(item.memory_id, item.awaiting, item.attempts, item.last_error)
    cursor = page[-1].cursor
```

Enrichment rows lack vectors; formation rows are searchable and owe only typed formation. Call `settle()` during your host's approved model-work window. Records at the default three-attempt ceiling remain queued until retried with a higher ceiling or explicit `settle(memory_ids=[...])`.

### Stopped-owner copy

1. Close the owner and verify no process holds its directory lock.
2. Copy the complete directory, including any SQLite side files.
3. Record the SenseMem version and model/index configuration.
4. Restore the copy to a temporary path and verify known text and media records.
5. Restart the original owner.

`state.sqlite3` and `assets/` are required. Zvec can be omitted if restore capacity permits rebuilding it. For online backups, use a SQLite-aware snapshot mechanism rather than copying a live database file alone.

## Restore

Restore into a new empty path while the target service is stopped:

1. Restore the complete snapshot, or SQLite plus `assets/` with Zvec absent.
2. Apply the service account's ownership and restrictive permissions.
3. Open one owner with the recorded runtime and embedding, transcription, face, and index configuration.
4. Verify `get`/`list` results and representative text and media searches.
5. Route traffic only after verification.

Missing Zvec rebuilds from SQLite embeddings without re-embedding historical content. Incompatible spaces or recipes fail on open.

## Index maintenance and repair

Use the live owner for normal maintenance:

```python
count = memory.reindex()
memory.optimize()
print(f"reindexed {count} memories")
```

`reindex()` replaces derived state from SQLite FP32 vectors, replays concurrent writes, and merges remaining segments. `optimize()` drains pending index work, merges staged vectors, and flushes. Neither repairs missing media nor converts embedding spaces.

Startup, add, delete, and search drain the index outbox. Applied rows are searchable immediately; flush and outbox acknowledgement run at 256 applied rows, on deletion, during `optimize()`/`reindex()`, and at `close()`.

If Zvec cannot open or is corrupt:

1. Stop the owner and retain a tested backup.
2. Move only `zvec/` aside; keep `state.sqlite3` and `assets/`.
3. Start one owner with its original configuration and wait for rebuilding.
4. Verify searches before removing the old index.

Never edit the outbox, replace the index, or move authoritative files while an owner is live. Do not delete RocksDB journal files with filename globs.

### Space and quantization

In-place optimization normally merges without replacing the collection; interrupted compaction cleans up and rebuilds derived state on next open. Supported quantization modes are `none`, `fp16`, `int8` (unrotated), and `rabitq` (64–4095 dimensions). Changing quantization rebuilds Zvec from stored FP32 vectors without model calls, and quantized candidates receive exact local FP32 rescoring within the discovered pool.

Applications can configure Zvec process resources once before constructing `Memory`:

```python
import zvec

zvec.init(query_threads=4, optimize_threads=2, memory_limit_mb=2048)
```

## Capacity

Record pressure and physical retention are separate: `memory_budget_records` creates a deliberation signal, while declared retention runs only when the host calls it:

```python
report = memory.apply_retention(dry_run=True)
print(report.deleted, report.asset_ids, report.capture_memory_ids)
```

Review direct and cascading IDs before applying; physical deletion is irreversible. See [retention configuration](configuration/lifecycle.md#retention-policy) and [data-subject operations](api/python-identity.md#data-subject-rights).

Monitor SQLite, media, and index disk usage, free inodes, headroom for a second index, file descriptors, operation and model latency, error codes, capture backlog, and tested recovery time.

## Changing the runtime or stored representations

1. Create and restore-test a backup.
2. Stop the owner.
3. Update code and configuration together.
4. Reopen with compatible identities or replay sources via `replay_sources()` into a fresh directory.
5. Verify records, media, retrieval, and telemetry before routing traffic.

Startup validates store metadata and markers rather than rewriting existing spaces in place:

| Change | Required action |
| --- | --- |
| Embedding space or dimension | Replay sources into a fresh directory |
| Persisted transcription or face space | Keep a compatible backend or replay into a fresh directory |
| Ingestion-stage recipe | Replay sources into a fresh directory |
| Index quantization | Reopen to rebuild Zvec from stored FP32 vectors |
| Missing Zvec | Reopen to rebuild from SQLite |
| Evidence projection policy | Replay sources into a fresh directory with the intended policy |
| Storage schema | Read sources with the matching runtime and replay into a fresh directory |

Do not edit schema metadata or the outbox by hand. Run the [model quality gate](model-quality.md) for model or prompt changes and verify the deployed path with [observability](observability.md).
