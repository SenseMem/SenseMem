# Operations

Keep memory available and recoverable with explicit lifecycle, backup, and index-maintenance
procedures. Run these operations through the process that owns the data directory.
[Deployment](deployment.md) covers topology; [troubleshooting](troubleshooting.md) covers errors.

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

Overload raises retryable `StorageError(reason="capacity_exceeded", stage="admission")`.
REST returns 503 with `Retry-After: 1`, and MCP preserves the reason and retryability. Bound
producers and retry with bounded backoff. `/healthz` stays outside admission.

### Admission limits

Directory limits apply to one physical `data_dir`. Process limits are shared across directories.

| Resource | Limit |
| --- | --- |
| Active memory operations across SDK, REST, and MCP | 32 per directory, 128 per process |
| Operations that can invoke a model | 28 per directory and per process |
| Submitted async SDK calls, including live vision descriptions | 32 per directory, 128 per process |
| Pinned answer-stream workers shared by `AsyncMemory` and REST | 8 per directory, 32 per process |
| Index-route workers | 4 per directory, 16 per process |
| Speech-overlap workers | 2 per directory, 8 per process |
| SQLite handles, borrowed and idle | 32 per directory, 128 per process |
| REST `/v1` requests and MCP tool calls sharing one directory | 32 per directory, 128 per process |
| Open local stores | 16 per process |
| Cached candidate identity projections | 1,024 per directory |
| Cached hydrated plain-text records | 1,024 per directory, at most 2,048 content and metadata characters each |
| Materialized text and media retained by active operations | 512 MiB per operation and per process |
| Inputs in one SDK `add_many()` | 1,024 |
| Captures awaiting enrichment or formation | 10,000 per directory |

Admission is immediate. Text is charged by UTF-8 size and media by byte size before materialization.
Inputs above the per-operation byte cap raise `ValidationError`. Otherwise valid inputs raise
retryable `capacity_exceeded` if the process budget is full. A full capture backlog refuses new
queued records atomically and still accepts idempotent duplicate captures. Settle existing captures
before retrying. Cache reuse bypasses active writes and invalidates at transaction end. A working
set above its bound disables that cache until the next write generation. Hydrated record caching
excludes scoped reads, media, and semantic context, and retains at most 8 MiB of UTF-8 content and
metadata per directory plus object overhead.

### Workers and cancellation

The model budget leaves four directory operation slots available to lightweight reads. Backend
inference locks and unrelated host work can still delay progress. Formation calls on distinct
sources can overlap. Concurrent writes of the same source share a formation call, with serialized
and transactionally checked application. Naming and rollback use the
[atomic write path](architecture.md#write-consistency).

The process creates and reuses up to 64 asynchronous parent threads, 16 index threads, eight
speech threads, and 32 pinned stream threads. Index routes run in batches of up to four per
operation, including temporal supplements. At most 128 index jobs and 32 speech-overlap jobs can
be pending or running. Index jobs acquire directory quota before dispatch. Failed submissions
wait for earlier routes before releasing assets. Parent calls have their own pool so child routes
can run. Cancelling an async waiter retains capacity until synchronous work stops. Close abandoned
streams with `close()` or `aclose()`. A stream worker stays pinned through cleanup and is reusable
before terminal delivery.

REST reads a request body within 30 seconds total and allows up to 30 seconds for each response
send. Body timeout returns a retryable 503 `timeout` envelope. A send timeout ends the response
and frees request capacity. MCP admits tool calls before framework worker dispatch. Configure HTTP
connection and MCP session limits at the host.

SQLite and native index allocation preserve file-descriptor headroom under shared admission.
Descriptor exhaustion reports `capacity_exceeded`, including at open. The process retains at most
64 idle SQLite handles, with each directory's first four idle handles protected from eviction.
Oldest surplus handles close first. Active requests can reclaim idle handles before reporting
overload. Handles closing during eviction still count against the directory limit. Directory
closure waits for its own borrowers, evictions, and jobs without shutting down other directories'
workers. Each closing directory can use one cleanup thread.

These bounds cover runtime-owned work. Size process RSS, GPU memory, decoded media, caller buffers,
native allocations, provider quotas, and framework queues separately. Measure throughput, tail
latency, and rejections with the [local SDK load test](benchmarks/local.md#measure-local-sdk-concurrency).
Use finite backend timeouts and a host-side producer bound. Uvicorn's
[concurrency limit](https://uvicorn.dev/server-behavior/#resource-limits) can bound connections and
in-flight tasks in addition to the runtime's admission.

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
cursor = None
while page := memory.pending_captures(limit=100, cursor=cursor):
    for item in page:
        print(item.memory_id, item.awaiting, item.attempts, item.last_error)
    cursor = page[-1].cursor
```

Enrichment rows lack vectors; formation rows are searchable and owe only typed formation.
Each page is bounded; its cursor remains usable after the preceding record leaves the queue.
Track enqueue age, attempts, and stage.
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
support. SQLite retains authoritative FP32 vectors, so measure total directory size as well as
the derived index. Quantized candidates receive exact local rescoring within the discovered pool.

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

Only SQLite schema 25 is accepted. Selector-piece ordering and outbox pagination use their primary
keys without duplicate secondary indexes. Startup validates tables, constraints, provenance, and policy
markers; it performs no migration or automatic re-embedding. Initial face setup is allowed only
without existing face-analysis/space markers and pins both face identities together.
Operation logs require current complete payloads; incompatible logs raise StorageError.

| Change | Required action |
| --- | --- |
| Embedding space/dimension | Replay into a fresh directory |
| Persisted transcription/face space | Keep a compatible backend or replay |
| Ingestion stage recipe changed or missing | Replay into a fresh directory |
| Current-recipe quantization | Rebuild from stored FP32 vectors |
| Missing Zvec | Reopen and rebuild from SQLite |
| Evidence projection policy | Fresh directory with the intended policy |
| SQLite schema | Read sources with the original runtime, then replay |

Do not bypass checks by editing schema metadata or the outbox. Rebuilds preserve semantic/evidence
history and selectors; they do not recover physical deletion. Run the [model quality gate](model-quality.md)
for model/prompt changes and use [observability](observability.md) to verify the deployed path.
