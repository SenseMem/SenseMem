# Runtime limits

Bound producer concurrency and size the host using SenseMem's admission, worker, and input
budgets. The tables below distinguish per-directory limits from process-wide budgets shared
across directories.

For overload handling and sizing procedures, see [Operations](operations.md#concurrency-and-overload).

## Admission limits

Each physical `data_dir` owns its admission limits and worker leases. Process budgets are shared
across directories. The fixed runtime bounds are:

| Resource | Limit |
| --- | --- |
| Active memory operations across SDK, REST, and MCP | 32 per directory; 128 per process |
| Operations that can invoke a model | 28 per directory and per process |
| Submitted asynchronous SDK calls, including live vision descriptions | 32 per directory; 128 per process |
| Pinned answer-stream workers shared by AsyncMemory and REST | 8 per directory; 32 per process |
| Index-route workers | 4 per directory; 16 per process |
| Speech-overlap workers | 2 per directory; 8 per process |
| SQLite connections, including borrowed and idle handles | 32 per directory; 128 per process |
| REST `/v1` requests and MCP tool calls sharing one directory | 32 per directory; 128 per process |
| Open local stores | 16 per process |
| Cached candidate identity projections | 1,024 per directory |
| Cached hydrated plain-text records | 1,024 per directory; at most 2,048 content and metadata characters each |
| Materialized input text and media retained by active operations | 512 MiB per operation and per process |
| Inputs in one SDK `add_many()` | 1,024 |
| Captures awaiting enrichment or formation | 10,000 per directory |

Admission is immediate: overload raises `StorageError(reason="capacity_exceeded",
stage="admission")`, which is retryable. REST returns the shared 503 envelope with
`Retry-After: 1`; MCP reports the same reason and retryability. Apply bounded backoff when
retrying. `/healthz` remains outside request admission.

## Worker and cancellation behavior

Model work leaves four of a directory's operation slots available to lightweight reads. The
process model budget also leaves room in shared SDK and framework worker pools; it does not
guarantee progress through backend locks or unrelated host work. Local model adapters retain
their inference safety locks. Formation calls over distinct sources may overlap; concurrent
writes of the same source share one formation call, and application remains serialized and
transactionally checked. Naming and identity rollback commit registry names, speech text,
vectors, and repair outbox together, including when embedding fails.

REST reads a request body within 30 seconds total and allows at most 30 seconds for each response
send. Body timeout returns the usual retryable 503 `timeout` envelope. A send timeout ends the
response and returns request capacity. A terminal answer releases its pinned worker before
network delivery. MCP tool admission runs before framework worker dispatch; HTTP connection and
MCP session limits still belong to the serving host.

Index routes use batches of at most four per admitted operation; temporal supplements reuse
the same pool. At most 128 route jobs and 32 speech-overlap jobs can be pending or running.
Each route obtains its directory's quota before entering the shared worker pool, so waiting for
that quota does not occupy another directory's worker. A failed submission still waits for any
earlier routes to finish before releasing the operation's assets.
Parent asynchronous calls use a separate pool so they cannot occupy the workers their child
routes need. Cancelling an asynchronous waiter retains capacity until its synchronous work
actually stops. Stream workers remain pinned through queued cleanup and are reusable before
the terminal result is delivered. Close an abandoned stream with `close()` or `aclose()`.

The process shares 64 asynchronous parent threads, 16 index threads, eight speech threads, and
at most 32 pinned stream threads, created as needed and reused across directories. Closing one
directory waits for its own jobs without shutting down another directory's workers. Each closing
directory can also use one cleanup thread. File-descriptor exhaustion is reported as
`capacity_exceeded`, including during open, rather than as a missing search index.
SQLite connection opens and native index allocations share one descriptor admission lock and
preserve filesystem descriptor headroom before allocating; the numeric connection ceiling is
also subject to the process's actual file-descriptor limit.
SQLite retains at most 64 idle connections across the process. Every handle remains in its own
directory's pool. Each directory's first four idle handles are protected from idle-budget
eviction; spare budget lets a busy directory retain more. The oldest surplus handles close when
that budget fills. Active requests can reclaim idle handles before reporting process overload.
Directory closure waits for both its borrowers and any eviction already closing one of its
handles. The active limits remain 32 per directory and 128 per process.
Handles already being closed by eviction still count toward the directory limit.

## Read caches

Read caches admit stable working sets, bypass active writes, and invalidate when a write transaction
ends. A working set larger than the cache bound disables its cache until the next write generation.
Hydrated-record reuse excludes scoped
reads, media, and semantic contexts. Its retained UTF-8 content and metadata total at most 8 MiB
per directory, plus object overhead; the smaller candidate cache retains IDs and occurrence times.

## What the bounds cover

These bounds cover SenseMem-owned submissions, workers, stores, and SQLite handles. The byte
budget charges UTF-8 text and media sizes before materialization and releases them on operation
exit. Inputs whose own cumulative size exceeds 512 MiB raise `ValidationError`; contention for
otherwise valid input raises retryable `capacity_exceeded`. This is not an RSS or GPU-memory
ceiling. Caller-owned buffers and threads, framework bodies,
decoded media, native model/Zvec allocations, backend request sizes, and provider quotas still
need host sizing. A full capture backlog refuses new queued records atomically; settle existing
captures before retrying. Duplicate captures remain idempotent at the limit.
Choose producer concurrency from measurements within these bounds.
Use the [local SDK load test](benchmarks/local.md#measure-local-sdk-concurrency) to measure throughput,
tail latency, and overload rejections for a physical store before tuning its producers.
Use backend timeouts and bound the host's own producers. At the HTTP server boundary,
Uvicorn's [concurrency limit](https://uvicorn.dev/server-behavior/#resource-limits) can
also reject excess requests based on open connections and in-flight tasks. A worker count alone
does not bound queued submissions;
Python's [executor documentation](https://docs.python.org/3/library/concurrent.futures.html)
also describes nested-pool deadlocks and cancellation limits.
