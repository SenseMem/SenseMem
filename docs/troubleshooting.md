# Troubleshooting

Find the failing stage, take the corresponding action, and retry when the error says the request can succeed later. SenseMem's structured errors share the same classification across interfaces. This example assumes an open `memory` with a configured answerer:

```python
from sensemem import SenseMemError

try:
    result = memory.ask("What happened yesterday?")
except SenseMemError as error:
    print(error.code, error.reason, error.stage, error.subject, error.retryable)
```

`code` is the stable family, `reason` narrows the cause, `stage` locates the failed pipeline step, `subject` identifies the affected item when available, and `retryable` states whether the identical request can succeed later. CLI, REST, and MCP envelopes return the same classification fields and add a `trace_id`; preserve it with request metadata. See the [REST error contract](api/rest.md#codes-and-reasons).

For composition failures, run the same selection with `sensemem ... doctor`. Bundled recipes exercise their loaders, `--app` verifies the import without calling a factory that could open the store, and `--url` checks `/healthz`.

## Error triage

| Signal | Meaning | Action |
| --- | --- | --- |
| `validation_error`, REST 422 | Input violates its contract | Correct shape, media type, event bounds, IDs, or declared modality support. |
| `model_error` | A configured model stage failed | Check the slot, endpoint, credentials, timeout, and capability declaration. |
| `model_output_truncated` | Generation reached its output limit | Increase output capacity or reduce selected evidence. |
| `index_unavailable` | Search projection work failed | Inspect Zvec, disk, and file descriptors; check SQLite for the committed record. |
| `storage_error` | Local state or ownership failed validation or I/O | Check the directory owner, filesystem, assets, SQLite, schema, and model identities. |
| `memory_not_found`, REST 404 | The authoritative record is absent | Verify the ID and directory with `get()` or `list()`. |

Retry only when `retryable` is true. The retryable reasons are `connection_failed`, `capacity_exceeded`, `data_dir_in_use`, `flush_failed`, `index_missing`, `provider_unavailable`, `rate_limited`, and `timeout`. REST returns `Retry-After: 1` only for retryable 503 responses.

## Memory does not open

### A configuration filename is rejected

`Memory.from_config()` accepts a mapping or `MemorySpec`, not a path. Decode JSON with `json.loads()` or YAML with the host's `yaml.safe_load()` before passing it. Unknown fields, providers, and invalid values are rejected at this boundary; see [file loading](configuration.md#loading-a-configuration-file).

### The directory is in use

`reason="data_dir_in_use"` means a live `Memory` owns the physical directory. Close that owner, call its REST interface, or choose a different directory. Do not delete `.sensemem.lock`; the file is only the operating-system lock target and deleting it does not release an owner.

The error is retryable because the current owner may exit. Python, CLI, and MCP retain the local path in `subject`; unauthenticated REST redacts storage, index, and internal subjects.

### The schema is unsupported

`reason="schema_unsupported"` means this SenseMem version cannot open the SQLite schema or required tables. Reopen with the matching runtime and replay sources into a fresh directory. Do not edit `state.sqlite3` or `PRAGMA user_version` by hand.

### Store metadata mismatch

The store records embedding space and dimension, transcription space, configured face spaces, ingestion-stage recipe, and index recipe. Reopen with matching configuration or replay sources into a fresh directory. `reindex()` rebuilds Zvec from stored FP32 vectors without converting embedding spaces.

### Zvec is missing or damaged

A missing `zvec/` directory rebuilds automatically on next open without re-embedding stored content. If Zvec is damaged, follow the [index repair runbook](operations.md#index-maintenance-and-repair): stop the owner, retain a backup, move only `zvec/` aside, and reopen. Never move `state.sqlite3` or `assets/`.

## A write returned an index error

SenseMem commits SQLite before applying Zvec changes and flushes Zvec in batches. An `index_unavailable` error can therefore mean the record or deletion is already durable in SQLite, with outbox rows pending until the next drain succeeds:

1. Preserve the error envelope and `trace_id`.
2. Check authoritative state with `get()` or `list()` if the ID is known.
3. Correct the Zvec, disk-space, or file-descriptor failure.
4. Reopen the owner or run `reindex()` / `optimize()`.
5. Retry an add only with the exact same canonical input (its content-derived ID makes it idempotent).

Do not delete SQLite rows or clear `search_index_queue` manually.

## Search returns no hits

A `SearchResult` with `hits=()` is a valid result. Candidates below `minimum_relevance` are rejected, and an unresolved top-two tie may be withheld when `limit=1`. `minimum_relevance` compares against `gate_relevance`, not the final `SearchHit.score`; use `search(..., explain=True)` to inspect both.

Before changing thresholds, check whether a captured record is still unsettled via `pending_captures()`. An `awaiting="enrichment"` row has no vectors yet; call `settle()` (or `settle(memory_ids=...)` past the retry ceiling) and search again. A record with `forgotten_at` set is excluded by policy until its operation is rolled back.

Then:

1. Confirm the record exists with `get()` or `list()`.
2. Confirm query and stored modalities are supported by the embedder.
3. Remove unintended memory-type, event-time, bitemporal, or spatial filters.
4. Run `search(..., explain=True)` and inspect `rejected_by` (`stale_index`, `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`, `ambiguity`, `limit`).
5. If authoritative records exist but the projection is suspect, run [index repair](operations.md#index-maintenance-and-repair).

Check `retrieval_mode` as well: `lexical` skips query embedding and dense candidates, so an uncaptioned image without text has no lexical match. See [runtime settings](configuration/settings.md).

## A stored media record fails to load

SQLite cannot recreate original media bytes. If `get()`, search hydration, or model work reports a storage failure for a known media record, verify that the digest-named regular file exists under `assets/` and restore it from a tested backup. Do not create an empty placeholder or edit the SQLite descriptor.

## A model operation fails

### Backend is not configured

Every `Memory` needs an embedder, and `ask()` needs an answerer. With `kinds=None`, `perceive()` uses configured speech and face backends and reports per-asset `coverage`; explicit `kinds` require the matching capability. See [configuration](configuration.md) and [memory operations](api/python-memory.md).

### Optional dependency is missing

Install the extra for the selected adapter (`local`, `openai`, `face`, `reranking`, `gazetteer`, `server`, `mcp`); see the [extras table](configuration.md#1-install-only-the-surfaces-you-use).

### Modality is unsupported

Select a backend that declares the required modality, or configure a transcription/vision backend for source-backed text fallback. If neither native input nor a usable derivation is available, the operation fails explicitly.

### Provider request or output fails

Inspect `reason` and `stage`, then check credentials, base URL, proxy, timeout, rate limit, provider status, and response shape. SenseMem preserves the provider exception as Python `__cause__` without serializing provider bodies in REST or MCP envelopes.

- **`context_length_exceeded`**: the provider rejected the request against its context window. Reduce selected evidence (`limit`, `evidence_budget_chars=None`) or choose a larger-context model; answers are not retried by silently dropping evidence.
- **`model_output_truncated`**: generation hit its output-token ceiling. Raise the adapter's `max_tokens` or reduce `ask()` evidence.
- **`payload_too_large`**: request exceeded SenseMem's byte ceilings or an upstream HTTP 413 limit. Reduce request size or increase serving limits. See [Python adapters](api/python-adapters.md) and [REST input limits](api/rest.md#input-limits).

## Content is rejected or not fetched

SenseMem never fetches HTTP(S) URLs: in Python a URL-shaped string is treated as text, and REST/MCP reject remote URLs and server paths. Download remote media in application code, then pass a `Blob` or regular local `Path`. REST rejects invalid payloads with 422 and bodies over 8 MiB with 413.

## REST or MCP cannot share the store

Separate Python, REST, and MCP processes cannot open the same `data_dir`. Wrap one constructed `Memory` with the needed adapters, call the running REST owner, or allocate separate directories. See [deployment](deployment.md#choose-a-topology).

## Latency or telemetry looks wrong

`/healthz` reports liveness, not model or retrieval readiness. Compare operation, stage, and model spans in [observability](observability.md) to separate model latency from SQLite, index sync, and ranking. Check `sensemem.token_usage.complete` when token totals appear lower than expected; SenseMem never estimates missing provider usage.

For face or speaker recognition, inspect `sensemem.identity.faces` and `sensemem.identity.speakers` spans (`observations`, `matched_existing`, `created`) to diagnose fragmentation, and calibrate thresholds per [identity thresholds](configuration/settings.md#identity-thresholds).
