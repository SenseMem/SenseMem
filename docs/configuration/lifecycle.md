# Formation, consolidation, and retention configuration

Derive useful assertions after capture, maintain them as evidence changes, and apply a retention policy on your schedule. Formation, consolidation, and retention are optional and independent.

[Configuration](../configuration.md) covers construction, [typed memory](../typed-memory.md) explains evidence semantics, and [memory management](../memory-management.md) explains deliberate operations.

## Automatic memory formation

Add a `formation` slot to derive entity, event, state, relation, affect, and trait records after source commit:

```yaml
formation:
  provider: openai
  model: your-formation-model
  base_url: https://your-endpoint.example/v1
  modalities: [text]
  temperature: 0.0
  max_tokens: 2048
  response_format: json_schema
```

Use `response_format: json_object` when your endpoint does not support strict JSON Schema. Source observations remain durable if formation fails and can be retried via `settle()` without duplicating already-formed sources. Formation issues one completion per `add()` or batched `add_many()`.

### Input and output requirements

- Configuring `generation` does not enable formation; their clients, models, and credentials are independent.
- Native modalities and labelled caption/transcript fallback determine usable inputs; unsupported observations without fallback are skipped diagnostically.
- Requests include spatial frame and anchor, not metric coordinates.
- Strict schema requires one envelope item per observation alias (allowing zero proposals); JSON-object mode still validates locally.
- Duplicate/missing observations and invalid evidence references fail validation; individually ungrounded proposals increment `sensemem.formation.refused_proposals`.
- Length termination raises `ModelOutputTruncatedError`; provider refusal raises `model_refused`; malformed JSON gets at most one bounded correction attempt.
- Response policies require host authorization with `response_feedback` basis and cannot be granted by formation.

### Bounded formation history

| Setting | Default | Meaning |
| --- | --- | --- |
| `formation_history_max_rows` | `8` | Read-only background rows per target (`0` disables; maximum `128`) |
| `formation_history_budget_chars` | `6000` | Positive combined content budget, excluding protocol overhead |

History ends at the target's recording time, excludes batch targets and forgotten records, and keeps whole records (stopping when a recent row exceeds the budget). Enabling history lets the formation model reuse existing property keys and interpret state changes against earlier claims; set `formation_history_max_rows: 0` to send only the current batch.

Each target receives a snapshot-grounded history aliased as `history_N` and may cite only itself, its batch, and its own active history. Commit rechecks cited sources atomically so forgotten or newly hidden/retired support rejects derived commits while keeping the source observation durable.

## Agentic memory management

```yaml
consolidation:
  provider: openai
  model: your-consolidation-model
  modalities: [text, image]
  response_format: json_schema
  max_tokens: 4096
```

The `consolidation` slot builds a consolidation backend with the shared completion controls and runs only when the host calls `consolidate()` or `consolidate_pending()`. Evidence assets in declared modalities attach natively; undeclared modalities are omitted without failing the pass. Identical generation/formation configuration can reuse one adapter instance.

### Running the loop inside a server process

Schedule periodic consolidation inside the process that owns the directory:

```python
from sensemem.api import create_app

app = create_app(memory=memory, deliberate_every=900.0)
```

For MCP, pass `deliberate_every` to `build_mcp_server(memory, ...)`. The interval is in positive seconds, defaults to `None` (disabled), and requires a configured consolidation backend. Each round runs in a worker thread, logs aggregate counts at `INFO` under `sensemem.api.deliberation`, and logs a warning without crashing on failure.

## Retention policy

Retention is an explicit top-level section:

```yaml
retention:
  media_days: 30
  forgotten_days: 90
  capture_failure_days: 7
```

| Field | Default | Effect when the host runs retention |
| --- | --- | --- |
| `media_days` | `None` | Delete aged assets and every memory referencing them |
| `forgotten_days` | `None` | Delete records cognitively forgotten longer ago than this |
| `capture_failure_days` | `None` | Abandon aged failed capture-queue rows, keeping the memory |

Ages are positive days from each material's recorded, enqueued, or forgotten timestamp (`None` disables that rule). Nothing runs automatically: preview with `memory.apply_retention(dry_run=True)` or `sensemem apply-retention --dry-run` before applying. Physical deletion is irreversible; see [data-subject operations](../api/python-identity.md#data-subject-rights).

Direct construction uses `MemorySettings(retention=RetentionPolicy(...))`. Declarative files use the top-level `retention` section (declaring retention in both places is rejected).
