# Formation, consolidation, and retention configuration

Derive useful assertions after capture, maintain them as evidence changes, and apply a retention
policy on your schedule. Formation, consolidation, and retention are optional and independent.

[Configuration](../configuration.md) covers construction,
[typed memory](../typed-memory.md) explains evidence semantics, and
[memory management](../memory-management.md) explains deliberate operations.

## Automatic memory formation

Add a formation slot to derive entity/event/state/relation/affect/trait records after source commit:

```yaml
formation:
  provider: openai
  model: qwen3-8b
  base_url: http://127.0.0.1:8080/v1
  api_key: unused-by-this-local-server
  modalities: [text]
  temperature: 0.0
  max_tokens: 2048
  response_format: json_object
```

This example targets a compatible local server. Use `json_schema` when it supports strict schema.
The endpoint implements chat completions with one nonempty message or an explicit refusal.
Source observations remain durable; failed formation can be retried without duplicating work
already completed under the same recipe. Formation is one completion per add or batched add-many.

### Input and output requirements

- Configuring generation does not enable formation; their clients/models/keys are independent.
- Native modalities and labelled caption/transcript fallback determine usable inputs. Unsupported
  observations without fallback are skipped diagnostically.
- The request includes spatial frame/anchor, not metric x/y coordinates.
- Strict schema requires exactly one envelope item per observation alias, allowing zero proposals.
  The provider must support item bounds and string enums; JSON-object mode still validates locally.
- Duplicate/missing observations and invalid evidence references fail validation. Individually
  unreadable proposals increment adapter dropped counts; ungrounded proposals increment kernel
  refused counts.
- Length termination raises `ModelOutputTruncatedError`; raise the output budget rather than accept
  partial proposals. Refusal raises `model_refused`. Streamed structural responses follow the same
  validation. Malformed JSON has one bounded correction attempt, separate from transport retries.
- Response policies require host authorization with response-feedback basis; formation cannot
  grant them.

### Optional formation history

| Setting | Default | Meaning |
| --- | --- | --- |
| `formation_history_max_rows` | `0` | Off; at most 128 read-only background rows per target |
| `formation_history_budget_chars` | `6000` | Positive combined content budget, excluding protocol overhead |

History ends at the target's recording time, excludes batch targets and forgotten records, and
keeps whole records; an oversized recent row stops selection. Typed versions are selected at that
cutoff, with retirement status. No-known-version rows consume no window. It is recent background,
not semantic history search or another formation target.

Each target has one read-snapshot-grounded history. Identical ordered snapshots serialize once as
read-only groups; different versions/provenance/truncation remain separate. Shared serialization
does not broaden citation authority. The adapter aliases background rows `history_N`; a target
may cite only itself, its batch, and its own active history. History participates in the recipe
identity for both JSON modes.

With history enabled, a named event/state/relation/trait/affect must jointly cite a standing entity
record naming its subject. Retired, missing, or competing names withhold that proposal; supported
anonymous events remain possible. Current identity binding also rejects competing names.
`history_truncated` means counterevidence may be absent. Stored text/metadata and in-place
confidence retain existing semantics; this is not complete historical replay.

Commit rechecks cited sources. Forgotten or newly hidden/retired support prevents derived commits
and raises `ModelError(reason="response_invalid", stage="form")`, leaving the observation stored.
Changed models/prompts change `formation_space`; old derived records remain, and a different recipe
may add new records on reprocessing.

## Agentic memory management

```yaml
consolidation:
  provider: openai
  model: gpt-5-mini-2025-08-07
  modalities: [text, image]
  response_format: json_schema
  max_tokens: 4096
```

The slot builds a consolidation backend with the shared completion controls. It runs only when
the host calls `consolidate()` or `consolidate_pending()`. An absent backend raises
`backend_not_configured` for an explicit evidence pass or due work; an empty loop calls no model.
The SDK/doctor capability document lists consolidation; REST/MCP do not expose it.

Evidence assets in declared modalities attach natively. Undeclared modalities are omitted without
failing the pass. Identical generation/formation configuration can reuse an adapter implementing
consolidation; routed or evidence-wrapped answerers need a separate adapter. Any changed model,
endpoint, credential, or other value creates another adapter.

### Running the loop inside a server process

Give scheduling to the process that already owns the directory:

```python
from sensemem.api import create_app

app = create_app(memory=memory, deliberate_every=900.0)
```

For MCP, pass the same `deliberate_every` argument to `build_mcp_server(memory, ...)`.
The positive seconds interval is off by default and requires configured consolidation.
Construction rejects invalid intervals or missing backends with `ValidationError`.

Each lifespan round runs in a worker thread and logs aggregate rounds/weighed/applied/rejected at
INFO under `sensemem.api.deliberation`. Failure logs a warning and the next round still runs.
No external route/tool or second directory owner is introduced. The host should choose an interval
that fits its latency, power, and inference budget.

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

Ages are positive days from each material's recorded/enqueue/forget time, not its last read.
`None` disables that policy. Nothing runs automatically.

Start with `memory.apply_retention(dry_run=True)` and review its IDs/cascades before applying.
The CLI exposes `sensemem apply-retention`. Physical deletion cannot be rolled back.
The [data-subject API](../api/python-identity.md#data-subject-rights) owns report fields and limits.

Direct injection uses `MemoryConfig(retention=RetentionPolicy(...))`. Declarative files prefer
the top-level section; declaring retention there and inside settings is rejected.
