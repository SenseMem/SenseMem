# Command-line API

Operate memory from a shell or script. The `sensemem` command opens a local composition or calls an existing REST owner, returning one JSON document per invocation.

Jump to [invocation](#invoke-the-cli), [recipes](#bundled-recipes), [commands](#commands), [content and JSONL](#content-and-jsonl-input), [output](#output), [diagnostics](#doctor-and-explain), or [exit codes](#exit-codes).

## Surface

Each `sensemem` invocation runs one command in one process and emits one JSON result. Local commands dispatch to the corresponding `Memory` method (except `doctor`); remote commands forward to a running owner's `/v1` route. Benchmark commands belong to `sensemem-bench`, documented in [benchmarking](../benchmarking.md).

`read_asset`, `replay_sources`, `prepare_input`, and `call_tool` are Python SDK methods without CLI commands.

## Invoke the CLI

```text
sensemem [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

The console entry point calls `sensemem.cli.main(argv: Sequence[str] | None = None) -> int`. Choose exactly one composition per invocation:

```bash
sensemem --app my_application:build_memory search "where is the spare key"
sensemem --embedder openai add "The spare key is in the blue toolbox."
sensemem --url http://127.0.0.1:8000 list
```

Global options precede the command. Use `sensemem COMMAND --help` for command-specific flags.

| Option | Meaning |
| --- | --- |
| `--app MODULE:ATTR` | `Memory` instance or zero-argument callable returning one |
| `--embedder NAME` | Construct a bundled embedding recipe |
| `--url URL` | Address a running owner over `/v1` |
| `--data-dir PATH` | Local directory for `--embedder` (default `.sensemem`) |
| `--timeout SECONDS` | Positive finite remote timeout for `--url` (default `30`) |
| `--answerer NAME` | Generation recipe for `--embedder` |
| `--former NAME` | Formation recipe for `--embedder` (omit to disable) |
| `--consolidator NAME` | Consolidation recipe for `--embedder` (omit to disable) |
| `--transcriber NAME` | Speech recipe for `--embedder` |
| `--vision NAME` | Visual description recipe for `--embedder` (omit to disable) |
| `--index-speech`, `--no-index-speech` | Enable or disable transcript indexing (default enabled) |
| `--minimum-relevance FLOAT` | Relevance floor (default `0.10`) |
| `--ambiguity-margin FLOAT` | Top-two gate when `limit=1` (default `0.01`) |
| `--decay-half-life-days FLOAT` | Optional positive recency half-life |
| `--explain` | Print resolved composition and execute no operation |
| `-q`, `--quiet` | Suppress the composition banner on stderr |
| `-V`, `--version` | Print the installed version |

`--data-dir`, backend options, and tuning options apply only to `--embedder`; `--timeout` applies only to `--url`. Use `--app MODULE:ATTR` for index quantization, face analysis, custom thresholds, custom backends, or retention policies. The product CLI uses the SDK's `strict` answer policy; use the Python SDK or REST `AnswerRequest` for `best_effort`. See [configuration](../configuration.md).

## Lifecycle and ownership

Local `--app` and `--embedder` commands open one `Memory`, perform one operation, and close it before exit. One physical `data_dir` allows only one live owner; when another process owns it, use `--url` or a different directory.

## Contract

### Bundled recipes

| Recipe | Allowed slot | Default identity |
| --- | --- | --- |
| `funasr` | `--transcriber` | Pinned FunASR model and component revisions |
| `openai` | `--embedder`, `--answerer`, `--former`, `--consolidator`, `--transcriber`, `--vision` | `text-embedding-3-small`, `gpt-5-mini-2025-08-07`, `whisper-1` |

Use a model suffix with `openai`, such as `--former openai:gpt-5-mini-2025-08-07`:

```bash
sensemem --embedder openai --former openai --vision openai add @panel.png
```

### Commands

| Command | Operands and options | JSON result | `--url` |
| --- | --- | --- | --- |
| `add` | content; `--occurred-at`; `--occurred-end`; `--metadata`; `--memory-type`; `--context` | memory object | yes |
| `add-many` | optional JSONL source; `--memory-type` | `{"memories":[...]}` | yes |
| `add-stream` | optional JSONL source; `--memory-type`; `--capture` | `{"memories":[...]}` | no |
| `capture` | same operands and options as `add` | memory object | yes |
| `settle` | optional `MEMORY_ID...`; `--limit`; `--max-attempts` | `{"settled":int}` | yes |
| `pending-captures` | optional `MEMORY_ID...`; `--limit` | `{"pending":[...]}` | yes |
| `search` | content; `--explain`; `--limit`; `--memory-type`; `--reference-at`; `--scope`; `--occurred-from`; `--occurred-until` | `{"hits":[...],"trace":null}` | yes |
| `ask` | content; `--limit`; `--memory-type`; `--reference-at`; `--scope` | answer object | yes |
| `compile` | content; `--max-chars`; `--max-items`; `--max-media-items`; repeatable `--memory-type`; `--min-confidence`; `--freshness-seconds`; `--max-latency-ms`; `--reference-at`; `--scope`; `--allow-partial-sources`; `--selected-proofs` | context bundle plus `rendered` | yes |
| `get` | `MEMORY_ID` | memory object | yes |
| `perceive` | `MEMORY_ID`; optional repeatable `--kind speech` / `--kind faces` | `{"speech":[...],"faces":[...],"coverage":[...]}` | yes (`embodied_operations`) |
| `register-identity` | `IDENTITY_ID NAME`; `--relationship` | `{}` | yes (`identity_operations`) |
| `identity` | `IDENTITY_ID` | `{"identity":{...}}` or `{"identity":null}` | yes (`identity_operations`) |
| `record-consent` | `IDENTITY_ID STATE`; `--note` | `{"operation":{...}}` or `{"operation":null}` | yes (`identity_operations`) |
| `consent` | `IDENTITY_ID` | `{"consent":"granted"}` or `{"consent":null}` | yes (`identity_operations`) |
| `forget-identity` | `IDENTITY_ID` | erasure counts | yes (`identity_operations`) |
| `unlink-identity` | `ALIAS_ID` | `{"restored_identity_id":...}` | yes (`identity_operations`) |
| `reinforce` | one or more `MEMORY_ID` values | `{"reinforced":int}` | yes |
| `consolidation-candidates` | `--limit`; `--idle` | `{"candidates":[...]}` | no |
| `consolidate` | optional goal content; `--evidence-id`; `--limit`; `--trigger` | `ConsolidationReport` object | no |
| `consolidate-pending` | `--limit`; `--max-rounds`; `--idle` | `ConsolidationReport` object | no |
| `apply` | `--operation` | `{"operation":{...}}` | no |
| `record-outcome` | `OPERATION_ID OUTCOME`; `--note` | `{"recorded":bool}` | no |
| `forget` | one or more `MEMORY_ID` values | `{"operation":{...}}` or `{"operation":null}` | no |
| `rollback` | `OPERATION_ID` | `{"rolled_back":bool}` | no |
| `operations` | `--limit` | `{"operations":[...]}` | no |
| `export` | `--identity-id` or repeatable `--memory-id` | export bundle | yes (`identity_operations`) |
| `apply-retention` | `--dry-run` | retention report | yes (`identity_operations`) |
| `list` | `--limit`; `--cursor` | `{"items":[...],"next_cursor":...}` | yes |
| `delete` | `MEMORY_ID` | `{"deleted":bool}` | yes |
| `reindex` | none | `{"memories":int}` | no |
| `optimize` | none | `{}` | no |
| `doctor` | none | composition and loader report | yes |

#### Defaults and options

Defaults match the SDK: `memory_type=semantic`; `search` `limit=10`; `ask` `limit=5`; `list`, `settle`, `pending-captures`, and `operations` `limit=100`; `settle` `max-attempts=3`; `consolidate` and `consolidation-candidates` `limit=32` (`trigger=manual`); `consolidate-pending` `max-rounds=4`. Timestamps must be timezone-aware ISO 8601 values, and cursors are opaque. `compile` mirrors [`ContextBudget` defaults](../context-compilation.md#budget), with `--selected-proofs` and `--allow-partial-sources` disabled by default.

#### Control operations and replay

`operations` prints the append-only operation log (see [operation values](python-values.md#control-operations)). `apply --operation` replays a complete operation JSON object (including explicit `null` and `[]` fields) through kernel validation, such as:

```json
{
  "intent": "forget",
  "evidence_ids": [],
  "target_ids": ["memory-id"],
  "proposal": null,
  "claim": null,
  "consent": null,
  "identity": null,
  "rationale": null
}
```

#### Consent, export, retention, and identity binding

`record-consent` accepts `granted`, `withheld`, or `withdrawn` (see [consent](python-identity.md#consent)). `export` prints every retained version, identity row, and operation touching the subject without printing media bytes. `apply-retention` acts on the composition's configured [retention policy](../configuration/lifecycle.md#retention-policy); start with `--dry-run`. To confirm a cross-modal voice/face merge, pass a `merge` operation through local `apply --operation`; see [cross-modal identity binding](python-identity.md#cross-modal-identity-binding).

### Content and JSONL input

Content commands (`add`, `capture`, `search`, `ask`, `compile`, `consolidate`) accept either:

- Positional atoms: bare values are text, `@PATH` is a local file, `@@TEXT` escapes a leading `@`, and `-` reads stdin.
- `--content-json VALUE`: a JSON string or REST-shaped content-parts array (`VALUE` is literal JSON, `@PATH`, or `-`).

Local `--content-json` also accepts `{"type":"input_file","path":"./media/panel.png"}`; `input_file.path` and `@PATH` are rejected under `--url` (send inline base64 media per [REST content input](rest.md#content-input)).

`--metadata`, `--context`, and `--scope` accept literal JSON, `@PATH`, or `-`. `add-many` and `add-stream` read JSONL objects containing required `content` and optional `occurred_at`, `occurred_end`, `metadata`, and `context`:

```bash
sensemem --app my_application:build_memory add "The mug is on the table" \
  --context '{"basis":"observation","source_id":"camera-1:42","confidence":0.94}'

sensemem --app my_application:build_memory search "Where is the mug?" \
  --scope '{"near":{"frame_id":"home/map","anchor":"subject","x":2,"y":1},"radius_m":0.75}'
```

### Operations without a command

`ask_stream` is not a CLI command because each invocation returns one JSON document; use `ask` for completed answers or SDK `ask_stream()` / REST `POST /v1/answers/stream` for streaming.

### Output

On success, stdout receives one JSON document and a trailing newline matching the [REST response vocabulary](rest.md#response-objects), with UTC timestamps rendered with a trailing `Z`. Unless `--quiet` is passed, stderr prints the resolved composition JSON before execution. Runtime failures write one [shared error envelope](rest.md#error-envelope) to stderr and nothing to stdout.

### Doctor and explain

`--explain` prints the resolved composition without running the command. `doctor` reports installed versions and composition checks: `--embedder` exercises each recipe's loader probe and prints `capabilities` without opening storage; `--app` resolves the target without invoking factories; `--url` calls `GET /healthz`.

## Errors and limits

### Exit codes

| Exit | `code` | Meaning |
| --- | --- | --- |
| 0 | — | Success |
| 1 | `internal_error` or unmapped code | Unexpected failure |
| 2 | — | Argparse syntax or type error |
| 3 | `validation_error` | Input rejected |
| 4 | `memory_not_found` | Memory does not exist |
| 6 | `model_error` | Inspect `reason` and `retryable` |
| 7 | `storage_error` | Storage or remote request failure |
| 8 | `index_unavailable` | Vector index unavailable |
| 9 | `storage_error/data_dir_in_use` | Another live process owns the directory; use `--url` |
| 10 | `configuration_error` | Composition or remote-capability failure |
| 11 | `model_output_truncated` | Generation hit its output limit |
| 12 | `identity_not_found` | Face/voice identity does not exist |
| 130 | — | Interrupted |

### Operations without a remote route

Eleven local commands exit 10 with `unsupported_in_remote_mode` under `--url` because they have no `/v1` route:

- Owner-only control plane ([public and trust boundaries](../architecture.md#public-and-trust-boundaries)): `consolidation-candidates`, `consolidate`, `consolidate-pending`, `apply`, `record-outcome`, `forget`, `rollback`, `operations`.
- Local ingestion and index maintenance: `add-stream`, `reindex`, `optimize`.

Gated identity/perception commands also exit 10 when the remote owner did not enable the matching switch; see [REST operations without a route](rest.md#operations-without-a-route).

### Input limits

Local mode applies Python bounds (128 content parts, 65,536 characters, 262,144 metadata bytes, 512 MiB per local asset, `add-many` up to 1,024 items, search/answer/page limits 1–100). Remote `--url` mode applies REST bounds (16 content parts, 8 MiB request body, `add-many` up to 100 items).
