# Maintain prompts

SenseMem's built-in model instructions and their text composition live in
[`src/sensemem/prompts/`](../src/sensemem/prompts/). Edit the component that owns an operation,
then check its direct callers and the matching quality fixtures. These modules are internal
implementation components; the supported SDK imports remain those exported from `sensemem`.

## Runtime components

| Component | Owns |
| --- | --- |
| `generation.py` | Answers, formation, consolidation, event segmentation, vision descriptions, recall plans, output protocols, and their conditional instructions. |
| `recall.py` | Recall coverage, evidence expansion, replanning, reference time, and host query-budget explanations. |
| `events.py` | Evidence boundaries for event summaries, grouped media, and unfinished-event continuation. |
| `evidence_operations.py` | The native evidence tool's instructions, examples, description, and recovery composition. |
| `agent.py` | The bundled agent's default system instructions. |
| `embedding.py` | Checkpoint-defined query and document prefixes. |
| `reranking.py` | The checkpoint classification template and media-query fallback text. |
| `context.py` | Context reading guides and rendering of selected evidence, partial sources, projections, and provenance. |
| `tools.py` | Descriptions of the SDK's bundled tools. |
| `mcp.py` | MCP server instructions, tool descriptions, parameter guidance, and resource descriptions. |
| `demo.py` | The local audio/video demo's session instructions and tool description. |

Prompt components are pure: they do not open storage, call a model, or import the kernel,
provider adapters, API transports, or benchmark package. The kernel decides retrieval scope,
budgets, evidence membership, and authority. Components render those decisions; adapters encode
the rendered text and media in the provider's request format. User questions, stored content,
timestamps, source aliases, schemas, and attachment markers remain structured input data.

The text-only recall backend contract carries a host-generated capability suffix in
`corpus_digest`. Its rendering and budget recognition share one template in `recall.py`, so
the adapter does not duplicate the kernel's prose. The execution plane still enforces the
configured query budget.

Caller-supplied agent system instructions and transcription hints remain caller inputs.
The local EmbeddingGemma encoder obtains its trained prefixes from the pinned model's own
configuration; the remote adapter uses the corresponding component constants.

## Review prompt behavior

Use OpenAI's [prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering),
[reasoning](https://developers.openai.com/api/docs/guides/reasoning-best-practices),
[structured output](https://developers.openai.com/api/docs/guides/structured-outputs), and
[function calling](https://developers.openai.com/api/docs/guides/function-calling) guidance for
the model and operation being configured. These are authoring and evaluation practices, not a
certification of model accuracy.

| Component | What to check |
| --- | --- |
| `generation.py` | The evidence policy remains consistent across fragments; JSON types match the schema; unreadable or irrelevant inputs have a non-fabricating response. |
| `recall.py` | Read coverage is distinct from answer-set completeness; trusted budgets and reference time remain explicit. |
| `events.py` | Model summaries are quoted data with separate original-media context; an unfinished synopsis cannot establish completion. |
| `evidence_operations.py` | Instructions describe function arguments, not an invented executor result; uncertainty remains represented in the existing schema. |
| `agent.py` | Tools serve the user's task, arguments follow their schemas, and errors are handled according to their actual meaning. |
| `embedding.py` | Prefixes match the pinned checkpoint's trained format. |
| `reranking.py` | The instruction and chat template match the checkpoint; general generation advice must not rewrite them. |
| `context.py` | Stored and temporary evidence have the same trust boundary; rendering and character accounting use the same guides. |
| `tools.py` | Descriptions state when to use each tool, where IDs come from, and what its returned references mean. |
| `mcp.py` | Descriptions match the runtime, optional planners, and host-selected authority switches. |
| `demo.py` | Historical answers wait for search results; empty results preserve uncertainty and sampled frames do not imply continuous motion. |

Keep tasks and contracts direct. Use headings or quoted structured values to separate instructions,
examples, and dynamic evidence. Retain examples that explain a difficult wire contract and verify
them against the actual parser; examples never supply facts for a real request. Request a short
evidence justification only where the schema requires it, rather than a private reasoning trace.
State what the model should do when evidence is insufficient, preserving the caller's selected
strict or best-effort policy.

## Benchmark components

Benchmark-only prompts stay under
[`src/sensemem/benchmarks/`](../src/sensemem/benchmarks/). Product components must not import them.

- `prompts.py` owns dataset query specifications and shared question wrappers.
- `baseline_prompts.py` owns blind, full-context, and compile-reader instructions.
- `_official/query_prompts.py` and `_official/judge_prompts.py` hold extracted query and judge
  wording alongside the existing upstream-specific components.
- [`_official/NOTICE.md`](../src/sensemem/benchmarks/_official/NOTICE.md) records pinned upstream
  sources and retained adaptations.

Reuse the pinned official query or judge component when one exists. Preserve its wording,
rubric, and provenance. A task-specific SDK adaptation must remain explicit; changing an
evaluation protocol requires a new version and renewed calibration. Dataset answers and
scorer-only fields must never enter answer-generation prompts.

## Validate a change

The component boundaries preserve public schemas and configured model controls. Prompt wording
has its own revisions. The component-boundary tests prevent provider or benchmark dependencies
from entering runtime prompts and detect inline instruction definitions at model and tool
boundaries. Existing request, context, MCP, and example tests cover the rendered results.

For a wording or policy change, also follow the [model quality gate](model-quality.md): record
the model revision, input fixtures, pass criteria, and real-service checks. Passing structural
tests establishes request compatibility, not model quality. Main formation, vision,
consolidation, and segmentation prompts participate in their existing derived recipe identities;
the shared JSON instruction also participates in schema recipe identities. Changed wording in
these components changes the corresponding default recipe. Use fresh directories for paired
evaluations and apply the existing store compatibility rules to previously derived content.
