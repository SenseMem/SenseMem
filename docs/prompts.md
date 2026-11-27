# Maintain prompts

SenseMem's built-in model instructions and text composers live in [`src/sensemem/prompts/`](../src/sensemem/prompts/). Edit the component that owns an operation, then verify its direct callers and quality fixtures. These modules are internal components; supported SDK imports remain those exported from `sensemem`.

## Runtime components

| Component | Owns |
| --- | --- |
| `generation.py` | Answers, formation, consolidation, event segmentation, vision descriptions, recall plans, output protocols, and conditional instructions. |
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

Prompt components are pure: they do not open storage, call a model, or import the kernel, provider adapters, API transports, or benchmark package. The kernel decides retrieval scope, budgets, evidence membership, and authority; components render those decisions and adapters encode them in the provider's request format.

The text-only recall backend contract carries a host-generated capability suffix in `corpus_digest`; rendering and budget recognition share one template in `recall.py`. Caller-supplied agent instructions and transcription hints remain caller inputs.

## Review prompt behavior

| Component | What to check |
| --- | --- |
| `generation.py` | Evidence policy remains consistent across fragments; JSON types match the schema; unreadable or irrelevant inputs produce a non-fabricating response. |
| `recall.py` | Read coverage stays distinct from answer-set completeness; trusted budgets and reference time remain explicit. |
| `events.py` | Model summaries are quoted data with separate original-media context; an unfinished synopsis cannot establish completion. |
| `evidence_operations.py` | Instructions describe function arguments rather than invented executor results; uncertainty remains represented in the schema. |
| `agent.py` | Tools serve the user's task, arguments follow their schemas, and errors are handled by their actual meaning. |
| `embedding.py` | Prefixes match the checkpoint's trained format. |
| `reranking.py` | Instruction and chat template match the checkpoint rather than general generation instructions. |
| `context.py` | Stored and temporary evidence share the same trust boundary; rendering and character accounting use the same guides. |
| `tools.py` | Descriptions state when to use each tool, where IDs come from, and what returned references mean. |
| `mcp.py` | Descriptions match the runtime, optional planners, and host-selected authority switches. |
| `demo.py` | Historical answers wait for search results; empty results preserve uncertainty and sampled frames do not imply continuous motion. |

Keep tasks and contracts direct. Use headings or quoted structured values to separate instructions, examples, and dynamic evidence. Retain examples that clarify a wire contract and verify them against the actual parser. State what the model should do when evidence is insufficient under both strict and best-effort policies.

## Benchmark components

Benchmark-only prompts stay under [`src/sensemem/benchmarks/`](../src/sensemem/benchmarks/); product modules must not import them:

- `prompts.py` owns dataset query specifications and shared question wrappers.
- `baseline_prompts.py` owns blind, full-context, and compile-reader instructions.
- `_official/query_prompts.py` and `_official/judge_prompts.py` hold pinned upstream query and judge wording.
- [`_official/NOTICE.md`](../src/sensemem/benchmarks/_official/NOTICE.md) records pinned upstream sources and retained adaptations.

Reuse pinned official query or judge components when available and preserve their wording, rubric, and provenance. Dataset answers and scorer-only fields must never enter answer-generation prompts.

## Validate a change

Component-boundary tests keep provider and benchmark dependencies out of runtime prompts and verify rendered schemas and examples. For wording or policy changes, run the relevant suites in the [model quality gate](model-quality.md) and record model revision, fixtures, and pass outcomes. Formation, vision, consolidation, and event-segmentation prompt changes alter their derived recipe identities, so use fresh directories for paired evaluations.
