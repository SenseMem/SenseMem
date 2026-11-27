# Model quality gate

Validate prompt, model, and retrieval-policy changes against focused real-service fixtures before deploying them. Unlike credential-free unit tests, these opt-in suites exercise real model behavior through the public SDK and backend contracts. Use [Benchmarking](benchmarking.md) for full dataset comparisons.

## Answering and workflow gates

Set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for a compatible endpoint ending in `/v1`), then choose the model snapshot in `MODEL_QUALITY_MODEL`:

```bash
uv run --frozen python -m sensemem.benchmarks.model_quality \
  --model "$MODEL_QUALITY_MODEL" \
  --response-format json_schema \
  --repeats 3 \
  --output .benchmarks/model-quality/report.json

uv run --frozen python -m sensemem.benchmarks.workflow_quality \
  --model "$MODEL_QUALITY_MODEL" \
  --response-format json_schema \
  --repeats 3 \
  --output .benchmarks/model-quality/workflow.json
```

Use `--response-format json_object` for endpoints that require JSON-object mode. Both runners allocate a fresh store per case and repeat, require every repeat to pass, and write aggregate reports containing fixture version, configured and returned model IDs, case outcomes, timing, and stable error codes without logging prompts, answers, credentials, or provider error bodies.

- **Answering gate (`sensemem.benchmarks.model_quality`):** tests grounded facts, insufficient-evidence abstention, prompt injection inside stored content, wrong-person attribution, conflicting reports, unsupported citation instructions, and explicit corrections.
- **Workflow gate (`sensemem.benchmarks.workflow_quality`):** tests relative-date formation, recall-planned collection counting, visual prompt injection, on-demand media diagnosis, adversarial consolidation preserving host response policies, control-plane consolidation and rollback, and multilingual answering.

## Opt-in integration fixtures

Focused pytest integration suites verify individual capabilities when enabled by environment flags. Supply `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and any suite-specific model/endpoint variables in the process environment.

| Capability | Command | Key variables |
| --- | --- | --- |
| Prompt stage contracts | `SENSEMEM_PROMPT_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_prompt_quality.py` | `SENSEMEM_PROMPT_EVAL_MODEL`, `SENSEMEM_PROMPT_EVAL_MODEL_REVISION` |
| Affect and preferences | `SENSEMEM_AFFECTIVE_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_affective_memory_quality.py` | `SENSEMEM_AFFECTIVE_EVAL_MODEL`, `SENSEMEM_AFFECTIVE_MODEL_REVISION` |
| EmbeddingGemma 2 loading | `SENSEMEM_EMBEDDINGGEMMA2_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_embeddinggemma2_quality.py` | Local CUDA GPU with BF16 (`sensemem[local]`) |
| Formation relative time | `SENSEMEM_FORMATION_TIME_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_formation_time_quality.py` | Standard OpenAI SDK variables |
| Event segmentation | `SENSEMEM_EVENT_SEGMENTATION_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_event_segmentation_quality.py` | `SENSEMEM_EVENT_MODEL` |
| Native audio/video segmentation | `SENSEMEM_EVENT_NATIVE_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_event_segmentation_quality.py -k native` | `SENSEMEM_EVENT_NATIVE_API_KEY`, `SENSEMEM_EVENT_NATIVE_BASE_URL`, `SENSEMEM_EVENT_NATIVE_MODEL`, `SENSEMEM_EVENT_NATIVE_MODEL_REVISION` |
| Synopsis and native-count delivery | `SENSEMEM_SYNOPSIS_DELIVERY_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_event_synopsis_delivery_quality.py` | `SENSEMEM_PROJECTION_*` endpoint, key, and model variables |
| Positive native audio count | `SENSEMEM_NATIVE_COUNT_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_native_count_positive_quality.py` | `SENSEMEM_PROJECTION_*` endpoint, key, and model variables |
| Goal-directed context compilation | `SENSEMEM_CONTEXT_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_goal_context_quality.py` | `SENSEMEM_CONTEXT_EVAL_MODEL`, `SENSEMEM_EMBEDDING_BASE_URL` |
| Agent tool reasoning | `SENSEMEM_AGENT_TOOL_EVAL=1 uv run --frozen pytest -W error -rP tests/integration/test_agent_tool_quality.py` | Standard OpenAI SDK variables |

### Check EmbeddingGemma 2 loading

`tests/integration/test_embeddinggemma2_quality.py` compares automatic BF16 and selective modality loading against a full-modal FP32 baseline on a CUDA GPU. It checks cosine agreement, weight-memory reduction, and multilingual retrieval isolation on fresh SDK stores.

### Check affective and attribution behavior

`tests/integration/test_affective_memory_quality.py` covers subject attribution, text/image cue disagreement, transient affect vs. corroborated traits, preference replacement, embedded instructions, and typed target/elicitor attribution. See [affective memory](affective-memory.md) for the representation and visibility rules.

### Check agent tool reasoning

`tests/integration/test_agent_tool_quality.py` exercises `OpenAIAgent` and `AgentRunner` across required tool execution, recovery from local argument validation errors, and resistance to instructions embedded in tool output. See [optional agents](api/python-agents.md).

## Configure CI

The [model-quality workflow](../.github/workflows/model-quality.yml) runs the answering and workflow gates on pull requests that touch product code, dependencies, integration fixtures, or the gates, and on weekly schedules when `MODEL_QUALITY_ENABLED=true` is set.

1. Set the `MODEL_QUALITY_API_KEY` repository secret and `MODEL_QUALITY_MODEL` repository variable.
2. Optionally set `MODEL_QUALITY_BASE_URL` and `MODEL_QUALITY_RESPONSE_FORMAT` for a compatible endpoint.
3. Set the `MODEL_QUALITY_ENABLED` repository variable to `true` to enable pull-request and scheduled weekly runs.

Manual workflow dispatches accept model, response-format, and repeat count inputs. Fork pull requests do not receive repository secrets and require a maintainer-triggered run.
