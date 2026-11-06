# Configuration examples

These files are starting points. They do not install model servers, supply credentials, or
establish benchmark scores. Run repository commands from the repository root, using the locked
environment in [Contributing](../../CONTRIBUTING.md#setup).

## Choose an example

| File | Purpose | Prerequisites |
| --- | --- | --- |
| [eval.example.yaml](eval.example.yaml) | Annotated evaluation template | Configure an embedding endpoint and reader; optional local speech requires `local` |
| [mosaic-embeddinggemma2-qwen38.yaml](mosaic-embeddinggemma2-qwen38.yaml) | Native multimodal MOSAIC baseline | Local multimodal EmbeddingGemma 2 endpoint, Qwen reader and judge, and `benchmarks,openai`; see the [run guide](../benchmarks/mosaic-native.md) |
| [atm-sgm-max-thinking-vllm.yaml](atm-sgm-max-thinking-vllm.yaml) | ATM source descriptions with bounded recall, reader reasoning, and remote reranking | Local ATM SGM corpus and compatible WeMM, DeepSeek, vLLM rerank, and judge endpoints; install `benchmarks,openai` |
| [mem-gallery-qwen-wemm.yaml](mem-gallery-qwen-wemm.yaml) | Image-dialogue evaluation with query vision | Compatible WeMM and Qwen endpoints, `benchmarks,openai`, and Mem-Gallery inputs |

Each YAML file contains a harness-owned `benchmark` section and is an input to
`sensemem-bench eval --config`. To use a file with `Memory.from_config()`, decode it in the host,
remove `benchmark`, and set the application's own `data_dir`. Product configuration accepts no
`benchmark` key. The evaluator replaces `data_dir` with physically isolated per-unit stores.

## Customize locally

```bash
mkdir -p .benchmarks/configs
cp docs/examples/eval.example.yaml .benchmarks/configs/my-run.yaml
```

Edit the local copy before running it. Replace endpoint and model placeholders, verify the
embedding dimension and declared modalities, select the tasks, and choose a new output directory.
The annotated template selects full datasets; use `--limit 1` for an initial bounded run, noting
that one unit can contain many questions.
Model examples omit an explicit `revision`; local loaders resolve the Hub commit at load time.
Keep the resolved model identity with results, and preserve compatible model/cache state when
reopening a stored embedding space.

Credentials can come from each slot's explicit `api_key` or the documented environment lookup.
An explicit placeholder such as `EMPTY` overrides environment lookup. Remove or replace it when a
service requires a real key. These YAML files do not interpolate `${ENV}` values. Keep real
credentials in ignored local files or the process environment.

Check task names and input readiness before an evaluation:

```bash
uv run --frozen sensemem-bench eval --list-tasks
uv run --frozen sensemem-bench eval \
  --config .benchmarks/configs/my-run.yaml \
  --tasks locomo-refined --check-integrity --no-download
```

The integrity command requires the selected corpus locally and makes no model calls. It does
not validate model-service readiness or quality. See [benchmarking](../benchmarking.md) for
downloads, sampling units, resource requirements, and result interpretation.
