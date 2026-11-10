# Configuration examples

Choose a configuration for your evaluation, connect its model services, and run it with
`sensemem-bench eval --config`. The catalog below lists each example's purpose and prerequisites.

Run repository commands from the repository root with the locked environment in
[Contributing](../../CONTRIBUTING.md#setup).

## Choose an example

| File | Purpose | Prerequisites |
| --- | --- | --- |
| [eval.example.yaml](eval.example.yaml) | Annotated evaluation template | Configure an embedding endpoint and reader; optional local speech requires `local` |
| [mosaic-embeddinggemma2-qwen38.yaml](mosaic-embeddinggemma2-qwen38.yaml) | Native multimodal MOSAIC baseline | Local multimodal EmbeddingGemma 2 endpoint, Qwen reader and judge, and `benchmarks,openai`; see the [run guide](../benchmarks/mosaic-native.md) |
| [atm-sgm-max-thinking-vllm.yaml](atm-sgm-max-thinking-vllm.yaml) | ATM source descriptions with bounded recall, reader reasoning, and remote reranking | Local ATM SGM corpus and compatible WeMM, DeepSeek, vLLM rerank, and judge endpoints; install `benchmarks,openai` |
| [mem-gallery-qwen-wemm.yaml](mem-gallery-qwen-wemm.yaml) | Image-dialogue evaluation with query vision | Compatible WeMM and Qwen endpoints, `benchmarks,openai`, and Mem-Gallery inputs |

Each YAML combines product model configuration with a harness-owned `benchmark` section.
The evaluator allocates physically isolated stores for independent units.

To reuse a composition in your application, decode the YAML, remove `benchmark`, and set your
own `data_dir` before calling `Memory.from_config()`.

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

The integrity command checks the local corpus without model calls. Continue with
[benchmarking](../benchmarking.md) to download inputs, run a bounded evaluation, and interpret
results. [Benchmark results](../benchmarks/results.md) documents measured configurations.
