# Example catalog

Replay and recall real audio and video with the Python SDK, or choose a configuration for an
evaluation. Run commands from the repository root with the locked environment in
[Contributing](../../CONTRIBUTING.md#setup).

## Replay and recall media in a notebook

The [best practices notebook](../../examples/best_practices.ipynb) uses the bundled
[audio and video](../../examples/media/README.md) to demonstrate concurrent replay, semantic
event formation, time-filtered recall, and playback of the original evidence. It needs the
`local` and `openai` extras, Jupyter, and `ffmpeg` on `PATH` for event playback.

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` (the provider's OpenAI-compatible `/v1` endpoint)
in the process that launches Jupyter. You can copy [`.env.example`](../../.env.example) to a
local `.env`, set both values, and load that file explicitly when launching Jupyter:

```bash
uv run --locked --default-index https://pypi.org/simple --extra local --extra openai \
  --with jupyterlab --env-file .env jupyter lab examples/best_practices.ipynb
```

If you export the variables in your shell, omit `--env-file .env`. Never put a real API key
in a notebook cell. Set `MODEL` in the first configuration cell to your deployed generation
model (the example uses `qwen3.8-omni-flash`). The endpoint must accept audio and video inputs
and structured responses for event segmentation; adjust provider-specific `extra_body` settings
if needed. The `embedding` block separately selects the local `google/embeddinggemma-2` model.
See [model setup](../configuration/models.md#embeddinggemma-2) for its requirements.

Run the cells in order with `examples/` as the kernel's working directory so `media/` resolves.
Searches run against a fresh temporary store; the notebook reports empty results rather than
assuming a model will always find media or form events.

## Choose an evaluation configuration

Connect the model services for your chosen example and run it with `sensemem-bench eval --config`.

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
