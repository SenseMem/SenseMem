# Configuration examples

These files are starting points. They do not install model servers, supply credentials, or
establish benchmark scores. Run repository commands from the repository root, using the locked
environment in [Contributing](../../CONTRIBUTING.md#setup).

## Choose an example

| File | Purpose | Prerequisites |
| --- | --- | --- |
| [eval.example.yaml](eval.example.yaml) | Annotated evaluation template | Configure an embedding endpoint and reader; optional local speech requires `local` |
| [atm-raw-qwen-wemm.yaml](atm-raw-qwen-wemm.yaml) | Raw ATM-Bench media with descriptions and bounded recall | Compatible WeMM and Qwen endpoints, `benchmarks,openai`, and the ATM corpus |
| [atm-sgm-detail-qwen3-vl.yaml](atm-sgm-detail-qwen3-vl.yaml) | Fused source descriptions plus native visual reranking | Caller-supplied fused corpus, embedding/reader/judge endpoints, and CUDA reranking; see the [recipe](../atm-fused-retrieval.md) |
| [atm-sgm-max-thinking-vllm.yaml](atm-sgm-max-thinking-vllm.yaml) | Fused ATM corpus, maximum reader thinking, and remote text/media reranking | WeMM, DeepSeek, vLLM rerank and judge endpoints; see the [profile](../atm-max-thinking.md) |
| [mem-gallery-qwen-wemm.yaml](mem-gallery-qwen-wemm.yaml) | Image-dialogue evaluation with query vision | Compatible WeMM and Qwen endpoints, `benchmarks,openai`, and Mem-Gallery inputs |
| [baselines/rtx5090-qwen38-wemm9b-text.yaml](baselines/rtx5090-qwen38-wemm9b-text.yaml) | Reference text-workload measurement profile | Access to the declared remote services; review the [baseline protocol](../benchmarking.md#rtx-5090-qwen38-and-wemm-reference-baseline) |
| [baselines/rtx5090-qwen38-wemm9b-media.yaml](baselines/rtx5090-qwen38-wemm9b-media.yaml) | Reference media-workload profile with local speech | The same services plus local FunASR, media tools, and the selected corpora |

All seven files contain a harness-owned `benchmark` section and are inputs to
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
Model examples omit an explicit `revision`; local loaders resolve the Hub commit at load time.
Keep the resolved model identity with results, and preserve compatible model/cache state when
reopening a stored embedding space.

Credentials can come from each slot's explicit `api_key` or the documented environment lookup.
An explicit placeholder such as `EMPTY` overrides environment lookup. Remove or replace it when a
service requires a real key. These YAML files do not interpolate `${ENV}` values. Keep real
credentials in ignored local files or the process environment.

The RTX 5090 profiles describe a particular reference deployment. Their remote service hardware
is not the benchmark client's GPU, and their URLs are not public hosted SenseMem endpoints.
Replace them only in a local copy and report that change as a different measurement condition.

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
