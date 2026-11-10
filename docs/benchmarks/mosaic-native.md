# MOSAIC native multimodal baseline

Run SenseMem with native text, image, audio, and video evidence on MOSAIC. The configuration
combines `google/embeddinggemma-2` and `qwen3.8-omni-flash` and reaches **77.76% mean judge
score** on answerable questions, using `qwen3.7-plus` as judge.

Original media goes directly to the embedder and reader, with no separate captioner, speech
recognizer, or reranker. See [benchmark results](results.md) for the headline and task breakdowns.
This guide provides the service setup, input protocol, and reproduction commands.

| Metric | Result | Population |
| --- | ---: | --- |
| Mean judge score | 0.777551 | 784 answerable questions |
| Correct refusal rate | 53.2374% | 74 / 139 unanswerable questions |
| False refusal rate | 0.5102% | 4 / 784 answerable questions |

The 2026-11-06 measurement covers all 923 questions across 70 personas. Scheduled answerable
questions remain in the denominator, with request failures scored as zero. The
[configuration](../examples/mosaic-embeddinggemma2-qwen38.yaml) and
[measurement summary](runs/mosaic-native-2026-11-06.json) retain the exact values and completion counts.

## Model services

Install the benchmark client from the repository root:

```bash
uv sync --locked --default-index https://pypi.org/simple --extra benchmarks --extra openai
```

The embedding service must accept OpenAI-compatible multimodal `/embeddings` messages.
The measured service used vLLM 0.33.0, an 8,192-token context, and 768-dimensional embeddings.
For a separate environment with that vLLM version:

```bash
vllm serve google/embeddinggemma-2 --runner pooling \
  --max-model-len 8192 --host 127.0.0.1 --port 8000
```

The reader and judge use a compatible Chat Completions service. Supply its endpoint and key
through the environment; the YAML does not interpolate environment variables:

```bash
export SENSEMEM_GENERATION_BASE_URL='https://your-gateway.example/v1'
export SENSEMEM_GENERATION_API_KEY='your-reader-key'
```

The judge uses the same endpoint and key by default. Set `SENSEMEM_JUDGE_BASE_URL` and
`SENSEMEM_JUDGE_API_KEY` if it is served separately. Both configured model IDs must be available.
Reader thinking is disabled with `reasoning_effort: none`; the MOSAIC judge uses temperature 0,
4,096 output tokens, `thinking.type: disabled`, and `enable_thinking: false`.

Record the resolved embedding, reader, and judge revisions with new measurements. Service aliases
identify the requested models; the measurement summary retains the available model identities.

## Download and run

Download the pinned annotations and original media, then check the corpus without model calls:

```bash
uv run --frozen sensemem-bench eval \
  --config docs/examples/mosaic-embeddinggemma2-qwen38.yaml \
  --check-integrity --download
```

Run the full evaluation:

```bash
uv run --frozen sensemem-bench eval \
  --config docs/examples/mosaic-embeddinggemma2-qwen38.yaml \
  --limit all --no-download \
  --data-root .benchmarks/runs/mosaic-native-seed0 \
  --output-path .benchmarks/results/mosaic-native-seed0 \
  --run-id mosaic-native-seed0 \
  --use-cache .benchmarks/cache/mosaic-native-seed0
```

Use `--limit 1` with separate run, output, and cache paths for a one-question smoke run.
It still indexes that question's complete persona history. An interrupted run can continue
with the same command plus `--resume`. For a completed run with failed requests, keep its
outputs and use a new output path with the same run ID, store root, cache, and `--resume`.
Successful cached answers and verdicts are reused. See [cache and resume](reference.md#cache-and-resume).

The evaluator writes `config.yaml`, `results.jsonl`, and `samples.jsonl`.
Despite its name, `results.jsonl` contains one JSON document. Keep these files with the source
commit and model-service versions when recording a new measurement.

## Input and scoring protocol

Annotations use the [pinned MOSAIC release](https://huggingface.co/datasets/gzone0111/MOSAIC/tree/c4567178d7430d87484db1ab207f5c8aaa26199a).
Reader instructions and scoring follow the
[upstream evaluator](https://github.com/HKUST-KnowComp/MOSAIC/tree/82dc5ccf8063f41b28145f0c0e49747549aff457).
The summary and task catalog retain exact input and scorer identities.

Each persona owns a separate physical SenseMem store. Sessions are deduplicated by session ID
and ordered by timestamp; text chunks retain visible conversation text and media chunks retain
their original image, audio, or video. Answers, aliases, generation prompts, and evidence labels
stay outside memory inputs. Labels are used only for scoring.

MOSAIC text records split losslessly at 16,000 characters, preserving the original source ID
and fitting the model's shared context window. The embedding adapter applies
`task: search result | query:` to text queries and `title: none | text:` to documents,
each followed by a space, with special tokens enabled.

The public CLI sends inline media. The recorded measurement also used verified original assets
from the official CDN after SDK media admission, preserving evidence selection and part order.
Its summary records that transport condition. Account for inline payload size when sizing your
gateway.

The headline is the mean judge score over answerable questions. Refusal rates use MOSAIC's
refusal-phrase matcher and separate answerable/unanswerable denominators. The measurement covers
the full embedding, memory, and reader combination. Use the [benchmarking guide](../benchmarking.md)
to add a blind-reader control when evaluating the contribution of memory.
