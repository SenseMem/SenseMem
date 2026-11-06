# MOSAIC native multimodal baseline

SenseMem with `google/embeddinggemma-2` and `qwen3.8-omni-flash` scored **77.7551%**
on MOSAIC on 2026-11-06. The judge was `qwen3.7-plus`.
The [configuration](../examples/mosaic-embeddinggemma2-qwen38.yaml) and
[run summary](runs/mosaic-native-2026-11-06.json) record the settings and measurements.

| Metric | Result | Population |
| --- | ---: | --- |
| Mean judge score | 0.777551 | 784 answerable questions |
| Correct refusal rate | 53.2374% | 74 / 139 unanswerable questions |
| False refusal rate | 0.5102% | 4 / 784 answerable questions |

All 923 questions across 70 personas were attempted; 921 completed with reader answers and
judge verdicts. Two reader requests returned HTTP 400 `data_inspection_failed` after retries.
Both count as zero in the 784-question denominator. The run summary includes their question IDs
and scores by question type.

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

The observed local embedding cache revision was
`f9972ac22e696a105c482202646979e33e4b5a1e`; the running server did not pin its revision.
Reader and judge IDs are service aliases. Record resolved model revisions with new runs when
the services expose them.

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

Annotations come from
[`gzone0111/MOSAIC@c4567178d743`](https://huggingface.co/datasets/gzone0111/MOSAIC/tree/c4567178d7430d87484db1ab207f5c8aaa26199a).
Reader instructions and scoring follow
[`HKUST-KnowComp/MOSAIC@82dc5ccf8063`](https://github.com/HKUST-KnowComp/MOSAIC/tree/82dc5ccf8063f41b28145f0c0e49747549aff457).
Exact revisions, annotation and question-roster hashes are in the run summary and catalog.

Each persona owns a separate physical SenseMem store. Sessions are deduplicated by session ID
and ordered by timestamp; text chunks retain visible conversation text and media chunks retain
their original image, audio, or video. Answers, aliases, generation prompts, and evidence labels
stay outside memory inputs. Labels are used only for scoring.

MOSAIC text records split losslessly at 16,000 characters, preserving the original source ID.
This produced 26,910 SDK records from 26,884 native chunks; all actual SDK text inputs, including
source labels and embedding prefixes, fit within 6,004 tokens. The embedding adapter applies
`task: search result | query:` to text queries and `title: none | text:` to documents,
each followed by a space, with special tokens enabled.

Images, audio, and video go directly to the embedder and reader. There is no separate ASR,
captioner, reranker, or auxiliary reader. The public CLI sends inline media; the measured run
also used signed official CDN URLs for original assets after matching their SHA-256 and size
to the pinned release. This transport substitution occurred after the SDK's media admission,
preserving selected evidence, part order, and fitted payloads. Large inline requests can take
longer on some gateways.

The headline is the mean judge score over answerable questions, including failed questions at
zero. Refusal rates use MOSAIC's refusal-phrase matcher and separate answerable/unanswerable
denominators. This measures the full embedding, memory, and reader combination; no blind reader
control was run.

Before the full run, real-service synthetic checks passed for image text (`ORCHID 714`), a tone
audio clip, red-before-blue video order, combined media, ignoring an instruction embedded beside
a stored fact, and abstaining on an unknown bank PIN. These checks used no MOSAIC answers.
