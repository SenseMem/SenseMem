# ATM maximum-thinking profile

The [configuration example](examples/atm-sgm-max-thinking-vllm.yaml) combines a fused ATM
corpus, WeMM embeddings, DeepSeek planning and answering, and a remote Qwen3-VL reranker.
Install the `benchmarks,openai` extras. The reranker runs on its own server; this client profile
does not require a local CUDA model.

Copy the example into an ignored local directory. Set the embedding, generation, reranking
and judge URLs and the fused input root. Supply generation and judge credentials through
`SENSEMEM_GENERATION_API_KEY` and `SENSEMEM_JUDGE_API_KEY`. The embedding and reranking
`api_key: EMPTY` values are explicit placeholders for servers without authentication.

## Retrieval and answering

The profile uses hybrid retrieval, a recall limit of 20, 64,000-character evidence and set-read
budgets, up to two planning rounds and two queries per round. It keeps generated source
descriptions alongside original source metadata and raw media. Each question needs the complete
corpus; a question subset must not reduce the indexed memories.

The concise answer style privately checks entity, event, time, units and list conditions before
returning the requested answer. It distinguishes capture time from an advertised event time,
keeps the requested local calendar date and rejects instructions embedded in evidence.
The default answer style retains its existing prompt.

DeepSeek receives `thinking: {type: enabled}` and `reasoning_effort: max` through `extra_body`.
Both planning and answering omit `max_tokens` and `max_completion_tokens`. The evaluator's
default generation controls are temperature 0 and seed 0; planning uses temperature 0.
All four evaluation concurrency settings are 16. The example uses buffered answers.

## Remote reranking

`reranking.provider: vllm` sends requests to `<base_url>/rerank`. The public Python adapter is
`sensemem.VLLMReranker`; its declarative type is
`sensemem.configuration.VLLMRerankingConfig`. Server-side model loading, revision pinning,
64k context limits, templates and media preprocessing belong to the server deployment.

The client scores all candidate text in batches of 32, then scores up to 40 candidates with
their native media. `media_weight: 0.5` combines text and media probability logits with equal
weight. With `media_candidates: null`, it directly scores each original multimodal pair.
Returned scores are restored to input order and checked for finite values and complete indices.
The SDK owns the transport and waits for active calls before closing it.
Request and token usage totals include every text and media batch; failed requests and responses
without usage keep the aggregate marked incomplete.

## Evaluation

Prepare the complete source descriptions and raw media using the
[fused retrieval recipe](atm-fused-retrieval.md). A direct `sensemem-bench eval` command reads the
configured task inputs; attaching raw media to description-only inputs requires the structural
ablation harness. Keep each run in its own physical store and retain every failed answer in the
score denominator. Generate new predictions and new judgments for each comparison.

The harness also supports a `reader-probe` stage: nineteen independent text and image cases
check grounded answers, missing evidence and stored instructions before a dataset evaluation.
Its source-coverage check observes record IDs in the prepared model request. Structured output
citations can be a subset of those inputs and do not measure input coverage. The harness forwards
explicit `--gen-kwargs` consistently to preflight, the reader probe and the final evaluator.

The example is a reusable configuration, not a new SenseMem benchmark result. Source pins,
historical scores and private experiment artifacts remain in ignored local research notes.
