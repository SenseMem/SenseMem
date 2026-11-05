# Use a local WeMM embedding service

Connect an already-running WeMM service through the OpenAI-compatible embedding adapter. This
example expects a `messages` endpoint serving `tencent/WeMM-Embedding-2B` with 2,048-dimensional
vectors. Configure generation separately. SenseMem does not install or launch either model service.

```yaml
data_dir: /absolute/path/to/a-new-sensemem-store

embedding:
  provider: openai
  base_url: http://127.0.0.1:18871/v1
  api_key: EMPTY
  model: tencent/WeMM-Embedding-2B
  dimension: 2048
  modalities: [text, image, video]
  request_format: messages
  timeout: 120.0
  max_retries: 0

generation:
  provider: openai
  base_url: https://your-qwen-endpoint.example/v1
  api_key: replace-me
  model: Qwen3.8-27B
  modalities: [text, image, video]
  video_limit: 8
  extra_body:
    chat_template_kwargs:
      enable_thinking: false

speech:
  provider: funasr
  device: cuda

settings:
  index_speech: true
```

Replace the generation URL, model name, and credential placeholders with your deployment's
values, then save the YAML as `config.yaml`. Remove a slot's `api_key` to use SDK environment
lookup instead; an explicit placeholder overrides that lookup. Install the SDK surfaces and the
host's YAML parser:

```bash
uv add "sensemem[openai,local]" pyyaml
```

Decode the file before opening the configuration with the public SDK:

```python
from pathlib import Path

import yaml

from sensemem import Memory

config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
with Memory.from_config(config) as memory:
    stored = memory.add("The spare key is in the blue toolbox.")
    print(stored.id)
    print(memory.search("Where is the spare key?").hits)
```

Give every live instance a different `data_dir`. The directory is the durable memory domain and
has one owner at a time.

The local WeMM service in this recipe has a 2048-vector `messages` contract. It accepts text,
images, and videos; it does not accept audio embeddings. Configure FunASR to turn audio and video
audio into searchable text, then keep `index_speech` enabled. This adds ASR work on ingestion.

Verify your service's batch, inline-byte, and video-frame limits before ingestion. The SDK's
own transport bounds remain in force; see the [adapter limits](api/python-adapters.md).

The local server runtime, model checkpoint, and service script are machine-managed dependencies;
they are not installed by SenseMem. Use the official
[WeMM-Embedding-2B model card](https://huggingface.co/tencent/WeMM-Embedding-2B) to obtain the
checkpoint and its serving prerequisites, then configure the service to expose the `messages`,
2048-dimension contract shown above. Keep its runtime and model files outside the SenseMem
environment. Pin its model revision and serving recipe so reopening a store uses the same
embedding space; a matching model alias alone does not establish compatibility.

## Optional typed-memory formation

Formation is off unless the configuration includes a `formation` slot. It uses a separate
OpenAI-compatible completion client on the write path, so choose the endpoint's output budget and
reasoning settings explicitly:

```yaml
formation:
  provider: openai
  base_url: https://your-qwen-endpoint.example/v1
  api_key: replace-me
  model: Qwen3.8-27B
  modalities: [text]
  temperature: 0.0
  seed: 0
  max_tokens: 1024
  timeout: 120.0
  max_retries: 0
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
```

`add_many` makes one formation request for the supplied `Sequence[ContentInput]`. If an endpoint
needs smaller requests, choose that application-level batch size before calling the public method:

```python
from sensemem import Memory

# Reuse the decoded config above after adding its optional formation section.
with Memory.from_config(config) as memory:
    observations = (
        "Ada said she prefers tea.",
        "Ada brought a blue mug to the meeting.",
    )

    for start in range(0, len(observations), 2):
        memory.add_many(observations[start : start + 2])

    # After addressing a formation failure, resume its committed sources.
    pending = memory.pending_captures()
    if pending:
        memory.settle(memory_ids=[item.memory_id for item in pending])
```

Choose a batch size within the SDK's 1,024-input `add_many()` limit and the endpoint's request
and output budgets. A source is committed before its formation call, so a formation failure does
not discard the source. Inspect `memory.pending_captures()` and, after addressing the failure,
resume the returned memory IDs with `memory.settle`. Settlement can retry model work; its attempt
and selection controls are documented with the
[public memory operations](api/python-memory.md).
