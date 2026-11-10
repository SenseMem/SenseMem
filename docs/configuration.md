# Configuration

Choose your models without changing the memory API. Configure an embedding backend to enable
storage and retrieval, then add answering, perception, formation, or maintenance independently.
All construction paths validate the composition before opening the same embedded kernel.

## 1. Install only the surfaces you use

| Extra | Adds |
| --- | --- |
| `local` | Sentence Transformers embedding/text reranking and FunASR speech |
| `openai` | OpenAI SDK 3.x, its httpx2 transport, and media handling |
| `face` | Local OpenCV face analysis |
| `gazetteer` | Lookup in a caller-supplied pinned map index |
| `reranking` | Local Qwen3-VL vLLM classifier on Linux x86-64/CUDA |
| `server` / `mcp` | REST / MCP transport |
| `observability` | OpenTelemetry export |
| `benchmarks` | Evaluation/download tooling; datasets download separately |
| `all` | Union of the optional surfaces |

The base package includes local storage, Zvec, public values, and backend protocols.
The runtime supports Python 3.11–3.14; verify the selected optional runtimes on your target platform.

```bash
uv add "sensemem[local,openai]"
```

## 2. Declare a composition

`embedding` is required; `data_dir` defaults to `.sensemem`. Other slots are optional.
This configuration uses local embedding and an OpenAI-compatible answerer:

```python
from sensemem import Memory

config = {
    "data_dir": "./data/assistant",
    "embedding": {
        "provider": "sentence-transformers",
        "model": "google/embeddinggemma-2",
        "device": "cpu",
    },
    "generation": {
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07",
    },
}
with Memory.from_config(config) as memory:
    memory.add("The spare key is in the blue toolbox.")
    print(memory.ask("Where is the spare key?").answer)
```

Set `OPENAI_API_KEY` in the host environment. Each OpenAI slot can instead declare its own
endpoint, credentials, timeout, and retries; see [provider fields](configuration/providers.md).
Unknown fields/providers and out-of-range values are rejected. Declarative composition owns its
adapters. `AsyncMemory.from_config()` accepts the same shape; use
`SenseMemConfig.model_validate()` to validate without opening storage.

### Loading a configuration file

The SDK accepts a mapping or `SenseMemConfig`. The host decodes files:

```python
import json
from pathlib import Path

from sensemem import Memory

config = json.loads(Path("config.json").read_text(encoding="utf-8"))
with Memory.from_config(config) as memory:
    print(memory.capabilities.embedding_model)
```

For YAML, install a host parser and use `yaml.safe_load()`. The product SDK does not depend on
YAML; the benchmark extra's `--config` handles it separately. SenseMem does not automatically
load `.env`; hosts can use `uv run --env-file .env python application.py`.

## 3. Add capabilities

Each slot enables its own capability. Setting `generation` does not enable formation, vision,
or consolidation.

| Need | Slot or policy | Guide |
| --- | --- | --- |
| Select local/remote embedding | `embedding` | [Embedding choices](configuration/models.md#embedding-choices) |
| Rerank text or multimodal candidates | `reranking` | [Semantic reranking](configuration/models.md#semantic-reranking) |
| Transcribe and identify speech | `speech` | [FunASR](configuration/models.md#local-speech-with-funasr) |
| Index visual captions/OCR | `vision` | [Visual descriptions](configuration/models.md#visual-descriptions) |
| Analyze faces | `face` | [Provider fields](configuration/providers.md#local-provider-fields) |
| Derive typed memory after writes | `formation` | [Automatic formation](configuration/lifecycle.md#automatic-memory-formation) |
| Propose deliberate memory changes | `consolidation` | [Memory management](configuration/lifecycle.md#agentic-memory-management) |
| Change retrieval, grounding, or identity thresholds | `settings` | [Runtime settings](configuration/settings.md) |
| Declare explicit physical retention | Top-level `retention` | [Retention](configuration/lifecycle.md#retention-policy) |

For native media, declare the modalities the deployed endpoint accepts. Add matching speech or
vision backends for caption/transcript routes. `memory.capabilities` describes the resolved
composition; use the [quality gate](model-quality.md) to qualify it for your tasks.

## 4. Choose an ownership boundary

| Construction | Use when |
| --- | --- |
| `Memory.from_config()` | Bundled providers and pure data are enough |
| `Memory.from_plugins()` | Constructed backends should travel as one `MemoryPlugins` value |
| `Memory(...)` | The host injects custom backends or owns provider clients |

### Direct adapter injection

```python
from openai import OpenAI

from sensemem import Memory, OpenAIModels, SentenceTransformersEmbedder

with OpenAI(timeout=30.0, max_retries=3) as client:
    models = OpenAIModels(
        generation_client=client,
        generation_model="gpt-5-mini-2025-08-07",
    )
    with Memory(
        "./data/assistant",
        embedder=SentenceTransformersEmbedder(model_id="google/embeddinggemma-2"),
        answerer=models,
    ) as memory:
        memory.add("The spare key is in the blue toolbox.")
        print(memory.ask("Where is the spare key?").answer)
```

`Memory` closes each distinct adapter once. `OpenAIModels` leaves caller-supplied clients open;
the outer context owns that client. One compatible adapter may fill several slots, such as
`answerer`, `former`, and `consolidator`. Pass policy through `config=MemoryConfig(...)`
and tracing through the separate `tracer=` argument.

`resolve_memory_config()` returns a `MemoryComposition` for hosts needing to assemble before
opening storage. Close it unless its plugins are transferred to a `Memory`. Exact constructors
and custom inference protocols are in the [SDK](api/python-sdk.md) and
[backend reference](api/python-backends.md).

## Before reopening a store

One physical directory has one live owner. Embedding space/dimension, persisted transcription/face
recipes, and evidence policy must match the store. Deliberate representation changes need replay
into a fresh directory; query-time policies such as reranking can change on reopen.

Use immutable model revisions for reproducible deployments and validate model/prompt changes
through the [quality gate](model-quality.md). See [representation changes](operations.md#changing-the-runtime-or-stored-representations)
for the required action per change, and [example configurations](examples/README.md) for runnable
compositions.
