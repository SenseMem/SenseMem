# Application plugin factories

Compose custom models, computation policies, ingestion stages, and typed tools around the SenseMem kernel. `Memory.from_config(..., factories=...)` constructs application-owned integrations from configuration data using explicitly supplied Python callables, sharing the same capability validation and resource ownership as bundled providers.

## Kernel and plugin boundary

Plugins supply inference, scoring, selection, or interpretation; the kernel retains memory identity, validation, SQLite and media-CAS durability, the index outbox, SQLite hydration, consent, evidence closure, and budget admission. Composition stays fixed for one `Memory` lifetime—changing a backend requires closing and reopening the instance.

- **[Backend protocols](../api/python-backends.md):** integrate model inference into built-in memory slots (`embedder`, `answerer`, `transcriber`, `vision_describer`, `face_analyzer`, `former`, `consolidator`, `reranker`, `EventSegmentationBackend`). Compose multiple models cleanly with `GenerationRouter`, `EmbeddingRouter` (shared vector space and dimension), and `EvidenceGeneration` (media selection and source metadata enrichment).
- **[Computation policies and stages](../api/python-policies.md):** replace `QueryPolicy`, `CandidateSource`, `RankingPolicy`, and `ContextSelector`, or append ordered `IngestionStage` interpretations before built-in perception and embedding. Built-in defaults implement the same public protocols.
- **[Typed tools](../api/python-tools.md):** register arbitrary Pydantic-validated computations in `ToolRegistry` without adding a kernel slot, shared by fixed pipelines (`memory.call_tool`) and [optional agents](../api/python-agents.md).

Any backend whose output is persisted (`embedding_space`, `transcription_space`, `face_space`, `formation_space`, `vision_space`, or `IngestionStage.space`) must change its declared space whenever its model, prompt, or preprocessing recipe changes so cached derivations never mix incompatible recipes. Request-local components (`RerankingBackend`, `QueryPolicy`, `CandidateSource`, `RankingPolicy`, `ContextSelector`) have no persisted space and can change on reopen without replay.

## Define factories in the host

A factory has signature `Callable[[Mapping[str, object]], Closable]`. Backend factories return an object implementing the slot's backend protocol; tool factories return a `Tool`. Factories validate their own options:

```python
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from sensemem import (
    FunASRTranscriber,
    PluginFactory,
    SentenceTransformersEmbedder,
    Tool,
)


class EncoderOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    device: str = "cpu"


def build_encoder(options: Mapping[str, object]) -> SentenceTransformersEmbedder:
    values = EncoderOptions.model_validate(options)
    return SentenceTransformersEmbedder(model_id=values.model, device=values.device)


class SpeechOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "cpu"
    ncpu: int = 2


def build_asr(options: Mapping[str, object]) -> FunASRTranscriber:
    values = SpeechOptions.model_validate(options)
    return FunASRTranscriber(device=values.device, ncpu=values.ncpu)


class LookupOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: dict[str, str]


class LookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str


class LookupOutput(BaseModel):
    definition: str | None


def build_lookup(options: Mapping[str, object]) -> Tool[LookupInput, LookupOutput]:
    values = LookupOptions.model_validate(options)

    def lookup(value: LookupInput) -> LookupOutput:
        return LookupOutput(definition=values.entries.get(value.term))

    return Tool("lookup", "Look up an application glossary term", LookupInput, LookupOutput, lookup)


factories: dict[str, PluginFactory] = {
    "application_encoder": build_encoder,
    "application_asr": build_asr,
    "application_lookup": build_lookup,
}
```

Install the optional extras required by the adapters your factories construct (such as `sensemem[local]` above).

## Construct backend slots through factories

`MemorySpec` accepts either a bundled `provider` declaration or a `PluginConfig` (`factory` + `options`) in each backend field (`embedding`, `generation`, `speech`, `vision`, `face`, `formation`, `consolidation`, `reranking`, `query_policy`, `candidate_source`, `ranking_policy`, `context_selector`), plus `ingestion_stages` and `tools` lists:

```python
from sensemem import Memory, MemorySpec

config = MemorySpec.model_validate(
    {
        "data_dir": "./data/application",
        "embedding": {
            "factory": "application_encoder",
            "options": {"model": "google/embeddinggemma-2", "device": "cpu"},
        },
        "speech": {
            "factory": "application_asr",
            "options": {"device": "cpu", "ncpu": 2},
        },
        "tools": [
            {
                "factory": "application_lookup",
                "options": {"entries": {"episodic": "An observed event"}},
            }
        ],
    }
)

with Memory.from_config(config, factories=factories) as memory:
    result = memory.call_tool("lookup", {"term": "episodic"})
    print(result.document())
```

`Memory.from_config()` and `AsyncMemory.from_config()` also accept plain mappings. You can mix bundled `provider` slots with custom `factory` slots and `tools` in the same `MemorySpec`:

```python
from sensemem import Memory, MemorySpec

config = MemorySpec.model_validate(
    {
        "data_dir": "./data/bundled-with-tools",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "google/embeddinggemma-2",
        },
        "tools": [
            {
                "factory": "application_lookup",
                "options": {"entries": {"episodic": "An observed event"}},
            }
        ],
    }
)

with Memory.from_config(config, factories=factories) as memory:
    print(memory.call_tool("lookup", {"term": "episodic"}).document())
```

## Bundled factories and computation policies

Bundled factory names are reserved and cannot be overridden by the host. The catalog is per-instance with no global registry or dynamic module import:

| Factory | Role |
| --- | --- |
| `sensemem.embedding.openai` | `embedder` |
| `sensemem.embedding.sentence-transformers` | `embedder` |
| `sensemem.generation.openai` | `answerer` |
| `sensemem.formation.openai` | `former` |
| `sensemem.consolidation.openai` | `consolidator` |
| `sensemem.vision.openai` | `vision_describer` |
| `sensemem.speech.openai` | `transcriber` |
| `sensemem.speech.funasr` | `transcriber` |
| `sensemem.face.opencv` | `face_analyzer` |
| `sensemem.reranking.sentence-transformers` | `reranker` |
| `sensemem.reranking.qwen3-vl` | `reranker` |
| `sensemem.reranking.vllm` | `reranker` |
| `sensemem.query.default` | `query_policy` |
| `sensemem.candidates.default` | `candidate_source` |
| `sensemem.ranking.default` | `ranking_policy` |
| `sensemem.context.default` | `context_selector` |

Model factories accept the corresponding bundled provider's options without a `provider` field; the four default computation factories accept no options:

```python
from sensemem import DefaultRankingPolicy, Memory

with Memory.from_config(
    {
        "data_dir": "./data/mixed",
        "embedding": {
            "factory": "sensemem.embedding.sentence-transformers",
            "options": {"model": "google/embeddinggemma-2"},
        },
        "ranking_policy": {"factory": "application_ranking"},
        "context_selector": {"factory": "sensemem.context.default"},
    },
    factories={"application_ranking": lambda options: DefaultRankingPolicy()},
) as memory:
    memory.add("The spare key is in the workshop")
```

See [computation plugins](../api/python-policies.md) for the exact `QueryPolicy`, `CandidateSource`, `RankingPolicy`, `ContextSelector`, and `IngestionStage` contracts.

## Ordered ingestion stages

An `IngestionStage` declares a nonempty `name`, a nonempty durable recipe `space`, and `process(ModelInput) -> StageResult`. Stages run in declared order before built-in perception and embedding, appending labelled `model_interpretation` parts without altering original text, assets, identity, or metadata:

```python
from sensemem import ModelInput, StageResult


class WorkshopStage:
    name = "workshop-context"
    space = "workshop-context:v1"

    def process(self, input: ModelInput) -> StageResult:
        return StageResult("The source concerns a workshop" if "workshop" in input.text else "")

    def close(self) -> None:
        pass
```

Register a factory returning this object and declare `"ingestion_stages": [{"factory": "application_workshop"}]` in `MemorySpec`, or pass `ingestion_stages=(WorkshopStage(),)` to `Memory` / `MemoryPlugins`. Stage names must be unique, stages must be thread-safe, and the ordered stage recipe is pinned in SQLite metadata for populated stores; changing stages requires replaying sources into a fresh directory.

## Validation and ownership

Unknown factory names, unsupported slots, and invalid capability declarations fail before local storage opens. Configuration names registered dictionary keys and never imports arbitrary module paths.

`build_memory_composition(config, factories=factories)` validates and constructs a `MemoryComposition` without opening storage; close it unless its plugins are transferred into `Memory`. `Memory.from_config()` transfers ownership automatically and closes already-constructed resources once on partial construction failure, including shared clients declared via `Tool(..., resources=(client,))`.
