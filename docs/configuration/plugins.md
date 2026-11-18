# Application plugin factories

Use `Memory.from_config(..., factories=...)` to construct application-owned integrations from
data. The host supplies a mapping of factory names to already imported Python callables. A
`PluginConfig` names a bundled or host factory and passes its `options` object to it. Both use
the same per-instance catalog, capability validation and resource ownership.

`MemorySpec` is the only declarative composition shape. Each backend field accepts either
a typed bundled `provider` declaration or a `PluginConfig` with `factory` and `options`.
Already validated provider values go directly to their typed builders; only factory options
are validated by the selected factory. Tools and ingestion stages use factory declarations.

A factory has the signature `Callable[[Mapping[str, object]], Closable]`. Backend factories
return an object implementing the slot's [backend protocol](../api/python-backends.md). Tool
factories return a [registered tool](../api/python-tools.md), typically a `Tool` with Pydantic
input and output models. Factories validate their own options, including model revisions and
runtime settings specific to their implementation.

## Define factories in the host

These examples reuse bundled adapters; a factory can construct a third-party protocol
implementation through the same interface.

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

Install the optional runtimes required by those particular adapters, such as `sensemem[local]`
for this example. The registry itself introduces no model runtime dependency.

## Construct backend slots through factories

The declarative fields are `embedding` (required), `generation`, `speech`, `vision`, `face`,
`formation`, `consolidation`, `reranking`, `query_policy`, `candidate_source`, `ranking_policy`,
and `context_selector`. They are the same fields used for bundled providers. Declare ordered
computations in `ingestion_stages`, and additional tools in the separate `tools` list.

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

`Memory.from_config()` also accepts this shape as a plain mapping. `settings` and top-level
`retention` retain their existing meanings. Declaring a registered backend does not add an
automatic Pipeline stage; its existing memory capability determines where the kernel uses it.
The host chooses when to invoke additional tools directly or through an agent.

## Add tool factories to the bundled provider catalog

Keep `MemorySpec` when bundled providers cover the backend slots. Its `tools` field accepts
the same factory declarations:

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

The `provider` fields remain concise declarations for bundled models. To mix bundled models
with third-party backends, put a factory declaration directly in the corresponding field.
`AsyncMemory.from_config()` accepts the same configurations and explicit factory mapping.

## Bundled factories and computation policies

Bundled names are reserved and cannot be overridden by the host. The catalog is built for each
composition; registration changes neither other instances nor process-wide state.

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

Model factories accept the corresponding bundled provider's options without a `provider` field.
The four default computation factories accept no options. For example, combine a bundled encoder with
an application ranking policy:

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

`QueryPolicy.plan(QueryRequest)` selects query text and representations; the kernel executes
embedding calls against the declared capabilities. `CandidateSource.search(CandidateRequest,
CandidateReadPort)` proposes native index references or existing local memory IDs. Sources with
`requires_embeddings=False` skip query embeddings. The scoped read port cannot broaden the
request's bounds or mutate storage. See [computation plugins](../api/python-policies.md) for
contracts and examples.

`RankingPolicy.rank(RankingRequest)` proposes scores for the supplied authoritative candidates.
`ContextSelector.select(ContextSelectionRequest)` returns `ContextSelection` with candidate
anchor IDs in priority order and an optional diversity preference.
Candidates report full closure costs and optional `excerpt_chars` and `projection_chars` costs.
The request prices the corresponding section headings through `excerpt_section_chars` and
`projection_section_chars`. These are alternative delivery costs; a text-only projection has
omitted native media and cannot satisfy a required evidence closure. The compiler admits that projection when the complete record exceeds character or media limits.
Selectors do not simulate or consume budgets; one compiler admission ledger owns those decisions.
The kernel validates results and retains identity, visibility, consent, evidence closure and budget
checks. `Memory` selects the corresponding default implementation for each omitted computation
role. Object compositions pass `query_policy=`, `candidate_source=`, `ranking_policy=`, and
`context_selector=` to `Memory` or `MemoryPlugins`. `MemorySpec` also accepts factory
declarations in these four fields.

## Ordered ingestion stages

An `IngestionStage` declares a nonempty `name`, a nonempty durable recipe `space`, and
`process(ModelInput) -> StageResult`. Stages run in declared order before built-in perception and
embedding. Each receives the preceding interpretations and verified input assets. The result's
`interpretation` adds explicitly labelled, untrusted model context; it cannot replace the original
source, assets, identity or metadata.

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

Register a factory returning this object and declare `"ingestion_stages": [{"factory":
"application_workshop"}]` alongside either configuration shape. Object compositions use
`ingestion_stages=(WorkshopStage(),)`. Stage names must be unique within one composition, and
stage objects must be thread-safe. `Memory` owns them, including their `close()` calls.

The ordered names and spaces are pinned to the physical store. Keep the same recipe when
reopening a populated store; changing or removing stages, reordering them, or adding a stage to
existing data is rejected. Use a fresh directory and ingest original sources to change that
recipe. Empty stores may change their stage composition.

## Validation and ownership

Unknown factory names, unsupported backend slots, and incompatible returned capabilities fail
before local storage opens. Factory names are dictionary keys supplied by the host; configuration
does not import a module, evaluate an expression, or discover executable entry points.

`build_memory_composition(config, factories=factories)` constructs a `MemoryComposition` without
opening its store. Its caller must close the composition or transfer its plugins into `Memory`.
Normal `Memory.from_config()` construction transfers ownership automatically. If later construction
fails, already constructed resources are closed once, including clients shared between backend
slots and registered tools.

Factories own creation of their clients. Declare nested tool resources with `resources=(client,)`
so the common resource traversal can deduplicate them. A factory that raises before returning an
object is responsible for releasing any resources it allocated before that failure.
