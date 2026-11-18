"""Typed declarative composition over SenseMem's existing backend protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SecretStr,
    StringConstraints,
    Tag,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

import sensemem.recipes as recipes
from sensemem._resources import Closable, close_quietly
from sensemem._validation import absolute_http_url, server_base_url
from sensemem.candidate_sources import CandidateSource, DefaultCandidateSource
from sensemem.context_selection import ContextSelector, DefaultContextSelector
from sensemem.control_values import RetentionPolicy
from sensemem.exceptions import ValidationError
from sensemem.infrastructure.local.gazetteer import OfflineGazetteer
from sensemem.infrastructure.local.source_metadata import GazetteerEnricher
from sensemem.kernel.answer_evidence import EvidenceGeneration
from sensemem.models._openai_defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
)
from sensemem.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    RerankingBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from sensemem.models.funasr import FunASRTranscriber
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder
from sensemem.models.openai_speech import OpenAITranscriber
from sensemem.models.opencv_face import OpenCVFaceAnalyzer
from sensemem.models.qwen3_vl_reranking import Qwen3VLReranker
from sensemem.models.routing import GenerationRouter
from sensemem.models.sentence_transformers import (
    SentenceTransformersEmbedder,
    SentenceTransformersReranker,
)
from sensemem.models.vllm_reranking import VLLMReranker
from sensemem.plugins import MemoryPlugins
from sensemem.processing import IngestionStage
from sensemem.query_policy import DefaultQueryPolicy, QueryPolicy
from sensemem.ranking_policy import DefaultRankingPolicy, RankingPolicy
from sensemem.settings import MemorySettings
from sensemem.tools import RegisteredTool, ToolRegistry
from sensemem.types import Modality

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_Temperature = Annotated[float, Field(strict=True, ge=0, le=2)]
_Seed = Annotated[int, Field(strict=True, ge=0, lt=2**63)]
_UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1)]
_DISCRIMINATED_SLOTS = frozenset({"embedding", "reranking", "speech"})
PluginFactory = Callable[[Mapping[str, object]], Closable]
_ConfigT = TypeVar("_ConfigT", bound=BaseModel)


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    @field_serializer("modalities", check_fields=False, when_used="json")
    def _sorted_modalities(self, modalities: frozenset[Modality] | None) -> list[str] | None:
        # Serialized configurations enter benchmark cache and resume identities.
        return None if modalities is None else sorted(modality.value for modality in modalities)


class _OpenAIConfig(_ConfigModel):
    provider: Literal["openai"]
    base_url: _Text | None = None
    # Each OpenAI-compatible slot builds its own client, so each needs its own credential:
    # without this field a composition pointing embedding at a local server and generation at a
    # hosted one could only ever present the single key `OPENAI_API_KEY` names. `SecretStr` keeps
    # the value out of `model_dump`, `repr`, and therefore out of anything that serialises a
    # configuration. Leaving it unset falls back to the SDK's own environment lookup.
    api_key: SecretStr | None = None
    timeout: _PositiveFloat | None = None
    max_retries: _NonNegativeInt | None = None

    @field_validator("base_url")
    @classmethod
    def _absolute_http_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not absolute_http_url(value):
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        return value


class OpenAIEmbeddingConfig(_OpenAIConfig):
    model: _Text = DEFAULT_EMBEDDING_MODEL
    dimension: _PositiveInt = DEFAULT_EMBEDDING_DIMENSION
    space: _Text | None = None
    # An OpenAI-compatible server may host a multimodal embedding model. Declaring which
    # modalities it accepts is what lets routing send image, video, and audio memories to it
    # instead of failing the write; `messages` is the request shape those servers read media
    # parts from. Both already existed on the backend and were unreachable from configuration.
    # At least one: an empty set declares a model that can embed nothing, which builds a `Memory`
    # whose every write fails with "does not support: text". Configuration rejects out-of-range
    # values rather than deferring them to the first `add`.
    modalities: Annotated[frozenset[Modality], Field(min_length=1)] = frozenset({Modality.TEXT})
    request_format: Literal["input", "messages"] = "input"
    image_max_pixels: Annotated[int, Field(strict=True, ge=1, le=64_000_000)] | None = None


class SentenceTransformersEmbeddingConfig(_ConfigModel):
    provider: Literal["sentence-transformers"]
    model: _Text
    revision: _Text | None = None
    dimension: _PositiveInt | None = None
    device: _Text | None = None
    batch_size: _PositiveInt = 32
    dtype: Literal["auto", "float32", "bfloat16"] = "auto"
    modalities: Annotated[frozenset[Modality], Field(min_length=1)] | None = None


class SentenceTransformersRerankingConfig(_ConfigModel):
    provider: Literal["sentence-transformers"]
    model: _Text
    revision: _Text | None = None
    device: _Text | None = None
    batch_size: _PositiveInt = 32
    max_length: _PositiveInt = 1024


class Qwen3VLRerankingConfig(_ConfigModel):
    provider: Literal["qwen3-vl"]
    model: _Text
    revision: _Text | None = None
    device: Literal["cuda"] | None = None
    batch_size: _PositiveInt = 32
    max_length: _PositiveInt = 8192
    gpu_memory_utilization: Annotated[float, Field(strict=True, gt=0, le=1)] = 0.9
    image_limit: _PositiveInt = 8
    video_limit: _PositiveInt = 2
    image_max_pixels: _PositiveInt = 262144
    video_frames: Annotated[int, Field(strict=True, ge=2, le=64)] = 8
    video_cache_bytes: _NonNegativeInt = 268435456


class VLLMRerankingConfig(_ConfigModel):
    provider: Literal["vllm"]
    model: _Text
    base_url: _Text
    api_key: SecretStr | None = None
    timeout: _PositiveFloat = 60.0
    batch_size: _PositiveInt = 32
    request_concurrency: _PositiveInt = 32
    media_candidates: _PositiveInt | None = None
    media_weight: Annotated[float, Field(strict=True, gt=0, le=1)] = 0.5
    modalities: Annotated[
        frozenset[Literal[Modality.TEXT, Modality.IMAGE, Modality.VIDEO]], Field(min_length=1)
    ] = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        return server_base_url(value)


class _OpenAICompletionConfig(_OpenAIConfig):
    """Knobs `OpenAICompletion` reads for any chat-completion role."""

    model: _Text = DEFAULT_GENERATION_MODEL
    modalities: frozenset[Modality] = frozenset({Modality.TEXT})
    temperature: _Temperature | None = None
    seed: _Seed | None = None
    max_tokens: _PositiveInt | None = None
    response_format: Literal["json_object", "json_schema"] = "json_schema"
    store: Annotated[bool, Field(strict=True)] = False
    extra_body: Mapping[str, object] | None = None
    stream: bool = False


class OfflineGazetteerConfig(_ConfigModel):
    """An explicitly pinned external-world map resource used only while answering."""

    index_path: Path
    index_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$", strict=True)]
    radius_m: Annotated[float, Field(strict=True, gt=0, le=500)] = 500
    limit: Annotated[int, Field(strict=True, ge=1, le=8)] = 8
    budget_chars: _PositiveInt = 16_000


class OpenAIGenerationConfig(_OpenAICompletionConfig):
    routes: tuple[OpenAIGenerationConfig, ...] = ()
    request_max_bytes: _PositiveInt | None = None
    answer_style: Literal["default", "concise"] = "default"
    answer_format: Literal["text", "structured"] = "text"
    image_source_prose: Literal["include", "context_when_supplied"] = "include"
    evidence_operation_tool: Annotated[bool, Field(strict=True)] = False
    source_metadata_key: _Text = "source_id"
    image_max_pixels: Annotated[int, Field(strict=True, ge=1, le=64_000_000)] | None = None
    video_limit: _PositiveInt | None = 8
    # Below this duration a video is answered as four ordered stills instead. The option already
    # existed on `OpenAICompletion` and was unreachable from configuration, which left every caller
    # that builds its models from a file -- the benchmark harness included -- unable to set the
    # floor its endpoint needs. Answer-only, like `video_limit`, so it is not on the shared
    # completion base that formation also reads.
    min_video_seconds: _PositiveFloat | None = None
    media_policy: Literal["all", "on_demand"] = "all"
    media_max_items: Annotated[int, Field(strict=True, ge=1, le=8)] = 2
    media_max_tokens: _PositiveInt = 8192
    offline_gazetteer: OfflineGazetteerConfig | None = None

    @model_validator(mode="after")
    def _evidence_operation_answer_format(self) -> Self:
        if self.evidence_operation_tool and self.answer_format != "text":
            raise ValueError("evidence_operation_tool requires answer_format='text'")
        return self


class OpenAIFormationConfig(_OpenAICompletionConfig):
    """Chat-completion knobs for the adapter that proposes derived typed memories.

    Formation is a separate LLM round-trip on the write path and usually wants its own model,
    token budget, and endpoint, so it is its own slot rather than a reuse of `generation`;
    leaving the slot out keeps that round-trip off, which is the default. It reads the same
    completion knobs as generation, inherited here. `video_limit` is answer-only;
    `image_max_pixels` bounds native image transport without changing the observation.
    `formation_model` and `formation_space` are derived from these values.

    Two inherited fields matter more here than they do for answering. `modalities` becomes
    `formation_capabilities`, which gates which observations reach the former at all: an
    observation whose modalities are not covered is skipped, silently and by design, so declare
    the media the endpoint really accepts or image and video sources will never form anything.
    And formation returns JSON, so a `max_tokens` truncation is a hard error rather than a
    partial parse.
    """

    image_max_pixels: Annotated[int, Field(strict=True, ge=1, le=64_000_000)] | None = None


class OpenAIVisionConfig(_OpenAICompletionConfig):
    """Chat-completion knobs for the adapter that captions visual memories for the text index.

    A memory whose content is one image has no words, so its full-text document is empty and the
    lexical half of retrieval cannot reach it however good the embedder is. This slot pays one
    chat completion per write batch to derive that text; the caption is unioned into the indexed
    document beside whatever the caller wrote, never substituted for it, and the asset is still
    embedded natively. Leaving the slot out keeps the write path exactly as cheap as it is today:
    no describer is constructed, and the derived-text branch returns its input unchanged.

    Like `formation` this is its own slot and its own client rather than a reuse of `generation`,
    because captioning usually wants a smaller model than answering and because configuring an
    answerer must never start spending on writes. `modalities` is the visual capability set and
    accepts only `image` and `video`; an asset outside it is left undescribed. Video is described
    from four ordered stills decoded locally rather than by uploading the clip, so it costs
    four image parts per memory; that is why the default is image alone.
    """

    modalities: Annotated[frozenset[Modality], Field(min_length=1)] = frozenset({Modality.IMAGE})
    video_frames: Annotated[int, Field(strict=True, ge=2, le=64)] = 4
    image_max_pixels: Annotated[int, Field(strict=True, ge=1, le=64_000_000)] | None = None

    @field_validator("modalities")
    @classmethod
    def _visual_modalities_only(cls, value: frozenset[Modality]) -> frozenset[Modality]:
        # `Memory` rejects the same set at construction. Rejecting it here means a host that
        # validates a document before opening storage sees it too.
        if value - {Modality.IMAGE, Modality.VIDEO}:
            raise ValueError("must contain only image or video")
        return value


class OpenAIConsolidationConfig(_OpenAICompletionConfig):
    """Chat-completion knobs for the backend that proposes control-plane operations.

    Declared like `formation` and off unless it is configured: consolidation is a paid reasoning
    call over a bounded evidence set, and a deployment that only wants recall should not start
    making it. When these values match a `generation` or `formation` slot exactly, the composition
    reuses that adapter instead of opening a second client against the same endpoint.

    `modalities` matters here for the same reason it does for formation: it becomes
    `generation_capabilities`, which decides which evidence media the adapter attaches to the
    request. A native image or audio memory with no derived description is otherwise opaque to
    the loop, which can then only propose forgetting it.
    """


class OpenAITranscriptionConfig(_OpenAIConfig):
    model: _Text = DEFAULT_TRANSCRIPTION_MODEL
    space: _Text | None = None


class FunASRSpeechConfig(_ConfigModel):
    provider: Literal["funasr"]
    device: _Text = "auto"
    ncpu: _PositiveInt = 4
    vad_device: _Text | None = None
    speaker_device: _Text | None = None


class OpenCVFaceConfig(_ConfigModel):
    provider: Literal["opencv"]
    detector_model: Path
    recognizer_model: Path
    score_threshold: _UnitInterval = 0.9
    nms_threshold: _UnitInterval = 0.3
    top_k: _PositiveInt = 5000
    frame_interval_ms: _PositiveInt = 1000
    max_video_frames: _PositiveInt = 300


EmbeddingProviderConfig = Annotated[
    OpenAIEmbeddingConfig | SentenceTransformersEmbeddingConfig,
    Field(discriminator="provider"),
]
RerankingProviderConfig = Annotated[
    SentenceTransformersRerankingConfig | Qwen3VLRerankingConfig | VLLMRerankingConfig,
    Field(discriminator="provider"),
]
SpeechProviderConfig = Annotated[
    OpenAITranscriptionConfig | FunASRSpeechConfig,
    Field(discriminator="provider"),
]


class PluginConfig(_ConfigModel):
    """A named bundled or application factory, with factory-owned options."""

    factory: _Text
    options: Mapping[str, object] = Field(default_factory=dict)


def _declaration_kind(value: object) -> str:
    if isinstance(value, PluginConfig) or (isinstance(value, Mapping) and "factory" in value):
        return "factory"
    return "provider"


ProviderOrFactory = Annotated[
    Annotated[_ConfigT, Tag("provider")] | Annotated[PluginConfig, Tag("factory")],
    Discriminator(_declaration_kind),
]


class MemorySpec(_ConfigModel):
    """One declarative shape for bundled providers and host-registered factories."""

    data_dir: Path = Path(".sensemem")
    embedding: ProviderOrFactory[EmbeddingProviderConfig]
    generation: ProviderOrFactory[OpenAIGenerationConfig] | None = None
    reranking: ProviderOrFactory[RerankingProviderConfig] | None = None
    # Both reachable, but never implicit, and omitted by default. Configuring `generation` must
    # not enable either one: a former is an LLM call per observation and a describer an LLM call
    # per visual, both on the write path, and the only bundled adapters are cloud calls, which a
    # local-first deployment should opt into rather than discover. What each derives is also a
    # union with the raw sources rather than a replacement for them.
    formation: ProviderOrFactory[OpenAIFormationConfig] | None = None
    vision: ProviderOrFactory[OpenAIVisionConfig] | None = None
    # The agentic memory control plane. Same rule as `formation`: reachable, never implicit. A
    # deployment that never sets it keeps `consolidate()` unavailable and everything else
    # unchanged, which is what a host that only recalls should get.
    consolidation: ProviderOrFactory[OpenAIConsolidationConfig] | None = None
    speech: ProviderOrFactory[SpeechProviderConfig] | None = None
    face: ProviderOrFactory[OpenCVFaceConfig] | None = None
    query_policy: PluginConfig | None = None
    candidate_source: PluginConfig | None = None
    ranking_policy: PluginConfig | None = None
    context_selector: PluginConfig | None = None
    ingestion_stages: tuple[PluginConfig, ...] = ()
    tools: tuple[PluginConfig, ...] = ()
    settings: MemorySettings = Field(default_factory=MemorySettings)
    # Its own section rather than a `settings` field: every other setting shapes what recall
    # returns and can be changed back, and this one deletes. It reaches `Memory` as
    # `MemorySettings.retention`, which is the same value under the name `from_plugins` reads;
    # declaring it twice is refused rather than silently resolved.
    retention: RetentionPolicy | None = None


@dataclass(frozen=True, slots=True)
class MemoryComposition:
    """Constructed plugins and settings whose resources the caller owns.

    Close the composition unless its plugins are transferred to a ``Memory`` instance.
    Building this value does not open local storage.
    """

    data_dir: Path
    plugins: MemoryPlugins
    settings: MemorySettings

    def close(self) -> None:
        """Release constructed plugins, including resources shared across slots."""
        close_quietly(
            self.plugins.tools,
            self.plugins.query_policy,
            self.plugins.candidate_source,
            self.plugins.ranking_policy,
            self.plugins.context_selector,
            *self.plugins.ingestion_stages,
            self.plugins.face_analyzer,
            self.plugins.vision_describer,
            self.plugins.transcriber,
            self.plugins.consolidator,
            self.plugins.former,
            self.plugins.answerer,
            self.plugins.reranker,
            self.plugins.embedder,
        )


def build_memory_composition(  # noqa: C901 - each optional capability is wired explicitly
    spec: MemorySpec | Mapping[str, object],
    *,
    factories: Mapping[str, PluginFactory] | None = None,
) -> MemoryComposition:
    """Validate a specification and construct its plugins without opening storage.

    Provider construction may load models and allocate clients. The caller must close the
    returned composition or transfer its plugins to a ``Memory`` instance.
    """
    config = _validated_config(spec)
    settings = _with_retention(config)
    completions: list[tuple[_OpenAICompletionConfig, Closable]] = []
    catalog = _builtin_factories(completions=completions)
    if factories:
        if factories.keys() & catalog.keys():
            raise ValidationError("application factories cannot replace bundled factories")
        catalog.update(factories)
    resources: list[Closable] = []
    with ExitStack() as cleanup:
        cleanup.callback(lambda: close_quietly(*resources))
        embedder = (
            cast(
                EmbeddingBackend,
                _registered_plugin(config.embedding, catalog, (EmbeddingBackend,), resources),
            )
            if isinstance(config.embedding, PluginConfig)
            else _build_embedding(config.embedding)
        )
        resources.append(embedder)
        answerer = None
        if config.generation is not None:
            answerer = (
                cast(
                    GenerationBackend,
                    _registered_plugin(config.generation, catalog, (GenerationBackend,), resources),
                )
                if isinstance(config.generation, PluginConfig)
                else _build_generation(config.generation)
            )
            resources.append(answerer)
        if isinstance(config.generation, OpenAIGenerationConfig) and answerer is not None:
            completions.append((config.generation, answerer))
        former = None
        if config.formation is not None:
            former = (
                cast(
                    FormationBackend,
                    _registered_plugin(config.formation, catalog, (FormationBackend,), resources),
                )
                if isinstance(config.formation, PluginConfig)
                else _build_formation(config.formation)
            )
            resources.append(former)
        if isinstance(config.formation, OpenAIFormationConfig) and former is not None:
            completions.append((config.formation, former))
        consolidator = None
        if config.consolidation is not None:
            consolidator = (
                cast(
                    ConsolidationBackend,
                    _registered_plugin(
                        config.consolidation, catalog, (ConsolidationBackend,), resources
                    ),
                )
                if isinstance(config.consolidation, PluginConfig)
                else _build_consolidation(config.consolidation, completions=completions)
            )
            resources.append(consolidator)
        transcriber = None
        if config.speech is not None:
            transcriber = (
                cast(
                    SpeechBackend | TranscriptionBackend,
                    _registered_plugin(
                        config.speech, catalog, (SpeechBackend, TranscriptionBackend), resources
                    ),
                )
                if isinstance(config.speech, PluginConfig)
                else _build_speech(config.speech)
            )
            resources.append(transcriber)
        vision = None
        if config.vision is not None:
            vision = (
                cast(
                    VisionDescriptionBackend,
                    _registered_plugin(
                        config.vision, catalog, (VisionDescriptionBackend,), resources
                    ),
                )
                if isinstance(config.vision, PluginConfig)
                else _build_vision(config.vision)
            )
            resources.append(vision)
        face = None
        if config.face is not None:
            face = (
                cast(
                    FaceBackend, _registered_plugin(config.face, catalog, (FaceBackend,), resources)
                )
                if isinstance(config.face, PluginConfig)
                else _build_face(config.face)
            )
            resources.append(face)
        reranker = None
        if config.reranking is not None:
            reranker = (
                cast(
                    RerankingBackend,
                    _registered_plugin(config.reranking, catalog, (RerankingBackend,), resources),
                )
                if isinstance(config.reranking, PluginConfig)
                else _build_reranking(config.reranking)
            )
            resources.append(reranker)
        stages = tuple(
            cast(IngestionStage, _registered_plugin(stage, catalog, (IngestionStage,), resources))
            for stage in config.ingestion_stages
        )
        tools = _build_tools(config.tools, catalog, resources)
        plugins = MemoryPlugins(
            embedder=embedder,
            answerer=answerer,
            transcriber=transcriber,
            vision_describer=vision,
            face_analyzer=face,
            former=former,
            consolidator=consolidator,
            reranker=reranker,
            query_policy=None
            if config.query_policy is None
            else cast(
                QueryPolicy,
                _registered_plugin(config.query_policy, catalog, (QueryPolicy,), resources),
            ),
            candidate_source=None
            if config.candidate_source is None
            else cast(
                CandidateSource,
                _registered_plugin(config.candidate_source, catalog, (CandidateSource,), resources),
            ),
            ranking_policy=None
            if config.ranking_policy is None
            else cast(
                RankingPolicy,
                _registered_plugin(config.ranking_policy, catalog, (RankingPolicy,), resources),
            ),
            context_selector=None
            if config.context_selector is None
            else cast(
                ContextSelector,
                _registered_plugin(config.context_selector, catalog, (ContextSelector,), resources),
            ),
            ingestion_stages=stages,
            tools=tools,
        )
        cleanup.pop_all()
    return MemoryComposition(config.data_dir, plugins, settings)


def _registered_plugin(
    config: PluginConfig,
    factories: Mapping[str, PluginFactory],
    expected: tuple[type[object], ...],
    resources: list[Closable],
) -> Closable:
    factory = factories.get(config.factory)
    if factory is None:
        raise ValidationError(f"unregistered plugin factory: {config.factory}")
    plugin = factory(config.options)
    resources.append(plugin)
    if not isinstance(plugin, expected):
        raise ValidationError(
            f"plugin factory {config.factory} returned an incompatible capability"
        )
    return plugin


def _build_tools(
    configs: tuple[PluginConfig, ...],
    factories: Mapping[str, PluginFactory],
    resources: list[Closable],
) -> ToolRegistry:
    registry = ToolRegistry()
    resources.append(registry)
    for config in configs:
        tool = cast(
            RegisteredTool, _registered_plugin(config, factories, (RegisteredTool,), resources)
        )
        registry.register(tool)
    return registry


def _builtin_factories(
    *,
    completions: list[tuple[_OpenAICompletionConfig, Closable]],
) -> dict[str, PluginFactory]:
    """Create an instance-local catalog; bundled factories follow the application contract."""

    def configured(
        schema: type[_ConfigT],
        build: Callable[[_ConfigT], Closable],
        provider: str | None = None,
        *,
        share_completion: bool = False,
    ) -> PluginFactory:
        def factory(options: Mapping[str, object]) -> Closable:
            values = dict(options) if provider is None else {"provider": provider, **options}
            try:
                declared = schema.model_validate(values)
            except PydanticValidationError:
                raise ValidationError("invalid bundled plugin options") from None
            plugin = build(declared)
            if share_completion:
                assert isinstance(declared, _OpenAICompletionConfig)
                completions.append((declared, plugin))
            return plugin

        return factory

    return {
        "sensemem.embedding.openai": configured(OpenAIEmbeddingConfig, _build_embedding, "openai"),
        "sensemem.embedding.sentence-transformers": configured(
            SentenceTransformersEmbeddingConfig, _build_embedding, "sentence-transformers"
        ),
        "sensemem.generation.openai": configured(
            OpenAIGenerationConfig, _build_generation, "openai", share_completion=True
        ),
        "sensemem.formation.openai": configured(
            OpenAIFormationConfig, _build_formation, "openai", share_completion=True
        ),
        "sensemem.vision.openai": configured(OpenAIVisionConfig, _build_vision, "openai"),
        "sensemem.consolidation.openai": configured(
            OpenAIConsolidationConfig,
            lambda declared: _build_consolidation(declared, completions=completions),
            "openai",
        ),
        "sensemem.speech.openai": configured(OpenAITranscriptionConfig, _build_speech, "openai"),
        "sensemem.speech.funasr": configured(FunASRSpeechConfig, _build_speech, "funasr"),
        "sensemem.face.opencv": configured(OpenCVFaceConfig, _build_face, "opencv"),
        "sensemem.reranking.sentence-transformers": configured(
            SentenceTransformersRerankingConfig, _build_reranking, "sentence-transformers"
        ),
        "sensemem.reranking.qwen3-vl": configured(
            Qwen3VLRerankingConfig, _build_reranking, "qwen3-vl"
        ),
        "sensemem.reranking.vllm": configured(VLLMRerankingConfig, _build_reranking, "vllm"),
        "sensemem.ranking.default": configured(_ConfigModel, lambda _: DefaultRankingPolicy()),
        "sensemem.query.default": configured(_ConfigModel, lambda _: DefaultQueryPolicy()),
        "sensemem.candidates.default": configured(_ConfigModel, lambda _: DefaultCandidateSource()),
        "sensemem.context.default": configured(_ConfigModel, lambda _: DefaultContextSelector()),
    }


def _build_reranking(config: RerankingProviderConfig) -> RerankingBackend:
    if isinstance(config, VLLMRerankingConfig):
        return VLLMReranker(
            model_id=config.model,
            base_url=config.base_url,
            api_key=None if config.api_key is None else config.api_key.get_secret_value(),
            timeout=config.timeout,
            batch_size=config.batch_size,
            request_concurrency=config.request_concurrency,
            modalities=config.modalities,
            media_candidates=config.media_candidates,
            media_weight=config.media_weight,
        )
    if isinstance(config, Qwen3VLRerankingConfig):
        return Qwen3VLReranker(
            model_id=config.model,
            revision=config.revision,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            gpu_memory_utilization=config.gpu_memory_utilization,
            image_limit=config.image_limit,
            video_limit=config.video_limit,
            image_max_pixels=config.image_max_pixels,
            video_frames=config.video_frames,
            video_cache_bytes=config.video_cache_bytes,
        )
    return SentenceTransformersReranker(
        model_id=config.model,
        revision=config.revision,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )


def _with_retention(config: MemorySpec) -> MemorySettings:
    """Fold the top-level `retention` section into the settings `Memory` is constructed from."""
    if config.retention is None:
        return config.settings
    if config.settings.retention != RetentionPolicy():
        raise ValidationError(
            "config.retention: declared both as its own section and as settings.retention"
        )
    return replace(config.settings, retention=config.retention)


def _validated_config(value: MemorySpec | Mapping[str, object]) -> MemorySpec:
    if isinstance(value, MemorySpec):
        return value
    if not isinstance(value, Mapping):
        raise ValidationError("config must be a MemorySpec or mapping")
    try:
        return MemorySpec.model_validate(value)
    except PydanticValidationError as error:
        issues = []
        for issue in error.errors(include_url=False, include_context=False, include_input=False):
            location = tuple(issue["loc"])
            if len(location) >= 2 and location[1] in {"provider", "factory"}:
                location = (location[0], *location[2:])
            if issue["type"] in {"union_tag_invalid", "union_tag_not_found"}:
                location = (*location, "provider")
            elif len(location) >= 2 and location[0] in _DISCRIMINATED_SLOTS:
                location = (location[0], *location[2:])
            path = ".".join(("config", *(str(part) for part in location)))
            issues.append(f"{path}: {issue['msg']}")
        raise ValidationError("; ".join(issues)) from None


def _openai_values(config: _OpenAIConfig) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "base_url": config.base_url,
            "api_key": None if config.api_key is None else config.api_key.get_secret_value(),
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }.items()
        if value is not None
    }


def _build_embedding(config: EmbeddingProviderConfig) -> EmbeddingBackend:
    if isinstance(config, OpenAIEmbeddingConfig):
        values = _openai_values(config)
        values.update(
            embedding_model=config.model,
            embedding_dimension=config.dimension,
            embedding_capabilities=config.modalities,
            embedding_request_format=config.request_format,
        )
        if config.space is not None:
            values["embedding_space"] = config.space
        if config.image_max_pixels is not None:
            values["image_max_pixels"] = config.image_max_pixels
        return recipes._owned_openai(OpenAIEmbedder, **values)
    return SentenceTransformersEmbedder(
        model_id=config.model,
        revision=config.revision,
        dimension=config.dimension,
        device=config.device,
        batch_size=config.batch_size,
        dtype=config.dtype,
        modalities=config.modalities,
    )


def _completion_values(config: _OpenAICompletionConfig) -> dict[str, object]:
    values = _openai_values(config)
    values.update(
        generation_model=config.model,
        generation_capabilities=config.modalities,
        generation_response_format=config.response_format,
        generation_store=config.store,
    )
    optional = {
        "generation_temperature": config.temperature,
        "generation_seed": config.seed,
        "generation_max_tokens": config.max_tokens,
        "generation_extra_body": config.extra_body,
        "generation_stream": True if config.stream else None,
    }
    values.update((key, value) for key, value in optional.items() if value is not None)
    return values


def _build_generation(
    config: OpenAIGenerationConfig,
) -> GenerationBackend:
    if config.routes:
        with ExitStack() as cleanup:
            base = _build_native_generation(config)
            cleanup.callback(base.close)
            backends = [base]
            for route in config.routes:
                backend = _build_generation(route)
                cleanup.callback(backend.close)
                backends.append(backend)
            router = GenerationRouter(backends)
            cleanup.pop_all()
            return router
    return _build_native_generation(config)


def _build_native_generation(config: OpenAIGenerationConfig) -> GenerationBackend:
    values = _completion_values(config)
    for name, value, default in (
        ("generation_request_max_bytes", config.request_max_bytes, None),
        ("generation_answer_format", config.answer_format, "text"),
        ("generation_image_source_prose", config.image_source_prose, "include"),
        ("generation_evidence_operation_tool", config.evidence_operation_tool, False),
        ("generation_source_metadata_key", config.source_metadata_key, "source_id"),
        ("generation_answer_style", config.answer_style, "default"),
        ("generation_media_max_tokens", config.media_max_tokens, 8192),
    ):
        if value != default:
            values[name] = value
    if config.image_max_pixels is not None:
        values["image_max_pixels"] = config.image_max_pixels
    if config.video_limit != 8:
        values["generation_video_limit"] = config.video_limit
    if config.min_video_seconds is not None:
        values["generation_min_video_seconds"] = config.min_video_seconds
    backend = recipes._owned_openai(OpenAICompletion, **values)
    if config.media_policy == "all" and config.offline_gazetteer is None:
        return backend
    with ExitStack() as cleanup:
        cleanup.callback(backend.close)
        answerer = EvidenceGeneration(
            backend,
            selector=backend if config.media_policy == "on_demand" else None,
            enricher=GazetteerEnricher(OfflineGazetteer(**config.offline_gazetteer.model_dump()))
            if config.offline_gazetteer is not None
            else None,
            media_max_items=config.media_max_items,
        )
        cleanup.pop_all()
        return answerer


def _build_formation(config: OpenAIFormationConfig) -> FormationBackend:
    # The bundled adapter reads the generation controls for `form`; `video_limit` is answer-only.
    values = _completion_values(config)
    if config.image_max_pixels is not None:
        values["image_max_pixels"] = config.image_max_pixels
    return recipes._owned_openai(OpenAICompletion, **values)


def _build_vision(config: OpenAIVisionConfig) -> VisionDescriptionBackend:
    # `describe` reads the same generation controls; its capability set is the visual one.
    values = _completion_values(config)
    if config.video_frames != 4:
        values["vision_video_frames"] = config.video_frames
    if config.image_max_pixels is not None:
        values["image_max_pixels"] = config.image_max_pixels
    return recipes._owned_openai(OpenAICompletion, **values)


def _build_consolidation(
    config: OpenAIConsolidationConfig,
    *,
    completions: Sequence[tuple[_OpenAICompletionConfig, Closable]],
) -> ConsolidationBackend:
    """Build the consolidation adapter, reusing an identical generation or formation client.

    `OpenAICompletion` implements every reasoning protocol on one generation client, so a slot that
    declares exactly the same completion values wants that same object, not a second HTTP client
    and connection pool against the same endpoint.
    """
    values = _completion_values(config)
    for declared, built in completions:
        if isinstance(built, ConsolidationBackend) and _completion_values(declared) == values:
            return built
    return recipes._owned_openai(OpenAICompletion, **values)


def _build_speech(config: SpeechProviderConfig) -> SpeechBackend | TranscriptionBackend:
    if isinstance(config, FunASRSpeechConfig):
        return FunASRTranscriber(
            device=config.device,
            ncpu=config.ncpu,
            vad_device=config.vad_device,
            speaker_device=config.speaker_device,
        )
    values = _openai_values(config)
    values["transcription_model"] = config.model
    if config.space is not None:
        values["transcription_space"] = config.space
    return recipes._owned_openai(OpenAITranscriber, **values)


def _build_face(config: OpenCVFaceConfig) -> FaceBackend:
    return OpenCVFaceAnalyzer(
        config.detector_model,
        config.recognizer_model,
        score_threshold=config.score_threshold,
        nms_threshold=config.nms_threshold,
        top_k=config.top_k,
        frame_interval_ms=config.frame_interval_ms,
        max_video_frames=config.max_video_frames,
    )


__all__ = [
    "EmbeddingProviderConfig",
    "FunASRSpeechConfig",
    "MemoryComposition",
    "MemorySpec",
    "OpenAIConsolidationConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIFormationConfig",
    "OpenAIGenerationConfig",
    "OpenAITranscriptionConfig",
    "OpenAIVisionConfig",
    "OpenCVFaceConfig",
    "PluginConfig",
    "PluginFactory",
    "Qwen3VLRerankingConfig",
    "RerankingProviderConfig",
    "SentenceTransformersEmbeddingConfig",
    "SentenceTransformersRerankingConfig",
    "SpeechProviderConfig",
    "VLLMRerankingConfig",
    "build_memory_composition",
]
