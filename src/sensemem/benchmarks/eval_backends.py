"""Shared model ownership, borrowed SDK backends, and benchmark perception caches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    CandidateSource,
    ContextSelector,
    FaceAnalysis,
    IngestionStage,
    Memory,
    MemorySpec,
    Modality,
    QueryPolicy,
    RankingPolicy,
    RegisteredTool,
    SearchHit,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    build_memory_composition,
)
from sensemem._perception_codec import decode_description, encode_description
from sensemem._telemetry import mark_model_requests, record_unmetered_model_usage
from sensemem.benchmarks.eval_adapters import MemoryItem
from sensemem.benchmarks.eval_arms import _media_asset_ref
from sensemem.benchmarks.eval_cache import DescriptionCache
from sensemem.benchmarks.eval_config import _MODALITY_BY_SUFFIX
from sensemem.benchmarks.model_config import ModelConfig
from sensemem.benchmarks.prepare_media import _has_audio
from sensemem.configuration import OpenAIGenerationConfig
from sensemem.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    EmbeddingInputSupport,
    EmbedTask,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    GenerationRoutingBackend,
    ModelInput,
    RerankingBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionAnalysisBackend,
    VisionDescriptionBackend,
)
from sensemem.perception import SpeechAnalysis, VisualDescription

# Runtime protocols inspect attributes statically, so `__getattr__` cannot expose optional
# capabilities. Bind only methods the pooled backend actually supports; role-specific wrappers
# below expose the required members.
_OPTIONAL_CAPABILITY_METHODS = (
    "select",
    "plan_recall",
    "stream_answer",
    "describe_analysis",
    "grounding_assets",
    "select_media",
    "enrich",
    "plan",
    "rank",
    "search",
    "process",
)


class _BorrowedBackend:
    """Forward a shared backend while making per-store ``close`` a no-op."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        if isinstance(backend, CandidateSource):
            self.requires_embeddings = backend.requires_embeddings
        if isinstance(backend, IngestionStage):
            self.name = backend.name
            self.space = backend.space
        if isinstance(backend, EmbeddingInputSupport):
            self.embedding_routes = backend.embedding_routes
        if isinstance(backend, GenerationRoutingBackend):
            self.generation_routes = backend.generation_routes
        for name in _OPTIONAL_CAPABILITY_METHODS:
            # A subclass that wraps the call itself owns the name; anything else binds the
            # pooled backend's own method, which also keeps its real signature.
            if hasattr(type(self), name):
                continue
            method = getattr(backend, name, None)
            if callable(method):
                setattr(self, name, method)

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def resources(self) -> tuple[()]:
        return ()

    def close(self) -> None:
        return None


class _BorrowedToolRegistry(ToolRegistry):
    """Lend the concrete registry API without transferring ownership of its nested tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def closed(self) -> bool:
        return self._registry.closed

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._registry.definitions

    def register(self, tool: RegisteredTool) -> None:
        self._registry.register(tool)

    def get(self, name: str) -> RegisteredTool | None:
        return self._registry.get(name)

    def call(self, name: str, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult:
        return self._registry.call(name, arguments, call_id=call_id)

    def resources(self) -> tuple[()]:
        return ()

    def close(self) -> None:
        return None


class _BorrowedEmbeddingBackend(_BorrowedBackend):
    """Expose required members to Python's static runtime-protocol inspection."""

    @property
    def embedding_capabilities(self) -> frozenset[Modality]:
        return cast(EmbeddingBackend, self._backend).embedding_capabilities

    @property
    def embedding_model(self) -> str:
        return cast(EmbeddingBackend, self._backend).embedding_model

    @property
    def embedding_space(self) -> str:
        return cast(EmbeddingBackend, self._backend).embedding_space

    @property
    def embedding_dimension(self) -> int:
        return cast(EmbeddingBackend, self._backend).embedding_dimension

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return cast(EmbeddingBackend, self._backend).embed(inputs, task=task)


class _BorrowedRerankingBackend(_BorrowedBackend):
    """Preserve the runtime protocol without giving a store ownership of the model."""

    @property
    def reranking_capabilities(self) -> frozenset[Modality]:
        return cast(RerankingBackend, self._backend).reranking_capabilities

    @property
    def reranking_model(self) -> str:
        return cast(RerankingBackend, self._backend).reranking_model

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        return cast(RerankingBackend, self._backend).rerank(query, documents)


class _BorrowedFormationBackend(_BorrowedBackend):
    """Expose the configured formation contract to static runtime-protocol inspection."""

    def __init__(self, backend: FormationBackend) -> None:
        super().__init__(backend)
        self.formation_capabilities = backend.formation_capabilities
        self.formation_model = backend.formation_model
        self.formation_space = backend.formation_space
        self.form = backend.form


class _BorrowedConsolidationBackend(_BorrowedBackend):
    """Expose the configured consolidation contract without owning its model."""

    def __init__(self, backend: ConsolidationBackend) -> None:
        super().__init__(backend)
        self.consolidation_model = backend.consolidation_model
        self.consolidation_recipe = backend.consolidation_recipe
        self.consolidate = backend.consolidate


class _BorrowedTranscriptionBackend(_BorrowedBackend):
    """Keep plain transcription distinct from speaker analysis."""

    def __init__(self, backend: TranscriptionBackend) -> None:
        super().__init__(backend)
        self.transcription_capabilities = backend.transcription_capabilities
        self.transcription_model = backend.transcription_model
        self.transcription_space = backend.transcription_space
        self.transcribe = backend.transcribe


class _BorrowedVisionDescriber(_BorrowedBackend):
    """Expose visual description even when no benchmark description cache is configured."""

    def __init__(self, backend: VisionDescriptionBackend) -> None:
        super().__init__(backend)
        self.vision_capabilities = backend.vision_capabilities
        self.vision_model = backend.vision_model
        self.vision_space = backend.vision_space
        self.describe = backend.describe


class _CachedVisionDescriber:
    """Describe only the visuals not yet described in this benchmark invocation.

    The endpoint returns a different caption for the same image on every call, so without this
    two ingests of one corpus build different full-text documents and a paired arm cannot be
    compared with itself. Keys use the asset's own SHA-256, while the cache path is namespaced by
    run so a repeat still executes the same measured model workload.
    """

    def __init__(self, backend: VisionDescriptionBackend, cache: DescriptionCache) -> None:
        self._backend = backend
        self._cache = cache
        self._key_locks: dict[str, Lock] = {}
        self._registry_lock = Lock()

    @property
    def vision_capabilities(self) -> frozenset[Modality]:
        return self._backend.vision_capabilities

    @property
    def vision_model(self) -> str:
        return self._backend.vision_model

    @property
    def vision_space(self) -> str:
        return self._backend.vision_space

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        return tuple(value.content for value in self.describe_analysis(inputs))

    def describe_analysis(self, inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]:
        batch = tuple(inputs)
        keys = tuple(
            None if (key := _description_digest(value)) is None else "typed-v1:" + key
            for value in batch
        )
        with self._registry_lock:
            locks = [
                self._key_locks.setdefault(key, Lock())
                for key in sorted({key for key in keys if key is not None})
            ]
        with ExitStack() as acquired:
            for lock in locks:
                acquired.enter_context(lock)
            # A prior batch may have filled a shared digest while this batch waited.
            return self._describe_batch(batch, keys)

    def _describe_batch(
        self, batch: tuple[ModelInput, ...], keys: tuple[str | None, ...]
    ) -> tuple[VisualDescription, ...]:
        known = tuple(None if key is None else self._cache.get(key) for key in keys)
        pending = tuple(value for value, found in zip(batch, known, strict=True) if found is None)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("sensemem.model.batch_size", len(pending))
            span.set_attribute(
                "sensemem.input.modalities",
                tuple(
                    sorted({modality.value for value in pending for modality in value.modalities})
                ),
            )
        # One call for the whole miss set, so a partially cached batch still costs one request.
        fresh: Iterator[VisualDescription]
        if not pending:
            mark_model_requests(0, token_usage_expected=0)
            fresh = iter(())
        elif isinstance(self._backend, VisionAnalysisBackend):
            fresh = iter(self._backend.describe_analysis(pending))
        else:
            fresh = iter(VisualDescription(value) for value in self._backend.describe(pending))
        described: list[VisualDescription] = []
        written: list[tuple[str, str]] = []
        for key, found in zip(keys, known, strict=True):
            if found is not None:
                described.append(decode_description(found))
                continue
            value = next(fresh)
            if key is not None:
                written.append((key, encode_description(value)))
            described.append(value)
        # One commit for the batch: the cache is `synchronous=FULL`, so a per-caption write cost
        # an fsync each.
        self._cache.put_many(written)
        return tuple(described)

    def close(self) -> None:
        return None


def _description_digest(value: ModelInput) -> str | None:
    """Identify one description input by the content of the assets it carries.

    An input with no resolved digest is describable but not cacheable; it is passed through
    rather than silently sharing a key with everything else in its shape.
    """
    digests = tuple(asset.sha256 for asset in value.assets if asset.sha256)
    if not digests or len(digests) != len(value.assets):
        return None
    return ":".join((*digests, value.text))


class _BorrowedGenerationBackend(_BorrowedBackend):
    """Lend one answerer without transferring ownership to a per-unit memory."""

    @property
    def generation_capabilities(self) -> frozenset[Modality]:
        return cast(GenerationBackend, self._backend).generation_capabilities

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return cast(GenerationBackend, self._backend).answer(
            question, hits, answer_policy=answer_policy, exhaustive=exhaustive
        )


class _BorrowedFaceBackend(_BorrowedBackend):
    """Lend one face analyzer while preserving the runtime-checkable protocol."""

    @property
    def face_capabilities(self) -> frozenset[Modality]:
        return cast(FaceBackend, self._backend).face_capabilities

    @property
    def face_model(self) -> str:
        return cast(FaceBackend, self._backend).face_model

    @property
    def face_space(self) -> str:
        return cast(FaceBackend, self._backend).face_space

    @property
    def face_analysis_space(self) -> str:
        return cast(FaceBackend, self._backend).face_analysis_space

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return cast(FaceBackend, self._backend).analyze(assets)


class _BorrowedSpeechBackend(_BorrowedBackend):
    """Skip visual-only videos before lending the shared speech backend, and analyse ahead.

    The store ingests a chunk as speech, then embedding, then write, so the GPU idles while the
    embedding request is in flight and the network idles while the speech model runs -- measured
    at about 5 s and 3.7 s of a 10 s batch. `prefetch()` analyses the next chunk's clips on one
    background thread while the store embeds and writes the current one; the store's own
    `analyze()` for those clips then finds the result here. Entries are keyed by content digest,
    which is also the store's asset id, so the source clip and the copy the store analyses are
    the same bytes. Only the analysis runs early: the store still writes chunk by chunk, so
    corpus order and every stored row are what they were.
    """

    def __init__(self, backend: object) -> None:
        super().__init__(backend)
        self._ahead: dict[str, Future[dict[str, SpeechAnalysis]]] = {}
        self._ahead_lock = Lock()
        # One worker: the speech model serialises on its own lock, and a second thread would
        # only queue behind it.
        self._ahead_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-ahead")

    @property
    def transcription_capabilities(self) -> frozenset[Modality]:
        return cast(SpeechBackend, self._backend).transcription_capabilities

    def prefetch(self, items: Sequence[MemoryItem]) -> None:
        """Analyse these items' audible media in the background, once per distinct clip."""
        refs = []
        for item in items:
            for atom in item.content:
                if not isinstance(atom, Path):
                    continue
                modality = _MODALITY_BY_SUFFIX.get(atom.suffix.casefold())
                if modality not in self.transcription_capabilities:
                    continue
                if modality is Modality.VIDEO and not _has_audio(atom):
                    continue
                refs.append(_media_asset_ref(atom, modality))
        with self._ahead_lock:
            fresh = tuple({ref.id: ref for ref in refs if ref.id not in self._ahead}.values())
            if not fresh:
                return
            future = self._ahead_worker.submit(self._analyze_by_id, fresh)
            for ref in fresh:
                self._ahead[ref.id] = future

    def _analyze_by_id(self, assets: Sequence[AssetRef]) -> dict[str, SpeechAnalysis]:
        analyses = cast(SpeechBackend, self._backend).analyze(assets)
        if len(analyses) != len(assets):
            raise RuntimeError("speech backend returned the wrong number of analyses")
        return {asset.id: analysis for asset, analysis in zip(assets, analyses, strict=True)}

    @property
    def transcription_model(self) -> str:
        return cast(SpeechBackend, self._backend).transcription_model

    @property
    def transcription_space(self) -> str:
        return cast(SpeechBackend, self._backend).transcription_space

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        selected = tuple(
            asset.modality is not Modality.VIDEO or asset.path is None or _has_audio(asset.path)
            for asset in assets
        )
        audible = tuple(asset for asset, include in zip(assets, selected, strict=True) if include)
        with self._ahead_lock:
            futures = {
                asset.id: self._ahead.pop(asset.id) for asset in audible if asset.id in self._ahead
            }
        ahead: dict[str, SpeechAnalysis] = {}
        for asset_id, future in futures.items():
            # A prefetch that failed is simply analysed again here, where the error belongs to
            # the write that needed it.
            with suppress(Exception):
                ahead[asset_id] = future.result()[asset_id]
        missing = tuple(asset for asset in audible if asset.id not in ahead)
        if missing:
            generated = cast(SpeechBackend, self._backend).analyze(missing)
        else:
            record_unmetered_model_usage(request_count=0)
            generated = ()
        if len(generated) != len(missing):
            raise RuntimeError("speech backend returned the wrong number of analyses")
        fresh = dict(zip((asset.id for asset in missing), generated, strict=True))
        return tuple(
            (ahead.get(asset.id) or fresh[asset.id])
            if include
            else SpeechAnalysis(turns=(), speakers=())
            for asset, include in zip(assets, selected, strict=True)
        )

    def shutdown_prefetch(self) -> None:
        """Finish running look-ahead, cancel queued work, and reject later submissions."""
        self._ahead_worker.shutdown(wait=True, cancel_futures=True)
        with self._ahead_lock:
            self._ahead.clear()


class _BackendPool:
    """Load model weights once and lend them to every isolated store."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        memory_config: MemorySpec | None = None,
        tracer: Tracer | None = None,
        description_cache: Path | None = None,
    ) -> None:
        self._description_cache: DescriptionCache | None = None
        self._tracer = trace.get_tracer("sensemem.benchmarks.eval") if tracer is None else tracer
        self.embedding_warmup_count = 0
        if memory_config is None:
            raise ValueError(
                "benchmark execution requires --config with an explicit embedding section"
            )
        if isinstance(memory_config.generation, OpenAIGenerationConfig):
            # The harness owns both of these for a run: `--model-args` and the environment
            # name the modalities, and the video floor belongs to the corpus rather than to
            # the endpoint. Injecting them here is what makes the configured path honour
            # them -- the resolved backends are built from this document, so a floor left
            # only on `ModelConfig` would be reported in the result artifact while short
            # videos still went to the endpoint whole.
            update: dict[str, object] = {"modalities": config.generation_capabilities}
            if config.generation_min_video_seconds is not None:
                update["min_video_seconds"] = config.generation_min_video_seconds
            memory_config = memory_config.model_copy(
                update={"generation": memory_config.generation.model_copy(update=update)}
            )
        composition = build_memory_composition(memory_config)
        self._composition = composition
        plugins = composition.plugins
        embedder = cast(EmbeddingBackend, _BorrowedEmbeddingBackend(plugins.embedder))
        answerer = (
            None
            if plugins.answerer is None
            else cast(GenerationBackend, _BorrowedGenerationBackend(plugins.answerer))
        )
        reranker = (
            None
            if plugins.reranker is None
            else cast(RerankingBackend, _BorrowedRerankingBackend(plugins.reranker))
        )
        transcriber = (
            None
            if plugins.transcriber is None
            else cast(
                SpeechBackend | TranscriptionBackend,
                (
                    _BorrowedSpeechBackend(plugins.transcriber)
                    if isinstance(plugins.transcriber, SpeechBackend)
                    else _BorrowedTranscriptionBackend(plugins.transcriber)
                ),
            )
        )
        face_analyzer = (
            None
            if plugins.face_analyzer is None
            else cast(FaceBackend, _BorrowedFaceBackend(plugins.face_analyzer))
        )
        former = (
            None
            if plugins.former is None
            else cast(FormationBackend, _BorrowedFormationBackend(plugins.former))
        )
        consolidator = (
            None
            if plugins.consolidator is None
            else cast(ConsolidationBackend, _BorrowedConsolidationBackend(plugins.consolidator))
        )
        if plugins.vision_describer is None:
            vision_describer = None
        else:
            borrowed = cast(
                VisionDescriptionBackend, _BorrowedVisionDescriber(plugins.vision_describer)
            )
            # Opened only when `vision:` is configured, so a run without the slot touches no
            # cache file at all and behaves exactly as it does today.
            if description_cache is not None:
                self._description_cache = DescriptionCache(
                    description_cache, plugins.vision_describer.vision_space
                )
                borrowed = cast(
                    VisionDescriptionBackend,
                    _CachedVisionDescriber(borrowed, self._description_cache),
                )
            vision_describer = borrowed
        self._plugins = replace(
            plugins,
            embedder=embedder,
            answerer=answerer,
            reranker=reranker,
            transcriber=transcriber,
            face_analyzer=face_analyzer,
            former=former,
            consolidator=consolidator,
            vision_describer=vision_describer,
            query_policy=(
                None
                if plugins.query_policy is None
                else cast(QueryPolicy, _BorrowedBackend(plugins.query_policy))
            ),
            candidate_source=(
                None
                if plugins.candidate_source is None
                else cast(CandidateSource, _BorrowedBackend(plugins.candidate_source))
            ),
            ranking_policy=(
                None
                if plugins.ranking_policy is None
                else cast(RankingPolicy, _BorrowedBackend(plugins.ranking_policy))
            ),
            context_selector=(
                None
                if plugins.context_selector is None
                else cast(ContextSelector, _BorrowedBackend(plugins.context_selector))
            ),
            ingestion_stages=tuple(
                cast(IngestionStage, _BorrowedBackend(stage)) for stage in plugins.ingestion_stages
            ),
            tools=None if plugins.tools is None else _BorrowedToolRegistry(plugins.tools),
        )
        # Answer-time reinforcement is a product behaviour, not a measured one: it makes a
        # question's retrieval depend on which earlier questions ran, and under concurrency on
        # the order their updates committed, so a run stops being reproducible from its seed.
        self._settings = replace(composition.settings, reinforce_on_answer=False)
        self._plugins.embedder.embed(
            (ModelInput(text="SenseMem benchmark warmup"),),
            task=EmbedTask.QUERY,
        )
        self.embedding_warmup_count = 1

    def prefetch_speech(self, items: Sequence[MemoryItem]) -> None:
        """Analyse the next chunk's speech while a store embeds and writes the current one."""
        if isinstance(self._plugins.transcriber, _BorrowedSpeechBackend):
            self._plugins.transcriber.prefetch(items)

    def memory(self, data_dir: Path) -> AsyncMemory:
        # Keep the validated policy object intact across the public SDK boundary.
        return AsyncMemory(
            Memory.from_plugins(
                data_dir,
                plugins=self._plugins,
                tracer=self._tracer,
                settings=self._settings,
            )
        )

    def close(self) -> None:
        if isinstance(self._plugins.transcriber, _BorrowedSpeechBackend):
            self._plugins.transcriber.shutdown_prefetch()
        if self._description_cache is not None:
            with suppress(Exception):
                self._description_cache.close()
            self._description_cache = None
        self._composition.close()
