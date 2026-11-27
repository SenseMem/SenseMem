"""Evaluation arms: the product arm and the blind and full-context baseline generators."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

try:
    import openai
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
except ImportError:
    openai = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    ChatCompletion = None  # type: ignore[assignment,misc]

from opentelemetry import trace
from opentelemetry.trace import Tracer

from sensemem import AbstentionReason, AnswerPolicy, AssetRef, MemorySpec, Modality, SearchHit
from sensemem._telemetry import (
    MODEL_MODULE,
    SPAN_KIND,
    mark_model_requests,
    model_span,
)
from sensemem.benchmarks.baseline_prompts import baseline_system_prompt
from sensemem.benchmarks.eval_config import (
    BASELINE_ARMS,
    MODALITY_BY_SUFFIX,
    resolve_generation_temperature,
)
from sensemem.benchmarks.eval_results import DEFAULT_ARM
from sensemem.benchmarks.model_config import ModelConfig
from sensemem.benchmarks.prompts import task_answer_surface
from sensemem.configuration import OpenAIGenerationConfig
from sensemem.models._openai_protocol import (
    _json_text,
    output_limit_parameter,
    reasoning_model,
)
from sensemem.models._openai_telemetry import (
    _model_usage,
    _record_openai_provenance,
    _record_usage_batch,
)
from sensemem.models.base import ModelInput
from sensemem.models.openai_completion import OpenAICompletion


@dataclass(frozen=True, slots=True)
class Arm:
    """One evaluation arm: what it may read, and what generates its answer."""

    name: str
    generator: BaselineGenerator | None = None
    seed: int = 0
    allow_partial_sources: bool = False

    @property
    def retrieves(self) -> bool:
        return self.name in {DEFAULT_ARM, "random"}

    @property
    def generates(self) -> bool:
        return self.name != "random"

    @property
    def reads_memory(self) -> bool:
        """Report whether this arm needs the ingested store at all."""
        return self.retrieves or self.name == "compile"


PRODUCT_ARM = Arm(DEFAULT_ARM)


class BaselineGenerator:
    """Call the configured generation model directly for the no-retrieval baseline arms.

    Deliberately outside the product path: Memory.ask abstains before it reaches the model
    when no hit survives grounding, so neither baseline could exist through it.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int,
        gen_kwargs: str,
        generation: MemorySpec | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if openai is None:
            raise RuntimeError("baseline arms require sensemem[openai]")
        self._tracer = trace.get_tracer("sensemem.benchmarks.eval") if tracer is None else tracer
        options = dict(item.split("=", 1) for item in gen_kwargs.split(",") if "=" in item)
        self._model = config.generation_model
        self._seed = None if reasoning_model(self._model) else seed
        self._max_tokens = (
            None if "max_tokens" not in options else int(options["max_tokens"]) or None
        )
        thinking = options.get("enable_thinking")
        extra_body: dict[str, object] = (
            {}
            if thinking is None
            else {"chat_template_kwargs": {"enable_thinking": thinking == "true"}}
        )
        # The declarative generation stanza reaches the product answerer, so a baseline that
        # ignored it would not be the same request. extra_body is the one that decides whether
        # a thinking model answers at all: without the deployment's reasoning_effort the model
        # spends its budget on reasoning tokens and every reply ends finish_reason=length, which
        # would score as "the baseline knows nothing" rather than "the baseline was misconfigured".
        # Sampling controls are sent only where the shared provider policy supports them.
        # Explicit controls in extra_body still reach the adapter's validation.
        stanza = None if generation is None else generation.generation
        if not isinstance(stanza, OpenAIGenerationConfig):
            stanza = None
        self._generation_store = False if stanza is None else stanza.store
        self._answer_style: Literal["default", "concise"] = (
            "default" if stanza is None else stanza.answer_style
        )
        if stanza is not None:
            if stanza.extra_body is not None:
                extra_body.update(stanza.extra_body)
            if stanza.max_tokens is not None and self._max_tokens is None:
                self._max_tokens = stanza.max_tokens
        self._extra_body = extra_body or None
        self._temperature = resolve_generation_temperature(self._model, options, self._extra_body)
        self._client = openai.AsyncOpenAI(
            api_key=config.generation_api_key,
            base_url=config.generation_base_url,
            timeout=config.timeout_seconds,
        )
        self._generation_capabilities = config.generation_capabilities
        self._query_asset_cache: dict[Path, AssetRef] = {}
        self._native_media = OpenAICompletion(
            generation_model=self._model,
            generation_capabilities=self._generation_capabilities,
            generation_seed=self._seed,
            generation_temperature=self._temperature,
            generation_max_tokens=self._max_tokens,
            generation_min_video_seconds=config.generation_min_video_seconds,
            generation_video_limit=(8 if stanza is None else stanza.video_limit),
            generation_extra_body=self._extra_body,
            generation_answer_style=self._answer_style,
            generation_store=self._generation_store,
        )

    async def answer(
        self,
        question: str,
        context: str | None,
        *,
        question_assets: Sequence[Path] = (),
        evidence_hits: Sequence[SearchHit] = (),
        system_prompt: str | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> str:
        system = baseline_system_prompt(
            has_context=context is not None,
            override=system_prompt,
            answer_policy=answer_policy,
            answer_style=self._answer_style,
        )
        user = (
            question if context is None else _json_text({"context": context, "question": question})
        )
        resolved_question_assets = tuple(self._query_asset(path) for path in question_assets)
        if resolved_question_assets or any(hit.assets for hit in evidence_hits):
            request, modalities = self._native_media_request(
                question,
                user,
                system,
                resolved_question_assets,
                evidence_hits,
            )
        else:
            request = {
                "model": self._model,
                "store": self._generation_store,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if self._temperature is not None:
                request["temperature"] = self._temperature
            if self._seed is not None:
                request["seed"] = self._seed
            if self._max_tokens is not None:
                request[output_limit_parameter(self._model)] = self._max_tokens
            if self._extra_body is not None:
                request["extra_body"] = self._extra_body
            modalities = frozenset({Modality.TEXT})
        with model_span(
            self._tracer,
            "sensemem.model.generation",
            attributes={
                SPAN_KIND: "model",
                MODEL_MODULE: "generation",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self._model,
                "sensemem.model.batch_size": 1,
                "sensemem.input.modalities": tuple(
                    sorted(modality.value for modality in modalities)
                ),
            },
        ):
            usages = []
            try:
                mark_model_requests(1)
                create = cast(
                    "Callable[..., Awaitable[ChatCompletion]]",
                    self._client.chat.completions.create,
                )
                response = await create(**request)
                _record_openai_provenance(response)
                usages.append(
                    _model_usage(
                        response,
                        input_modalities=modalities,
                        output_modalities=frozenset({Modality.TEXT}),
                    )
                )
                return str(response.choices[0].message.content or "").strip()
            finally:
                _record_usage_batch(usages, request_count=1)

    def _query_asset(self, path: Path) -> AssetRef:
        resolved = path.expanduser().resolve(strict=True)
        cached = self._query_asset_cache.get(resolved)
        if cached is not None:
            return cached
        modality = MODALITY_BY_SUFFIX.get(resolved.suffix.casefold())
        if modality is None:
            raise ValueError(f"benchmark query media has unsupported suffix: {resolved}")
        asset = media_asset_ref(resolved, modality)
        self._query_asset_cache[resolved] = asset
        return asset

    def _native_media_request(
        self,
        question: str,
        user: str,
        system: str,
        question_assets: tuple[AssetRef, ...],
        evidence_hits: Sequence[SearchHit],
    ) -> tuple[dict[str, object], frozenset[Modality]]:
        # Reuse the product adapter's capability checks, integrity checks, media-size fitting,
        # video limits, and native part preparation. The benchmark then restores its existing
        # context prompt; no benchmark labels or reference answers participate in this request.
        hits = tuple(evidence_hits)
        if not hits:
            hits = (
                SearchHit(
                    id="benchmark-compiled-context",
                    content=user,
                    score=1.0,
                    created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
                ),
            )
        prepared = self._native_media._requests.answer_request(
            ModelInput(text=question, assets=question_assets), hits
        )
        if isinstance(prepared, AbstentionReason):
            raise RuntimeError("native-media preparation rejected a compiled context")
        request, grounded, _prepared_modalities = prepared
        messages = cast(list[dict[str, object]], request["messages"])
        native_content = messages[-1]["content"]
        if isinstance(native_content, str):
            content: str | list[dict[str, object]] = user
        else:
            content = self._bound_native_content(
                user,
                question_assets,
                grounded,
                cast(Sequence[dict[str, object]], native_content),
            )
        request["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        modalities = {Modality.TEXT}
        kind_to_modality = {
            "image_url": Modality.IMAGE,
            "video_url": Modality.VIDEO,
            "input_audio": Modality.AUDIO,
        }
        if not isinstance(content, str):
            modalities.update(
                kind_to_modality[kind]
                for part in content
                if isinstance((kind := part.get("type")), str) and kind in kind_to_modality
            )
        return request, frozenset(modalities)

    @classmethod
    def _bound_native_content(
        cls,
        user: str,
        question_assets: Sequence[AssetRef],
        grounded: Sequence[SearchHit],
        native_content: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        groups = cls._native_media_groups(native_content)
        if len(groups) != len(grounded) + 1:
            raise RuntimeError("native-media preparation did not align with selected hits")
        content: list[dict[str, object]] = [{"type": "text", "text": user}]
        seen_assets: set[str] = set()
        query_media = groups[0]
        query_asset_ids = cls._unseen_asset_ids(question_assets, seen_assets)
        if bool(query_media) != bool(query_asset_ids):
            raise RuntimeError("native query media did not align with its asset references")
        if query_media:
            content.append({"type": "text", "text": _json_text({"query_assets": query_asset_ids})})
            content.extend(query_media)
        for hit, media in zip(grounded, groups[1:], strict=True):
            asset_ids = cls._unseen_asset_ids(hit.assets, seen_assets)
            if bool(media) != bool(asset_ids):
                raise RuntimeError("native evidence media did not align with its source hit")
            if media:
                content.append(
                    {
                        "type": "text",
                        "text": _json_text({"memory_assets": cls._media_binding(hit, asset_ids)}),
                    }
                )
                content.extend(media)
        return content

    @staticmethod
    def _unseen_asset_ids(assets: Sequence[AssetRef], seen: set[str]) -> tuple[str, ...]:
        ids: list[str] = []
        for asset in assets:
            if asset.id in seen:
                continue
            seen.add(asset.id)
            ids.append(asset.id)
        return tuple(ids)

    @staticmethod
    def _native_media_groups(
        content: Sequence[dict[str, object]],
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        """Split product-prepared native parts at their existing JSON text labels.

        Only a JSON label -- the question or a memory payload -- opens a group. The product also
        writes a plain-text attachment marker right before each asset's media parts; that marker
        belongs with the media it announces and travels inside the group, so the rebuilt request
        keeps the same label beside the same picture.
        """
        groups: list[tuple[dict[str, object], ...]] = []
        media: list[dict[str, object]] | None = None
        for part in content:
            if part.get("type") == "text" and str(part.get("text", "")).startswith("{"):
                if media is not None:
                    groups.append(tuple(media))
                media = []
            elif media is None:
                raise RuntimeError("native-media preparation emitted an unlabeled media part")
            else:
                media.append(part)
        if media is None:
            raise RuntimeError("native-media preparation emitted no text labels")
        groups.append(tuple(media))
        return tuple(groups)

    @staticmethod
    def _media_binding(hit: SearchHit, asset_ids: Sequence[str]) -> dict[str, object]:
        """Bind media to the already-rendered memory without repeating its content."""
        context = hit.context
        source_id = None if context is None else context.source_id
        if source_id is None:
            candidate = hit.metadata.get("source_id")
            source_id = candidate if isinstance(candidate, str) and candidate else None
        binding: dict[str, object] = {
            "memory_id": hit.id,
            "memory_type": hit.memory_type.value,
            "event_time": (hit.occurred_at or hit.created_at).isoformat(),
            "asset_ids": tuple(asset_ids),
        }
        if source_id is not None:
            binding["source_id"] = source_id
        if hit.occurred_end is not None:
            binding["event_end"] = hit.occurred_end.isoformat()
        if context is not None:
            binding["kind"] = context.kind.value
            binding["basis"] = context.basis.value
        return binding

    async def close(self) -> None:
        await self._client.close()


def media_asset_ref(path: Path, modality: Modality) -> AssetRef:
    """Describe a local media file the way the store will, keyed by its content digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    asset_id = digest.hexdigest()
    return AssetRef(
        id=asset_id,
        modality=modality,
        media_type=_MEDIA_TYPE_BY_SUFFIX[path.suffix.casefold()],
        size_bytes=path.stat().st_size,
        sha256=asset_id,
        name=path.name,
        path=path,
    )


_MEDIA_TYPE_BY_SUFFIX = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def answers_through_ask(arm: Arm | str, task_name: str) -> bool:
    """Report whether this arm reaches Memory.ask on this task.

    Only that path exposes the in-answer ranked list; baseline generators request an answer
    policy too, while the product's compile surface follows its host-assistant prompt.
    """
    name = arm if isinstance(arm, str) else arm.name
    return name == DEFAULT_ARM and task_answer_surface(task_name) == "ask"


def requests_answer_policy(arm: Arm | str, task_name: str) -> bool:
    name = arm if isinstance(arm, str) else arm.name
    return name in {"blind", "full-context", "compile"} or answers_through_ask(name, task_name)


def generator_arms(arms: Sequence[str], task_names: Sequence[str]) -> frozenset[str]:
    """Name the arms that answer through the harness generator.

    The generating baselines always do. The product arm does only when a selected task answers
    through Memory.compile, whose bundle is handed to the same generator standing in for the
    host assistant; on every other task it answers through Memory.ask and needs none.
    """
    names = {name for name in arms if name in BASELINE_ARMS and Arm(name).generates}
    if DEFAULT_ARM in arms and not all(
        answers_through_ask(DEFAULT_ARM, name) for name in task_names
    ):
        names.add(DEFAULT_ARM)
    return frozenset(names)
