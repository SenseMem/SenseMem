"""OpenAI generation adapter over one explicitly owned SDK client."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from opentelemetry import trace

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion

from sensemem._telemetry import (
    GEN_AI_FINISH_REASONS,
    GEN_AI_TTFC,
    GROUNDING_MEDIA_FALLBACK,
    MODEL_TTFT,
    mark_model_requests,
    nested_model_span,
)
from sensemem._validation import nonempty_text
from sensemem.content import AssetRef
from sensemem.control_values import MemoryOperation, MemoryTrigger, QueryFailure
from sensemem.events import EventSegment
from sensemem.evidence import (
    retain_source_complete_hits,
)
from sensemem.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from sensemem.models import _openai_completion as completion
from sensemem.models import _openai_media as media
from sensemem.models import _openai_output as output
from sensemem.models import _openai_protocol as protocol
from sensemem.models import _openai_requests as requests
from sensemem.models import _openai_telemetry as telemetry
from sensemem.models._evidence_operations import (
    EvidenceCatalogue,
    InvalidEvidenceOperation,
    execute_evidence_operation,
)
from sensemem.models._openai_client import OpenAIClient
from sensemem.models._openai_defaults import (
    DEFAULT_GENERATION_MODEL,
)
from sensemem.models._openai_protocol import (
    output_limit_parameter,
    reasoning_model,
    sampling_supported,
    validate_sampling,
)
from sensemem.models.base import (
    EventObservation,
    FormationInput,
    ModelInput,
    _modalities,
)
from sensemem.perception import VisualDescription
from sensemem.prompts import generation as prompts
from sensemem.prompts import recall as recall_prompts
from sensemem.prompts.recall import SEMANTIC_QUERY_CAPABILITY
from sensemem.records import AnswerResult, MemoryRecord, SearchHit
from sensemem.semantics import FormationProposal
from sensemem.types import AbstentionReason, AnswerPolicy, Modality

_JsonResult = TypeVar("_JsonResult")


class OpenAICompletion(OpenAIClient):
    def __init__(  # noqa: C901 - independent completion controls at the adapter boundary
        self,
        client: OpenAI | None = None,
        *,
        owns_client: bool = False,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
        generation_seed: int | None = None,
        generation_temperature: float | None = None,
        generation_max_tokens: int | None = None,
        generation_media_max_tokens: int = 8192,
        generation_response_format: Literal["json_object", "json_schema"] = "json_schema",
        generation_store: bool = False,
        generation_min_video_seconds: float | None = None,
        generation_video_limit: int | None = 8,
        generation_extra_body: Mapping[str, object] | None = None,
        generation_stream: bool = False,
        generation_answer_style: Literal["default", "concise"] = "default",
        generation_answer_format: Literal["text", "structured"] = "text",
        generation_evidence_operation_tool: bool = False,
        generation_image_source_prose: Literal["include", "context_when_supplied"] = "include",
        generation_source_metadata_key: str = "source_id",
        vision_video_frames: int = 4,
        image_max_pixels: int | None = None,
        generation_request_max_bytes: int | None = None,
        generation_audio_data_url: bool = False,
    ) -> None:
        super().__init__(client, owns_client=owns_client)
        if generation_request_max_bytes is not None and (
            isinstance(generation_request_max_bytes, bool)
            or not isinstance(generation_request_max_bytes, int)
            or generation_request_max_bytes < 1
        ):
            raise ValidationError("generation_request_max_bytes must be a positive integer")
        self._generation_request_max_bytes = generation_request_max_bytes
        if not isinstance(generation_answer_format, str) or generation_answer_format not in {
            "text",
            "structured",
        }:
            raise ValidationError("generation_answer_format must be text or structured")
        self._generation_answer_format = generation_answer_format
        if not isinstance(generation_evidence_operation_tool, bool):
            raise ValidationError("generation_evidence_operation_tool must be a boolean")
        if generation_evidence_operation_tool and generation_answer_format != "text":
            raise ValidationError("generation_evidence_operation_tool requires text answer format")
        self._generation_evidence_operation_tool = generation_evidence_operation_tool
        if not isinstance(
            generation_image_source_prose, str
        ) or generation_image_source_prose not in {
            "include",
            "context_when_supplied",
        }:
            raise ValidationError(
                "generation_image_source_prose must be include or context_when_supplied"
            )
        self._generation_image_source_prose = generation_image_source_prose
        self._generation_source_metadata_key = nonempty_text(
            generation_source_metadata_key, "generation_source_metadata_key"
        )
        if not isinstance(generation_answer_style, str) or generation_answer_style not in {
            "default",
            "concise",
        }:
            raise ValidationError("generation_answer_style must be default or concise")
        self._generation_answer_style = generation_answer_style
        if image_max_pixels is not None and (
            isinstance(image_max_pixels, bool)
            or not isinstance(image_max_pixels, int)
            or not 1 <= image_max_pixels <= 64_000_000
        ):
            raise ValidationError("image_max_pixels must be between 1 and 64000000")
        self._image_max_pixels = image_max_pixels
        if (
            isinstance(vision_video_frames, bool)
            or not isinstance(vision_video_frames, int)
            or not 2 <= vision_video_frames <= 64
        ):
            raise ValidationError("vision_video_frames must be between 2 and 64")
        self._vision_video_frames = vision_video_frames
        generation_model = nonempty_text(generation_model, "generation_model")
        if generation_response_format not in {"json_object", "json_schema"}:
            raise ValidationError("generation_response_format must be json_object or json_schema")
        if generation_seed is not None and (
            isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
            or not 0 <= generation_seed < 2**63
        ):
            raise ValidationError("generation_seed must be an integer between 0 and 2^63 - 1")
        if generation_temperature is not None and (
            isinstance(generation_temperature, bool)
            or not isinstance(generation_temperature, int | float)
            or not math.isfinite(generation_temperature)
            or not 0 <= generation_temperature <= 2
        ):
            raise ValidationError("generation_temperature must be between zero and two")
        if generation_max_tokens is not None and (
            isinstance(generation_max_tokens, bool)
            or not isinstance(generation_max_tokens, int)
            or generation_max_tokens <= 0
        ):
            raise ValidationError("generation_max_tokens must be a positive integer")
        if (
            isinstance(generation_media_max_tokens, bool)
            or not isinstance(generation_media_max_tokens, int)
            or generation_media_max_tokens <= 0
        ):
            raise ValidationError("generation_media_max_tokens must be a positive integer")
        if generation_min_video_seconds is not None and (
            isinstance(generation_min_video_seconds, bool)
            or not isinstance(generation_min_video_seconds, int | float)
            or not math.isfinite(generation_min_video_seconds)
            or generation_min_video_seconds <= 0
        ):
            raise ValidationError("generation_min_video_seconds must be a positive finite number")
        if generation_video_limit is not None and (
            isinstance(generation_video_limit, bool)
            or not isinstance(generation_video_limit, int)
            or generation_video_limit <= 0
        ):
            raise ValidationError("generation_video_limit must be a positive integer")
        if generation_extra_body is not None and (
            not isinstance(generation_extra_body, Mapping)
            or any(not isinstance(key, str) or not key.strip() for key in generation_extra_body)
        ):
            raise ValidationError("generation_extra_body must have non-empty string keys")
        try:
            json.dumps(generation_extra_body, allow_nan=False, sort_keys=True)
        except (RecursionError, TypeError, ValueError):
            raise ValidationError("generation_extra_body must be JSON-compatible") from None
        if generation_evidence_operation_tool and {
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        }.intersection(generation_extra_body or {}):
            raise ValidationError(
                "generation_extra_body must not override the evidence operation tool"
            )
        if not isinstance(generation_stream, bool):
            raise ValidationError("generation_stream must be a boolean")
        if not isinstance(generation_store, bool):
            raise ValidationError("generation_store must be a boolean")
        if not isinstance(generation_audio_data_url, bool):
            raise ValidationError("generation_audio_data_url must be a boolean")
        validate_sampling(
            generation_model,
            temperature=generation_temperature,
            seed=generation_seed,
            extra_body=generation_extra_body,
        )
        self._generation_model = generation_model
        self._generation_response_format = generation_response_format
        self._generation_capabilities = _modalities(generation_capabilities, "generation")
        self._generation_seed = generation_seed
        self._generation_stream = generation_stream
        self._generation_store = generation_store
        self._generation_audio_data_url = generation_audio_data_url
        self._generation_temperature = generation_temperature
        self._generation_max_tokens = generation_max_tokens
        self._generation_media_max_tokens = generation_media_max_tokens
        self._generation_min_video_seconds = (
            None if generation_min_video_seconds is None else float(generation_min_video_seconds)
        )
        self._generation_video_limit = generation_video_limit
        self._generation_extra_body = (
            None if generation_extra_body is None else dict(generation_extra_body)
        )
        self._formation_space = protocol.default_formation_space(
            generation_model,
            capabilities=self._generation_capabilities,
            seed=generation_seed,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
            extra_body=self._generation_extra_body,
            image_max_pixels=image_max_pixels,
        )
        self._vision_space = protocol.default_vision_space(
            generation_model,
            capabilities=self._generation_capabilities,
            seed=(
                generation_seed
                if generation_seed is not None or reasoning_model(generation_model)
                else protocol.VISION_SEED
            ),
            temperature=(
                generation_temperature
                if generation_temperature is not None
                else 0.0
                if sampling_supported(generation_model, generation_extra_body)
                else None
            ),
            max_tokens=generation_max_tokens,
            extra_body=self._generation_extra_body,
            video_frames=vision_video_frames,
            image_max_pixels=image_max_pixels,
        )
        self._consolidation_recipe = protocol.default_reasoning_recipe(
            generation_model,
            prompt=prompts._CONSOLIDATION_SYSTEM_PROMPT,
            label="consolidation",
            capabilities=self._generation_capabilities,
            seed=generation_seed,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
            extra_body=self._generation_extra_body,
            # Counterevidence and complete wire examples change the reasoning recipe, which
            # participates in operation keys and derived record identities.
            version="v5",
        )
        self._event_recipe = protocol.default_reasoning_recipe(
            generation_model,
            prompt=prompts._EVENT_SEGMENTATION_SYSTEM_PROMPT,
            label="event-segmentation",
            capabilities=self._generation_capabilities,
            seed=generation_seed,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
            extra_body=self._generation_extra_body,
            version="v3",
        )
        self._event_recipe += (
            protocol.schema_recipe("event_segmentation")
            if generation_response_format == "json_schema"
            else ":json-object-v1"
        )
        if image_max_pixels is not None:
            self._event_recipe += media.image_transport_recipe(image_max_pixels)
        if generation_response_format == "json_schema":
            self._formation_space += protocol.schema_recipe("formation")
            self._vision_space += protocol.schema_recipe("vision")
            self._consolidation_recipe += protocol.schema_recipe("consolidation")
        self._formation_space += ":source-parts-v2"
        self._consolidation_recipe += ":source-parts-v2"
        self._requests = requests.OpenAIRequestBuilder(
            model=self._generation_model,
            response_format=self._generation_response_format,
            capabilities=self._generation_capabilities,
            seed=self._generation_seed,
            temperature=self._generation_temperature,
            max_tokens=self._generation_max_tokens,
            extra_body=self._generation_extra_body,
            store=self._generation_store,
            video_limit=self._generation_video_limit,
            min_video_seconds=self._generation_min_video_seconds,
            image_max_pixels=self._image_max_pixels,
            request_max_bytes=self._generation_request_max_bytes,
            answer_style=self._generation_answer_style,
            answer_format=self._generation_answer_format,
            evidence_operation_tool=self._generation_evidence_operation_tool,
            source_metadata_key=self._generation_source_metadata_key,
            audio_data_url=self._generation_audio_data_url,
        )

    @property
    def generation_capabilities(self) -> frozenset[Modality]:
        return self._generation_capabilities

    @property
    def generation_model(self) -> str:
        return self._generation_model

    @property
    def event_model(self) -> str:
        return self._generation_model

    @property
    def event_recipe(self) -> str:
        return self._event_recipe

    @property
    def formation_capabilities(self) -> frozenset[Modality]:
        return self._generation_capabilities

    @property
    def formation_model(self) -> str:
        return self._generation_model

    @property
    def formation_space(self) -> str:
        return self._formation_space

    @property
    def vision_capabilities(self) -> frozenset[Modality]:
        # Description is a chat completion, so the visual modalities the endpoint accepts are the
        # generation ones, exactly as formation reads them. Memory narrows this further: a
        # describer declaring anything but image or video is rejected at construction.
        return self._generation_capabilities

    @property
    def vision_model(self) -> str:
        return self._generation_model

    @property
    def consolidation_model(self) -> str:
        return self._generation_model

    @property
    def consolidation_recipe(self) -> str:
        return self._consolidation_recipe

    @property
    def vision_space(self) -> str:
        return self._vision_space

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        """Propose complete event prefixes while preserving source order and native media."""
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(observations, (str, bytes)):
            raise ValidationError("observations must contain EventObservation values")
        batch = tuple(observations)
        if any(not isinstance(value, EventObservation) for value in batch):
            raise ValidationError("observations must contain EventObservation values")
        if not isinstance(end_of_stream, bool):
            raise ValidationError("end_of_stream must be a boolean")
        if not batch:
            return ()
        modalities = frozenset(modality for value in batch for modality in value.content.modalities)
        protocol.require_capabilities(
            "event segmentation", modalities, self._generation_capabilities
        )
        parts = requests.event_content(
            batch,
            end_of_stream=end_of_stream,
            image_max_pixels=self._image_max_pixels,
            audio_data_url=self._generation_audio_data_url,
        )
        request = self._requests.json_request(
            prompts._EVENT_SEGMENTATION_SYSTEM_PROMPT, parts, schema="event_segmentation"
        )
        return self._json_completion(
            request,
            subject="event segmentation",
            stage="segment_events",
            input_modalities=modalities,
            parse=lambda content: output.event_segments(content, len(batch)),
        )

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]:
        """Propose source-grounded typed memories with the configured generation model."""
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(inputs, (str, bytes)):
            raise ValidationError("inputs must contain FormationInput values")
        batch = tuple(inputs)
        if any(not isinstance(value, FormationInput) for value in batch):
            raise ValidationError("inputs must contain FormationInput values")
        if not batch:
            return ()
        modalities = frozenset(modality for value in batch for modality in value.content.modalities)
        protocol.require_capabilities("formation", modalities, self.formation_capabilities)
        assets = tuple(asset for value in batch for asset in value.content.assets)
        media.require_consistent_assets(assets)
        verified_assets = media._require_asset_integrity(assets)
        image_parts = media._image_transport_parts(assets, self._image_max_pixels)
        media.require_verified_inline_size(
            tuple(verified for verified in verified_assets if verified[0].id not in image_parts)
        )
        content = requests._formation_content(
            batch,
            image_parts=image_parts,
            audio_data_url=self._generation_audio_data_url,
        )
        if not isinstance(content, str):
            media.require_inline_media_parts(content)
        request = self._requests.json_request(
            prompts._FORMATION_SYSTEM_PROMPT,
            content,
            schema="formation",
            formation_count=len(batch),
        )
        return self._json_completion(
            request,
            subject="formation",
            stage="form",
            input_modalities=modalities,
            parse=lambda content: output._formation_results(
                content, batch, source_aliases=requests._formation_aliases(batch)
            ),
            retry_instruction=lambda _content: prompts.FORMATION_RETRY_INSTRUCTION,
        )

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        """Caption every supplied visual in one chat completion, for the text index.

        The write path calls this so that an image-only memory has a full-text document at all.
        What travels is pixels, an ordinal, safe native capture metadata, and the input's
        text -- in practice the clip's diarised transcript: no memory ID, file name, or path,
        because the caption is stored beside the caller's own text and anything else in the
        prompt becomes provenance leaked into a searchable document. A video is sent as the
        configured number of ordered stills, never as the file; one caption preserves the
        observations and event order across those samples.

        A caption is one string per visual holding labelled lines separated by single newlines,
        never a blank line: the write path splits it into the memory's description and facts
        sections, and a blank line inside it would end the section it belongs to.

        The request asks for temperature 0 and a fixed seed unless the composition set its
        own, which removes the sampler as a cause of drift. It does not make captions
        reproducible: a measured endpoint returned four distinct completions for four identical
        requests at temperature 0 and seed 0, so a caller that needs two ingests of one corpus
        to build the same documents must cache descriptions by asset, not rely on these values.

        One malformed reply is retried once. An HTTP 200 carrying invalid JSON is outside what
        the SDK's max_retries covers and is transient in practice; a truncation or any other
        failure is not retried, because an identical second request cannot clear it.
        """
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(inputs, (str, bytes)):
            raise ValidationError("inputs must contain ModelInput values")
        batch = tuple(inputs)
        if any(not isinstance(value, ModelInput) for value in batch):
            raise ValidationError("inputs must contain ModelInput values")
        if not batch:
            return ()
        modalities = frozenset(modality for value in batch for modality in value.modalities)
        # Only the media are a routing question. A visual's derived context travels as this
        # input's text, which every chat completion takes, while vision_capabilities is
        # narrowed to the visual modalities -- so requiring text of it would refuse a request
        # the endpoint accepts. modalities still carries it, because it is billed input.
        protocol.require_capabilities(
            "vision",
            modalities - {Modality.TEXT},
            self.vision_capabilities,
        )
        assets = tuple(asset for value in batch for asset in value.assets)
        media.require_consistent_assets(assets)
        # Video files are never uploaded by this operation. Verify the source descriptors first,
        # reject oversized source images before reading them, then enforce provider limits against
        # the image data URLs the request actually carries.
        verified_assets = media._require_asset_integrity(assets)
        capture_metadata: dict[str, dict[str, object]] = {}
        image_parts = media._image_transport_parts(
            assets, self._image_max_pixels, capture_metadata=capture_metadata
        )
        media.require_verified_inline_size(
            tuple(
                verified
                for verified in verified_assets
                if verified[0].modality is not Modality.VIDEO and verified[0].id not in image_parts
            )
        )
        content = requests.vision_content(
            batch,
            video_frames=self._vision_video_frames,
            image_parts=image_parts,
            capture_metadata=capture_metadata,
        )
        media.require_inline_media_parts(content)
        request = self._requests.json_request(
            prompts._VISION_SYSTEM_PROMPT, content, schema="vision"
        )
        if sampling_supported(self._generation_model, self._generation_extra_body):
            request.setdefault("temperature", 0.0)
        if not reasoning_model(self._generation_model):
            request.setdefault("seed", protocol.VISION_SEED)
        return self._json_completion(
            request,
            subject="vision description",
            stage="describe",
            input_modalities=modalities,
            parse=lambda content: output.vision_captions(content, len(batch)),
            retry_instruction=lambda content: prompts._vision_retry_instruction(
                content, len(batch)
            ),
        )

    def describe_analysis(self, inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]:
        return tuple(output._visual_description(value) for value in self.describe(inputs))

    def _json_completion(
        self,
        request: dict[str, object],
        *,
        subject: str,
        stage: str,
        input_modalities: frozenset[Modality],
        parse: Callable[[str], _JsonResult],
        retry_instruction: Callable[[str], str] | None = None,
    ) -> _JsonResult:
        """Send one JSON chat completion and parse it, retrying a malformed reply once.

        An HTTP 200 carrying invalid JSON is outside what the SDK's max_retries covers and is
        transient in practice. A caller may add a shape-only correction to its second request;
        other callers repeat the original request. A truncation or any other failure is not
        retried. Both attempts are metered, even when the second one also fails: the first request
        was billed whether or not its reply parsed, so the usage is summed in finally rather
        than recorded per attempt, which would report only the last.
        """
        create_completion = cast(
            Callable[..., object], self._client("generation").chat.completions.create
        )
        usages: list[telemetry._ModelUsage | None] = []
        attempted = 0
        invalid_content: str | None = None

        def remember_usage(response: object) -> None:
            usages.append(
                telemetry._model_usage(
                    response,
                    input_modalities=input_modalities,
                    output_modalities=frozenset({Modality.TEXT}),
                )
            )
            telemetry._record_openai_provenance(response)

        def attempt() -> _JsonResult:
            nonlocal attempted, invalid_content
            attempted += 1
            mark_model_requests(attempted)
            try:
                cleanup_error: Exception | None = None
                response = create_completion(
                    **request,
                    **(
                        {"stream": True, "stream_options": {"include_usage": True}}
                        if self._generation_stream
                        else {}
                    ),
                )
                if self._generation_stream:
                    response, cleanup_error = completion._collect_streamed_completion(
                        response, subject=subject, stage=stage, on_failure=remember_usage
                    )
            except ModelError:
                raise
            except Exception as error:
                raise ModelError(
                    f"{subject} request failed",
                    reason=protocol._provider_reason(error),
                    stage=stage,
                ) from error
            remember_usage(response)
            content = output._json_completion_text(response, subject=subject, stage=stage)
            if isinstance(cleanup_error, Exception):
                raise ModelError(
                    f"{subject} response cleanup failed",
                    reason=protocol._provider_reason(cleanup_error),
                    stage=stage,
                ) from cleanup_error
            try:
                return parse(content)
            except output._InvalidStructuredResponse:
                invalid_content = content
                raise

        try:
            try:
                return attempt()
            except output._InvalidStructuredResponse:
                if retry_instruction is not None and invalid_content is not None:
                    request = requests.json_retry_request(
                        request, retry_instruction(invalid_content)
                    )
            return attempt()
        finally:
            telemetry._record_usage_batch(usages, request_count=attempted)

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]:
        """Propose memory operations over one bounded evidence set.

        The model cites evidence and targets by their integer position in the numbered evidence
        list, so a hallucinated memory identifier is structurally impossible. An index outside
        the list fails the whole response as response_invalid rather than being dropped: a
        proposal that miscounts its own evidence is not one the kernel should partially apply.
        The kernel still checks every resolved ID against the set it showed.

        Evidence media travels natively, the same way form and answer send it: an asset whose
        modality the generation model declares is attached as its own content part after the JSON
        for the record that owns it. Without that, a native image or audio memory with no derived
        description reached the slow plane as an empty content string and could only ever be
        forgotten, never reasoned about. An asset in a modality the model does not declare stays
        out of the request rather than failing the pass.
        """
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(evidence, (str, bytes)):
            raise ValidationError("evidence must contain MemoryRecord values")
        batch = tuple(evidence)
        if any(not isinstance(value, MemoryRecord) for value in batch):
            raise ValidationError("evidence must contain MemoryRecord values")
        if not isinstance(trigger, MemoryTrigger):
            raise ValidationError("trigger must be a MemoryTrigger")
        if not batch:
            return ()
        content = requests.consolidation_content(
            batch,
            trigger=trigger,
            query_failure=query_failure,
            capabilities=self._generation_capabilities,
            audio_data_url=self._generation_audio_data_url,
        )
        request = self._requests.json_request(
            prompts._CONSOLIDATION_SYSTEM_PROMPT, content, schema="consolidation"
        )
        return self._json_completion(
            request,
            subject="consolidation",
            stage="consolidate",
            input_modalities=media.generation_modalities(content),
            parse=lambda text: output._consolidation_results(text, batch),
        )

    def plan_recall(
        self,
        question: str,
        *,
        reference_at: datetime,
        corpus_digest: str,
        attempted: str = "",
    ) -> str | None:
        """Return the model's own JSON recall plan, or None when it produced nothing usable.

        The kernel validates the text, so this reports what the model said rather than deciding
        what it meant. Temperature is pinned to zero whatever the answer temperature is: a plan
        is a structural decision about which reads to make, and sampling it would make the same
        question read a different corpus twice.
        """
        mark_model_requests(0, token_usage_expected=0)
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("question must be non-empty text")
        if not isinstance(reference_at, datetime) or reference_at.utcoffset() is None:
            raise ValidationError("reference_at must be a timezone-aware datetime")
        if not isinstance(corpus_digest, str):
            raise ValidationError("corpus_digest must be text")
        if not isinstance(attempted, str):
            raise ValidationError("attempted must be text")
        content = protocol._json_text(
            {
                "reference_at": reference_at.isoformat(),
                "corpus_digest": corpus_digest.strip(),
                "question": question.strip(),
                **({"attempted": attempted.strip()} if attempted.strip() else {}),
            }
        )
        budget = recall_prompts.semantic_query_budget(corpus_digest)
        system = prompts.recall_system_prompt(
            semantic=SEMANTIC_QUERY_CAPABILITY in corpus_digest.splitlines(),
            additional_queries=budget,
        )
        request = self._requests.json_request(system, content, schema="recall_plan")
        if sampling_supported(self._generation_model, self._generation_extra_body):
            request["temperature"] = 0.0
        return self._json_completion(
            request,
            subject="recall plan",
            stage="plan",
            input_modalities=frozenset({Modality.TEXT}),
            # The plan stays text: validating it is the kernel's job, and a completion this
            # backend cannot even read as text already failed above.
            parse=lambda content: content,
        )

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        """Answer only from supplied hits, preserving native media content parts."""
        mark_model_requests(0, token_usage_expected=0)
        prepared = self._requests.answer_request(
            question,
            hits,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
            image_source_context=self._generation_image_source_prose == "context_when_supplied",
        )
        if isinstance(prepared, AbstentionReason):
            mark_model_requests(0, token_usage_expected=0)
            return AnswerResult(
                answer=prompts.UNKNOWN_ANSWER,
                abstained=True,
                abstention_reason=prepared,
            )
        operation_request = self._requests.evidence_operation_request(prepared, stream=False)
        if operation_request is not None:
            return self._answer_with_evidence_operation(
                question,
                hits,
                prepared,
                operation_request,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
            )
        response, grounded, modalities, request_count, _request = self._create_answer(
            question,
            hits,
            prepared,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
        )
        telemetry._record_openai_usage(
            response,
            input_modalities=modalities,
            output_modalities=frozenset({Modality.TEXT}),
            request_count=request_count,
        )
        answer = output.answer_text(response)
        return output._finish_answer(
            answer,
            question.evidence_sources or hits,
            grounded,
            answer_policy=answer_policy,
            answer_format=self._generation_answer_format,
            source_metadata_key=self._generation_source_metadata_key,
        )

    def stream_answer(  # noqa: C901 - stream validation and usage share one response lifecycle
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        """Yield grounded text deltas while recording first-token and final usage data.

        The deltas are the provider's own, marker included, so a best_effort stream shows the
        low-confidence line before the answer. The completion carries the cleaned answer, which
        is what a buffering caller reports.
        """
        mark_model_requests(0, token_usage_expected=0)
        prepared = self._requests.answer_request(
            question,
            hits,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
            stream=True,
            image_source_context=self._generation_image_source_prose == "context_when_supplied",
        )
        if isinstance(prepared, AbstentionReason):
            mark_model_requests(0, token_usage_expected=0)
            yield prompts.UNKNOWN_ANSWER
            return AnswerResult(
                answer=prompts.UNKNOWN_ANSWER, hits=(), abstained=True, abstention_reason=prepared
            )
        operation_request = self._requests.evidence_operation_request(prepared, stream=True)
        if operation_request is not None:
            result = self._answer_with_evidence_operation(
                question,
                hits,
                prepared,
                operation_request,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                stream=True,
            )
            yield result.answer
            return result
        from openai.lib.streaming.chat import ChatCompletionStreamState

        started = time.perf_counter()
        usage_response: object | None = None
        responses, grounded, modalities, request_count, _request = self._create_answer(
            question,
            hits,
            prepared,
            answer_policy=answer_policy,
            stream=True,
            exhaustive=exhaustive,
        )
        stream_completed = False
        telemetry._record_openai_provenance(responses)
        try:
            finish_reason: object = None
            refused = False
            emitted = False
            state: ChatCompletionStreamState[object] = ChatCompletionStreamState()
            snapshot: ParsedChatCompletion[object] | None = None
            first_chunk = False
            for response in cast("Iterable[ChatCompletionChunk]", responses):
                elapsed = time.perf_counter() - started
                span = trace.get_current_span()
                telemetry._record_openai_provenance(response)
                if not first_chunk:
                    span.set_attribute(GEN_AI_TTFC, elapsed)
                    first_chunk = True
                if getattr(response, "usage", None) is not None:
                    usage_response = response
                choices = getattr(response, "choices", None)
                if not choices:
                    continue
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or getattr(choices[0], "index", None) != 0
                ):
                    raise ModelError(
                        "generation response was invalid",
                        reason="response_invalid",
                        stage="generate",
                    )
                chunk_finish_reason = getattr(choices[0], "finish_reason", None)
                if chunk_finish_reason is not None:
                    finish_reason = chunk_finish_reason
                delta = getattr(choices[0], "delta", None)
                refused = refused or bool(getattr(delta, "refusal", None))
                if refused:
                    continue
                content = getattr(delta, "content", None)
                if content is not None and not isinstance(content, str):
                    raise ModelError(
                        "generation response was invalid",
                        reason="response_invalid",
                        stage="generate",
                    )
                state.handle_chunk(response)
                snapshot = state.current_completion_snapshot
                if content:
                    if not emitted:
                        span.set_attribute(MODEL_TTFT, elapsed)
                    emitted = True
                    if self._generation_answer_format == "text":
                        yield content
            output.record_finish_reason(finish_reason)
            if refused or finish_reason == "content_filter":
                raise output.model_refused("generation", "generate")
            if finish_reason == "length":
                raise ModelOutputTruncatedError(output.TRUNCATED_ANSWER_ERROR, stage="generate")
            streamed = (snapshot.choices[0].message.content or "") if snapshot is not None else ""
            if finish_reason != "stop" or not streamed.strip():
                raise ModelError(
                    "generation response was invalid", reason="response_invalid", stage="generate"
                )
            result = output._finish_answer(
                streamed,
                question.evidence_sources or hits,
                grounded,
                answer_policy=answer_policy,
                answer_format=self._generation_answer_format,
                source_metadata_key=self._generation_source_metadata_key,
            )
            stream_completed = True
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(
                "generation request failed",
                reason=protocol._provider_reason(error),
                stage="generate",
            ) from error
        finally:
            try:
                close = getattr(responses, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                if stream_completed:
                    raise ModelError(
                        "generation response cleanup failed",
                        reason=protocol._provider_reason(error),
                        stage="generate",
                    ) from error
            finally:
                if usage_response is not None:
                    telemetry._record_openai_usage(
                        usage_response,
                        input_modalities=modalities,
                        output_modalities=frozenset({Modality.TEXT}),
                        request_count=request_count,
                    )
        # Strict streams report the deltas, including a refusal marker. Best-effort results
        # carry the cleaned prose while preserving the marker's structured abstention status.
        if self._generation_answer_format == "structured":
            yield result.answer
            return result
        return result if answer_policy == "best_effort" else replace(result, answer=streamed)

    def _answer_with_evidence_operation(  # noqa: C901 - one first response and one bounded recovery
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        prepared: tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]],
        operation: tuple[dict[str, object], EvidenceCatalogue],
        *,
        answer_policy: AnswerPolicy,
        exhaustive: bool,
        stream: bool = False,
    ) -> AnswerResult:
        attempted = 0
        usages: list[telemetry._ModelUsage | None] = []
        finish_reasons: list[str] = []
        recovery_prepared = prepared
        may_recover = False
        submitted = False
        rejection: telemetry.EvidenceOperationRejection | None = None

        def attempt() -> None:
            nonlocal attempted
            attempted += 1
            mark_model_requests(attempted)

        def remember(response: object, modalities: frozenset[Modality]) -> None:
            telemetry._record_openai_provenance(response)
            choices = getattr(response, "choices", None)
            if isinstance(choices, list) and len(choices) == 1:
                reason = getattr(choices[0], "finish_reason", None)
                if isinstance(reason, str) and reason:
                    finish_reasons.append(reason)
            usages.append(
                telemetry._model_usage(
                    response,
                    input_modalities=modalities,
                    output_modalities=frozenset({Modality.TEXT}),
                )
            )

        def read(
            ready: tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]],
            *,
            first: bool = False,
        ) -> tuple[object, tuple[SearchHit, ...], dict[str, object]]:
            nonlocal may_recover, recovery_prepared
            may_recover = False
            response, grounded, modalities, _request_count, request = self._create_answer(
                question,
                hits,
                ready,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                stream=stream,
                on_request=attempt,
            )
            if first and "tools" not in request:
                recovery_prepared = request, grounded, modalities
            may_recover = True
            cleanup_error: Exception | None = None
            try:
                if stream:
                    response, cleanup_error = completion._collect_streamed_completion(
                        response,
                        subject="generation",
                        stage="generate",
                        on_failure=lambda value: remember(value, modalities),
                        allow_tool_calls=True,
                        require_usage=False,
                    )
            except ModelError as error:
                if (
                    not isinstance(error, output._InvalidStructuredResponse)
                    or error._cleanup_failed
                ):
                    may_recover = False
                raise
            except Exception as error:
                may_recover = False
                raise ModelError(
                    "generation request failed",
                    reason=protocol._provider_reason(error),
                    stage="generate",
                ) from error
            remember(response, modalities)
            if isinstance(cleanup_error, Exception):
                may_recover = False
                raise ModelError(
                    "generation response cleanup failed",
                    reason=protocol._provider_reason(cleanup_error),
                    stage="generate",
                ) from cleanup_error
            return response, grounded, request

        try:
            try:
                response, grounded, request = read(
                    (operation[0], prepared[1], prepared[2]), first=True
                )
                answer, arguments, rejection = output._evidence_operation_response(
                    response, tools_allowed="tools" in request, hits=operation[1].hits
                )
                if answer is not None:
                    return output._finish_answer(
                        answer,
                        question.evidence_sources or hits,
                        grounded,
                        answer_policy=answer_policy,
                        answer_format=self._generation_answer_format,
                        source_metadata_key=self._generation_source_metadata_key,
                    )
                submitted = True
                telemetry.evidence_operation_event("submitted")
                if rejection is not None:
                    raise InvalidEvidenceOperation("native evidence operation protocol was invalid")
                outcome = execute_evidence_operation(
                    cast(str, arguments), operation[1], question=question.text
                )
                telemetry.evidence_operation_event(outcome.status)
                if outcome.status == "executed":
                    return AnswerResult(answer=cast(str, outcome.answer), hits=outcome.hits)
                return AnswerResult(
                    answer=outcome.answer
                    if answer_policy == "best_effort" and outcome.answer
                    else prompts.UNKNOWN_ANSWER,
                    hits=outcome.hits,
                    abstained=True,
                    abstention_reason=outcome.reason or AbstentionReason.INSUFFICIENT_EVIDENCE,
                )
            except InvalidEvidenceOperation as error:
                telemetry.evidence_operation_event(
                    "rejected", rejection=rejection or error.category
                )
            except ModelError as error:
                if not may_recover or error.reason != "response_invalid":
                    raise
                if submitted:
                    telemetry.evidence_operation_event("rejected", rejection="response_invalid")
                else:
                    telemetry.evidence_operation_event("response_invalid")
            final = requests.evidence_operation_recovery(recovery_prepared[0])
            if not self._requests.evidence_operation_fits(final, stream=stream):
                raise ModelError(
                    "generation response was invalid", reason="response_invalid", stage="generate"
                )
            telemetry.evidence_operation_event("recovery")
            response, grounded, _request = read((final, recovery_prepared[1], recovery_prepared[2]))
            answer = output.evidence_operation_recovery_answer(response)
            return output._finish_answer(
                answer,
                question.evidence_sources or hits,
                grounded,
                answer_policy=answer_policy,
                answer_format=self._generation_answer_format,
                source_metadata_key=self._generation_source_metadata_key,
            )
        finally:
            telemetry._record_usage_batch(usages, request_count=attempted)
            if finish_reasons:
                trace.get_current_span().set_attribute(GEN_AI_FINISH_REASONS, tuple(finish_reasons))

    def _create_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        prepared: tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]],
        *,
        answer_policy: AnswerPolicy = "strict",
        stream: bool = False,
        exhaustive: bool = False,
        on_request: Callable[[], None] | None = None,
    ) -> tuple[object, tuple[SearchHit, ...], frozenset[Modality], int, dict[str, object]]:
        request, grounded, modalities = prepared
        create_completion = cast(
            Callable[..., object], self._client("generation").chat.completions.create
        )

        def create() -> object:
            if on_request is not None:
                on_request()
            if stream:
                return create_completion(
                    **request,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            return create_completion(**request)

        mark_model_requests(1)
        try:
            return create(), grounded, modalities, 1, request
        except ModelError:
            raise
        except Exception as error:
            fallback = self._media_rejection_fallback(
                question,
                hits,
                grounded,
                error,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                stream=stream,
            )
            if fallback is None:
                raise ModelError(
                    "generation request failed",
                    reason=protocol._provider_reason(error),
                    stage="generate",
                ) from error
            request, grounded, modalities = fallback
            mark_model_requests(2)
            try:
                return create(), grounded, modalities, 2, request
            except ModelError:
                raise
            except Exception as retry_error:
                raise ModelError(
                    "generation request failed",
                    reason=protocol._provider_reason(retry_error),
                    stage="generate",
                ) from retry_error

    def _media_rejection_fallback(
        self,
        question: ModelInput,
        retrieved: Sequence[SearchHit],
        grounded: Sequence[SearchHit],
        error: Exception,
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
        stream: bool = False,
    ) -> tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]] | None:
        rejected = protocol.rejected_media_modality(error)
        if rejected is None:
            return None
        # ponytail: the provider does not identify the rejected clip; keep every hit's text and
        # other media, then replace this with asset-specific fallback if APIs expose an ID.
        reduced = tuple(
            replace(
                hit,
                assets=assets,
                modality=ModelInput(text=hit.content, assets=assets).modality,
            )
            for hit in grounded
            if (assets := tuple(asset for asset in hit.assets if asset.modality is not rejected))
            or hit.content.strip()
        )
        reduced = retain_source_complete_hits(reduced, grounded)
        if reduced == tuple(grounded):
            return None
        fallback = self._requests.answer_request(
            question,
            reduced,
            omission_source_hits=question.evidence_sources or retrieved,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
            stream=stream,
            image_source_context=self._generation_image_source_prose == "context_when_supplied",
        )
        if isinstance(fallback, AbstentionReason):
            return None
        telemetry.record_grounding_fit(tuple(retrieved), fallback[1])
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(GROUNDING_MEDIA_FALLBACK, True)
        return fallback

    def grounding_assets(
        self, question: ModelInput, hits: Sequence[SearchHit]
    ) -> tuple[AssetRef, ...]:
        fitted = media._fit_grounding_media(
            question,
            hits,
            video_limit=self._generation_video_limit,
            sampled_parts={},
            sample_videos=Modality.IMAGE in self.generation_capabilities,
            image_max_pixels=self._image_max_pixels,
            capture_metadata={},
        )
        return (*question.assets, *(asset for hit in fitted for asset in hit.assets))

    def select_media(
        self, payload: Mapping[str, object], *, count: int, max_items: int
    ) -> tuple[int, ...]:
        chosen: tuple[int, ...] = ()
        text = protocol._json_text(payload)
        if (
            len(prompts._MEDIA_DIAGNOSTIC_PROMPT.encode("utf-8")) + len(text.encode("utf-8"))
            > requests._MAX_GROUNDED_TEXT_BYTES
        ):
            raise ModelError(
                "media diagnosis exceeds 4 MiB; lower the answer limit",
                reason="payload_too_large",
                stage="generate",
            )
        with nested_model_span(
            "sensemem.model.media_diagnosis",
            attributes={
                "sensemem.span.kind": "model",
                "sensemem.model.module": "generation",
                "sensemem.stage": "generate",
                "gen_ai.operation.name": "diagnose_media",
                "gen_ai.request.model": self._generation_model,
                "sensemem.input.modalities": ("text",),
            },
        ) as span:
            request = self._requests.json_request(
                prompts._MEDIA_DIAGNOSTIC_PROMPT, text, schema="media_selection"
            )
            request[output_limit_parameter(self._generation_model)] = min(
                self._generation_max_tokens or self._generation_media_max_tokens,
                self._generation_media_max_tokens,
            )
            response = None
            span.set_attribute("sensemem.grounding.media_diagnosis.truncated", False)
            mark_model_requests(1)
            try:
                response = cast(
                    Callable[..., object], self._client("generation").chat.completions.create
                )(**request)
                content = output._json_completion_text(
                    response, subject="media diagnosis", stage="generate"
                )
                selected = output._parse_media_requests(content, count, max_items)
                chosen = selected
                span.set_attribute("sensemem.grounding.media_diagnosis.valid", True)
            except Exception as error:
                # A failed diagnostic adds no media; the text-only answer can still proceed.
                span.set_attribute("sensemem.grounding.media_diagnosis.valid", False)
                span.set_attribute(
                    "sensemem.grounding.media_diagnosis.truncated",
                    isinstance(error, ModelOutputTruncatedError),
                )
                reason = (
                    error.reason
                    if isinstance(error, ModelError)
                    else protocol._provider_reason(error)
                )
                if reason is not None:
                    span.set_attribute("sensemem.grounding.media_diagnosis.error_reason", reason)
            finally:
                if response is not None:
                    telemetry._record_openai_usage(
                        response,
                        input_modalities=frozenset({Modality.TEXT}),
                        output_modalities=frozenset({Modality.TEXT}),
                    )
            span.set_attribute("sensemem.grounding.media_diagnosis.selected", len(chosen))
        return chosen
