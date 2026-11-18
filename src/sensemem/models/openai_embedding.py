"""OpenAI embedding adapter over one explicitly owned SDK client."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from openai import OpenAI

from sensemem._telemetry import (
    mark_model_requests,
)
from sensemem._validation import nonempty_text
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models import _openai_media as media
from sensemem.models import _openai_output as output
from sensemem.models import _openai_protocol as protocol
from sensemem.models import _openai_requests as requests
from sensemem.models import _openai_telemetry as telemetry
from sensemem.models._openai_client import OpenAIClient
from sensemem.models._openai_defaults import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL
from sensemem.models.base import (
    EmbedTask,
    ModelInput,
    _modalities,
)
from sensemem.prompts import embedding as embedding_prompts
from sensemem.types import Modality


class OpenAIEmbedder(OpenAIClient):
    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        owns_client: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_space: str | None = None,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        embedding_request_format: Literal["input", "messages"] = "input",
        embedding_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
        image_max_pixels: int | None = None,
    ) -> None:
        super().__init__(client, owns_client=owns_client)
        if image_max_pixels is not None and (
            isinstance(image_max_pixels, bool)
            or not isinstance(image_max_pixels, int)
            or not 1 <= image_max_pixels <= 64_000_000
        ):
            raise ValidationError("image_max_pixels must be between 1 and 64000000")
        self._image_max_pixels = image_max_pixels
        embedding_model = nonempty_text(embedding_model, "embedding_model")
        if (
            isinstance(embedding_dimension, bool)
            or not isinstance(embedding_dimension, int)
            or embedding_dimension <= 0
        ):
            raise ValidationError("embedding_dimension must be a positive integer")
        embedding_request_format = protocol._embedding_format(embedding_request_format)
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._embedding_request_format = embedding_request_format
        self._embeddinggemma2 = (
            embedding_model == "google/embeddinggemma-2" and embedding_request_format == "messages"
        )
        recipe = "messages-ordered-input-v2:" if embedding_request_format == "messages" else ""
        if self._embeddinggemma2:
            recipe = "embeddinggemma2-ordered-text-prompts-special-tokens-v3:"
        self._embedding_space = (
            f"{embedding_model}:{embedding_dimension}:{recipe}l2-v1"
            if embedding_space is None
            else nonempty_text(embedding_space, "embedding_space")
        )
        if image_max_pixels is not None:
            self._embedding_space += media._image_transport_recipe(image_max_pixels)
        self._embedding_capabilities = _modalities(embedding_capabilities, "embedding")

    @property
    def embedding_capabilities(self) -> frozenset[Modality]:
        return self._embedding_capabilities

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def embedding_space(self) -> str:
        return self._embedding_space

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode one batch with the configured OpenAI-compatible API shape.

        The generic OpenAI embeddings contract has no task field. EmbeddingGemma 2 in messages
        mode selects the checkpoint's retrieval prefix for text using ``task``.
        """
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(inputs, (str, bytes)):
            raise ValidationError("inputs must be a sequence of model inputs")
        try:
            selected_task = EmbedTask(task)
        except ValueError:
            raise ValidationError("embedding task is invalid") from None
        batch = tuple(inputs)
        if any(not isinstance(value, ModelInput) for value in batch):
            raise ValidationError("inputs must be a sequence of ModelInput values")
        if not batch:
            mark_model_requests(0, token_usage_expected=0)
            return ()
        for value in batch:
            protocol._require_capabilities(
                "embedding", value.modalities, self.embedding_capabilities
            )
        if self._embeddinggemma2:
            prefix = (
                embedding_prompts.EMBEDDINGGEMMA_QUERY_PREFIX
                if selected_task is EmbedTask.QUERY
                else embedding_prompts.EMBEDDINGGEMMA_DOCUMENT_PREFIX
            )
            # Prefix text only, matching the local checkpoint recipe; media alone stays bare.
            batch = tuple(
                replace(
                    value,
                    text=prefix + value.text if value.text else "",
                    parts=tuple(
                        replace(part, value=prefix + part.value)
                        if part.kind != "asset"
                        and all(earlier.kind == "asset" for earlier in value.parts[:position])
                        else part
                        for position, part in enumerate(value.parts)
                    ),
                )
                for value in batch
            )
        embedding_assets = tuple(asset for value in batch for asset in value.assets)
        media._require_consistent_assets(embedding_assets)
        image_parts = media._image_transport_parts(embedding_assets, self._image_max_pixels)
        sampled_assets = media._embedding_video_sampling(embedding_assets, image_parts=image_parts)

        client = self._client("embedding")
        request_count = 1
        mark_model_requests(request_count)
        try:
            sampled_videos: list[int] = []
            response = self._embedding_response(
                client,
                batch,
                sample_video=False,
                sampled=sampled_videos,
                sampled_assets=sampled_assets,
                image_parts=image_parts,
            )
            if sampled_assets:
                telemetry._record_video_sampling(len(sampled_videos))
        except ModelError as error:
            # A video long enough to overrun the model's context is refused outright, and one
            # refusal would otherwise lose the whole memory. Ordered stills keep the visual
            # evidence at a fraction of the tokens; they cannot keep the motion, so this is a
            # degradation and it is recorded as one. A second failure raises on its own, chained
            # to the refusal that prompted the retry.
            if (
                sampled_assets
                or not protocol._is_context_length_rejection(error.__cause__)
                or not media._has_video(batch)
            ):
                raise
            request_count = 2
            mark_model_requests(request_count)
            sampled: list[int] = []
            response = self._embedding_response(
                client, batch, sample_video=True, sampled=sampled, image_parts=image_parts
            )
            telemetry._record_video_sampling(len(sampled))
        telemetry._record_openai_usage(
            response,
            input_modalities=frozenset(
                modality for value in batch for modality in value.modalities
            ),
            output_modalities=frozenset(),
            request_count=request_count,
        )
        return output._embedding_vectors(response, len(batch), self.embedding_dimension)

    def _embedding_response(
        self,
        client: OpenAI,
        batch: Sequence[ModelInput],
        *,
        sample_video: bool,
        sampled: list[int] | None = None,
        sampled_assets: frozenset[str] = frozenset(),
        image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
    ) -> object:
        try:
            samples = (
                requests._embedding_samples(
                    batch,
                    sample_video=sample_video,
                    sampled=sampled,
                    sampled_assets=sampled_assets,
                    image_parts=image_parts,
                )
                if any(value.assets for value in batch)
                else []
            )
            media._require_inline_media_parts(
                tuple(
                    part
                    for sample in samples
                    for part in cast(list[dict[str, object]], sample[0]["content"])
                )
            )
            if self._embedding_request_format == "messages":
                from openai.types.create_embedding_response import CreateEmbeddingResponse

                if not samples:
                    samples = requests._embedding_samples(batch)
                body: dict[str, object] = {
                    "messages": samples[0] if len(samples) == 1 else samples,
                    "model": self.embedding_model,
                    # The declared dimension validates output; chat servers may reject it as
                    # an unsupported Matryoshka truncation request.
                    "encoding_format": "float",
                }
                if self._embeddinggemma2:
                    # Its chat template emits content only; the tokenizer must add BOS/EOS.
                    body["add_special_tokens"] = True
                return client.post(
                    "/embeddings",
                    cast_to=CreateEmbeddingResponse,
                    body=body,
                )
            create_embedding = cast(Callable[..., object], client.embeddings.create)
            return create_embedding(
                input=(
                    [value.text for value in batch]
                    if all(not value.assets for value in batch)
                    else samples
                ),
                model=self.embedding_model,
                dimensions=self.embedding_dimension,
                encoding_format="float",
            )
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(
                "embedding request failed",
                reason=protocol._provider_reason(error),
                stage="embed",
            ) from error
