"""OpenAI transcription adapter over one explicitly owned SDK client."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from openai import OpenAI

from sensemem._telemetry import (
    mark_model_requests,
)
from sensemem._validation import nonempty_text
from sensemem.content import AssetRef
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models import _openai_media as media
from sensemem.models import _openai_protocol as protocol
from sensemem.models import _openai_requests as requests
from sensemem.models import _openai_telemetry as telemetry
from sensemem.models._concurrency import ordered_calls
from sensemem.models._media import (
    resolved_asset,
)
from sensemem.models._openai_client import OpenAIClient
from sensemem.models._openai_defaults import DEFAULT_TRANSCRIPTION_MODEL
from sensemem.models.base import (
    _modalities,
)
from sensemem.types import Modality


class OpenAITranscriber(OpenAIClient):
    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        owns_client: bool = False,
        transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
        transcription_space: str | None = None,
        transcription_prompt: str | None = None,
        transcription_keywords: Sequence[str] | None = None,
        transcription_languages: Sequence[str] | None = None,
        transcription_capabilities: frozenset[Modality] = frozenset(
            {Modality.AUDIO, Modality.VIDEO}
        ),
    ) -> None:
        super().__init__(client, owns_client=owns_client)
        transcription_model = nonempty_text(transcription_model, "transcription_model")
        transcription_prompt, transcription_keywords, transcription_languages = (
            protocol.transcription_context(
                transcription_prompt,
                transcription_keywords,
                transcription_languages,
            )
        )
        self._transcription_model = transcription_model
        self._transcription_prompt = transcription_prompt
        self._transcription_keywords = transcription_keywords
        self._transcription_languages = transcription_languages
        self._transcription_space = (
            protocol.default_transcription_space(
                transcription_model,
                prompt=transcription_prompt,
                keywords=transcription_keywords,
                languages=transcription_languages,
            )
            if transcription_space is None
            else nonempty_text(transcription_space, "transcription_space")
        )
        self._transcription_capabilities = _modalities(transcription_capabilities, "transcription")

    @property
    def transcription_capabilities(self) -> frozenset[Modality]:
        return self._transcription_capabilities

    @property
    def transcription_model(self) -> str:
        return self._transcription_model

    @property
    def transcription_space(self) -> str:
        return self._transcription_space

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        """Transcribe resolved audio/video assets in input order."""
        mark_model_requests(0, token_usage_expected=0)
        if isinstance(assets, (str, bytes)):
            raise ValidationError("assets must contain AssetRef values")
        batch = tuple(assets)
        if any(not isinstance(asset, AssetRef) for asset in batch):
            raise ValidationError("assets must contain AssetRef values")
        if not batch:
            mark_model_requests(0, token_usage_expected=0)
            return ()
        usages: list[telemetry._ModelUsage | None] = []
        attempted: list[None] = []

        def transcribe_one(asset: AssetRef) -> str:
            modality, media_type, path = resolved_asset(asset)
            protocol.require_capabilities(
                "transcription", frozenset({modality}), self.transcription_capabilities
            )
            create_transcription = cast(
                Callable[..., object],
                self._client("transcription").audio.transcriptions.create,
            )
            try:
                with media.transcription_file(asset, modality, media_type, path) as file:
                    if file is None:
                        # A video with no audio stream carries no speech to preserve, so it
                        # costs neither a request nor a usage slot.
                        return ""
                    attempted.append(None)
                    response = create_transcription(
                        **requests.transcription_request(
                            self.transcription_model,
                            file,
                            prompt=self._transcription_prompt,
                            keywords=self._transcription_keywords,
                            languages=self._transcription_languages,
                        )
                    )
            except ModelError:
                raise
            except Exception as error:
                raise ModelError(
                    "transcription request failed",
                    reason=protocol._provider_reason(error),
                    stage="transcribe",
                ) from error
            usages.append(
                telemetry._model_usage(
                    response,
                    input_modalities=frozenset({modality}),
                    output_modalities=frozenset({Modality.TEXT}),
                )
            )
            telemetry._record_openai_provenance(response)
            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise ModelError(
                    "transcription response was invalid",
                    reason="response_invalid",
                    stage="transcribe",
                ) from None
            return text.strip()

        try:
            return ordered_calls(tuple(partial(transcribe_one, asset) for asset in batch))
        finally:
            telemetry._record_usage_batch(usages, request_count=len(attempted))
