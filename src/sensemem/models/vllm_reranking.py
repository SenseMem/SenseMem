"""HTTP adapter for vLLM's text and multimodal rerank endpoint."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from pydantic import ValidationError as PydanticValidationError

from sensemem._telemetry import (
    mark_model_requests,
    record_model_provenance,
    record_model_request_id,
    record_model_usage,
)
from sensemem._validation import nonempty_text, optional_text, positive_int
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models._media import asset_url_part
from sensemem.models.base import ModelInput, _modalities
from sensemem.models.reranking import reranking_inputs
from sensemem.types import Modality

if TYPE_CHECKING:
    import httpx2

_CAPABILITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})


class _RerankResult(BaseModel):
    model_config = ConfigDict(strict=True)

    index: int = Field(ge=0)
    relevance_score: FiniteFloat


class _RerankUsage(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _RerankResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    results: list[_RerankResult]
    id: str | None = None
    model: str | None = None
    usage: _RerankUsage | None = None


@dataclass
class _RequestUsage:
    requests: int = 0
    reported: int = 0
    input_tokens: int = 0
    total_tokens: int = 0


def _server_base_url(value: str) -> str:
    value = nonempty_text(value, "base_url")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValidationError("base_url must be an absolute HTTP or HTTPS URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValidationError(
            "base_url must be an absolute HTTP or HTTPS URL without query or fragment"
        )
    return value.rstrip("/")


class VLLMReranker:
    """Score via ``<base_url>/rerank`` and restore input-order scores.

    Model loading, revision pinning, templates, context limits, and media preprocessing
    belong to the server. The owning Memory closes the HTTP client, including an injected one.
    Calls may run concurrently; closing waits for active calls and rejects new ones.
    Optional text/media fusion refines a bounded shortlist using probability logits.
    """

    def __init__(
        self,
        client: httpx2.Client | None = None,
        *,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        batch_size: int = 32,
        modalities: Iterable[Modality] = _CAPABILITIES,
        media_candidates: int | None = None,
        media_weight: float = 0.5,
    ) -> None:
        self._model_id = nonempty_text(model_id, "model_id")
        self._url = _server_base_url(base_url) + "/rerank"
        self._api_key = optional_text(api_key, "api_key")
        self._batch_size = positive_int(batch_size, "batch_size")
        self._capabilities = _modalities(modalities, "reranking")
        self._media_candidates = (
            None if media_candidates is None else positive_int(media_candidates, "media_candidates")
        )
        if (
            isinstance(media_weight, bool)
            or not isinstance(media_weight, (int, float))
            or not math.isfinite(media_weight)
            or not 0 < media_weight <= 1
        ):
            raise ValidationError("media_weight must be in (0, 1]")
        self._media_weight = float(media_weight)
        if not self._capabilities or self._capabilities - _CAPABILITIES:
            raise ValidationError("reranking modalities must contain only text, image, or video")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValidationError("timeout must be a positive finite number")
        self._timeout = float(timeout)
        if client is None:
            try:
                import httpx2
            except ImportError:
                raise ModelError(
                    "HTTP transport is unavailable; install SenseMem with the openai extra",
                    stage="reranking",
                ) from None
            client = httpx2.Client()
        self._client = client
        self._condition = threading.Condition()
        self._closed = False
        self._active_requests = 0

    @property
    def reranking_capabilities(self) -> frozenset[Modality]:
        return self._capabilities

    @property
    def reranking_model(self) -> str:
        return self._model_id

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        batch = reranking_inputs(query, documents, self._capabilities)
        with self._condition:
            if self._closed:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            self._active_requests += 1
        usage = _RequestUsage()
        try:
            if self._media_candidates is None:
                return self._score(query, batch, usage)
            text_query = ModelInput(
                text=query.text or "Retrieve candidates relevant to the query media."
            )
            scores = list(
                self._score(
                    text_query,
                    tuple(ModelInput(text=row.text or "Media evidence") for row in batch),
                    usage,
                )
            )
            eligible = sorted(
                (index for index, row in enumerate(batch) if row.assets or query.assets),
                key=lambda index: (-scores[index], index),
            )[: self._media_candidates]
            if eligible:
                from opentelemetry.trace import get_current_span

                get_current_span().set_attribute(
                    "sensemem.reranking.media_candidates", len(eligible)
                )
                refined = self._score(query, tuple(batch[index] for index in eligible), usage)
                for index, value in zip(eligible, refined, strict=True):
                    if not 0 <= scores[index] <= 1 or not 0 <= value <= 1:
                        raise ModelError(
                            "fusion requires probability scores",
                            reason="response_invalid",
                            stage="reranking",
                        )
                    base = min(max(scores[index], 1e-7), 1 - 1e-7)
                    media = min(max(value, 1e-7), 1 - 1e-7)
                    logit = (1 - self._media_weight) * math.log(
                        base / (1 - base)
                    ) + self._media_weight * math.log(media / (1 - media))
                    scores[index] = 1 / (1 + math.exp(-logit))
            return tuple(scores)
        finally:
            record_model_usage(
                input_tokens=usage.input_tokens if usage.reported else None,
                output_tokens=0,
                total_tokens=usage.total_tokens if usage.reported else None,
                request_count=usage.requests,
                expected_requests=usage.requests,
                reported_requests=usage.reported,
            )
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    def _score(
        self, query: ModelInput, batch: Sequence[ModelInput], usage: _RequestUsage
    ) -> tuple[float, ...]:
        import httpx2

        query_payload = _score_input(query, {})
        headers = {} if self._api_key is None else {"Authorization": f"Bearer {self._api_key}"}
        scores: list[float] = []
        for start in range(0, len(batch), self._batch_size):
            chunk = batch[start : start + self._batch_size]
            cache: dict[str, str] = {}
            payload = {
                "model": self._model_id,
                "query": query_payload,
                "documents": [_score_input(document, cache) for document in chunk],
                "top_n": len(chunk),
            }
            usage.requests += 1
            mark_model_requests(usage.requests)
            try:
                response = self._client.post(
                    self._url, json=payload, headers=headers, timeout=self._timeout
                )
            except httpx2.TimeoutException:
                raise ModelError(
                    "reranking request timed out", reason="timeout", stage="reranking"
                ) from None
            except httpx2.RequestError:
                raise ModelError(
                    "reranking connection failed", reason="connection_failed", stage="reranking"
                ) from None
            request_id = response.headers.get("x-request-id")
            if request_id:
                record_model_request_id(request_id)
            if not response.is_success:
                reason = {
                    401: "auth_failed",
                    403: "auth_failed",
                    408: "timeout",
                    413: "payload_too_large",
                    429: "rate_limited",
                }.get(response.status_code)
                if reason is None:
                    reason = (
                        "provider_unavailable"
                        if response.status_code >= 500
                        else "request_rejected"
                    )
                raise ModelError("reranking request failed", reason=reason, stage="reranking")
            scores.extend(_response_scores(response.content, len(chunk), usage))
        return tuple(scores)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.wait_for(lambda: self._active_requests == 0)
            if not self._client.is_closed:
                self._client.close()


def _score_input(value: ModelInput, cache: dict[str, str]) -> str | dict[str, object]:
    if not value.assets:
        return value.text
    parts: list[dict[str, object]] = (
        [] if not value.text else [{"type": "text", "text": value.text}]
    )
    try:
        parts.extend(asset_url_part(asset, cache) for asset in value.assets)
    except ModelError as error:
        error.stage = "reranking"
        raise
    return {"content": parts}


def _response_scores(content: bytes, count: int, usage: _RequestUsage) -> tuple[float, ...]:
    try:
        result = _RerankResponse.model_validate_json(content)
    except PydanticValidationError:
        raise ModelError(
            "reranking response was invalid", reason="response_invalid", stage="reranking"
        ) from None
    if len(result.results) != count or {row.index for row in result.results} != set(range(count)):
        raise ModelError(
            "reranking response indices were invalid", reason="response_invalid", stage="reranking"
        )
    if result.id:
        record_model_request_id(result.id)
    record_model_provenance(response_model=result.model, system_fingerprint=None)
    if result.usage is not None:
        usage.reported += 1
        usage.input_tokens += result.usage.prompt_tokens
        usage.total_tokens += result.usage.total_tokens
    return tuple(row.relevance_score for row in sorted(result.results, key=lambda row: row.index))
