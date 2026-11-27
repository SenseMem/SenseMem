"""Model request controls, recipe identities and provider error mapping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from re import fullmatch
from typing import Literal

from sensemem._validation import nonempty_text
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models._openai_media import image_transport_recipe
from sensemem.models._openai_schemas import SchemaName, response_schema
from sensemem.models._openai_telemetry import _member, record_provider_request_id
from sensemem.prompts import generation as prompts
from sensemem.types import Modality


def reasoning_model(model: str) -> bool:
    if model.startswith("gpt-5-chat"):
        return False
    return (
        model == "gpt-5"
        or model.startswith(("gpt-5-", "gpt-5."))
        or any(model == family or model.startswith(f"{family}-") for family in ("o1", "o3", "o4"))
    )


def sampling_supported(model: str, extra_body: Mapping[str, object] | None) -> bool:
    if not reasoning_model(model):
        return True
    # The original GPT-5 models have no non-reasoning mode. Later families allow temperature
    # only with reasoning explicitly disabled; the provider still validates model availability.
    original = any(
        model == family or model.startswith(f"{family}-")
        for family in ("gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o4")
    )
    default = "none" if fullmatch(r"gpt-5\.[12](?:-\d{4}-\d{2}-\d{2})?", model) else None
    return not original and (extra_body or {}).get("reasoning_effort", default) == "none"


def validate_sampling(
    model: str,
    *,
    temperature: float | None,
    seed: int | None,
    extra_body: Mapping[str, object] | None,
) -> None:
    body = extra_body or {}
    if not sampling_supported(model, body) and any(
        value is not None
        for value in (
            temperature,
            body.get("temperature"),
            body.get("top_p"),
            body.get("logprobs"),
            body.get("top_logprobs"),
        )
    ):
        raise ValidationError(f"{model} does not support the configured sampling controls")
    if reasoning_model(model) and (seed is not None or body.get("seed") is not None):
        raise ValidationError(f"{model} does not support generation_seed")
    reserved = {
        "model",
        "messages",
        "response_format",
        "stream",
        "stream_options",
        "store",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "temperature",
        "seed",
    } & body.keys()
    if reserved:
        raise ValidationError("generation_extra_body must not override the model request contract")


def output_limit_parameter(model: str) -> str:
    return "max_completion_tokens" if reasoning_model(model) else "max_tokens"


VISION_SEED = 0


_QUOTA_ERRORS = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)


_SAFETY_ERRORS = frozenset({"content_policy_violation", "content_filter"})


def schema_recipe(name: SchemaName) -> str:
    recipe: dict[str, object] = {
        "schema": response_schema(name),
        "instruction": prompts.STRICT_JSON_INSTRUCTION,
    }
    if name == "formation":
        # The schema varies with each request; persist the strategy rather than one batch.
        recipe["batch_envelope"] = "exact-item-count-and-input-aliases-v1"
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return ":json-schema-v1:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _text_sequence(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{name} must be a sequence of non-empty strings")
    result: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ValidationError(f"{name} must be a sequence of non-empty strings")
        result.append(entry.strip())
    return tuple(result)


def transcription_context(
    prompt: object,
    keywords: object,
    languages: object,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    normalized_prompt = None if prompt is None else nonempty_text(prompt, "transcription_prompt")
    normalized_keywords = _text_sequence(keywords, "transcription_keywords")
    normalized_languages = _text_sequence(languages, "transcription_languages")
    if any(any(character in keyword for character in "<>\r\n") for keyword in normalized_keywords):
        raise ValidationError(
            "transcription_keywords must contain one-line text without '<' or '>'"
        )
    return normalized_prompt, normalized_keywords, normalized_languages


def default_transcription_space(
    model: str,
    *,
    prompt: str | None,
    keywords: Sequence[str],
    languages: Sequence[str],
) -> str:
    context: dict[str, object] = {}
    if prompt is not None:
        context["prompt"] = prompt
    if keywords:
        context["keywords"] = list(keywords)
    if languages:
        context["languages"] = list(languages)
    if not context:
        return f"{model}:asr-v1"
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"openai-transcription-v1:{payload}".encode()).hexdigest()[:16]
    return f"{model}:asr-v1:{digest}"


def default_vision_space(
    model: str,
    *,
    capabilities: frozenset[Modality],
    seed: int | None,
    temperature: float | None,
    max_tokens: int | None,
    extra_body: Mapping[str, object] | None,
    video_frames: int = 4,
    image_max_pixels: int | None = None,
) -> str:
    """Bind cached captions to their prompt, model controls, and media transport recipe."""
    payload = json.dumps(
        {
            "prompt": prompts._VISION_SYSTEM_PROMPT,
            "capabilities": sorted(modality.value for modality in capabilities),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
            "source_capture_metadata": "exif-gps-time-v1",
            **({"video_frames": video_frames} if video_frames != 4 else {}),
            **(
                {"image_transport": image_transport_recipe(image_max_pixels)}
                if image_max_pixels is not None
                else {}
            ),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"sensemem-vision-v2:{payload}".encode()).hexdigest()[:16]
    return f"{model}:sensemem-vision-v2:{digest}"


def default_reasoning_recipe(
    model: str,
    *,
    prompt: str,
    label: str,
    capabilities: frozenset[Modality],
    seed: int | None,
    temperature: float | None,
    max_tokens: int | None,
    extra_body: Mapping[str, object] | None,
    version: str = "v1",
) -> str:
    """Identify every control that changes what this reasoning call proposes.

    version is bumped by hand when a prompt edit is meant to be a new recipe. The digest below
    already changes with the prompt, so the bump is a marker for the reader, not the mechanism.
    """
    payload = json.dumps(
        {
            "prompt": prompt,
            "capabilities": sorted(modality.value for modality in capabilities),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    version = f"sensemem-{label}-{version}"
    digest = hashlib.sha256(f"{version}:{payload}".encode()).hexdigest()[:16]
    return f"{model}:{version}:{digest}"


def default_formation_space(
    model: str,
    *,
    capabilities: frozenset[Modality],
    seed: int | None,
    temperature: float | None,
    max_tokens: int | None,
    extra_body: Mapping[str, object] | None,
    image_max_pixels: int | None = None,
) -> str:
    recipe = default_reasoning_recipe(
        model,
        prompt=prompts._FORMATION_SYSTEM_PROMPT,
        label="formation",
        capabilities=capabilities,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return recipe if image_max_pixels is None else recipe + image_transport_recipe(image_max_pixels)


def embedding_format(value: object) -> Literal["input", "messages"]:
    if value == "input":
        return "input"
    if value == "messages":
        return "messages"
    raise ValidationError("embedding_request_format must be 'input' or 'messages'")


def _provider_reason(error: Exception) -> str | None:
    """Classify a failed request with the official OpenAI SDK's own exception taxonomy.

    An unrecognized failure stays unclassified rather than being guessed into a retryable reason.
    """
    record_provider_request_id(error)
    import httpx2
    import openai

    if isinstance(error, openai.RateLimitError):
        # The SDK raises this for every 429, exhausted billing included, and an agent that retries
        # exhausted billing never stops. APIError.code is the SDK's own parsed error body, so
        # the permanent case is separated by the provider's code rather than by its message.
        return "quota_exhausted" if error.code in _QUOTA_ERRORS else "rate_limited"
    if _is_safety_rejection(error):
        return "model_refused"
    if is_context_length_rejection(error):
        return "context_length_exceeded"
    if _is_generation_abort_rejection(error):
        return "provider_unavailable"
    # Proxies can reject before returning a JSON error body. The SDK represents HTTP 413
    # as APIStatusError rather than BadRequestError; status is authoritative in either form.
    if isinstance(error, openai.APIStatusError) and error.status_code == 413:
        return "payload_too_large"
    # APITimeoutError subclasses APIConnectionError, so the narrower class is checked first.
    for provider_error, reason in (
        (openai.AuthenticationError, "auth_failed"),
        (openai.APITimeoutError, "timeout"),
        (openai.APIConnectionError, "connection_failed"),
        (openai.BadRequestError, "request_rejected"),
        (openai.InternalServerError, "provider_unavailable"),
        # The SDK translates transport failures only while the request itself is in flight. A
        # dropped response body or stream raises a transport exception directly. Match its
        # timeout subclass first.
        (httpx2.TimeoutException, "timeout"),
        (httpx2.TransportError, "connection_failed"),
    ):
        if isinstance(error, provider_error):
            return reason
    return None


def _is_generation_abort_rejection(error: Exception) -> bool:
    """Classify this compatible gateway failure within the OpenAI adapter only."""
    import openai

    if not isinstance(error, openai.BadRequestError):
        return False
    message = _member(error.body, "message")
    return isinstance(message, str) and (
        "model output became abnormal" in message.casefold()
        and "generation was aborted" in message.casefold()
    )


def _is_safety_rejection(error: Exception) -> bool:
    import openai

    if not isinstance(error, openai.APIStatusError):
        return False
    if error.code not in (None, 400, "400"):
        return error.code in _SAFETY_ERRORS
    # Compatible gateways may return an inspection message without a semantic code.
    message = _member(error.body, "message")
    return (
        isinstance(error, openai.BadRequestError)
        and isinstance(message, str)
        and "inappropriate content" in message.casefold()
    )


def is_context_length_rejection(error: BaseException | None) -> bool:
    """Prefer the provider's context error code over compatible-server message fallback.

    Compatible endpoints with no semantic code declare a maximum model or context length;
    some return the HTTP status as their code. An explicit different semantic code must not be
    reinterpreted from incidental words in the message.
    """
    import openai

    if not isinstance(error, openai.BadRequestError):
        return False
    if error.code not in (None, 400, "400"):
        return error.code == "context_length_exceeded"
    message = _member(error.body, "message")
    normalized = message.casefold() if isinstance(message, str) else ""
    return "maximum" in normalized and (
        "model length" in normalized or "context length" in normalized
    )


def rejected_media_modality(error: Exception) -> Modality | None:
    """Allow technical duration fallback only; never retry a content-safety refusal."""
    import openai

    if (
        not isinstance(error, openai.BadRequestError)
        or error.code not in (None, 400, "400")
        or is_context_length_rejection(error)
        or _provider_reason(error) == "model_refused"
    ):
        return None
    message = _member(error.body, "message")
    normalized = message.casefold() if isinstance(message, str) else ""
    if "too short" not in normalized:
        return None
    for modality, word in (
        (Modality.VIDEO, "video"),
        (Modality.IMAGE, "image"),
        (Modality.AUDIO, "audio"),
    ):
        if word in normalized:
            return modality
    return None


def require_capabilities(
    operation: str,
    required: frozenset[Modality],
    supported: frozenset[Modality],
) -> None:
    missing = required - supported
    if missing:
        names = ", ".join(sorted(value.value for value in missing))
        raise ModelError(
            f"configured {operation} model does not support: {names}",
            reason="unsupported_modality",
        )


def _json_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError):
        raise ModelError("grounding evidence is not JSON-compatible") from None
