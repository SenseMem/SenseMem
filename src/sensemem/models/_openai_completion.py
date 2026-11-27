"""Collect SDK chat deltas while retaining SenseMem's completion lifecycle checks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

from opentelemetry import trace

from sensemem._telemetry import GEN_AI_TTFC, MODEL_TTFT
from sensemem.models._openai_output import (
    _InvalidStructuredResponse,
    invalid_json_response,
    model_refused,
)
from sensemem.models._openai_telemetry import (
    _record_openai_provenance,
    record_provider_request_id,
)

if TYPE_CHECKING:
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion


def _collect_streamed_completion(  # noqa: C901 - one response lifecycle and admission boundary.
    responses: object,
    *,
    subject: str,
    stage: str,
    on_failure: Callable[[object], None] | None = None,
    allow_tool_calls: bool = False,
    require_usage: bool = True,
) -> tuple[ParsedChatCompletion[object], Exception | None]:
    """Use the SDK accumulator; keep completion, accounting and cleanup policy local."""
    from openai.lib.streaming.chat import ChatCompletionStreamState

    # Rich parsing may raise on a terminal chunk before the trailing usage arrives. The local
    # output boundary maps finish reasons after collection, so accumulate without a parser.
    state: ChatCompletionStreamState[object] = ChatCompletionStreamState()
    snapshot: ParsedChatCompletion[object] | None = None
    latest: ChatCompletionChunk | None = None
    usage: CompletionUsage | None = None
    model: str | None = None
    fingerprint: str | None = None
    finish_reason: str | None = None
    refused = False
    completed = False
    cleanup_error: Exception | None = None
    invalid: _InvalidStructuredResponse | None = None
    started = time.perf_counter()
    first_chunk = first_text = True

    def partial_response() -> object:
        if snapshot is None:
            return latest
        # The SDK overwrites usage/fingerprint with None on later chunks and retains the first
        # model. Preserve the latest reported accounting and provenance instead.
        snapshot.usage = usage
        if model is not None:
            snapshot.model = model
        snapshot.system_fingerprint = fingerprint
        return snapshot

    record_provider_request_id(responses)
    try:
        for response in cast("Iterable[ChatCompletionChunk]", responses):
            latest = response
            _record_openai_provenance(response)
            if allow_tool_calls and first_chunk:
                trace.get_current_span().set_attribute(GEN_AI_TTFC, time.perf_counter() - started)
                first_chunk = False
            if isinstance(response.model, str) and response.model:
                model = response.model
            if (
                isinstance(response_fingerprint := response.system_fingerprint, str)
                and response_fingerprint
            ):
                fingerprint = response_fingerprint
            if response.usage is not None:
                usage = response.usage
            choices = response.choices
            if not isinstance(choices, list):
                raise invalid_json_response(subject, stage)
            if choices:
                if len(choices) != 1 or choices[0].index != 0:
                    raise invalid_json_response(subject, stage)
                choice = choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                refused = refused or bool(delta.refusal)
                if delta.content is not None and not isinstance(delta.content, str):
                    raise invalid_json_response(subject, stage)
                if allow_tool_calls and delta.content and first_text:
                    trace.get_current_span().set_attribute(
                        MODEL_TTFT, time.perf_counter() - started
                    )
                    first_text = False
                if allow_tool_calls and delta.tool_calls is not None:
                    if not isinstance(delta.tool_calls, list):
                        raise invalid_json_response(subject, stage)
                    for fragment in delta.tool_calls:
                        if (
                            isinstance(fragment.index, bool)
                            or not isinstance(fragment.index, int)
                            or fragment.index != 0
                        ):
                            raise invalid_json_response(subject, stage)
                        if fragment.type is not None and fragment.type != "function":
                            raise invalid_json_response(subject, stage)
                        if fragment.function is not None and any(
                            value is not None and not isinstance(value, str)
                            for value in (fragment.function.name, fragment.function.arguments)
                        ):
                            raise invalid_json_response(subject, stage)
            try:
                state.handle_chunk(response)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                raise invalid_json_response(subject, stage) from None
            snapshot = state.current_completion_snapshot
        completed = True
    except Exception as error:
        if isinstance(error, _InvalidStructuredResponse):
            invalid = error
        if on_failure is not None:
            on_failure(partial_response())
        if refused or finish_reason == "content_filter":
            raise model_refused(subject, stage) from None
        raise
    finally:
        close = getattr(responses, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                if completed:
                    cleanup_error = error
                if invalid is not None:
                    invalid._cleanup_failed = True
    if (
        snapshot is None
        or (require_usage and usage is None and not refused and finish_reason != "content_filter")
        or (finish_reason is None and not refused)
    ):
        if on_failure is not None:
            on_failure(partial_response())
        invalid = invalid_json_response(subject, stage)
        invalid._cleanup_failed = cleanup_error is not None
        raise invalid
    partial_response()
    return snapshot, cleanup_error
