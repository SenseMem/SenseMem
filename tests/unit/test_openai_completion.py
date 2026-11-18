"""SDK accumulation retains the adapter's terminal, accounting and cleanup contracts."""

from collections.abc import Iterator

import pytest
from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion

from sensemem import ModelError, ModelOutputTruncatedError
from sensemem.models._openai_completion import _collect_streamed_completion
from sensemem.models._openai_output import _InvalidStructuredResponse, _json_completion_text


def _chunk(payload: dict[str, object]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "served-model",
            "choices": [],
        }
        | payload
    )


def test_sdk_accumulates_content_and_fragmented_function_arguments() -> None:
    chunks = (
        _chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Provisional ",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-fixture",
                                    "type": "function",
                                    "function": {"name": "calculate", "arguments": '{"value":'},
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "answer.",
                            "tool_calls": [{"index": 0, "function": {"arguments": "2}"}}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }
        ),
    )

    completion, cleanup = _collect_streamed_completion(
        chunks, subject="generation", stage="generate", allow_tool_calls=True
    )

    assert isinstance(completion, ParsedChatCompletion)
    assert cleanup is None
    (choice,) = completion.choices
    assert choice.message.content == "Provisional answer."
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    (call,) = choice.message.tool_calls
    assert call.type == "function"
    assert call.function.name == "calculate"
    assert call.function.arguments == '{"value":2}'


def test_accounting_and_provenance_survive_an_empty_trailing_chunk() -> None:
    completion, cleanup = _collect_streamed_completion(
        (
            _chunk(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "{}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "initial-model",
                }
            ),
            _chunk(
                {
                    "model": "final-model",
                    "system_fingerprint": "final-fingerprint",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                }
            ),
            _chunk({"model": "final-model"}),
        ),
        subject="planner",
        stage="plan",
    )

    assert cleanup is None
    assert completion.model == "final-model"
    assert completion.system_fingerprint == "final-fingerprint"
    assert completion.usage is not None and completion.usage.total_tokens == 7


@pytest.mark.parametrize("finish", ("length", "content_filter"))
def test_output_boundary_keeps_terminal_failures_distinct_after_usage_arrives(finish: str) -> None:
    completion, _cleanup = _collect_streamed_completion(
        (
            _chunk(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "partial"},
                            "finish_reason": finish,
                        }
                    ]
                }
            ),
            _chunk({"usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}),
        ),
        subject="planner",
        stage="plan",
    )

    assert completion.usage is not None and completion.usage.total_tokens == 7
    with pytest.raises(ModelError) as failure:
        _json_completion_text(completion, subject="planner", stage="plan")
    assert failure.value.reason == ("output_truncated" if finish == "length" else "model_refused")
    assert isinstance(failure.value, ModelOutputTruncatedError) == (finish == "length")


@pytest.mark.parametrize("ending", ("iterator_failure", "cleanup_failure"))
def test_partial_usage_and_refusal_survive_stream_failures(ending: str) -> None:
    observed: list[object] = []

    class Stream:
        close_calls = 0

        def __iter__(self) -> Iterator[ChatCompletionChunk]:
            yield _chunk(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "refusal": "private refusal"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                }
            )
            if ending == "iterator_failure":
                raise RuntimeError("private iterator failure")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("private cleanup failure")

    stream = Stream()
    with pytest.raises(ModelError) as failure:
        if ending == "iterator_failure":
            _collect_streamed_completion(
                stream, subject="planner", stage="plan", on_failure=observed.append
            )
        else:
            completion, cleanup = _collect_streamed_completion(
                stream, subject="planner", stage="plan", on_failure=observed.append
            )
            assert cleanup is not None
            observed.append(completion)
            _json_completion_text(completion, subject="planner", stage="plan")

    assert failure.value.reason == "model_refused"
    assert "private" not in str(failure.value)
    assert stream.close_calls == 1
    (partial,) = observed
    assert isinstance(partial, ParsedChatCompletion)
    assert partial.usage is not None and partial.usage.total_tokens == 7


@pytest.mark.parametrize("missing", ("finish", "usage"))
def test_incomplete_stream_cannot_become_a_successful_completion(missing: str) -> None:
    observed: list[object] = []
    chunk = _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "{}"},
                    "finish_reason": None if missing == "finish" else "stop",
                }
            ],
            "usage": (
                None
                if missing == "usage"
                else {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
            ),
        }
    )

    with pytest.raises(_InvalidStructuredResponse) as failure:
        _collect_streamed_completion(
            (chunk,), subject="planner", stage="plan", on_failure=observed.append
        )

    assert failure.value.reason == "response_invalid"
    assert len(observed) == 1 and isinstance(observed[0], ParsedChatCompletion)


def test_failure_before_first_chunk_does_not_fabricate_a_completion() -> None:
    observed: list[object] = []
    original = RuntimeError("iterator failed before a response")

    class Stream:
        close_calls = 0

        def __iter__(self) -> Iterator[ChatCompletionChunk]:
            raise original

        def close(self) -> None:
            self.close_calls += 1

    stream = Stream()
    with pytest.raises(RuntimeError) as failure:
        _collect_streamed_completion(
            stream, subject="planner", stage="plan", on_failure=observed.append
        )

    assert failure.value is original
    assert observed == [None]
    assert stream.close_calls == 1
