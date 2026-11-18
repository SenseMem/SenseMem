"""A single multimodal agent step over the caller-owned synchronous OpenAI SDK."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, cast

from sensemem._telemetry import mark_model_requests, nested_model_span
from sensemem._validation import nonempty_text, positive_int
from sensemem.agents import AgentStep, AgentTurn
from sensemem.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from sensemem.models import _openai_media as media
from sensemem.models import _openai_telemetry as telemetry
from sensemem.models._openai_protocol import _provider_reason, _require_capabilities
from sensemem.models.base import ModelInput, _modalities
from sensemem.prompts.agent import DEFAULT_AGENT_INSTRUCTIONS
from sensemem.tools import ToolCall, ToolDefinition
from sensemem.types import Modality

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion


class OpenAIAgent:
    """Translate neutral agent steps to Chat Completions without executing any tool.

    The SDK client owns authentication, endpoints, timeouts and retries, and remains caller-owned.
    Declare the endpoint's modalities and output-limit parameter explicitly. Strict tool schemas
    are opt-in because arbitrary Pydantic schemas can include defaults unsupported by strict mode.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        system_prompt: str = DEFAULT_AGENT_INSTRUCTIONS,
        capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
        max_tokens: int | None = None,
        max_tokens_parameter: Literal[
            "max_tokens", "max_completion_tokens"
        ] = "max_completion_tokens",
        strict_tools: bool = False,
        audio_data_url: bool = False,
    ) -> None:
        self._client = client
        self.agent_model = nonempty_text(model, "agent model")
        self._system_prompt = nonempty_text(system_prompt, "agent system prompt")
        self._capabilities = _modalities(capabilities, "agent")
        if not self._capabilities:
            raise ValidationError("agent capabilities must not be empty")
        self._max_tokens = None if max_tokens is None else positive_int(max_tokens, "max_tokens")
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValidationError("invalid agent output-limit parameter")
        self._max_tokens_parameter = max_tokens_parameter
        if not isinstance(strict_tools, bool) or not isinstance(audio_data_url, bool):
            raise ValidationError("agent protocol switches must be booleans")
        self._strict_tools = strict_tools
        self._audio_data_url = audio_data_url

    def step(
        self,
        input: ModelInput,
        *,
        tools: Sequence[ToolDefinition],
        history: Sequence[AgentTurn],
    ) -> AgentStep:
        _require_capabilities("agent", input.modalities, self._capabilities)
        media._require_inline_size(input.assets)
        cache: dict[str, str] = {}
        content = [
            {"type": "text", "text": item}
            if isinstance(item, str)
            else media._generation_asset_part(item, cache, audio_data_url=self._audio_data_url)
            for item in input.ordered_content
        ]
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]
        for turn in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.document()["arguments"],
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ),
                            },
                        }
                        for call in turn.step.calls
                    ],
                }
            )
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": json.dumps(result.document(), ensure_ascii=False, allow_nan=False),
                }
                for result in turn.results
            )
        request: dict[str, object] = {
            "model": self.agent_model,
            "messages": messages,
            "store": False,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.document()["input_schema"],
                        "strict": self._strict_tools,
                    },
                }
                for definition in tools
            ]
            request["tool_choice"] = "auto"
        if self._max_tokens is not None:
            request[self._max_tokens_parameter] = self._max_tokens
        with nested_model_span(
            "sensemem.model.agent",
            attributes={
                "sensemem.span.kind": "model",
                "sensemem.stage": "agent",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self.agent_model,
            },
        ):
            mark_model_requests(1)
            try:
                response = cast(
                    "Callable[..., ChatCompletion]", self._client.chat.completions.create
                )(**request)
            except Exception as error:
                raise ModelError(
                    "agent request failed", reason=_provider_reason(error), stage="agent"
                ) from None
            if response is None:
                raise ModelError(
                    "agent response was invalid", reason="response_invalid", stage="agent"
                )
            telemetry._record_openai_usage(
                response,
                input_modalities=input.modalities,
                output_modalities=frozenset({Modality.TEXT}),
            )
            return self._response(response)

    @staticmethod
    def _response(response: ChatCompletion) -> AgentStep:
        if len(response.choices) != 1 or response.choices[0].index != 0:
            raise ModelError("agent response was invalid", reason="response_invalid", stage="agent")
        choice = response.choices[0]
        message = choice.message
        if message.refusal or choice.finish_reason == "content_filter":
            raise ModelError("agent request was refused", reason="model_refused", stage="agent")
        if choice.finish_reason == "length":
            raise ModelOutputTruncatedError("agent response was truncated", stage="agent")
        try:
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                calls = []
                for call in message.tool_calls:
                    if call.type != "function":
                        raise ValueError("unsupported tool call type")
                    arguments: object = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    calls.append(ToolCall(call.id, call.function.name, arguments))
                return AgentStep(calls=tuple(calls))
            if choice.finish_reason == "stop" and not message.tool_calls:
                return AgentStep(content=message.content)
        except (ValueError, RecursionError):
            pass
        raise ModelError("agent response was invalid", reason="response_invalid", stage="agent")

    def close(self) -> None:
        """Leave the caller-owned SDK client open."""
