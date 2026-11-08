"""LLM-judge scoring with its response cache and transient-failure retries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import (
    Container,
    Mapping,
    Sequence,
)
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
    from openai.types.responses import Response, ResponseTextConfigParam
from sensemem import Modality
from sensemem._telemetry import (
    MODEL_MODULE,
    SPAN_KIND,
    mark_model_requests,
    model_span,
)
from sensemem.benchmarks.eval_adapters import LoadedTask
from sensemem.benchmarks.eval_arms import _Arm
from sensemem.benchmarks.eval_artifacts import _json_bytes
from sensemem.benchmarks.eval_cache import (
    CachedAnswer,
    ResponseCache,
)
from sensemem.benchmarks.eval_config import (
    _Arguments,
    _JudgeConfig,
)
from sensemem.benchmarks.eval_console import (
    _announce,
    _progress,
)
from sensemem.benchmarks.eval_results import (
    SampleResult,
    _retry_transient,
)
from sensemem.benchmarks.eval_telemetry import (
    BENCHMARK_ARM,
    BENCHMARK_JUDGE_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    JUDGE_PURPOSE,
)
from sensemem.benchmarks.official_scorers import (
    SCORER_VERSION,
    JudgeMessage,
    JudgePlan,
    combine_judge_scores,
    finalize_scores,
    judge_failure_scores,
    judge_model_is_official,
    judge_plan,
    parse_judge_response,
    sample_primary_metric,
)
from sensemem.models._openai_protocol import output_limit_parameter, sampling_supported
from sensemem.models.openai_sdk import (
    _model_usage,
    _record_openai_provenance,
    _record_usage_batch,
)

JUDGE_REQUEST_VERSION = "sensemem_judge_request_v2"


class _MosaicJudgeResponse(BaseModel):
    """Retain response model identity with cached MOSAIC verdicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    model: str | None

    def scores(self, plan: JudgePlan, requested_model: str) -> dict[str, float]:
        scores = dict(parse_judge_response(plan, self.text))
        if judge_model_is_official("mosaic", requested_model) and (
            self.model is None or not judge_model_is_official("mosaic", self.model)
        ):
            scores["judge_model_mismatch"] = 1.0
            scores["judge_score"] = 0.0
        return scores


def _atm_judge_text_format() -> ResponseTextConfigParam:
    """Require the official boolean verdict type without constraining its value."""
    return {
        "format": {
            "type": "json_schema",
            "name": "atm_judge",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "accuracy": {"type": "boolean"},
                },
                "required": ["explanation", "accuracy"],
                "additionalProperties": False,
            },
        }
    }


class _JudgeRequestRequired(TypedDict):
    store: bool
    model: str
    messages: list[ChatCompletionMessageParam]


class _JudgeRequest(_JudgeRequestRequired, total=False):
    temperature: float
    max_tokens: int
    max_completion_tokens: int
    extra_body: dict[str, object]
    reasoning_effort: Literal["medium"]


class _JudgeRequestError(Exception):
    """An exhausted SDK request failure, distinct from local scoring failures."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


async def _apply_judges(
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    *,
    arguments: _Arguments,
    config: _JudgeConfig,
    tracer: Tracer | None = None,
    already_judged: Container[str] = (),
) -> tuple[SampleResult, ...]:
    selected_tracer = trace.get_tracer("sensemem.benchmarks.eval") if tracer is None else tracer
    questions = {
        (task.spec.name, unit.unit_id, question.question_id): question
        for task in tasks
        for unit in task.units
        for question in unit.questions
    }
    planned: dict[str, JudgePlan] = {}
    for sample in samples:
        if (
            sample.error_code not in (None, "JudgeError")
            or sample.ingest_failure_count
            or not _Arm(sample.arm).generates
            # `--stream-results` already judged this one when its task finished. Re-judging would
            # spend the calls twice and let a non-deterministic judge overwrite a published score.
            or sample.sample_id in already_judged
        ):
            continue
        question = questions[(sample.task, sample.unit_id, sample.question_id)]
        plan = judge_plan(
            sample.task,
            question=question.source_question,
            references=question.references,
            prediction=sample.prediction,
            metadata=question.metadata,
        )
        if plan is not None:
            planned[sample.sample_id] = plan
    if not planned:
        return tuple(samples)
    if not arguments.quiet:
        _announce(f"judging {len(planned)} answers with {config.model}")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("official LLM scorers require sensemem[openai]") from None
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    cache = (
        None
        if arguments.use_cache is None
        else ResponseCache(
            arguments.use_cache,
            arguments.run_id,
            _judge_cache_namespace(config),
        )
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0

    async def judge(sample: SampleResult) -> SampleResult:
        nonlocal completed
        plan = planned.get(sample.sample_id)
        result = await _judge_sample(
            sample,
            plan,
            client=client,
            cache=cache,
            semaphore=semaphore,
            config=config,
            log_samples=arguments.log_samples,
            tracer=selected_tracer,
        )
        if plan is not None:
            completed += 1
            progress(completed, len(planned))
        return result

    try:
        with _progress(
            f"judging with {config.model}",
            "answer",
            total=len(planned),
            enabled=not arguments.quiet,
        ) as progress:
            judged = await asyncio.gather(*(judge(sample) for sample in samples))
    finally:
        await client.close()
        if cache is not None:
            cache.close()
    return tuple(judged)


async def _judge_sample(
    sample: SampleResult,
    plan: JudgePlan | None,
    *,
    client: AsyncOpenAI,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    log_samples: bool,
    tracer: Tracer,
) -> SampleResult:
    if plan is None:
        return sample
    try:
        outcomes = tuple(
            [
                await _traced_judge_call(
                    client,
                    messages,
                    sample=sample,
                    plan=plan,
                    call_index=index,
                    cache=cache,
                    semaphore=semaphore,
                    config=config,
                    tracer=tracer,
                )
                for index, messages in enumerate(plan.calls)
            ]
        )
        scores = combine_judge_scores(plan, tuple(item[0] for item in outcomes))
        previous = dict(sample.metrics)
        if plan.parser == "mosaic":
            previous.pop("judge_parse_error", None)
            previous.pop("judge_model_mismatch", None)
        metrics = finalize_scores(sample.task, {**previous, **scores})
        parse_failed = bool(scores.get("judge_parse_error"))
        model_mismatch = bool(scores.get("judge_model_mismatch"))
        responses = tuple(item[1] for item in outcomes)
        return replace(
            sample,
            score=metrics.get(sample_primary_metric(sample.task)),
            exact_match=metrics.get("exact_match"),
            metrics=metrics,
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
            judge_response=(json.dumps(responses, ensure_ascii=False) if log_samples else None),
            judge_cached=all(item[2] for item in outcomes),
            error_code=(
                "JudgeError"
                if parse_failed or model_mismatch
                else None
                if sample.error_code == "JudgeError"
                else sample.error_code
            ),
            scorer_error=(
                "MOSAIC judge response could not be parsed"
                if parse_failed
                else "MOSAIC judge response model does not match the official judge"
                if model_mismatch
                else None
            ),
        )
    except _JudgeRequestError as failure:
        error = failure.error
        message = " ".join(str(error).split())[:500]
        # An unavailable verdict scores zero only for metrics that depend on the judge.
        metrics = judge_failure_scores(sample.task, sample.metrics)
        return replace(
            sample,
            score=metrics.get(sample_primary_metric(sample.task)),
            metrics=metrics,
            error_code=sample.error_code or "JudgeError",
            scorer_error=f"{type(error).__name__}: {message}",
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
        )


async def _traced_judge_call(
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    tracer: Tracer,
) -> tuple[Mapping[str, float], str, bool]:
    cached = _cached_judge_call(
        cache, messages, sample=sample, plan=plan, call_index=call_index, config=config
    )
    if cached is not None:
        return cached
    with model_span(
        tracer,
        BENCHMARK_JUDGE_SPAN,
        attributes={
            BENCHMARK_TASK: sample.task,
            BENCHMARK_SAMPLE: sample.sample_id,
            BENCHMARK_ARM: sample.arm,
            BENCHMARK_PURPOSE: JUDGE_PURPOSE,
            SPAN_KIND: "model",
            MODEL_MODULE: "judge",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": config.model,
        },
    ):
        return await _judge_call(
            client,
            messages,
            sample=sample,
            plan=plan,
            call_index=call_index,
            cache=cache,
            semaphore=semaphore,
            config=config,
            read_cache=False,
        )


async def _judge_call(  # noqa: C901 - retry and provider fallback belong in one request path
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    read_cache: bool = True,
) -> tuple[Mapping[str, float], str, bool]:
    from httpx2 import TransportError
    from openai import APIError

    request_errors = (APIError, TransportError)
    cache_task, key = _judge_cache_coordinates(
        messages,
        plan=plan,
        call_index=call_index,
        config=config,
    )
    if read_cache:
        cached = _cached_judge_call(
            cache,
            messages,
            sample=sample,
            plan=plan,
            call_index=call_index,
            config=config,
        )
        if cached is not None:
            return cached
    request: _JudgeRequest = {
        "model": config.model,
        "store": False,
        "messages": [
            {"role": "system", "content": message.content}
            if message.role == "system"
            else {"role": "user", "content": message.content}
            for message in messages
        ],
    }
    if plan.max_tokens is not None:
        if output_limit_parameter(config.model) == "max_completion_tokens":
            request["max_completion_tokens"] = plan.max_tokens
        else:
            request["max_tokens"] = plan.max_tokens
    extra_body = dict(plan.extra_body or {})
    model_key = re.sub(r"[^a-z0-9]+", "", config.model.casefold())
    if (
        "qwen3" in model_key
        and plan.parser != "ama_bench"
        and not (
            "enable_thinking" in extra_body and judge_model_is_official(sample.task, config.model)
        )
    ):
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    if extra_body:
        request["extra_body"] = extra_body
    if sampling_supported(config.model, extra_body):
        request["temperature"] = 0.0
    if plan.parser == "longmemeval_v2" and judge_model_is_official(sample.task, config.model):
        request["reasoning_effort"] = "medium"
        request.pop("temperature", None)
    use_responses_api = plan.parser == "atm" and judge_model_is_official(sample.task, config.model)
    usages = []
    attempted = 0

    async def call() -> ChatCompletion | Response:
        nonlocal attempted
        async with semaphore:
            attempted += 1
            mark_model_requests(attempted)
            try:
                if use_responses_api:
                    return await client.responses.create(
                        model=config.model,
                        store=False,
                        input="\n".join(message.content for message in messages),
                        max_output_tokens=plan.max_tokens,
                        reasoning={"effort": "minimal"},
                        text=_atm_judge_text_format(),
                    )
                completion: ChatCompletion = await client.chat.completions.create(**request)
                return completion
            except request_errors as error:
                raise _JudgeRequestError(error) from error

    try:
        while True:
            try:
                response = await _retry_transient(call)
                break
            except _JudgeRequestError as failure:
                # MOSAIC's no-thinking and temperature controls are binding protocol settings.
                if plan.parser == "mosaic":
                    raise
                if "temperature" in request and _unsupported_temperature(failure.error):
                    request.pop("temperature")
                    continue
                if "extra_body" in request and _unsupported_extra_body(failure.error):
                    request.pop("extra_body")
                    continue
                raise
        if use_responses_api:
            text = str(cast("Response", response).output_text or "").strip()
        else:
            completion = cast("ChatCompletion", response)
            if not completion.choices:
                raise _JudgeRequestError(ValueError("judge response has no completion choices"))
            text = str(completion.choices[0].message.content or "").strip()
        _record_openai_provenance(response)
        usages.append(
            _model_usage(
                response,
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
            )
        )
        scores: Mapping[str, float]
        if plan.parser == "mosaic":
            verdict = _MosaicJudgeResponse(text=text, model=response.model)
            scores = verdict.scores(plan, config.model)
            cache_text = verdict.model_dump_json()
        else:
            scores = parse_judge_response(plan, text)
            cache_text = text
        if cache is not None and not (
            scores.get("judge_parse_error") or scores.get("judge_model_mismatch")
        ):
            cache.put(cache_task, sample.unit_id, key, CachedAnswer(cache_text, 0.0, ()))
        return scores, text, False
    finally:
        _record_usage_batch(usages, request_count=attempted)


def _judge_cache_coordinates(
    messages: Sequence[JudgeMessage],
    *,
    plan: JudgePlan,
    call_index: int,
    config: _JudgeConfig,
) -> tuple[str, str]:
    key = hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "request_version": JUDGE_REQUEST_VERSION,
                "model": config.model,
                "protocol": plan.protocol,
                "max_tokens": plan.max_tokens,
                "extra_body": plan.extra_body,
                "response_model_provenance": plan.parser == "mosaic",
                "call": call_index,
                "messages": tuple((message.role, message.content) for message in messages),
            }
        )
    ).hexdigest()
    return f"judge:{plan.protocol}:{config.model}", key


def _cached_judge_call(
    cache: ResponseCache | None,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    config: _JudgeConfig,
) -> tuple[Mapping[str, float], str, bool] | None:
    if cache is None:
        return None
    cache_task, key = _judge_cache_coordinates(
        messages,
        plan=plan,
        call_index=call_index,
        config=config,
    )
    cached = cache.get(cache_task, sample.unit_id, key)
    if cached is None:
        return None
    scores: Mapping[str, float]
    if plan.parser == "mosaic":
        try:
            verdict = _MosaicJudgeResponse.model_validate_json(cached.prediction)
        except ValidationError:
            return None
        scores = verdict.scores(plan, config.model)
        text = verdict.text
    else:
        scores = parse_judge_response(plan, cached.prediction)
        text = cached.prediction
    if scores.get("judge_parse_error") or scores.get("judge_model_mismatch"):
        return None
    return scores, text, True


def _unsupported_extra_body(error: BaseException) -> bool:
    message = str(error).casefold()
    field = "enable_thinking" in message or "chat_template_kwargs" in message
    marker = any(
        value in message
        for value in (
            "unknown",
            "unsupported",
            "unexpected",
            "unrecognized",
            "not allowed",
            "invalid parameter",
        )
    )
    return field and marker


def _unsupported_temperature(error: BaseException) -> bool:
    if getattr(error, "status_code", None) not in (400, 422):
        return False
    message = str(error).casefold()
    return bool(re.search(r"\btemperature\b", message)) and any(
        marker in message
        for marker in ("unsupported", "does not support", "not supported", "not allowed")
    )


def _judge_cache_namespace(config: _JudgeConfig) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "model": config.model,
                "base_url": config.base_url,
                "temperature": 0.0 if sampling_supported(config.model, None) else None,
            }
        )
    ).hexdigest()
