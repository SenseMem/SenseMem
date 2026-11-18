"""Immutable controls shared by one evaluation task and its question workers."""

from __future__ import annotations

from dataclasses import dataclass

from sensemem import AnswerPolicy, ContextBudget
from sensemem.benchmarks.eval_config import (
    DEFAULT_COMPILE_BUDGET,
    DEFAULT_FULL_CONTEXT_CHARS,
    DEFAULT_INGEST_MODE,
    INGEST_MODES,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationOptions:
    """Evaluation values only; stores, clients, caches and observers remain explicit inputs."""

    batch_size: int = 32
    unit_concurrency: int = 1
    request_concurrency: int = 1
    recall_limit: int = 10
    predict_only: bool = False
    log_samples: bool = False
    full_context_chars: int = DEFAULT_FULL_CONTEXT_CHARS
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET
    answer_policy: AnswerPolicy | None = None
    ingest_mode: str = DEFAULT_INGEST_MODE
    deliberate: bool = False
    ingest_concurrency: int = 1
    stream_answers: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.batch_size,
                self.unit_concurrency,
                self.request_concurrency,
                self.recall_limit,
                self.ingest_concurrency,
            )
            <= 0
        ):
            raise ValueError("batch size, concurrency, and recall limit must be positive")
        if self.recall_limit > 100:
            raise ValueError("recall limit must not exceed 100")
        if self.ingest_mode not in INGEST_MODES:
            raise ValueError(f"unknown ingest mode {self.ingest_mode!r}")
        if self.ingest_concurrency > 1 and (self.ingest_mode != "add" or self.deliberate):
            raise ValueError("concurrent ingest requires add without deliberation")
