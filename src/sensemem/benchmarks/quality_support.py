"""Shared result values and deterministic embedding fixture for model quality evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sensemem import EmbedTask, Modality, ModelInput


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    repeat: int
    passed: bool
    expected_found: bool
    forbidden_found: bool
    abstained: bool
    latency_ms: float
    error_code: str | None = None
    error_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QualityReport:
    fixture_version: str
    configured_model: str
    response_models: tuple[str, ...]
    repeats: int
    cases: tuple[CaseResult, ...]
    evaluation_scope: str = "grounded_answering"
    structured_output_capability_evaluated: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(case.passed for case in self.cases)


class FixtureEmbedder:
    """Make the sole fixture memory relevant without an external embedding model."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "reader-quality-fixture"
    embedding_space = "reader-quality-fixture:4:l2-v1"
    embedding_dimension = 4

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass
