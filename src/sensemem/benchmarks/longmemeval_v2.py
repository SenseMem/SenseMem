"""Typed readers for the official LongMemEval-V2 questions and trajectory haystacks."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from sensemem.benchmarks._contracts import ContractModel, Identifier
from sensemem.benchmarks._official.longmemeval_v2_scoring import parse_eval_spec

LONGMEMEVAL_V2_ADAPTER_VERSION = "longmemeval_v2_official_v1"
Domain = Literal["web", "enterprise"]
_Text = Annotated[str, Field(min_length=1)]
_Index = Annotated[int, Field(ge=0, strict=True)]


class LongMemEvalV2Question(ContractModel):
    id: Identifier
    domain: Domain
    environment: Identifier
    question_type: Literal[
        "static-environment",
        "static-environment-abs",
        "dynamic-environment",
        "dynamic-environment-abs",
        "procedure",
        "procedure-abs",
        "errors-gotchas",
    ]
    question: _Text
    image: str | None
    answer: _Text
    eval_function: str

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str | None) -> str | None:
        if value is not None:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[:1] != ("question_screenshots",)
            ):
                raise ValueError(
                    "LongMemEval-V2 question image must be under question_screenshots/"
                )
        return value

    @model_validator(mode="after")
    def validate_evaluator(self) -> LongMemEvalV2Question:
        rule = parse_eval_spec(self.eval_function)
        if (rule.name == "llm_abstention_checker") != self.question_type.endswith("-abs"):
            raise ValueError("LongMemEval-V2 abstention type and evaluator must agree")
        return self


class LongMemEvalV2State(ContractModel):
    state_index: _Index
    step: _Index | None
    url: str
    action: str | None
    thought: str | None
    accessibility_tree: str
    screenshot: str | None


class LongMemEvalV2Trajectory(ContractModel):
    id: Identifier
    domain: Domain
    environment: Identifier
    goal: _Text
    outcome: Literal["success", "failure"]
    start_url: str
    states: tuple[LongMemEvalV2State, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_states(self) -> LongMemEvalV2Trajectory:
        for index, state in enumerate(self.states):
            if state.state_index != index:
                raise ValueError("LongMemEval-V2 state indices must follow source order")
            if state.screenshot is not None:
                path = PurePosixPath(state.screenshot)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:2] != ("screenshots", self.id)
                ):
                    raise ValueError(
                        "LongMemEval-V2 screenshot must be under screenshots/<trajectory_id>/"
                    )
        return self


def load_longmemeval_v2_questions(path: Path) -> tuple[LongMemEvalV2Question, ...]:
    with path.open(encoding="utf-8") as stream:
        questions = tuple(
            LongMemEvalV2Question.model_validate_json(line) for line in stream if line.strip()
        )
    if not questions or len({question.id for question in questions}) != len(questions):
        raise ValueError("LongMemEval-V2 question IDs must be non-empty and unique")
    return questions


def load_longmemeval_v2_haystacks(path: Path) -> dict[str, tuple[str, ...]]:
    haystacks = TypeAdapter(dict[Identifier, tuple[Identifier, ...]]).validate_json(
        path.read_bytes()
    )
    if not haystacks or any(not ids or len(set(ids)) != len(ids) for ids in haystacks.values()):
        raise ValueError("LongMemEval-V2 haystacks must contain non-empty, unique trajectory IDs")
    return haystacks


def load_longmemeval_v2_trajectories(
    path: Path, selected_ids: Collection[str]
) -> dict[str, LongMemEvalV2Trajectory]:
    """Stream the 1.2 GB JSONL and retain only the selected haystacks."""
    selected = set(selected_ids)
    trajectories: dict[str, LongMemEvalV2Trajectory] = {}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            trajectory = LongMemEvalV2Trajectory.model_validate_json(line)
            if trajectory.id in seen:
                raise ValueError("LongMemEval-V2 trajectories contain duplicate IDs")
            seen.add(trajectory.id)
            if trajectory.id in selected:
                trajectories[trajectory.id] = trajectory
    if selected - trajectories.keys():
        raise ValueError("LongMemEval-V2 haystack references unknown trajectory IDs")
    return trajectories


def question_prompt(question: LongMemEvalV2Question) -> str:
    """Apply the upstream domain instruction using the SDK's query text surface."""
    environment = (
        "a web browsing environment that has a customized magento-based shopping website, "
        "a customized magento-based shopping admin cms website, as well as a customized forum "
        "website based on reddit/postmill"
        if question.domain == "web"
        else "a customized ServiceNow environment"
    )
    return (
        f"You are an experienced colleague in {environment}. "
        "Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed.\n\n"
        f"{question.question}"
    )
