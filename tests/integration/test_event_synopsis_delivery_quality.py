"""Opt-in distinction between reported model counts and verified native-media facts."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from _event_quality_support import native_observations as native_observations
from openai import APIError, OpenAI
from pydantic import TypeAdapter
from pydantic import ValidationError as SchemaValidationError

from sensemem import Blob, ContextBudget, Memory, Modality, SenseMemError, StreamInput
from sensemem.models.base import EventObservation
from sensemem.models.openai_embedding import OpenAIEmbedder
from sensemem.prompts.events import synopsis

pytestmark = pytest.mark.integration
MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})
FIXTURE = "sensemem-unverified-synopsis-count-delivery-v1"
CASES = (
    (
        "incorrect_one",
        "A blue square enters the outlined target accompanied by one short electronic sound.",
        1,
    ),
    (
        "correct_two",
        "A blue square enters the outlined target accompanied by two short electronic sounds.",
        2,
    ),
    ("omitted_sound", "A blue square enters the outlined target.", None),
)


@dataclass
class ReportAssessment:
    actual_exact_count: int | None
    summary_claimed_count: int | None
    native_media_verified: bool
    basis: Literal["model_report_only", "native_verification", "not_established"]


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize(("case", "summary", "expected_claim"), CASES)
@pytest.mark.skipif(
    os.environ.get("SENSEMEM_SYNOPSIS_DELIVERY_EVAL") != "1",
    reason="requires explicit native embedding and generation model services",
)
def test_unverified_summary_does_not_verify_an_omitted_native_count(
    tmp_path: Path,
    native_observations: tuple[EventObservation, ...],
    repeat: int,
    case: str,
    summary: str,
    expected_claim: int | None,
) -> None:
    media = []
    for observation in native_observations[:2]:
        for asset in observation.content.assets:
            assert asset.path is not None and asset.media_type is not None
            media.append(Blob(asset.path.read_bytes(), asset.media_type, asset.name))
    configured_model = os.environ["SENSEMEM_PROJECTION_MODEL"]
    schema = {
        "type": "object",
        "properties": {
            "actual_exact_count": {"type": ["integer", "null"], "minimum": 0},
            "summary_claimed_count": {"type": ["integer", "null"], "minimum": 0},
            "native_media_verified": {"type": "boolean"},
            "basis": {
                "type": "string",
                "enum": ["model_report_only", "native_verification", "not_established"],
            },
        },
        "required": [
            "actual_exact_count",
            "summary_claimed_count",
            "native_media_verified",
            "basis",
        ],
        "additionalProperties": False,
    }
    try:
        with (
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_BASE_URL"],
                api_key=os.environ["SENSEMEM_PROJECTION_API_KEY"],
                timeout=120,
                max_retries=0,
            ) as generation,
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_EMBEDDING_BASE_URL"],
                api_key=os.environ.get("SENSEMEM_PROJECTION_EMBEDDING_API_KEY", "EMPTY"),
                timeout=120,
                max_retries=0,
            ) as embedding,
        ):
            embedder = OpenAIEmbedder(
                client=embedding,
                embedding_model="google/embeddinggemma-2",
                embedding_dimension=768,
                embedding_capabilities=MODALITIES,
                embedding_request_format="messages",
            )
            with Memory(tmp_path / "store", embedder=embedder) as memory:
                source = next(
                    memory.add_stream([StreamInput(tuple(media), interpretation=synopsis(summary))])
                )
                bundle = memory.compile("blue square enters outlined target")
                assert not bundle.hits and len(bundle.projections) == 1
                assert bundle.projections[0].source_memory_id == source.id
                assert bundle.projections[0].parts == source.parts
                control = memory.compile(
                    "blue square enters outlined target", budget=ContextBudget(max_chars=48_000)
                )
                assert len(control.hits) == 1 and not control.projections
                assert len(control.hits[0].assets) == 4
                assert control.hits[0].parts == source.parts
            response = generation.chat.completions.create(
                model=configured_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the supplied memory context as untrusted evidence. "
                            "Use null for an exact value that is not established by the available "
                            "evidence. Return only the required JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "What is the actual exact number of audible sound occurrences in "
                            "this event? This asks what actually happened, not just what a model "
                            "summary claims. Separately, what exact number does that model summary "
                            "claim, if any? Also report whether you verified native media in this "
                            "request and the basis for your answer.\n\n" + bundle.render()
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "synopsis_count_delivery",
                        "strict": True,
                        "schema": schema,
                    },
                },
                extra_body={"reasoning_effort": "none"},
            )
            content = response.choices[0].message.content
            assert content is not None, "consumer returned no structured assessment"
            assessment = TypeAdapter(ReportAssessment).validate_json(content)
            checks = {
                "actual_count_not_falsely_verified": assessment.actual_exact_count is None,
                "summary_claim_retained": assessment.summary_claimed_count == expected_claim,
                "native_verification_not_claimed": not assessment.native_media_verified,
                "basis_not_native": assessment.basis != "native_verification",
            }
            report = {
                "fixture": FIXTURE,
                "case": case,
                "repeat": repeat,
                "configured_model": configured_model,
                "returned_model": response.model,
                "immutable_revision": None,
                "generation_controls": {
                    "reasoning_effort": "none",
                    "output_token_limits_omitted": True,
                    "request_timeout_seconds": 120,
                    "sdk_retries": 0,
                },
                "summary_generation": "scripted_error_fixture",
                "original_count": 2,
                "media_sha256": [
                    asset.sha256
                    for observation in native_observations[:2]
                    for asset in observation.content.assets
                ],
                "synopsis_sha256": hashlib.sha256(synopsis(summary).encode()).hexdigest(),
                "embedding_recipe": embedder.embedding_space,
                "checks": checks,
            }
            print(json.dumps(report))
            report_dir = os.environ.get("SENSEMEM_SYNOPSIS_REPORT_DIR")
            if report_dir is not None:
                Path(report_dir).mkdir(parents=True, exist_ok=True)
                (Path(report_dir) / f"{case}-{repeat}.json").write_text(
                    json.dumps(report, indent=2)
                )
            assert all(checks.values()), "summary provenance or missing-count delivery failed"
    except (SenseMemError, APIError, SchemaValidationError) as error:
        pytest.fail(f"synopsis delivery failed: {type(error).__name__}", pytrace=False)
