"""Neither original nor derived text grants authority to model-extracted claims."""

import json
from typing import Literal

import httpx2 as httpx
import pytest
from openai import OpenAI

from sensemem import ContentPart, EvidenceBasis, FormationInput, ModelInput, ObservationContext
from sensemem.models.openai_completion import OpenAICompletion

CLAIM = "Mina's current preferred drink is coffee."


@pytest.mark.parametrize(
    "derived_kind", ("audio_transcript", "visual_description", "model_interpretation")
)
@pytest.mark.parametrize("original", (False, True))
def test_source_part_role_does_not_authorize_a_model_subclaim(
    derived_kind: Literal["audio_transcript", "visual_description", "model_interpretation"],
    original: bool,
) -> None:
    raw = CLAIM if original else "This attachment is Mina's check-in."
    content = ModelInput(
        text=f"{raw}\n{CLAIM}",
        parts=(ContentPart("text", raw), ContentPart(derived_kind, CLAIM)),
    )

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "observation_id": "observation_0",
                                            "proposals": [
                                                {
                                                    "kind": "state",
                                                    "subject": "Mina",
                                                    "predicate": "preferred_drink",
                                                    "value": "coffee",
                                                    "content": CLAIM,
                                                    "confidence": 0.9,
                                                    "evidence_observation_ids": ["observation_0"],
                                                }
                                            ],
                                        }
                                    ]
                                }
                            )
                        },
                    }
                ],
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        former = OpenAICompletion(client, generation_model="fixture")
        (proposals,) = former.form(
            (
                FormationInput(
                    "source", content, ObservationContext(basis=EvidenceBasis.USER_STATEMENT)
                ),
            )
        )
    assert len(proposals) == 1
    assert proposals[0].basis is EvidenceBasis.MODEL_INFERENCE
