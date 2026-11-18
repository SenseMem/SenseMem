"""The optional reader style is explicit on configuration and SDK surfaces."""

from typing import Literal, cast

import pytest
from pydantic import ValidationError as PydanticValidationError

from sensemem import MemorySpec, ValidationError
from sensemem.configuration import OpenAIGenerationConfig
from sensemem.models.openai_completion import OpenAICompletion


@pytest.mark.parametrize("style", ["brief", "", True, [], None])
def test_answer_style_rejects_unsupported_values_before_inference(style: object) -> None:
    with pytest.raises(ValidationError, match="generation_answer_style"):
        OpenAICompletion(generation_answer_style=cast(Literal["default", "concise"], style))
    with pytest.raises(PydanticValidationError):
        MemorySpec.model_validate(
            {
                "embedding": {"provider": "openai"},
                "generation": {"provider": "openai", "answer_style": style},
            }
        )


def test_answer_style_defaults_to_the_existing_reader_and_is_answer_only() -> None:
    configured = MemorySpec.model_validate(
        {"embedding": {"provider": "openai"}, "generation": {"provider": "openai"}}
    )
    assert isinstance(configured.generation, OpenAIGenerationConfig)
    assert configured.generation.answer_style == "default"
    for slot in ("vision", "formation", "consolidation"):
        with pytest.raises(PydanticValidationError):
            MemorySpec.model_validate(
                {
                    "embedding": {"provider": "openai"},
                    slot: {"provider": "openai", "answer_style": "concise"},
                }
            )
