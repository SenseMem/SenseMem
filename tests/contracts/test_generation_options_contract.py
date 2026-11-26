"""Generation-only reader style and image-source display policy contracts."""

from typing import Literal, cast

import pytest
from pydantic import ValidationError as SchemaError

from sensemem import MemorySpec, ValidationError
from sensemem.configuration import OpenAIGenerationConfig
from sensemem.models.openai_completion import OpenAICompletion


@pytest.mark.parametrize("style", ["brief", "", True, [], None])
def test_answer_style_rejects_unsupported_values_before_inference(style: object) -> None:
    with pytest.raises(ValidationError, match="generation_answer_style"):
        OpenAICompletion(generation_answer_style=cast(Literal["default", "concise"], style))
    with pytest.raises(SchemaError):
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
        with pytest.raises(SchemaError):
            MemorySpec.model_validate(
                {
                    "embedding": {"provider": "openai"},
                    slot: {"provider": "openai", "answer_style": "concise"},
                }
            )


def test_image_source_prose_defaults_to_include() -> None:
    config = MemorySpec.model_validate(
        {"embedding": {"provider": "openai"}, "generation": {"provider": "openai"}}
    )
    assert isinstance(config.generation, OpenAIGenerationConfig)
    assert config.generation.image_source_prose == "include"


@pytest.mark.parametrize("value", ["include", "context_when_supplied"])
def test_explicit_image_source_prose_values(value: str) -> None:
    config = MemorySpec.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "image_source_prose": value},
        }
    )
    assert isinstance(config.generation, OpenAIGenerationConfig)
    assert config.generation.image_source_prose == value


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "native", "", "omit_when_supplied"])
def test_invalid_image_source_prose_is_rejected_at_both_boundaries(value: object) -> None:
    with pytest.raises(SchemaError, match="image_source_prose"):
        MemorySpec.model_validate(
            {
                "embedding": {"provider": "openai"},
                "generation": {"provider": "openai", "image_source_prose": value},
            }
        )
    with pytest.raises(ValidationError, match="generation_image_source_prose"):
        OpenAICompletion(generation_image_source_prose=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("slot", ["formation", "vision", "consolidation"])
def test_image_source_prose_is_not_a_shared_completion_option(slot: str) -> None:
    with pytest.raises(SchemaError, match="image_source_prose"):
        MemorySpec.model_validate(
            {
                "embedding": {"provider": "openai"},
                slot: {"provider": "openai", "image_source_prose": "context_when_supplied"},
            }
        )
