"""The image-source display policy is an explicit generation-only option."""

import pytest
from pydantic import ValidationError as SchemaError

from sensemem import MemorySpec, ValidationError
from sensemem.configuration import OpenAIGenerationConfig
from sensemem.models.openai_completion import OpenAICompletion


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
