"""Scoped operation execution is an explicit generation-only text capability."""

from collections.abc import Callable
from inspect import signature
from typing import cast

import pytest
from pydantic import ValidationError as SchemaError

import sensemem.configuration as configuration
import sensemem.recipes as recipes
from sensemem import OpenAIModels, SenseMemConfig, ValidationError
from sensemem.models.base import GenerationBackend, StreamingGenerationBackend


@pytest.fixture
def constructed_controls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Exercise real adapter validation without constructing a network client."""
    observed: list[dict[str, object]] = []

    def build(**controls: object) -> OpenAIModels:
        observed.append(controls.copy())
        return cast(Callable[..., OpenAIModels], OpenAIModels)(**controls)

    monkeypatch.setattr(recipes, "_owned_openai_models", build)
    return observed


def test_evidence_operation_tool_defaults_off_and_is_a_boolean_schema_field() -> None:
    config = SenseMemConfig.model_validate(
        {"embedding": {"provider": "openai"}, "generation": {"provider": "openai"}}
    )
    assert config.generation is not None
    assert config.generation.evidence_operation_tool is False
    field = SenseMemConfig.model_json_schema()["$defs"]["OpenAIGenerationConfig"]["properties"][
        "evidence_operation_tool"
    ]
    assert field["type"] == "boolean"
    assert field["default"] is False
    assert signature(OpenAIModels).parameters["generation_evidence_operation_tool"].default is False


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_booleans_round_trip_and_preserve_generation_ports(enabled: bool) -> None:
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "evidence_operation_tool": enabled},
        }
    )
    restored = SenseMemConfig.model_validate_json(config.model_dump_json())
    assert restored.generation is not None
    assert restored.generation.evidence_operation_tool is enabled
    backend = OpenAIModels(generation_evidence_operation_tool=enabled)
    try:
        assert isinstance(backend, GenerationBackend)
        assert isinstance(backend, StreamingGenerationBackend)
    finally:
        backend.close()


@pytest.mark.parametrize("value", [None, 0, 1, 0.0, 1.0, "true", "false", "", [], {}])
def test_non_boolean_values_are_rejected_at_config_and_direct_sdk(value: object) -> None:
    with pytest.raises(SchemaError, match="evidence_operation_tool"):
        SenseMemConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "generation": {"provider": "openai", "evidence_operation_tool": value},
            }
        )
    with pytest.raises(ValidationError, match="generation_evidence_operation_tool"):
        OpenAIModels(generation_evidence_operation_tool=value)  # type: ignore[arg-type]


def test_enabled_operation_tool_requires_text_at_both_boundaries() -> None:
    with pytest.raises(SchemaError, match=r"evidence_operation_tool.*answer_format"):
        SenseMemConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "generation": {
                    "provider": "openai",
                    "evidence_operation_tool": True,
                    "answer_format": "structured",
                },
            }
        )
    with pytest.raises(ValidationError, match="generation_evidence_operation_tool"):
        OpenAIModels(generation_evidence_operation_tool=True, generation_answer_format="structured")


def test_disabled_operation_tool_preserves_structured_answers() -> None:
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {
                "provider": "openai",
                "evidence_operation_tool": False,
                "answer_format": "structured",
            },
        }
    )
    assert config.generation is not None
    assert config.generation.answer_format == "structured"
    OpenAIModels(
        generation_evidence_operation_tool=False, generation_answer_format="structured"
    ).close()


@pytest.mark.parametrize("slot", ["embedding", "formation", "vision", "consolidation"])
def test_operation_tool_is_not_a_shared_completion_field(slot: str) -> None:
    with pytest.raises(SchemaError, match="evidence_operation_tool"):
        SenseMemConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                slot: {"provider": "openai", "evidence_operation_tool": True},
            }
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_config_forwards_option_without_changing_provider_extensions(
    enabled: bool, constructed_controls: list[dict[str, object]]
) -> None:
    extension = {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "vendor": {"tools": ["nested provider data"], "optional": None},
    }
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {
                "provider": "openai",
                "model": "test-reader",
                "evidence_operation_tool": enabled,
                "extra_body": extension,
            },
        }
    )
    assert config.generation is not None
    backend = configuration._build_generation(config.generation)
    try:
        assert len(constructed_controls) == 1
        controls = constructed_controls[0]
        assert controls.get("generation_evidence_operation_tool", False) is enabled
        assert controls["generation_extra_body"] == extension
        assert config.generation.extra_body == extension
        assert extension["vendor"] == {"tools": ["nested provider data"], "optional": None}
    finally:
        backend.close()


def test_generation_routes_choose_their_operation_option_independently(
    constructed_controls: list[dict[str, object]],
) -> None:
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {
                "provider": "openai",
                "model": "base-reader",
                "routes": [
                    {
                        "provider": "openai",
                        "model": "image-reader",
                        "modalities": ["text", "image"],
                        "evidence_operation_tool": True,
                    }
                ],
            },
        }
    )
    assert config.generation is not None
    backend = configuration._build_generation(config.generation)
    try:
        assert len(constructed_controls) == 2
        base, route = constructed_controls
        assert base["generation_model"] == "base-reader"
        assert base.get("generation_evidence_operation_tool", False) is False
        assert route["generation_model"] == "image-reader"
        assert route["generation_evidence_operation_tool"] is True
    finally:
        backend.close()


@pytest.mark.parametrize("key", ["tools", "tool_choice", "parallel_tool_calls"])
def test_enabled_owned_tool_keys_are_rejected_by_sdk_and_config_build(
    key: str, constructed_controls: list[dict[str, object]]
) -> None:
    with pytest.raises(ValidationError, match="generation_extra_body"):
        OpenAIModels(generation_evidence_operation_tool=True, generation_extra_body={key: None})
    # Unknown provider JSON is validated by the SDK owner, not parsed twice in configuration.
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {
                "provider": "openai",
                "evidence_operation_tool": True,
                "extra_body": {key: None},
            },
        }
    )
    assert config.generation is not None
    with pytest.raises(ValidationError, match="generation_extra_body"):
        configuration._build_generation(config.generation)
    assert constructed_controls[0]["generation_evidence_operation_tool"] is True


@pytest.mark.parametrize("key", ["tools", "tool_choice", "parallel_tool_calls"])
def test_disabled_tool_preserves_existing_provider_tool_extensions(
    key: str, constructed_controls: list[dict[str, object]]
) -> None:
    OpenAIModels(
        generation_evidence_operation_tool=False, generation_extra_body={key: None}
    ).close()
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "extra_body": {key: None}},
        }
    )
    assert config.generation is not None
    backend = configuration._build_generation(config.generation)
    try:
        assert constructed_controls[0]["generation_extra_body"] == {key: None}
    finally:
        backend.close()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("key", ["model", "messages", "response_format", "stream", "max_tokens"])
def test_operation_option_does_not_weaken_existing_extra_body_protections(
    enabled: bool, key: str
) -> None:
    with pytest.raises(ValidationError, match="generation_extra_body"):
        OpenAIModels(generation_evidence_operation_tool=enabled, generation_extra_body={key: None})


@pytest.mark.parametrize("value", [1, [], {"tools": object()}])
def test_enabled_tool_validates_unknown_extra_body_before_owned_key_inspection(
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="generation_extra_body"):
        OpenAIModels(
            generation_evidence_operation_tool=True,
            generation_extra_body=value,  # type: ignore[arg-type]
        )
