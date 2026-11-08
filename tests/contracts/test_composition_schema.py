"""Public composition values, structured visual proposals, and declarative routes."""

from dataclasses import asdict
from pathlib import Path

import pytest

from sensemem import (
    EmbeddingBackend,
    EmbeddingInputSupport,
    EmbeddingRouter,
    EvidenceGeneration,
    GenerationBackend,
    GenerationRouter,
    GenerationRoutingBackend,
    MediaSelectionBackend,
    MemoryPlugins,
    Modality,
    ModelInput,
    OpenAIModels,
    SenseMemConfig,
    SenseMemError,
    SpeakerName,
    ValidationError,
    VisionAnalysisBackend,
    VisualDescription,
)


def test_branded_public_values_use_the_sensemem_namespace() -> None:
    config = SenseMemConfig.model_validate({"embedding": {"provider": "openai"}})
    assert config.data_dir == Path(".sensemem")
    assert config.model_json_schema()["title"] == "SenseMemConfig"
    assert SenseMemError.code == "sensemem_error"


@pytest.mark.parametrize("modalities", [None, ["text"], ["video", "text", "audio"]])
def test_local_embedding_loading_options_round_trip(modalities: list[str] | None) -> None:
    config = SenseMemConfig.model_validate(
        {
            "embedding": {
                "provider": "sentence-transformers",
                "model": "google/embeddinggemma-2",
                "dtype": "bfloat16",
                "modalities": modalities,
            }
        }
    )
    encoded = config.model_dump(mode="json")
    assert encoded["embedding"]["modalities"] == (
        None if modalities is None else sorted(modalities)
    )
    assert SenseMemConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("invalid", [{"dtype": "float16"}, {"modalities": []}])
def test_local_embedding_schema_rejects_unsafe_precision_and_no_modalities(
    invalid: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SenseMemConfig.model_validate(
            {
                "embedding": {
                    "provider": "sentence-transformers",
                    "model": "google/embeddinggemma-2",
                    **invalid,
                }
            }
        )


def test_composed_backends_implement_the_public_protocols() -> None:
    model = OpenAIModels()
    embedding = EmbeddingRouter((model,))
    generation = GenerationRouter((EvidenceGeneration(model, selector=model),))
    assert isinstance(embedding, EmbeddingBackend)
    assert isinstance(embedding, EmbeddingInputSupport)
    assert isinstance(generation, GenerationBackend)
    assert isinstance(generation, GenerationRoutingBackend)
    assert isinstance(model, MediaSelectionBackend)
    assert isinstance(model, VisionAnalysisBackend)
    assert MemoryPlugins(embedder=embedding, answerer=generation).answerer is generation


def test_declared_generation_routes_have_independent_modalities_and_controls() -> None:
    config = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {
                "provider": "openai",
                "model": "text-fixture",
                "routes": [
                    {
                        "provider": "openai",
                        "model": "visual-fixture",
                        "modalities": ["text", "image"],
                        "max_tokens": 512,
                    }
                ],
            },
        }
    )
    assert config.generation is not None
    assert config.generation.model == "text-fixture"
    route = config.generation.routes[0]
    assert route.modalities == {Modality.TEXT, Modality.IMAGE} and route.max_tokens == 512


def test_structured_visual_names_are_explicit_values() -> None:
    value = VisualDescription(
        "one person", ("speaker_1 is called Lily",), (SpeakerName("speaker_1", "Lily"),)
    )
    assert asdict(value) == {
        "text": "one person",
        "facts": ("speaker_1 is called Lily",),
        "speaker_names": ({"speaker_label": "speaker_1", "name": "Lily"},),
    }
    assert VisualDescription("Fact: speaker_1 is called Lily").speaker_names == ()
    with pytest.raises(ValidationError):
        SpeakerName("identity_global", "Lily")


def test_model_input_runtime_evidence_does_not_change_embedding_key_identity() -> None:
    plain = ModelInput("question")
    enriched = ModelInput("question", source_metadata={"asset": {"untrusted": True}})
    assert plain == enriched and hash(plain) == hash(enriched)
