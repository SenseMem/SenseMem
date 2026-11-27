"""Ingestion stage interpretations and content-addressed source identity contracts."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest

from sensemem import (
    Blob,
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    StageResult,
    StreamInput,
    ValidationError,
)


class _SourceIdentityEmbedder:
    embedding_capabilities = frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
    )
    embedding_model = "source-identity-contract"
    embedding_space = "source-identity-contract:4"
    embedding_dimension = 4

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0, 0.0, 0.0),) * len(inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("interpretation", [None, 1, False, (), {}])
def test_stage_result_requires_text(interpretation: object) -> None:
    with pytest.raises(ValidationError, match="stage interpretation must be text"):
        StageResult(cast(str, interpretation))


def test_stage_result_normalizes_derived_text_and_is_immutable() -> None:
    result = StageResult("  A possible interpretation  ")
    assert result.interpretation == "A possible interpretation"
    assert StageResult().interpretation == StageResult(" \n ").interpretation == ""
    with pytest.raises(FrozenInstanceError):
        result.interpretation = "Original source"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["transcript", "description", "interpretation"])
def test_stream_derivation_keeps_the_original_observation_id(tmp_path: Path, field: str) -> None:
    media = Blob(b"original recording", "audio/wav" if field == "transcript" else "image/png")
    with Memory(tmp_path, embedder=_SourceIdentityEmbedder()) as memory:
        original = memory.add(("Original source words.", media), metadata={"batch": 7})
        item = StreamInput(
            ("Original source words.", media),
            metadata={"batch": 7},
            transcript="Derived words." if field == "transcript" else None,
            description="Derived scene." if field == "description" else None,
            interpretation="Derived meaning." if field == "interpretation" else None,
        )
        derived = next(memory.add_stream([item]))
        assert derived.id == original.id
        assert derived.content != original.content
        refreshed = next(
            memory.add_stream(
                [
                    replace(
                        item,
                        transcript="Refreshed words." if field == "transcript" else None,
                        description="Refreshed scene." if field == "description" else None,
                        interpretation="Refreshed meaning." if field == "interpretation" else None,
                    )
                ]
            )
        )
        assert refreshed.id == original.id
        assert "Refreshed" in refreshed.content and "Derived" not in refreshed.content
        different_metadata = memory.add(("Original source words.", media), metadata={"batch": 8})
        assert different_metadata.id != original.id
