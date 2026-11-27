"""Lossy stream interpretations keep their role through storage and model delivery."""

from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient
from fixtures.formation import HistoryFormer

from sensemem import Blob, ContentPart, Memory, MemorySettings, ModelInput, StreamInput
from sensemem.api.app import create_app
from sensemem.models._openai_requests import (
    _consolidation_evidence_payload,
    _formation_history_payload,
    _ordered_content_payload,
)


@pytest.mark.parametrize("capture", [False, True])
def test_interpretation_survives_settle_reopen_and_replay_does_not_promote_it(
    tmp_path: Path, capture: bool
) -> None:
    source = Blob(b"original independent media", "audio/wav", "source.wav")
    with Memory(tmp_path / "source", embedder=TinyEmbedder()) as memory:
        record = next(
            memory.add_stream(
                [StreamInput(("Original words", source), interpretation="A model synopsis")],
                capture=capture,
            )
        )
        if capture:
            assert memory.settle() == 1
        expected = (
            ContentPart("text", "Original words"),
            ContentPart("asset", record.assets[0].id),
            ContentPart(
                "model_interpretation",
                "[model interpretation: not original evidence]\nA model synopsis",
            ),
        )
        assert memory.get(record.id).parts == expected
        for payload in (
            _formation_history_payload(memory.get(record.id), "observation_0", {}),
            _consolidation_evidence_payload(record, 0, evidence_indices={record.id: 0}),
        ):
            assert payload["content_parts"] == [
                {"kind": "text", "text": expected[0].value},
                {"kind": "asset", "available": False},
                {"kind": "model_interpretation", "text": expected[2].value},
            ]
        with TestClient(create_app(memory=memory)) as client:
            response = client.get(f"/v1/memories/{record.id}")
            assert response.status_code == 200
            assert response.json()["parts"][-1]["kind"] == "model_interpretation"

    with (
        Memory(
            tmp_path / "source",
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0),
        ) as memory,
        Memory(tmp_path / "destination", embedder=TinyEmbedder()) as destination,
    ):
        assert memory.get(record.id).parts == expected
        assert memory.compile("Original words").hits[0].parts == expected
        replayed = tuple(memory.replay_sources(destination))
        assert len(replayed) == 1
        assert replayed[0].parts == expected[:2]
        assert replayed[0].content == "Original words"
        assert destination.read_asset(replayed[0].id, replayed[0].assets[0].id) == source


def test_text_only_model_delivery_retains_the_interpretation_role() -> None:
    original, interpretation = "Original words", "A model synopsis"
    value = ModelInput(
        text=f"{original}\n\n{interpretation}",
        parts=(ContentPart("text", original), ContentPart("model_interpretation", interpretation)),
    )
    assert _ordered_content_payload(value, {}) == {
        "content_parts": [
            {"kind": "text", "text": original},
            {"kind": "model_interpretation", "text": interpretation},
        ]
    }


def test_read_only_history_keeps_derived_roles_with_original_media_omitted(tmp_path: Path) -> None:
    former = HistoryFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(formation_history_max_rows=8),
    ) as memory:
        original = next(
            memory.add_stream(
                [
                    StreamInput(
                        ("Original words", Blob(b"source audio", "audio/wav")),
                        interpretation="A model synopsis",
                    )
                ]
            )
        )
        memory.add("A later observation")
        history = former.inputs[-1].history
        assert len(history) == 1
        assert history[0].id == original.id
        assert history[0].parts == original.parts
        assert history[0].assets == ()
        payload = _formation_history_payload(history[0], "history_0", {})
        assert payload["content_parts"] == [
            {"kind": "text", "text": original.parts[0].value},
            {"kind": "asset", "available": False},
            {"kind": "model_interpretation", "text": original.parts[2].value},
        ]
