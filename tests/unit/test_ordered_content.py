"""Text/media positions survive ingestion, deferred enrichment, reopening and model delivery."""

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from PIL import Image

from sensemem import (
    Blob,
    ContentPart,
    EmbedTask,
    Memory,
    MemorySettings,
    ModelInput,
    ObservationContext,
)
from sensemem.models._openai_media import _input_parts
from sensemem.models._openai_requests import _ordered_content_payload


class _RecordingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls: list[ModelInput] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.extend(inputs)
        return super().embed(inputs, task)


def _image(color: str) -> Blob:
    output = BytesIO()
    Image.new("RGB", (1, 1), color=color).save(output, format="PNG")
    return Blob(output.getvalue(), "image/png", f"{color}.png")


@pytest.mark.parametrize("capture", [False, True])
def test_interleaved_source_positions_survive_settlement_and_reopening(
    tmp_path: Path, capture: bool
) -> None:
    embedder = _RecordingEmbedder()
    red, blue = _image("red"), _image("blue")
    with Memory(
        tmp_path, embedder=embedder, settings=MemorySettings(minimum_relevance=0)
    ) as memory:
        record = (memory.capture if capture else memory.add)(
            ("Repair before:", red, "Repair after:", blue)
        )
        if capture:
            assert memory.settle() == 1
        combined = next(value for value in embedder.calls if len(value.assets) == 2)
        assert combined.ordered_content == (
            "Repair before:",
            combined.assets[0],
            "Repair after:",
            combined.assets[1],
        )
        wire, _ = _input_parts(combined, {})
        assert [part["type"] for part in wire] == ["text", "image_url", "text", "image_url"]
        expected = (
            ContentPart("text", "Repair before:"),
            ContentPart("asset", record.assets[0].id),
            ContentPart("text", "Repair after:"),
            ContentPart("asset", record.assets[1].id),
        )
        assert memory.get(record.id).parts == expected
    with Memory(
        tmp_path, embedder=TinyEmbedder(), settings=MemorySettings(minimum_relevance=0)
    ) as memory:
        assert memory.get(record.id).parts == expected
        assert memory.search("repair").hits[0].parts == expected
        bundle = memory.compile("repair")
        assert bundle.hits[0].parts == expected
        for rendered in (bundle.render(), bundle.compact().text):
            assert rendered.index("Repair before:") < rendered.index(record.assets[0].id)
            assert rendered.index(record.assets[0].id) < rendered.index("Repair after:")
            assert rendered.index("Repair after:") < rendered.index(record.assets[1].id)
        assert _ordered_content_payload(
            bundle.hits[0], {record.assets[0].id: "A1", record.assets[1].id: "A2"}
        ) == {
            "content_parts": [
                {"kind": "text", "text": "Repair before:"},
                {"kind": "asset", "media": "A1"},
                {"kind": "text", "text": "Repair after:"},
                {"kind": "asset", "media": "A2"},
            ]
        }


def test_source_replay_preserves_media_order_provenance_and_directory_isolation(
    tmp_path: Path,
) -> None:
    with (
        Memory(tmp_path / "source", embedder=TinyEmbedder()) as source,
        Memory(tmp_path / "destination", embedder=TinyEmbedder()) as destination,
    ):
        original = source.add(
            ("Before:", _image("red"), "After:", _image("blue")),
            context=ObservationContext(source_id="camera", place_id="workshop"),
        )
        discarded = source.add("An explicitly forgotten observation")
        source.forget((discarded.id,))
        replayed = tuple(source.replay_sources(destination))
        assert len(replayed) == 1
        assert replayed[0].id == original.id
        assert replayed[0].parts == original.parts
        assert replayed[0].place_id == "workshop"
        assert replayed[0].assets[0].path != original.assets[0].path
        assert destination.read_asset(replayed[0].id, replayed[0].assets[0].id) == _image("red")
