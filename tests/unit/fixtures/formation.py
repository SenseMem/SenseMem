"""Shared formation, identity, and image-prose helpers for unit tests."""

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from _feature_support import TinyEmbedder
from PIL import Image

from sensemem import (
    AssetRef,
    Blob,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryRecord,
    Modality,
    ModelInput,
    ObservationContext,
    RecordKind,
    RecordSemantics,
)


class _BeforeReturnHook(Protocol):
    def __call__(self) -> object: ...


class HistoryFormer:
    formation_capabilities = frozenset(TinyEmbedder.embedding_capabilities)
    formation_model = "history-test"
    formation_space = "history-test:v1"

    def __init__(self) -> None:
        self.inputs: list[FormationInput] = []
        self.before_return: _BeforeReturnHook | None = None
        self.history_content = "Ada is the visitor."

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.inputs.extend(inputs)
        results = []
        for target in inputs:
            historical = next(
                (r for r in target.history if r.content == self.history_content), None
            )
            results.append(
                ()
                if historical is None or target.content.text != "She left her notebook."
                else (
                    FormationProposal(
                        kind=RecordKind.EVENT,
                        content="Ada left her notebook.",
                        subject="Ada",
                        evidence_ids=(target.memory_id, historical.id),
                    ),
                )
            )
        if self.before_return is not None:
            self.before_return()
        return tuple(results)

    def close(self) -> None:
        pass


def history_input(*, retired: bool = False) -> FormationInput:
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = MemoryRecord(
        id="old",
        content="Ada is the visitor.",
        created_at=before,
        context=RecordSemantics(
            kind=RecordKind.ENTITY,
            subject="Ada",
            basis=EvidenceBasis.USER_STATEMENT,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=before,
            retired_at=datetime(2026, 1, 2, tzinfo=timezone.utc) if retired else None,
        ),
    )
    return FormationInput(
        memory_id="current",
        content=ModelInput(text="She left her notebook."),
        context=ObservationContext(),
        history=(history,),
    )


class Faces:
    face_capabilities = frozenset({Modality.IMAGE})
    face_model = "naming-premise-faces"
    face_space = "naming-premise-faces:2"
    face_analysis_space = "naming-premise-detector:v1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis(
                (
                    FaceEmbedding(
                        "face",
                        (1.0, 0.0) if asset.name == "first.png" else (0.0, 1.0),
                        (0.1, 0.1, 0.5, 0.5),
                    ),
                )
            )
            for asset in assets
        )

    def close(self) -> None:
        pass


def register_identity_fixture(memory: Memory, filename: str) -> tuple[str, str, int]:
    photo = memory.add(Blob(filename.encode(), "image/png", filename))
    person = memory.perceive(photo.id, kinds=frozenset({"faces"})).faces[0].identity_id
    memory.register_identity(person, "Rava")
    naming = next(
        operation
        for operation in memory.operations()
        if operation.operation.intent is MemoryIntent.IDENTIFY
        and operation.operation.claim is not None
        and operation.operation.claim.identity_id == person
    )
    return person, naming.created_ids[0], naming.operation_id


def make_image_asset(directory: Path, name: str) -> AssetRef:
    path = directory / f"{name}.png"
    Image.new("RGB", (1, 1), "blue").save(path)
    data = path.read_bytes()
    return AssetRef(
        name, Modality.IMAGE, "image/png", len(data), hashlib.sha256(data).hexdigest(), path=path
    )


def extract_memories(payload: dict[str, object]) -> list[dict[str, object]]:
    content = cast(list[dict[str, object]], payload["messages"])[1]["content"]
    if isinstance(content, str):
        return cast(list[dict[str, object]], json.loads(content)["hits"])
    return [
        cast(dict[str, object], json.loads(cast(str, part["text"]))["memory"])
        for part in cast(list[dict[str, object]], content)
        if part["type"] == "text" and cast(str, part["text"]).startswith('{"memory":')
    ]
