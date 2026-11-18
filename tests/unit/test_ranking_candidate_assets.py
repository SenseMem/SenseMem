"""Provisional ranking snapshots do not open or validate losing candidates' media."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from sensemem import (
    Blob,
    DefaultRankingPolicy,
    EmbedTask,
    Memory,
    MemorySettings,
    Modality,
    ModelInput,
    RankingRequest,
    RankingScore,
    StorageError,
)


class DistinctMediaEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "ranking-media-fixture"
    embedding_space = "ranking-media-fixture:v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) if "target" in value.text else (-1.0, 0.0) for value in inputs)

    def close(self) -> None:
        pass


class InspectAssets(DefaultRankingPolicy):
    def __init__(self) -> None:
        self.requests: list[RankingRequest] = []

    def rank(self, request: RankingRequest) -> tuple[RankingScore, ...]:
        self.requests.append(request)
        return super().rank(request)


@pytest.mark.parametrize("custom", [False, True])
def test_missing_losing_asset_does_not_block_text_search(tmp_path: Path, custom: bool) -> None:
    policy = InspectAssets() if custom else None
    with Memory(
        tmp_path,
        embedder=DistinctMediaEmbedder(),
        ranking_policy=policy,
        settings=MemorySettings(ambiguity_margin=0),
    ) as memory:
        target = memory.add("target answer")
        decoy = memory.add(("unrelated decoy", Blob(b"original image", "image/png")))
        assert decoy.assets[0].path is not None
        decoy.assets[0].path.unlink()
        result = memory.search("target", limit=1)
        assert [hit.id for hit in result.hits] == [target.id]
        if policy is not None:
            snapshot = next(
                candidate.memory
                for candidate in policy.requests[0].candidates
                if candidate.memory.id == decoy.id
            )
            asset = snapshot.assets[0]
            assert snapshot.modality is Modality.IMAGE
            assert asset.id == decoy.assets[0].id
            assert asset.sha256 == decoy.assets[0].sha256
            assert asset.size_bytes == len(b"original image")
            assert asset.media_type == "image/png"
            assert asset.path is None


def test_selected_asset_still_requires_a_valid_cas_file(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=DistinctMediaEmbedder(),
        settings=MemorySettings(ambiguity_margin=0),
    ) as memory:
        target = memory.add(("target image", Blob(b"original image", "image/png")))
        assert target.assets[0].path is not None
        target.assets[0].path.unlink()
        with pytest.raises(StorageError, match="local media"):
            memory.search("target", limit=1)
