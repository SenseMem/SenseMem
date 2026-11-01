"""Image and video refinement preserves positions, source pixels and explicit failures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError as SchemaValidationError

from mindbridge import AssetRef, Modality, ModelInput, MultimodalRerankingBackend, Qwen3VLReranker
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.qwen3_vl_reranking import _MediaSettings

REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"


class Engine:
    def __init__(self) -> None:
        self.media_calls: list[tuple[ModelInput, tuple[ModelInput, ...], _MediaSettings]] = []
        self.values: object = [0.1, 0.95]

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object:
        return [0.8, 0.7, 0.6, 0.5][: len(documents)]

    def score_media(
        self, query: ModelInput, documents: Sequence[ModelInput], settings: _MediaSettings
    ) -> object:
        self.media_calls.append((query, tuple(documents), settings))
        return self.values

    def close(self) -> None:
        pass


def documents(tmp_path: Path) -> tuple[ModelInput, ...]:
    return (
        ModelInput(text="email"),
        ModelInput(
            text="image caption",
            assets=(
                AssetRef(
                    id="image",
                    modality=Modality.IMAGE,
                    media_type="image/png",
                    path=tmp_path / "image",
                    size_bytes=1,
                    sha256="0" * 64,
                ),
            ),
        ),
        ModelInput(
            text="video caption",
            assets=(
                AssetRef(
                    id="video",
                    modality=Modality.VIDEO,
                    media_type="video/mp4",
                    path=tmp_path / "video",
                    size_bytes=1,
                    sha256="0" * 64,
                ),
            ),
        ),
        ModelInput(
            text="another image",
            assets=(
                AssetRef(
                    id="image2",
                    modality=Modality.IMAGE,
                    media_type="image/png",
                    path=tmp_path / "image2",
                    size_bytes=1,
                    sha256="0" * 64,
                ),
            ),
        ),
    )


def test_both_image_and_video_pixels_reach_shortlisted_engine_and_change_order(
    tmp_path: Path,
) -> None:
    engine = Engine()
    model = Qwen3VLReranker(
        engine,
        model_id="org/model",
        revision=REVISION,
        multimodal=True,
        media_candidates=2,
        media_weight=1,
    )
    assert isinstance(model, MultimodalRerankingBackend)
    assert model.reranking_capabilities == frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.VIDEO}
    )
    assert "native-image-video-v1" in model.reranking_model
    items = documents(tmp_path)
    query = ModelInput(text="Find the red bus.")
    assert model.rerank_media(query, items) == pytest.approx((0.8, 0.1, 0.95, 0.5))
    assert engine.media_calls[0][:2] == (query, items[1:3])
    assert engine.media_calls[0][2].video_max_frames == 8
    model.close()
    with pytest.raises(ModelError, match="closed"):
        model.rerank_media(query, items)


def test_text_only_keeps_original_path_and_fusion_is_in_logit_space(tmp_path: Path) -> None:
    engine = Engine()
    model = Qwen3VLReranker(
        engine,
        model_id="org/model",
        revision=REVISION,
        multimodal=True,
        media_candidates=2,
        media_weight=0.5,
    )
    scores = model.rerank_media(ModelInput(text="query"), documents(tmp_path))
    assert scores[0] == 0.8 and scores[3] == 0.5
    assert 0.1 < scores[1] < 0.7
    assert 0.6 < scores[2] < 0.95
    text_engine = Engine()
    text = Qwen3VLReranker(text_engine, model_id="org/model", revision=REVISION)
    assert text.rerank_media(ModelInput(text="query"), (ModelInput(text="email"),)) == (0.8,)
    assert not text_engine.media_calls
    with pytest.raises(ValidationError, match="not enabled"):
        text.rerank_media(ModelInput(text="query"), documents(tmp_path))


@pytest.mark.parametrize("values", [[0.5], [float("nan"), 0.5], [-1.0, 0.5], [1.1, 0.5]])
def test_malformed_media_scores_cannot_affect_ranking(tmp_path: Path, values: object) -> None:
    engine = Engine()
    engine.values = values
    model = Qwen3VLReranker(
        engine,
        model_id="org/model",
        revision=REVISION,
        multimodal=True,
        media_candidates=2,
    )
    with pytest.raises(ModelError) as error:
        model.rerank_media(ModelInput(text="query"), documents(tmp_path))
    assert error.value.reason == "response_invalid"
    assert error.value.stage == "reranking"


@pytest.mark.parametrize(
    "settings",
    [
        {"media_weight": 0},
        {"media_weight": float("nan")},
        {"video_max_frames": 2},
        {"video_fps": float("inf")},
        {"media_max_length": 0},
    ],
)
def test_invalid_media_policy_is_rejected(settings: dict[str, object]) -> None:
    from mindbridge import MindBridgeConfig

    with pytest.raises(SchemaValidationError):
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "reranking": {"provider": "qwen3-vl", "model": "org/model", **settings},
            }
        )
