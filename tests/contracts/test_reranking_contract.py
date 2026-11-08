"""The public scorer contract uses resolved multimodal values and aligned scores."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import httpx2 as httpx
import pytest

from sensemem import (
    AssetRef,
    Modality,
    ModelInput,
    Qwen3VLReranker,
    RerankingBackend,
    ValidationError,
    VLLMReranker,
)


@pytest.mark.parametrize("remote", [False, True])
def test_public_reranker_accepts_media_only_and_mixed_pairs_without_a_mode(
    tmp_path: Path, remote: bool
) -> None:
    class Engine:
        def score(
            self,
            query: ModelInput,
            documents: Sequence[ModelInput],
            *,
            max_length: int,
        ) -> object:
            assert query.assets == (image,)
            assert documents == candidates
            return (0.2, 0.9)

        def close(self) -> None:
            pass

    (tmp_path / "image.png").write_bytes(b"i")
    (tmp_path / "video.mp4").write_bytes(b"v")
    image = AssetRef(
        "image",
        modality=Modality.IMAGE,
        media_type="image/png",
        size_bytes=1,
        sha256=hashlib.sha256(b"i").hexdigest(),
        path=tmp_path / "image.png",
    )
    video = AssetRef(
        "video",
        modality=Modality.VIDEO,
        media_type="video/mp4",
        size_bytes=1,
        sha256=hashlib.sha256(b"v").hexdigest(),
        path=tmp_path / "video.mp4",
    )
    candidates = (ModelInput(text="source"), ModelInput(text="caption", assets=(video, image)))
    with_model: RerankingBackend
    if remote:

        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "score": 0.9},
                        {"index": 0, "score": 0.2},
                    ]
                },
            )

        with_model = VLLMReranker(
            httpx.Client(transport=httpx.MockTransport(respond)),
            model_id="org/model",
            base_url="https://fixture.test",
        )
    else:
        with_model = Qwen3VLReranker(Engine(), model_id="org/model", revision="a" * 40)
    assert isinstance(with_model, RerankingBackend)
    assert with_model.rerank(ModelInput(assets=(image,)), candidates) == (0.2, 0.9)
    with pytest.raises(ValidationError, match="ModelInput"):
        with_model.rerank("question", candidates)  # type: ignore[arg-type]
    audio = AssetRef(
        "audio",
        modality=Modality.AUDIO,
        media_type="audio/wav",
        size_bytes=1,
        sha256="c" * 64,
        path=tmp_path / "audio.wav",
    )
    with pytest.raises(ValidationError, match="unsupported modality"):
        with_model.rerank(ModelInput(assets=(audio,)), candidates)
    with_model.close()
