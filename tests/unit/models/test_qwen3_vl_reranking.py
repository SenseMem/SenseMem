"""Native vLLM pair scoring preserves modalities, positions and the trained binary head."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.qwen3_vl_reranking as reranking
from mindbridge import AssetRef, Modality, ModelInput, Qwen3VLReranker, RerankingBackend
from mindbridge.exceptions import ModelError, ValidationError

REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"


class Engine:
    def __init__(self, values: object) -> None:
        self.values = values
        self.calls: list[tuple[ModelInput, tuple[ModelInput, ...], int]] = []
        self.closed = 0

    def score(
        self,
        query: ModelInput,
        documents: Sequence[ModelInput],
        *,
        max_length: int,
    ) -> object:
        self.calls.append((query, tuple(documents), max_length))
        return self.values

    def close(self) -> None:
        self.closed += 1


def test_injected_engine_preserves_positions_settings_and_lifecycle() -> None:
    engine = Engine([0.125, 0.875])
    model = Qwen3VLReranker(
        engine, model_id="Qwen/Qwen3-VL-Reranker-8B", revision=REVISION, batch_size=2
    )
    query = ModelInput(text="Which booking was cancelled?")
    documents = (ModelInput(text="Confirmed"), ModelInput(text="Cancelled"))
    assert isinstance(model, RerankingBackend)
    assert model.reranking_capabilities == frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.VIDEO}
    )
    assert f"Qwen/Qwen3-VL-Reranker-8B@{REVISION}:" in model.reranking_model
    assert "max_length=8192" in model.reranking_model
    assert model.rerank(query, documents) == (0.125, 0.875)
    assert engine.calls == [(query, documents, 8192)]
    assert model.rerank(query, ()) == ()
    assert len(engine.calls) == 1
    model.close()
    model.close()
    assert engine.closed == 1
    with pytest.raises(ModelError, match="closed"):
        model.rerank(query, documents)


@pytest.mark.parametrize("values", [[], [True], [-0.1], [1.1]])
def test_invalid_probability_output_is_rejected(values: object) -> None:
    model = Qwen3VLReranker(Engine(values), model_id="org/model", revision=REVISION)
    with pytest.raises(ModelError) as error:
        model.rerank(ModelInput(text="question"), (ModelInput(text="source"),))
    assert error.value.reason == "response_invalid"
    assert error.value.stage == "reranking"


def test_resolve_revision_lazily_and_sanitize_inference_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reranking,
        "import_module",
        lambda name: SimpleNamespace(model_info=lambda _model: SimpleNamespace(sha=REVISION)),
    )
    engine = Engine([0.5])
    model = Qwen3VLReranker(engine, model_id="org/model")
    assert REVISION in model.reranking_model

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private source and provider body")

    monkeypatch.setattr(engine, "score", fail)
    with pytest.raises(ModelError) as error:
        model.rerank(ModelInput(text="private query"), (ModelInput(text="private source"),))
    assert str(error.value) == "reranking model failed"
    assert error.value.reason == "model_failed"


def test_validation_and_missing_vllm_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="immutable"):
        Qwen3VLReranker(Engine([]), model_id="org/model", revision="main")

    def missing(_name: str) -> object:
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(reranking, "import_module", missing)
    with pytest.raises(ModelError, match="reranking extra"):
        Qwen3VLReranker.load("org/model", revision=REVISION)


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_device_selection_is_explicit(device: str) -> None:
    with pytest.raises(ValidationError, match="CUDA_VISIBLE_DEVICES"):
        Qwen3VLReranker(Engine([]), model_id="org/model", revision=REVISION, device=device)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, math.nan, math.inf, True, "0.9"])
def test_gpu_memory_utilization_is_bounded(value: object) -> None:
    with pytest.raises(ValidationError, match="gpu_memory_utilization"):
        Qwen3VLReranker(
            Engine([]),
            model_id="org/model",
            revision=REVISION,
            gpu_memory_utilization=value,  # type: ignore[arg-type]
        )


def test_official_template_keeps_query_document_and_assistant_boundaries() -> None:
    assert reranking._prompt("question", "source") == (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n<Instruct>: Given a search query, retrieve relevant candidates "
        "that answer the query.<Query>:question\n<Document>:source<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize("fail", [False, True])
def test_vllm_batches_native_media_in_order_reuses_query_and_closes_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    calls: dict[str, object] = {}
    image = SimpleNamespace(close=lambda: closed.append("image"))
    frames = ("decoded-frames", {"fps": 4, "total_num_frames": 8})
    closed: list[str] = []
    fetched: list[tuple[str, object]] = []

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [1, 2, 3]

    class LLM:
        llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: closed.append("engine"))
        )

        def __init__(self, **kwargs: object) -> None:
            calls["load"] = kwargs

        def get_tokenizer(self) -> Tokenizer:
            return Tokenizer()

        def classify(self, prompts: object, **kwargs: object) -> list[object]:
            calls["classify"] = (prompts, kwargs)
            if fail:
                raise RuntimeError("private model response")
            return [SimpleNamespace(outputs=SimpleNamespace(probs=[v])) for v in (0.1, 0.9, 0.3)]

    def fetch_image(url: str) -> object:
        fetched.append(("image", url))
        return image

    def fetch_video(url: str, kwargs: dict[str, object]) -> object:
        fetched.append(("video", (url, kwargs)))
        return frames

    def importer(name: str) -> object:
        return {
            "vllm": SimpleNamespace(LLM=LLM),
            "vllm.multimodal.utils": SimpleNamespace(
                fetch_image=fetch_image, fetch_video=fetch_video
            ),
        }[name]

    monkeypatch.setattr(reranking, "import_module", importer)
    model = Qwen3VLReranker.load(
        "org/model", revision=REVISION, batch_size=2, gpu_memory_utilization=0.8
    )
    photo = AssetRef(
        id="photo",
        path=tmp_path / "image.png",
        media_type="image/png",
        modality=Modality.IMAGE,
        sha256="a" * 64,
        size_bytes=10,
    )
    video = AssetRef(
        id="video",
        path=tmp_path / "video.mp4",
        media_type="video/mp4",
        modality=Modality.VIDEO,
        sha256="b" * 64,
        size_bytes=10,
    )
    query = ModelInput(text="question", assets=(photo,))
    documents = (
        ModelInput(text="source"),
        ModelInput(assets=(video, photo)),
        ModelInput(text="中文"),
    )
    if fail:
        with pytest.raises(ModelError, match="reranking model failed"):
            model.rerank(query, documents)
    else:
        assert model.rerank(query, documents) == (0.1, 0.9, 0.3)
    marker = "<|vision_start|><|image_pad|><|vision_end|>"
    video_marker = "<|vision_start|><|video_pad|><|vision_end|>"
    assert calls["classify"] == (
        [
            {
                "prompt": reranking._prompt(marker + "question", "source"),
                "multi_modal_data": {"image": [image]},
            },
            {
                "prompt": reranking._prompt(marker + "question", video_marker + marker),
                "multi_modal_data": {"image": [image, image], "video": [frames]},
            },
            {
                "prompt": reranking._prompt(marker + "question", "中文"),
                "multi_modal_data": {"image": [image]},
            },
        ],
        {"use_tqdm": False},
    )
    assert fetched == [
        ("image", (tmp_path / "image.png").as_uri()),
        (
            "video",
            ((tmp_path / "video.mp4").as_uri(), {"num_frames": 8, "video_backend": "opencv"}),
        ),
    ]
    assert calls["load"] == {
        "model": "org/model",
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "runner": "pooling",
        "dtype": "auto",
        "trust_remote_code": False,
        "load_format": "safetensors",
        "hf_overrides": {
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "gpu_memory_utilization": 0.8,
        "limit_mm_per_prompt": {"image": 8, "video": 2},
        "mm_processor_kwargs": {"max_pixels": 262144, "do_sample_frames": False},
        "disable_log_stats": True,
    }
    assert closed == ["image"]
    model.close()
    model.close()
    assert closed == ["image", "engine"]


@pytest.mark.parametrize("probs", [[], [0.25, 0.75], [math.nan], [True]])
def test_vllm_rejects_invalid_classifier_outputs(
    monkeypatch: pytest.MonkeyPatch, probs: list[float]
) -> None:
    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3]),
        classify=lambda *args, **kwargs: [SimpleNamespace(outputs=SimpleNamespace(probs=probs))],
    )
    monkeypatch.setattr(
        reranking, "import_module", lambda name: SimpleNamespace(LLM=lambda **kwargs: llm)
    )
    model = Qwen3VLReranker.load("org/model", revision=REVISION)
    with pytest.raises(ModelError) as error:
        model.rerank(ModelInput(text="question"), (ModelInput(text="source"),))
    assert error.value.reason == "response_invalid"
    assert error.value.stage == "reranking"


def test_media_limit_applies_to_query_plus_document_and_never_drops_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    def classify(*args: object, **kwargs: object) -> object:
        pytest.fail("an over-limit pair must not reach inference")

    llm = SimpleNamespace(get_tokenizer=lambda: object(), classify=classify)
    loader = SimpleNamespace(
        fetch_image=lambda url: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(
        reranking,
        "import_module",
        lambda name: SimpleNamespace(LLM=lambda **kwargs: llm) if name == "vllm" else loader,
    )
    model = Qwen3VLReranker.load("org/model", revision=REVISION, image_limit=1)
    image = AssetRef(
        "image",
        modality=Modality.IMAGE,
        media_type="image/png",
        size_bytes=1,
        sha256="a" * 64,
        path=tmp_path / "image.png",
    )
    value = ModelInput(assets=(image,))
    with pytest.raises(ModelError) as error:
        model.rerank(value, (value,))
    assert error.value.reason == "payload_too_large"
    assert error.value.stage == "reranking"
    assert closed == [True]
