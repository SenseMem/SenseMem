"""Reject oversized reranking pairs without silently removing source content."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import sensemem.models.qwen3_vl_reranking as reranking
from sensemem import AssetRef, Modality, ModelInput, Qwen3VLReranker
from sensemem.exceptions import ModelError

REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        # Deterministic tokenization counts the entire rendered template and its contents.
        return list(text.encode("utf-8"))


@pytest.mark.parametrize(
    ("query", "documents", "max_length"),
    [
        ("long query " * 400, ("cancelled", "confirmed"), 2048),
        ("question", ("short source", "long document " * 400), 2048),
        ("question", ("source",), 12),
    ],
    ids=["oversized-query", "later-oversized-document", "template-does-not-fit"],
)
def test_oversized_pairs_fail_before_any_inference(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    documents: tuple[str, ...],
    max_length: int,
) -> None:
    def classify(*_args: object, **_kwargs: object) -> object:
        pytest.fail("no pair may reach inference when another pair exceeds its token budget")

    llm = SimpleNamespace(get_tokenizer=Tokenizer, classify=classify)
    monkeypatch.setattr(
        reranking, "import_module", lambda name: SimpleNamespace(LLM=lambda **kwargs: llm)
    )
    model = Qwen3VLReranker.load(
        "org/model", revision=REVISION, batch_size=1, max_length=max_length
    )
    with pytest.raises(ModelError) as error:
        model.rerank(ModelInput(text=query), tuple(ModelInput(text=text) for text in documents))
    assert error.value.reason == "payload_too_large"
    assert error.value.stage == "reranking"
    assert str(error.value) == "reranking model failed"


def test_pair_at_context_limit_keeps_query_document_and_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Which booking was cancelled?"
    documents = ("Confirmed", "Cancelled 中文")
    prompts = [reranking._prompt(query, document) for document in documents]
    submitted: list[object] = []

    def classify(rows: object, *, use_tqdm: bool) -> list[object]:
        assert not use_tqdm
        submitted.append(rows)
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[score])) for score in (0.1, 0.9)]

    llm = SimpleNamespace(get_tokenizer=Tokenizer, classify=classify)
    monkeypatch.setattr(
        reranking, "import_module", lambda name: SimpleNamespace(LLM=lambda **kwargs: llm)
    )
    model = Qwen3VLReranker.load(
        "org/model",
        revision=REVISION,
        batch_size=1,
        max_length=max(len(prompt.encode("utf-8")) for prompt in prompts),
    )
    assert model.rerank(
        ModelInput(text=query), tuple(ModelInput(text=text) for text in documents)
    ) == (0.1, 0.9)
    assert submitted == [[{"prompt": prompt} for prompt in prompts]]


@pytest.mark.parametrize("modality", [Modality.IMAGE, Modality.VIDEO])
@pytest.mark.parametrize(
    ("exception_type", "message", "reason"),
    [
        (
            ValueError,
            "The decoder prompt (length 9000) is longer than the maximum model length of 8192. "
            "Make sure that `max_model_len` is no smaller than the number of text tokens "
            "plus multimodal tokens.",
            "payload_too_large",
        ),
        (ValueError, "private invalid video or model configuration", "model_failed"),
        (
            RuntimeError,
            "The decoder prompt (length 9000) is longer than the maximum model length of 8192.",
            "model_failed",
        ),
        (
            ValueError,
            "private source quotes: The decoder prompt (length 9000) is longer than the maximum "
            "model length of 8192.",
            "model_failed",
        ),
    ],
    ids=["expanded-context-overflow", "unrelated-value-error", "runtime-error", "quoted-message"],
)
def test_visual_context_errors_are_classified_without_dropping_media_or_retrying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modality: Modality,
    exception_type: type[Exception],
    message: str,
    reason: str,
) -> None:
    closed: list[bool] = []
    media = (
        SimpleNamespace(close=lambda: closed.append(True))
        if modality is Modality.IMAGE
        else ("frames", {"fps": 4, "total_num_frames": 8, "frames_indices": list(range(8))})
    )
    submitted: list[object] = []

    def classify(prompts: object, *, use_tqdm: bool) -> list[object]:
        assert not use_tqdm
        submitted.append(prompts)
        raise exception_type(message)

    llm = SimpleNamespace(get_tokenizer=Tokenizer, classify=classify)
    loader = SimpleNamespace(fetch_image=lambda url: media, fetch_video=lambda url, kwargs: media)
    monkeypatch.setattr(
        reranking,
        "import_module",
        lambda name: SimpleNamespace(LLM=lambda **kwargs: llm) if name == "vllm" else loader,
    )
    model = Qwen3VLReranker.load("org/model", revision=REVISION)
    asset = AssetRef(
        "source",
        path=tmp_path / "source",
        modality=modality,
        media_type="image/png" if modality is Modality.IMAGE else "video/mp4",
        sha256="a" * 64,
        size_bytes=1,
    )
    with pytest.raises(ModelError) as error:
        model.rerank(ModelInput(text="private query"), (ModelInput(assets=(asset,)),))
    assert error.value.reason == reason
    assert error.value.stage == "reranking"
    assert str(error.value) == "reranking model failed"
    assert error.value.__cause__ is None
    marker = f"<|vision_start|><|{modality.value}_pad|><|vision_end|>"
    assert submitted == [
        [
            {
                "prompt": reranking._prompt("private query", marker),
                "multi_modal_data": {modality.value: [media]},
            }
        ]
    ]
    assert closed == ([True] if modality is Modality.IMAGE else [])
