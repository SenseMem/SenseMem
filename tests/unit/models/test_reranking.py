"""The optional cross-encoder preserves pair identity and rejects invalid output."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mindbridge.models.reranking as reranking
from mindbridge import Modality, ModelInput, RerankingBackend
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.reranking import SentenceTransformersReranker

REVISION = "a" * 40


class Scores:
    def __init__(self, values: object) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class Encoder:
    def __init__(self, values: object) -> None:
        self.values = values
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []
        self.closed = 0

    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> Scores:
        self.calls.append((inputs, kwargs))
        return Scores(self.values)

    def close(self) -> None:
        self.closed += 1


def test_reranker_preserves_pair_order_and_raw_logits() -> None:
    encoder = Encoder([-5.25, 8.5])
    model = SentenceTransformersReranker(
        encoder, model_id="org/cross-encoder", revision=REVISION, batch_size=7
    )
    assert isinstance(model, RerankingBackend)
    assert model.reranking_capabilities == frozenset({Modality.TEXT})
    assert model.reranking_model == f"org/cross-encoder@{REVISION}"
    assert model.rerank(
        ModelInput(text="Which booking was cancelled?"),
        (ModelInput(text="Confirmed booking."), ModelInput(text="Cancelled booking.")),
    ) == (
        -5.25,
        8.5,
    )
    pairs, kwargs = encoder.calls[0]
    assert pairs == [
        ("Which booking was cancelled?", "Confirmed booking."),
        ("Which booking was cancelled?", "Cancelled booking."),
    ]
    assert kwargs["batch_size"] == 7
    assert kwargs["apply_softmax"] is False
    assert kwargs["show_progress_bar"] is False
    activation = kwargs["activation_fn"]
    assert callable(activation) and activation(-5.25) == -5.25
    assert model.rerank(ModelInput(text="query"), ()) == ()
    assert len(encoder.calls) == 1
    model.close()
    model.close()
    assert encoder.closed == 1
    with pytest.raises(ModelError, match="closed"):
        model.rerank(ModelInput(text="query"), (ModelInput(text="document"),))


@pytest.mark.parametrize("values", [[], [True], [[0.2]], [float("nan")], [float("inf")]])
def test_reranker_rejects_incomplete_or_non_scalar_scores(values: object) -> None:
    model = SentenceTransformersReranker(Encoder(values), model_id="org/model", revision=REVISION)
    with pytest.raises(ModelError) as raised:
        model.rerank(ModelInput(text="query"), (ModelInput(text="document"),))
    assert raised.value.reason == "response_invalid"
    assert raised.value.stage == "reranking"


def test_load_resolves_revision_and_uses_safe_crossencoder_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []
    arguments: dict[str, object] = {}

    def factory(name: str, **kwargs: object) -> Encoder:
        arguments.update(model=name, **kwargs)
        return Encoder([0.5])

    def importer(name: str) -> object:
        imported.append(name)
        if name == "huggingface_hub":
            return SimpleNamespace(model_info=lambda _name: SimpleNamespace(sha=REVISION))
        assert name == "sentence_transformers"
        return SimpleNamespace(CrossEncoder=factory)

    monkeypatch.setattr(reranking, "import_module", importer)
    model = SentenceTransformersReranker.load("org/model", device="cpu", max_length=512)
    assert imported == ["huggingface_hub", "sentence_transformers"]
    assert arguments == {
        "model": "org/model",
        "revision": REVISION,
        "device": "cpu",
        "max_length": 512,
        "trust_remote_code": False,
    }
    assert model.rerank(ModelInput(text="query"), (ModelInput(text="document"),)) == (0.5,)


def test_reranker_sanitizes_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = Encoder([0.0])

    def failed(*_args: object, **_kwargs: object) -> Scores:
        raise RuntimeError("sensitive source text in provider response")

    monkeypatch.setattr(encoder, "predict", failed)
    model = SentenceTransformersReranker(encoder, model_id="org/model", revision=REVISION)
    with pytest.raises(ModelError) as raised:
        model.rerank(ModelInput(text="private question"), (ModelInput(text="private source"),))
    assert str(raised.value) == "reranking model failed"
    assert raised.value.reason == "model_failed"
    assert raised.value.stage == "reranking"


def test_reranker_rejects_mutable_revision_and_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match="immutable"):
        SentenceTransformersReranker(Encoder([]), model_id="org/model", revision="main")

    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(reranking, "import_module", missing)
    with pytest.raises(ModelError, match="local extra"):
        SentenceTransformersReranker.load("org/model", revision=REVISION)
