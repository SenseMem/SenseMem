"""Native Qwen3-VL reranking keeps source positions and uses the trained binary head."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import mindbridge.models.qwen3_vl_reranking as reranking
from mindbridge import RerankingBackend
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.qwen3_vl_reranking import Qwen3VLReranker

REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"


class Engine:
    def __init__(self, values: object) -> None:
        self.values = values
        self.calls: list[tuple[str, tuple[str, ...], int, int]] = []
        self.closed = 0

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object:
        self.calls.append((query, tuple(documents), batch_size, max_length))
        return self.values

    def close(self) -> None:
        self.closed += 1


def test_injected_engine_preserves_positions_settings_and_lifecycle() -> None:
    engine = Engine([0.125, 0.875])
    model = Qwen3VLReranker(
        engine, model_id="Qwen/Qwen3-VL-Reranker-8B", revision=REVISION, batch_size=2
    )
    assert isinstance(model, RerankingBackend)
    assert f"Qwen/Qwen3-VL-Reranker-8B@{REVISION}:" in model.reranking_model
    assert "max_length=2048" in model.reranking_model
    assert model.rerank("Which booking was cancelled?", ("Confirmed", "Cancelled")) == (
        0.125,
        0.875,
    )
    assert engine.calls == [("Which booking was cancelled?", ("Confirmed", "Cancelled"), 2, 2048)]
    assert model.rerank("question", ()) == ()
    assert len(engine.calls) == 1
    model.close()
    model.close()
    assert engine.closed == 1
    with pytest.raises(ModelError, match="closed"):
        model.rerank("question", ("source",))


@pytest.mark.parametrize("values", [[], [True], [[0.5]], [math.nan], [math.inf], [-0.1], [1.1]])
def test_invalid_probability_output_is_rejected(values: object) -> None:
    model = Qwen3VLReranker(Engine(values), model_id="org/model", revision=REVISION)
    with pytest.raises(ModelError) as error:
        model.rerank("question", ("source",))
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
        model.rerank("private query", ("private source",))
    assert str(error.value) == "reranking model failed"
    assert error.value.reason == "model_failed"


def test_validation_and_missing_native_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="immutable"):
        Qwen3VLReranker(Engine([]), model_id="org/model", revision="main")
    with pytest.raises(ValidationError, match="suffix"):
        Qwen3VLReranker(Engine([]), model_id="org/model", revision=REVISION, max_length=5)

    def missing(_name: str) -> object:
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(reranking, "import_module", missing)
    with pytest.raises(ModelError, match="local extra"):
        Qwen3VLReranker.load("org/model", revision=REVISION)


def test_official_text_messages_and_truncation_preserve_generation_suffix() -> None:
    messages = reranking._messages("question", "source")
    assert messages[0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "Judge whether the Document meets the requirements based on the Query and the Instruct "
                    'provided. Note that the answer can only be "yes" or "no".'
                ),
            }
        ],
    }
    assert messages[1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "<Instruct>: Given a search query, retrieve relevant candidates that answer the query.",
            },
            {"type": "text", "text": "<Query>:"},
            {"type": "text", "text": "question"},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "text", "text": "source"},
        ],
    }
    tokens = [101, *range(10, 30), 102, 201, 202, 203, 204, 205]
    trimmed = reranking._truncate_tokens(tokens, 12, {101, 102})
    assert len(trimmed) == 12
    assert trimmed[-5:] == tokens[-5:]
    assert trimmed[:7] == [101, 10, 11, 12, 13, 14, 102]
    assert reranking._truncate_tokens(tokens[:4], 12, {101, 102}) == tokens[:4]
    with pytest.raises(ModelError) as error:
        reranking._truncate_tokens([101] * 20, 12, {101})
    assert error.value.reason == "payload_too_large"


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_native_loader_uses_binary_weight_difference_and_last_nonpadding_hidden(  # noqa: C901 - one simulated native loader boundary
    monkeypatch: pytest.MonkeyPatch, padding_side: str
) -> None:
    torch = pytest.importorskip("torch")
    calls: dict[str, object] = {}

    class Tokenizer:
        all_special_ids = (0,)

        def get_vocab(self) -> dict[str, int]:
            return {"yes": 0, "no": 1}

        def pad(self, _ids: object, **kwargs: object) -> dict[str, object]:
            calls["pad"] = kwargs
            return {
                "input_ids": torch.tensor(
                    [[0, 3, 4], [4, 5, 6]] if padding_side == "left" else [[3, 4, 0], [4, 5, 6]]
                ),
                "attention_mask": torch.tensor(
                    [[0, 1, 1], [1, 1, 1]] if padding_side == "left" else [[1, 1, 0], [1, 1, 1]]
                ),
            }

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, pairs: object, **kwargs: object) -> list[str]:
            calls["template"] = (pairs, kwargs)
            return ["short source", "longer source"]

        def __call__(self, **kwargs: object) -> dict[str, list[list[int]]]:
            calls["processor"] = kwargs
            return {"input_ids": [[3, 4], [4, 5, 6]]}

    class Backbone:
        def to(self, _device: object) -> Backbone:
            return self

        def eval(self) -> Backbone:
            return self

        def __call__(self, **kwargs: object) -> object:
            ids = kwargs["input_ids"]
            assert isinstance(ids, torch.Tensor)
            calls["forward_use_cache"] = kwargs["use_cache"]
            return SimpleNamespace(
                last_hidden_state=torch.stack((ids.float(), torch.zeros_like(ids).float()), dim=-1)
            )

    def load_processor(model_id: str, **kwargs: object) -> Processor:
        calls["processor_load"] = (model_id, kwargs)
        return Processor()

    def load_model(model_id: str, **kwargs: object) -> object:
        calls["model_load"] = (model_id, kwargs)
        return SimpleNamespace(
            model=Backbone(),
            lm_head=SimpleNamespace(weight=torch.tensor([[0.25, 0.0], [0.0, 0.25]])),
        )

    def importer(name: str) -> object:
        if name == "torch":
            return torch
        assert name == "transformers"
        return SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        )

    monkeypatch.setattr(reranking, "import_module", importer)
    model = Qwen3VLReranker.load("org/model", revision=REVISION, device="cpu", batch_size=2)
    assert model.rerank("question", ("short source", "longer source")) == pytest.approx(
        [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(-1.5))]
    )
    assert calls["processor_load"] == (
        "org/model",
        {"revision": REVISION, "padding_side": "left", "trust_remote_code": False},
    )
    assert calls["model_load"] == (
        "org/model",
        {
            "revision": REVISION,
            "dtype": torch.float32,
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
            "use_safetensors": True,
        },
    )
    assert calls["forward_use_cache"] is False
    model.close()
