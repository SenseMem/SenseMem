"""Pinned native Qwen3-VL yes/no reranking of query and source-text windows."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.reranking import reranking_scores
from mindbridge.models.sentence_transformers import (
    _optional_text,
    _positive_integer,
    _revision,
    _text,
)

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
_RECIPE = "qwen3-vl-text-yes-no-float32-sigmoid-v1"
_SUFFIX_TOKENS = 5


class _ScoringEngine(Protocol):
    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object: ...

    def close(self) -> None: ...


def _messages(query: str, document: str) -> list[dict[str, object]]:
    # This is the publisher's text-only conversation, with no media or benchmark-specific prompt.
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<Instruct>: " + _INSTRUCTION},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "text", "text": document or "NULL"},
            ],
        },
    ]


def _truncate_tokens(tokens: list[int], max_length: int, special_ids: set[int]) -> list[int]:
    if len(tokens) <= max_length:
        return tokens
    prefix, suffix = tokens[:-_SUFFIX_TOKENS], tokens[-_SUFFIX_TOKENS:]
    # The publisher preserves the five-token generation suffix; it counts inside our hard limit.
    ordinary_budget = max_length - len(suffix) - sum(token in special_ids for token in prefix)
    if ordinary_budget < 0:
        raise ModelError(
            "reranking input exceeds its special-token budget",
            reason="payload_too_large",
            stage="reranking",
        )
    kept = []
    for token in prefix:
        if token in special_ids:
            kept.append(token)
        elif ordinary_budget:
            kept.append(token)
            ordinary_budget -= 1
    return kept + suffix


class _NativeScoringEngine:
    def __init__(self, model_id: str, revision: str, device: str | None) -> None:
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            factory = transformers.Qwen3VLForConditionalGeneration
        except (AttributeError, ImportError):
            raise ModelError(
                "native Qwen3-VL is unavailable; install MindBridge with the local extra",
                stage="reranking",
            ) from None
        self._torch = torch
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float32
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._processor = cast(
            "Qwen3VLProcessor",
            transformers.AutoProcessor.from_pretrained(
                model_id, revision=revision, padding_side="left", trust_remote_code=False
            ),
        )
        vocabulary = self._processor.tokenizer.get_vocab()
        yes, no = vocabulary["yes"], vocabulary["no"]
        lm = cast(
            "Qwen3VLForConditionalGeneration",
            factory.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=False,
                use_safetensors=True,
            ),
        )
        # Retain only the native backbone and the trained yes/no rows, not full vocabulary logits.
        self._weight = (lm.lm_head.weight[yes] - lm.lm_head.weight[no]).detach().unsqueeze(0)
        self._model = lm.model
        del lm
        self._model.to(self._device).eval()
        self._weight = self._weight.to(self._device)

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object:
        torch = self._torch
        scores: list[float] = []
        special_ids = set(self._processor.tokenizer.all_special_ids)
        for start in range(0, len(documents), batch_size):
            pairs = [
                _messages(query, document) for document in documents[start : start + batch_size]
            ]
            # The native annotation omits its documented multimodal content-block format.
            text = self._processor.apply_chat_template(
                pairs,  # type: ignore[arg-type]
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self._processor(text=text, truncation=False, padding=False)
            ids = cast(list[list[int]], encoded["input_ids"])
            trimmed = [_truncate_tokens(tokens, max_length, special_ids) for tokens in ids]
            padded = self._processor.tokenizer.pad(
                {"input_ids": trimmed}, padding=True, return_tensors="pt"
            )
            inputs = {
                name: cast("Tensor", value).to(self._device) for name, value in padded.items()
            }
            with torch.inference_mode():
                hidden = self._model(**inputs, use_cache=False).last_hidden_state
                mask = inputs["attention_mask"]
                positions = torch.arange(mask.shape[1], device=self._device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                selected = hidden[torch.arange(mask.shape[0], device=self._device), last]
                logits = torch.nn.functional.linear(selected, self._weight).squeeze(-1)
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return scores

    def close(self) -> None:
        del self._model
        del self._weight
        del self._processor


class Qwen3VLReranker:
    """Use the native trained yes/no head on text windows, retaining caller-owned positions.

    Scores are float32 sigmoid transformations of query-local yes/no log odds. They are neither
    answer confidence nor a replacement for the retrieval relevance gate.
    """

    def __init__(
        self,
        engine: _ScoringEngine | None = None,
        *,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 1,
        max_length: int = 2048,
    ) -> None:
        self._model_id = _text(model_id, "model_id")
        self._device = _optional_text(device, "device")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._max_length = _positive_integer(max_length, "max_length")
        if self._max_length <= _SUFFIX_TOKENS:
            raise ValidationError("max_length must exceed the five-token generation suffix")
        if revision is None:
            try:
                revision = import_module("huggingface_hub").model_info(self._model_id).sha
            except Exception:
                raise ModelError(
                    "failed to resolve an immutable reranking model revision", stage="reranking"
                ) from None
        self._revision = _revision(revision)
        if engine is None:
            try:
                engine = _NativeScoringEngine(self._model_id, self._revision, self._device)
            except ModelError:
                raise
            except Exception:
                raise ModelError("failed to load the reranking model", stage="reranking") from None
        self._engine: _ScoringEngine | None = engine
        self._lock = threading.RLock()

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 1,
        max_length: int = 2048,
    ) -> Qwen3VLReranker:
        return cls(
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

    @property
    def reranking_model(self) -> str:
        recipe = hashlib.sha256((_RECIPE + _SYSTEM + _INSTRUCTION).encode()).hexdigest()[:16]
        return f"{self._model_id}@{self._revision}:{recipe}:max_length={self._max_length}"

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        query = _text(query, "query")
        if isinstance(documents, (str, bytes)):
            raise ValidationError("documents must be a sequence of text values")
        try:
            batch = tuple(documents)
        except TypeError:
            raise ValidationError("documents must be a sequence of text values") from None
        if any(not isinstance(document, str) for document in batch):
            raise ValidationError("documents must be a sequence of text values")
        with self._lock:
            if self._engine is None:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            requests = (len(batch) + self._batch_size - 1) // self._batch_size
            mark_model_requests(requests, token_usage_expected=0)
            try:
                values = self._engine.score(
                    query, batch, batch_size=self._batch_size, max_length=self._max_length
                )
            except ModelError as error:
                raise ModelError(
                    "reranking model failed", reason=error.reason, stage="reranking"
                ) from None
            except Exception:
                raise ModelError(
                    "reranking model failed", reason="model_failed", stage="reranking"
                ) from None
            scores = reranking_scores(values, len(batch))
            if any(not 0 <= value <= 1 for value in scores):
                raise ModelError(
                    "reranking model returned invalid probabilities",
                    reason="response_invalid",
                    stage="reranking",
                )
            record_unmetered_model_usage(request_count=requests)
            return scores

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            if engine is None:
                return
            self._engine = None
            try:
                engine.close()
            except Exception:
                raise ModelError("failed to close reranking model", stage="reranking") from None
