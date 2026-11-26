"""Opt-in native image ordering gate through ingestion, retrieval, and grounded answering."""

import os
from io import BytesIO
from pathlib import Path

import pytest
from openai import OpenAI
from PIL import Image

from sensemem import Blob, Memory, Modality
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("colors", [("red", "blue"), ("blue", "red")])
@pytest.mark.skipif(
    os.environ.get("SENSEMEM_ORDERED_CONTENT_EVAL") != "1",
    reason="requires SENSEMEM_ORDERED_CONTENT_EVAL=1 and configured model services",
)
def test_native_image_order_quality(tmp_path: Path, colors: tuple[str, str], repeat: int) -> None:
    images = []
    for color in colors:
        stream = BytesIO()
        Image.new("RGB", (48, 48), color).save(stream, format="PNG")
        images.append(Blob(stream.getvalue(), "image/png", f"{color}.png"))
    with (
        OpenAI(timeout=120, max_retries=0) as generation,
        OpenAI(
            base_url=os.environ["SENSEMEM_QUALITY_EMBEDDING_BASE_URL"],
            api_key="EMPTY",
            timeout=120,
            max_retries=0,
        ) as embedding,
        Memory(
            tmp_path,
            embedder=OpenAIEmbedder(
                client=embedding,
                embedding_model="google/embeddinggemma-2",
                embedding_dimension=768,
                embedding_request_format="messages",
                embedding_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
            answerer=OpenAICompletion(
                client=generation,
                generation_model=os.environ["SENSEMEM_QUALITY_MODEL"],
                generation_max_tokens=None,
                generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
        ) as memory,
    ):
        memory.add(("Before repair:", images[0], "After repair:", images[1]))
        answer = memory.ask(
            "Which color is in the AFTER repair photo? Reply with only that color word."
        )
        assert not answer.abstained
        assert answer.answer.strip().strip(".*_` ").casefold() == colors[1]
