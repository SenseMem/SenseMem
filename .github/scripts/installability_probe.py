"""Prove one installed optional surface works, not merely that it imports.

Every heavy dependency in this package sits behind ``import_module`` inside a function so the
base import stays cold. That discipline hides missing declarations: importing
``sensemem.models.funasr`` succeeds without touching one ``funasr`` symbol, so an import-only
CI probe cannot tell an installable extra from a broken one. Each leg below therefore imports the
supported modules and then runs the loader those modules defer to.

Usage: ``python installability_probe.py <leg>`` inside a venv holding only that leg's extra.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path


class _StubEmbedder:
    """Smallest object satisfying ``EmbeddingBackend``.

    Every leg needs an embedding backend and no leg may pull a model in to get one, so this is
    also a standing check that the protocol stays small enough to implement by hand.
    """

    embedding_model = "installability-probe"
    embedding_space = "installability-probe:2:v1"
    embedding_dimension = 2

    def __init__(self) -> None:
        from sensemem.types import Modality

        self.embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self,
        inputs: Sequence[object],
        task: object = None,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        return None


def _memory_flow(tracer: object = None) -> None:
    """Write and read one record end to end: SQLite, the outbox, and the Zvec loader."""
    from sensemem import Memory

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder(), tracer=tracer) as memory,
    ):
        record = memory.add("the installability probe wrote this record")
        assert memory.get(record.id) is not None
        assert memory.search("installability probe", limit=1)
        assert memory.list(limit=1).items
        memory.reindex()
        assert memory.search("installability probe", limit=1)
        assert memory.delete(record.id)


def _base() -> None:
    _memory_flow()
    _product_cli()


def _product_cli() -> None:
    """Run the `sensemem` console script's entry point on the base dependency set.

    The script is declared whether or not its module imports, and every composition path except
    `--url` needs an extra, so the cheapest honest end-to-end check is a resolution that touches
    argparse, `sensemem.recipes`, and the JSON writer without constructing a backend.
    """
    from sensemem.cli import main

    assert main(["--url", "http://127.0.0.1:1", "-q", "--explain", "doctor"]) == 0


def _benchmarks() -> None:
    """Reach the download client and text scorers the catalog tasks load lazily."""
    from sensemem.benchmarks.cli import main

    assert main(["--help"]) == 0
    importlib.import_module("httpx")
    importlib.import_module("nltk")
    from sensemem.benchmarks._official.mosaic_scoring import parse_judge_score

    assert parse_judge_score("{score: 0.7,}") == (0.7, True)


def _observability() -> None:
    """Export real operation spans through the SDK this extra exists to supply."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from sensemem._telemetry import TRACER_NAME

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _memory_flow(provider.get_tracer(TRACER_NAME))
    provider.shutdown()
    assert {"sensemem.add", "sensemem.search"} <= {
        span.name for span in exporter.get_finished_spans()
    }


def _openai() -> None:
    """Build the adapter over a real SDK client without issuing a request."""
    import av
    from openai import OpenAI
    from PIL import Image

    from sensemem import OpenAIModels

    client = OpenAI(api_key="installability-probe", base_url="http://127.0.0.1:1/v1")
    models = OpenAIModels(client)
    assert callable(av.open) and Image.Image is not None
    assert models.embedding_dimension > 0
    models.close()


def _face() -> None:
    """Load the OpenCV APIs used by the face backend without downloading model weights."""
    cv = importlib.import_module("cv2")
    assert all(hasattr(cv, name) for name in ("FaceDetectorYN", "FaceRecognizerSF", "VideoCapture"))


def _local() -> None:
    """Load every third-party module the local backends resolve at first use.

    ``import funasr.utils.load_utils`` is not enough: since 1.4.7 FunASR guards ``torchaudio``
    behind ``funasr.utils.torchaudio_compat`` and falls back for decode and resampling, so that
    import succeeds with torchaudio absent. The one operation with no fallback is
    ``torchaudio.functional.forced_align``, which Fun-ASR-Nano -- the default recipe -- calls for
    its native timestamps, so assert that symbol instead.
    """
    import torchaudio.functional

    for name in ("funasr", "librosa", "sentence_transformers", "soundfile", "torch"):
        importlib.import_module(name)
    assert hasattr(torchaudio.functional, "forced_align")


def _server() -> None:
    """Build the real ASGI application over a real memory instance."""
    from sensemem import Memory
    from sensemem.api import create_app

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder()) as memory,
    ):
        app = create_app(memory=memory)
    # Generating the schema resolves every route's request and response model. Route objects are
    # not the assertion: a floating FastAPI can nest the /v1 router where `.path` does not reach.
    assert set(app.openapi()["paths"]) == {
        "/healthz",
        "/v1/answers",
        "/v1/capture",
        "/v1/answers/stream",
        "/v1/context",
        "/v1/memories",
        "/v1/memories/batch",
        "/v1/memories/reinforce",
        "/v1/memories/search",
        "/v1/memories/{memory_id}",
        "/v1/pending_captures",
        "/v1/settle",
    }


def _mcp() -> None:
    """Build the real MCP server and resolve the thirteen supported tool schemas."""
    import asyncio

    from sensemem import Memory
    from sensemem.api.mcp import build_mcp_server

    with (
        tempfile.TemporaryDirectory() as directory,
        Memory(Path(directory) / "store", embedder=_StubEmbedder()) as memory,
    ):
        server = build_mcp_server(memory)
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "add_memory",
        "ask_memory",
        "compile_context",
        "delete_memory",
        "forget_identity",
        "get_identity",
        "get_memory",
        "list_memories",
        "perceive_memory",
        "register_identity",
        "reinforce_memories",
        "search_memories",
        "unlink_identity",
    }


MODULES: dict[str, tuple[str, ...]] = {
    "base": (
        "sensemem",
        "sensemem.benchmarks",
        "sensemem.cli",
        "sensemem.benchmarks.isolation",
        "sensemem.benchmarks.local_index_benchmark",
        "sensemem.exceptions",
        "sensemem.infrastructure.local",
        "sensemem.memory",
        "sensemem.models.base",
        "sensemem.models.openai_sdk",
        "sensemem.recipes",
        "sensemem.types",
    ),
    "benchmarks": (
        "sensemem.benchmarks.cli",
        "sensemem.benchmarks.download",
        "sensemem.benchmarks.eval",
        "sensemem.benchmarks.official_scorers",
        "sensemem.benchmarks.prepare_media",
        "sensemem.benchmarks.task_catalog",
    ),
    "observability": ("sensemem", "sensemem._telemetry"),
    "openai": ("sensemem", "sensemem.models.openai_sdk"),
    "face": ("sensemem", "sensemem.models.opencv_face"),
    "local": (
        "sensemem.models.funasr",
        "sensemem.models.sentence_transformers",
    ),
    "server": ("sensemem.api", "sensemem.api.app"),
    "mcp": ("sensemem.api.mcp",),
}

LOADERS: dict[str, Callable[[], None]] = {
    "base": _base,
    "benchmarks": _benchmarks,
    "observability": _observability,
    "openai": _openai,
    "face": _face,
    "local": _local,
    "server": _server,
    "mcp": _mcp,
}


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1 or argv[0] not in LOADERS:
        print(f"usage: installability_probe.py {{{','.join(LOADERS)}}}", file=sys.stderr)
        return 2
    leg = argv[0]
    modules = MODULES[leg]
    for module in modules:
        importlib.import_module(module)
    LOADERS[leg]()
    print(f"{leg}: imported {len(modules)} modules and ran the loader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
