"""Native image/video rank discrimination and memory qualification, without ATM questions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mindbridge import AssetRef, MindBridgeConfig, Modality, ModelInput, Qwen3VLReranker
from mindbridge.benchmarks.eval import _borrow_reranker
from mindbridge.models.base import MultimodalRerankingBackend


def run_probe(config_path: Path, directory: Path) -> int:
    import av
    import torch
    from PIL import Image, ImageDraw

    payload = yaml.safe_load(config_path.read_text())
    payload.pop("benchmark")
    config = MindBridgeConfig.model_validate(payload).reranking
    if config is None or config.provider != "qwen3-vl" or not config.multimodal:
        raise ValueError("probe requires native multimodal Qwen reranking")
    fixture_root = directory / "fixtures"
    fixture_root.mkdir(exist_ok=True)
    images = {}
    for color in ("red", "blue"):
        image = Image.new("RGB", (384, 384), "white")
        ImageDraw.Draw(image).rectangle((96, 96, 288, 288), fill=color)
        images[color] = image
        image.save(fixture_root / (color + ".png"))
    for name, colors in (
        ("red", ["red"] * 8),
        ("blue", ["blue"] * 8),
        ("red-blue", ["red"] * 4 + ["blue"] * 4),
        ("blue-red", ["blue"] * 4 + ["red"] * 4),
    ):
        with av.open(str(fixture_root / (name + ".mp4")), mode="w") as container:
            stream = container.add_stream("libx264", rate=2)
            stream.width = stream.height = 384
            stream.pix_fmt = "yuv420p"
            for color in colors:
                for packet in stream.encode(av.VideoFrame.from_image(images[color])):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def document(name: str, kind: Modality) -> ModelInput:
        path = fixture_root / name
        return ModelInput(
            text="Visual evidence.",
            assets=(
                AssetRef(
                    id=name,
                    path=path,
                    modality=kind,
                    media_type="image/png" if kind is Modality.IMAGE else "video/mp4",
                    size_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ),
            ),
        )

    cases = (
        (
            "image-color",
            "An image showing a large red square on a white background.",
            document("red.png", Modality.IMAGE),
            document("blue.png", Modality.IMAGE),
        ),
        (
            "video-color",
            "A video showing a red square on a white background.",
            document("red.mp4", Modality.VIDEO),
            document("blue.mp4", Modality.VIDEO),
        ),
        (
            "video-order",
            "A video showing a red square first, followed by a blue square.",
            document("red-blue.mp4", Modality.VIDEO),
            document("blue-red.mp4", Modality.VIDEO),
        ),
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("multimodal-reranking-probe")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_id=config.model,
        revision=config.revision,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
        multimodal=True,
        media_candidates=config.media_candidates,
        media_weight=config.media_weight,
        media_max_length=config.media_max_length,
        image_max_pixels=config.image_max_pixels,
        video_max_pixels=config.video_max_pixels,
        video_max_frames=config.video_max_frames,
        video_fps=config.video_fps,
    )
    borrowed = _borrow_reranker(model)
    assert isinstance(borrowed, MultimodalRerankingBackend)
    results = []
    try:
        for name, query, positive, negative in cases:
            begin = time.perf_counter()
            with tracer.start_as_current_span(name):
                scores = borrowed.rerank_media(ModelInput(text=query), (positive, negative))
            results.append(
                {
                    "case": name,
                    "positive_score": scores[0],
                    "negative_score": scores[1],
                    "passed": scores[0] > scores[1],
                    "seconds": time.perf_counter() - begin,
                }
            )
        identity = model.reranking_model
        peak = torch.cuda.max_memory_reserved()
    finally:
        model.close()
    spans = [
        {
            "name": span.name,
            "numeric_evidence": dict(span.attributes or {}),
        }
        for span in exporter.get_finished_spans()
    ]
    pixels = {
        name: sum(
            int((span.attributes or {}).get("mindbridge.reranking." + name, 0))
            for span in exporter.get_finished_spans()
        )
        for name in ("image_pairs", "video_pairs", "video_frames", "media_forward_count")
    }
    assert pixels["image_pairs"] == 2 and pixels["video_pairs"] == 4
    assert pixels["video_frames"] >= 16 and pixels["media_forward_count"] == 6
    report = {
        "model": identity,
        "dispatch": "benchmark pool _borrow_reranker -> optional media protocol -> native adapter",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "results": results,
        "native_pixels": pixels,
        "spans": spans,
        "peak_reserved_bytes": peak,
        "wall_seconds": time.perf_counter() - started,
        "fixtures_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(fixture_root.iterdir())
        },
        "pass_criteria": "All three positive candidates outrank negatives; native image/video tensors and temporal frames present; no OOM.",
        "passed": all(row["passed"] for row in results),
        "not_an_ATM_score": True,
    }
    (directory / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1
