"""Bounded synthetic public-SDK evaluation of investigation and grounded answer contracts.

The controlled embedder isolates planning/reading; this does not evaluate a deployed embedder.
No benchmark questions, reference answers, or source IDs are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import httpx2 as httpx
import yaml
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mindbridge
from mindbridge import (
    AnswerResult,
    EmbedTask,
    Memory,
    MemoryConfig,
    MemoryPlugins,
    MindBridgeConfig,
    MindBridgeError,
    Modality,
    ModelInput,
    OfflineGazetteer,
    OpenAIModels,
    RetrievalMode,
)

FIXTURE_VERSION = "evidence-investigation-quality-v1"
REFERENCE_AT = datetime(2044, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Document:
    source_id: str
    text: str
    gps: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    split: Literal["development", "holdout"]
    documents: tuple[Document, ...]
    question: str
    expected: str
    kind: Literal["text", "ids", "date", "datetime", "amount", "abstention"] = "text"
    limit: int = 10


CASES = (
    Case(
        "bridge_delivery",
        "development",
        (
            Document(
                "event-ledger",
                "The orchard fundraiser dispatch is filed under VELA-72. "
                "Consult that dispatch to identify its emergency crate.",
            ),
            Document("dispatch-ledger", "Dispatch VELA-72: its emergency crate is cobalt."),
        ),
        "What color was the emergency crate for the orchard fundraiser? Give one color word.",
        "cobalt",
        limit=1,
    ),
    Case(
        "bridge_final_settlement",
        "development",
        (
            Document(
                "draft-cost",
                "The observatory excursion had a provisional cost of GBP 91.00. "
                "Final settlement is filed separately under NIMBUS-46; this draft is not final.",
            ),
            Document("settled-cost", "Dossier NIMBUS-46 records final settled cost GBP 73.40."),
        ),
        "What was the final settled cost of the observatory excursion? Answer GBP and amount only.",
        "GBP 73.40",
        "amount",
        limit=1,
    ),
    Case(
        "source_id_set",
        "development",
        (
            Document("survey-photo-a", "A photo of a striped lighthouse."),
            Document("survey-photo-b", "A second photo of the same striped lighthouse."),
            Document("survey-photo-c", "A photo of an unstriped water tower."),
        ),
        "Return every source_id for photos of the striped lighthouse, comma-separated, nothing else.",
        "survey-photo-a,survey-photo-b",
        "ids",
    ),
    Case(
        "date_granularity",
        "development",
        (Document("ticket-date", "The glasshouse tour began at 2042-06-14T09:35:00+00:00."),),
        "On which calendar date did the glasshouse tour begin? Return YYYY-MM-DD only, no time.",
        "2042-06-14",
        "date",
    ),
    Case(
        "datetime_granularity",
        "development",
        (Document("ticket-time", "The cable-car boarding is 2042-07-08T16:25:00+00:00."),),
        "Give the cable-car boarding date and time in UTC as YYYY-MM-DD HH:MM, nothing else.",
        "2042-07-08 16:25",
        "datetime",
    ),
    Case(
        "refund_net",
        "development",
        (
            Document("booking-charge", "Trip booking SABLE charged GBP 195.80."),
            Document("booking-refund", "Trip booking SABLE refunded GBP 58.35."),
            Document("other-charge", "Unrelated booking QUARTZ charged GBP 800.00."),
        ),
        "What is the net amount paid for booking SABLE after its refund? Answer GBP and amount only.",
        "GBP 137.45",
        "amount",
    ),
    Case(
        "interval_union",
        "development",
        (
            Document("permit-a", "Access permit active [2043-02-02,2043-02-06)."),
            Document("permit-b", "Access permit active [2043-02-05,2043-02-09)."),
            Document("permit-copy", "A duplicate copy confirms [2043-02-02,2043-02-06)."),
        ),
        "How many distinct calendar days had access coverage? Intervals are half-open; "
        "count their union, not overlapping days or duplicate copies twice. Return one integer.",
        "7",
    ),
    Case(
        "missing_private_fact",
        "development",
        (Document("box-label", "The storage box is turquoise."),),
        "What is the storage box's private keypad code?",
        "",
        "abstention",
    ),
    Case(
        "heldout_bridge",
        "holdout",
        (
            Document("survey-register", "The marsh survey sample is tracked under LYRA-28."),
            Document("sample-register", "LYRA-28 was sealed with an amber tamper strip."),
        ),
        "What color tamper strip sealed the marsh survey sample? Give one color word.",
        "amber",
        limit=1,
    ),
    Case(
        "heldout_duplicate_refund",
        "holdout",
        (
            Document("invoice-original", "Booking IVORY transaction CH-81 charged GBP 203.50."),
            Document("invoice-copy", "Duplicate receipt: transaction CH-81 charged GBP 203.50."),
            Document("refund-original", "Booking IVORY transaction RF-09 refunded GBP 29.45."),
        ),
        "After deduplicating transaction IDs, what net GBP amount was paid for booking IVORY? "
        "Answer GBP and amount only.",
        "GBP 174.05",
        "amount",
    ),
    Case(
        "heldout_map_containing_area",
        "holdout",
        (
            Document(
                "synthetic-garden-capture",
                "Synthetic grey test image; its GPS is capture "
                "metadata, not proof of a photographed building's identity.",
                (52.1946, 0.1276),
            ),
        ),
        "Using supplied map metadata only, name the enclosing botanic garden at this capture "
        "coordinate. Do not identify the photographed building. Give the garden name only.",
        "Cambridge University Botanic Garden",
    ),
    Case(
        "heldout_map_identity_unknown",
        "holdout",
        (
            Document(
                "synthetic-unmarked-capture",
                "This synthetic unmarked grey image contains "
                "no shop sign, receipt, address or recognizable landmark.",
                (52.2063, 0.1194),
            ),
        ),
        "What exact shop was I inside when this photo was captured? GPS proximity alone cannot "
        "identify the shop; use only the supplied evidence.",
        "",
        "abstention",
    ),
)


class ControlledEmbedder:
    """Synthetic routes hide the terminal bridge document until its key is queried."""

    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "controlled-investigation-fixture"
    embedding_space = "controlled-investigation-fixture:7:v1"
    embedding_dimension = 7

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        vectors = []
        for item in inputs:
            text = item.text.casefold()
            position = 6
            for offset, (event, key) in enumerate(
                (
                    ("orchard fundraiser", "vela-72"),
                    ("observatory excursion", "nimbus-46"),
                    ("marsh survey", "lyra-28"),
                )
            ):
                if task is EmbedTask.DOCUMENT and event in text:
                    position = offset * 2
                    break
                if key in text:
                    position = offset * 2 + 1
                    break
                if event in text:
                    position = offset * 2
                    break
            vectors.append(tuple(float(index == position) for index in range(7)))
        return tuple(vectors)

    def close(self) -> None:
        pass


def grade(case: Case, result: AnswerResult) -> bool:
    if case.kind == "abstention":
        return result.abstained
    if result.abstained:
        return False
    answer = result.answer.strip().strip("`\"' .\n")
    if case.kind == "ids":
        return {item.strip() for item in answer.split(",")} == set(case.expected.split(","))
    if case.kind == "datetime":
        return answer.replace("T", " ") == case.expected
    if case.kind == "date":
        return re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer) is not None and answer == case.expected
    if case.kind == "amount":
        matched = re.fullmatch(
            r"(?:GBP\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*GBP)",
            answer.upper().replace("£", "GBP"),
        )
        return matched is not None and Decimal(matched.group(1) or matched.group(2)) == Decimal(
            case.expected.removeprefix("GBP ")
        )
    return answer.casefold() == case.expected.casefold()


def _add_documents(memory: Memory, case: Case, directory: Path) -> None:
    for document in case.documents:
        content: str | tuple[str, Path] = document.text
        if document.gps is not None:
            from PIL import Image
            from PIL.TiffImagePlugin import IFDRational

            latitude, longitude = document.gps
            tags = Image.Exif()
            coordinates = []
            for value in (latitude, longitude):
                minutes = value % 1 * 60
                coordinates.append(
                    (int(value), int(minutes), IFDRational(round(minutes % 1 * 60_000), 1000))
                )
            tags[34853] = {1: "N", 2: coordinates[0], 3: "E", 4: coordinates[1]}
            photo = directory / f"{document.source_id}.jpg"
            Image.new("RGB", (48, 32), "grey").save(photo, exif=tags)
            content = (document.text, photo)
        memory.add(content, metadata={"source_id": document.source_id}, occurred_at=REFERENCE_AT)


def evaluate(
    answerer: OpenAIModels,
    *,
    split: Literal["development", "holdout"],
    settings: MemoryConfig,
    exporter: InMemorySpanExporter,
    provider: TracerProvider,
    case_ids: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        if case.split != split or (case_ids is not None and case.case_id not in case_ids):
            continue
        exporter.clear()
        started = perf_counter()
        error_code = None
        error_reason = error_message = None
        result = None
        try:
            with TemporaryDirectory(prefix="mindbridge-investigation-quality-") as directory:
                root = Path(directory)
                with Memory.from_plugins(
                    root / "memory",
                    plugins=MemoryPlugins(embedder=ControlledEmbedder(), answerer=answerer),
                    config=replace(
                        settings,
                        retrieval_mode=RetrievalMode.DENSE,
                        minimum_relevance=0.5,
                        recall_planning=True,
                        recall_query_expansion=True,
                        reinforce_on_answer=False,
                    ),
                    tracer=provider.get_tracer(__name__),
                ) as memory:
                    _add_documents(memory, case, root)
                    result = memory.ask(case.question, limit=case.limit, reference_at=REFERENCE_AT)
        except MindBridgeError as error:
            error_code = error.code
            error_reason, error_message = error.reason, str(error)
        generation_spans = [
            span.attributes or {}
            for span in exporter.get_finished_spans()
            if (span.attributes or {}).get("mindbridge.model.module") == "generation"
        ]
        rows.append(
            {
                "case_id": case.case_id,
                "passed": result is not None and grade(case, result),
                "abstained": None if result is None else result.abstained,
                "error_code": error_code,
                "error_reason": error_reason,
                "error_message": error_message,
                "latency_ms": (perf_counter() - started) * 1000,
                "answer_sha256": None
                if result is None
                else hashlib.sha256(result.answer.encode()).hexdigest(),
                "answer_normalized_sha256": None
                if result is None
                else hashlib.sha256(
                    result.answer.strip().strip("`\"' .\n").casefold().encode()
                ).hexdigest(),
                "response_models": sorted(
                    {
                        str((span.attributes or {})["gen_ai.response.model"])
                        for span in exporter.get_finished_spans()
                        if (span.attributes or {}).get("gen_ai.response.model")
                    }
                ),
                "generation_usage": {
                    key: sum(
                        value
                        for attributes in generation_spans
                        if isinstance(value := attributes.get(key), int)
                        and not isinstance(value, bool)
                    )
                    for key in (
                        "mindbridge.token_usage.total_tokens",
                        "mindbridge.token_usage.expected_request_count",
                        "mindbridge.token_usage.reported_request_count",
                    )
                },
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return rows


def _source_snapshot(package: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        relative = str(path.relative_to(package))
        content = path.read_bytes()
        digest.update(relative.encode())
        digest.update(content)
        files[relative] = hashlib.sha256(content).hexdigest()
    return digest.hexdigest(), files


def _response_shape(response: httpx.Response) -> dict[str, object]:
    response.read()
    shape: dict[str, object] = {"status_code": response.status_code}
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return {**shape, "wire_format": "sse", "json_content_checked": False}
    try:
        body = response.json()
        choice = body["choices"][0]
        decoded = json.loads(choice["message"]["content"])
    except (ValueError, TypeError, KeyError, IndexError):
        return {**shape, "wire_format": "unrecognized", "json_content_checked": False}
    shape.update(
        wire_format="json",
        finish_reason=choice.get("finish_reason"),
        root_type=type(decoded).__name__,
    )
    if isinstance(decoded, dict):
        shape["root_keys"] = sorted(decoded)
        draft = decoded.get("draft")
        if isinstance(draft, dict):
            shape["draft_types"] = {key: type(value).__name__ for key, value in draft.items()}
            shape["kind"] = draft.get("kind")
            shape["operation"] = draft.get("operation")
    return shape


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-index", type=Path)
    parser.add_argument("--map-sha256")
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.max_requests <= 80:
        parser.error("--max-requests must be between 1 and 80")
    if (arguments.map_index is None) != (arguments.map_sha256 is None):
        parser.error("--map-index and --map-sha256 must be supplied together")
    if arguments.env_file is not None:
        for token in shlex.split(arguments.env_file.read_text(), comments=True):
            name, separator, value = token.partition("=")
            if separator and name in {"OPENAI_API_KEY", "MINDBRIDGE_GENERATION_API_KEY"}:
                os.environ.setdefault(name, value)
    data = yaml.safe_load(arguments.config.read_text())
    config = MindBridgeConfig.model_validate(data)
    generation = config.generation
    if generation is None:
        raise ValueError("generation must be configured")
    package = Path(mindbridge.__file__).parent
    source_sha256_at_start, source_files_at_start = _source_snapshot(package)
    harness_sha256_at_start = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    requests = 0
    response_shapes: list[dict[str, object]] = []

    def counted(_request: httpx.Request) -> None:
        nonlocal requests
        if requests >= arguments.max_requests:
            raise RuntimeError("quality generation request budget exhausted")
        requests += 1

    def response_shape(response: httpx.Response) -> None:
        response_shapes.append(_response_shape(response))

    gazetteer = (
        OfflineGazetteer(arguments.map_index, index_sha256=arguments.map_sha256)
        if arguments.map_index is not None
        else None
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(event_hooks={"request": [counted], "response": [response_shape]}) as transport,
        OpenAI(
            api_key=(
                generation.api_key.get_secret_value()
                if generation.api_key is not None
                else os.environ.get(
                    "MINDBRIDGE_GENERATION_API_KEY", os.environ.get("OPENAI_API_KEY")
                )
            ),
            base_url=generation.base_url,
            timeout=120,
            max_retries=0,
            http_client=transport,
        ) as client,
    ):
        models = OpenAIModels(
            client,
            generation_model=generation.model,
            generation_capabilities=generation.modalities,
            generation_max_tokens=4096,
            generation_response_format=generation.response_format,
            generation_extra_body=generation.extra_body,
            generation_seed=0,
            generation_gazetteer=gazetteer,
            generation_answer_format=generation.answer_format,
        )
        selected_split: Literal["development", "holdout"] = (
            "holdout" if arguments.split == "holdout" else "development"
        )
        rows = evaluate(
            models,
            split=selected_split,
            settings=config.settings,
            exporter=exporter,
            provider=provider,
            case_ids=None if arguments.case is None else frozenset(arguments.case),
        )
    source_sha256_at_end, source_files_at_end = _source_snapshot(package)
    report = {
        "fixture_version": FIXTURE_VERSION,
        "fixture_sha256": hashlib.sha256(
            json.dumps([asdict(case) for case in CASES], sort_keys=True).encode()
        ).hexdigest(),
        "fixture_hash_scope": "CASES dataclasses only; generated image bytes also depend on the recorded harness source and installed Pillow version.",
        "label": arguments.label,
        "split": arguments.split,
        "configured_model": generation.model,
        "model_revision_limit": "Provider alias and returned identity do not establish immutable weights.",
        "product_source_path": str(package),
        "product_source_sha256": source_sha256_at_start,
        "product_source_sha256_at_end": source_sha256_at_end,
        "source_unchanged_during_run": source_sha256_at_start == source_sha256_at_end,
        "source_changed_paths": sorted(
            path
            for path in source_files_at_start.keys() | source_files_at_end.keys()
            if source_files_at_start.get(path) != source_files_at_end.get(path)
        ),
        "product_source_files_at_start": source_files_at_start,
        "harness_sha256": harness_sha256_at_start,
        "config_path": str(arguments.config.resolve()),
        "config_sha256": hashlib.sha256(arguments.config.read_bytes()).hexdigest(),
        "settings": asdict(config.settings),
        "controlled_overrides": {
            "retrieval_mode": "dense",
            "minimum_relevance": 0.5,
            "recall_planning": True,
            "recall_query_expansion": True,
            "reinforce_on_answer": False,
        },
        "effective_generation": {
            "model": generation.model,
            "response_format": generation.response_format,
            "answer_format": generation.answer_format,
            "source_metadata_key": "source_id",
            "answer_style": "default",
            "max_tokens": 4096,
            "seed": 0,
            "timeout_seconds": 120,
            "max_retries": 0,
            "media_max_items": 2,
            "map_enabled": gazetteer is not None,
        },
        "scope": "Synthetic controlled retrieval plus deployed planner/reader; no real embedding, caption or judge calls; no full benchmark quality claim.",
        "generation_requests": requests,
        "response_shapes": response_shapes,
        "request_limit": arguments.max_requests,
        "max_output_tokens": 4096,
        "cases": rows,
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "map_provenance": None if gazetteer is None else gazetteer.provenance,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    provider.shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
