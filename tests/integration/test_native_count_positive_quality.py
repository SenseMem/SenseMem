"""Opt-in affirmative native counting despite an incorrect model-derived synopsis."""

import base64
import hashlib
import json
import os
from pathlib import Path

import httpx2 as httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as SchemaValidationError
from test_event_segmentation_quality import native_observations as native_observations

from sensemem import Blob, Memory, Modality, SenseMemError, StreamInput
from sensemem.models.base import EventObservation
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder
from sensemem.prompts.events import synopsis

pytestmark = pytest.mark.integration
MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})
FIXTURE = "sensemem-positive-native-count-json-v1"
QUESTION = (
    "Count each separately audible tonal burst in the audio: a continuous tone separated "
    "from another tone by a silent gap is a separate burst, even within one notification. "
    'How many tonal bursts occur? Return only a JSON object with exactly one field, "count", '
    "whose value is an integer."
)
INCORRECT_SUMMARY = "The recording contains one short electronic tonal burst."


class CountAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    count: int


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.skipif(
    os.environ.get("SENSEMEM_NATIVE_COUNT_EVAL") != "1",
    reason="requires explicit native generation and embedding model services",
)
def test_native_audio_supports_an_affirmative_count_despite_an_incorrect_synopsis(
    tmp_path: Path, native_observations: tuple[EventObservation, ...], repeat: int
) -> None:
    original = native_observations[1].content.assets[1]
    assert original.path is not None and original.media_type is not None
    source_data = original.path.read_bytes()
    expected_sha = hashlib.sha256(source_data).hexdigest()
    wire: list[dict[str, object]] = []

    def inspect_request(request: httpx.Request) -> None:
        if not request.url.path.endswith("/chat/completions"):
            return
        body = json.loads(request.content)
        audio_sha = []
        for message in body["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part["type"] == "input_audio":
                        encoded = part["input_audio"]["data"].split("base64,", 1)[-1]
                        audio_sha.append(hashlib.sha256(base64.b64decode(encoded)).hexdigest())
        wire.append(
            {
                "audio_sha256": audio_sha,
                "max_tokens_present": "max_tokens" in body,
                "max_completion_tokens_present": "max_completion_tokens" in body,
                "reasoning_effort": body.get("reasoning_effort"),
            }
        )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configured_model = os.environ["SENSEMEM_PROJECTION_MODEL"]
    try:
        with (
            httpx.Client(event_hooks={"request": [inspect_request]}) as transport,
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_BASE_URL"],
                api_key=os.environ["SENSEMEM_PROJECTION_API_KEY"],
                http_client=transport,
                timeout=120,
                max_retries=0,
            ) as generation,
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_EMBEDDING_BASE_URL"],
                api_key=os.environ.get("SENSEMEM_PROJECTION_EMBEDDING_API_KEY", "EMPTY"),
                timeout=120,
                max_retries=0,
            ) as embedding,
        ):
            embedder = OpenAIEmbedder(
                client=embedding,
                embedding_model="google/embeddinggemma-2",
                embedding_dimension=768,
                embedding_capabilities=MODALITIES,
                embedding_request_format="messages",
            )
            answerer = OpenAICompletion(
                client=generation,
                generation_model=configured_model,
                generation_capabilities=MODALITIES,
                generation_extra_body={"reasoning_effort": "none"},
                generation_max_tokens=None,
            )
            with Memory(
                tmp_path / "store",
                embedder=embedder,
                answerer=answerer,
                tracer=provider.get_tracer(__name__),
            ) as memory:
                source = next(
                    memory.add_stream(
                        [
                            StreamInput(
                                (Blob(source_data, original.media_type, "observation.wav"),),
                                interpretation=synopsis(INCORRECT_SUMMARY),
                            )
                        ]
                    )
                )
                response = memory.ask(QUESTION)
            parsed = None
            parse_error_types = []
            try:
                parsed = CountAnswer.model_validate_json(response.answer)
            except SchemaValidationError as error:
                parse_error_types = [
                    detail["type"]
                    for detail in error.errors(
                        include_url=False, include_context=False, include_input=False
                    )
                ]
            checks = {
                "nonabstain": not response.abstained,
                "strict_json_count_correct": parsed is not None and parsed.count == 2,
                "source_retrieved": source.id in {hit.id for hit in response.hits},
                "original_native_audio_delivered": bool(wire)
                and all(request["audio_sha256"] == [expected_sha] for request in wire),
                "output_limits_omitted": all(
                    not request["max_tokens_present"]
                    and not request["max_completion_tokens_present"]
                    for request in wire
                ),
                "reasoning_control_retained": all(
                    request["reasoning_effort"] == "none" for request in wire
                ),
            }
            report = {
                "fixture": FIXTURE,
                "repeat": repeat,
                "configured_model": configured_model,
                "returned_models": sorted(
                    {
                        model
                        for span in exporter.get_finished_spans()
                        if isinstance(
                            model := (span.attributes or {}).get("gen_ai.response.model"), str
                        )
                    }
                ),
                "immutable_revision": None,
                "source_sha256": expected_sha,
                "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
                "synopsis_sha256": hashlib.sha256(synopsis(INCORRECT_SUMMARY).encode()).hexdigest(),
                "embedding_recipe": embedder.embedding_space,
                "request_timeout_seconds": 120,
                "sdk_retries": 0,
                "generation_requests": len(wire),
                "parse_valid": parsed is not None,
                "parse_error_types": parse_error_types,
                "answer_sha256": hashlib.sha256(response.answer.encode()).hexdigest(),
                "reported_count": None if parsed is None else parsed.count,
                "checks": checks,
            }
            print(json.dumps(report))
            report_dir = os.environ.get("SENSEMEM_NATIVE_COUNT_REPORT_DIR")
            if report_dir is not None:
                Path(report_dir).mkdir(parents=True, exist_ok=True)
                (Path(report_dir) / f"repeat-{repeat}.json").write_text(
                    json.dumps(report, indent=2)
                )
            assert all(checks.values()), "affirmative native count or its transport failed"
    except SenseMemError as error:
        pytest.fail(f"native positive count failed: {type(error).__name__}", pytrace=False)
    finally:
        provider.shutdown()
