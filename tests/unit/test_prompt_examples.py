"""Complete prompt examples must obey the runtime's wire shapes and local parsers."""

import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from sensemem.models._answer_output import parse_answer_draft
from sensemem.models._openai_schemas import SchemaName, response_schema
from sensemem.prompts import generation as prompts
from sensemem.recall import parse_recall_program


def _matches_structure(value: object, schema: Mapping[str, object]) -> bool:
    """Check closed objects, required fields, arrays, variants and JSON primitive types.

    This intentionally checks the structural subset used by the prompt examples; the local
    response parsers check the additional semantic constraints below.
    """
    variants = schema.get("anyOf")
    if isinstance(variants, list):
        return any(
            isinstance(variant, dict) and _matches_structure(value, variant) for variant in variants
        )
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        return False
    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_matches_structure(value, {**schema, "type": name}) for name in expected)
    if expected == "object":
        return _matches_object(value, schema)
    if expected == "array":
        items = schema.get("items")
        return (
            isinstance(value, list)
            and isinstance(items, dict)
            and all(_matches_structure(item, items) for item in value)
        )
    primitives: dict[str, tuple[type[object], ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
    }
    return type(value) in primitives.get(str(expected), ())


def _matches_object(value: object, schema: Mapping[str, object]) -> bool:
    if not isinstance(value, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required")
    assert isinstance(properties, dict) and isinstance(required, list)
    if not set(required) <= set(value):
        return False
    if schema.get("additionalProperties") is False and set(value) - set(properties):
        return False
    return all(
        isinstance(child, dict) and _matches_structure(value[key], child)
        for key, child in properties.items()
        if key in value
    )


@pytest.mark.parametrize(
    ("schema_name", "prompt"),
    [
        ("event_segmentation", prompts._EVENT_SEGMENTATION_SYSTEM_PROMPT),
        ("answer", prompts._STRUCTURED_ANSWER_PROMPT),
        ("recall_plan", prompts._RECALL_PLAN_SYSTEM_PROMPT),
        ("recall_plan", prompts._RECALL_SEMANTIC_SYSTEM_PROMPT),
        ("formation", prompts._FORMATION_SYSTEM_PROMPT),
        ("media_selection", prompts._MEDIA_DIAGNOSTIC_PROMPT),
        ("vision", prompts._VISION_SYSTEM_PROMPT),
        ("consolidation", prompts._CONSOLIDATION_SYSTEM_PROMPT),
    ],
    ids=("events", "answers", "recall", "semantic-recall", "formation", "media", "vision", "merge"),
)
def test_complete_prompt_examples_match_response_structure(
    schema_name: SchemaName, prompt: str
) -> None:
    examples = re.findall(r"```json\n(.*?)\n```", prompt, flags=re.DOTALL)
    assert examples, "Each structured task needs a complete JSON response example."
    schema = response_schema(schema_name)
    for example in examples:
        payload: object = json.loads(example)
        assert _matches_structure(payload, schema), example
        if schema_name == "answer":
            parse_answer_draft(example)
        elif schema_name == "recall_plan":
            assert (
                parse_recall_program(
                    example, reference_at=datetime(2026, 10, 26, tzinfo=timezone.utc)
                )
                is not None
            ), example
