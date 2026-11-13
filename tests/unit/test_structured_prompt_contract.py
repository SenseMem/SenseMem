"""Model examples must obey the same closed envelope as the local parser."""

import json
import re
from collections.abc import Mapping

import pytest

from sensemem.models._answer_output import parse_answer_draft
from sensemem.models._openai_schemas import response_schema
from sensemem.models.openai_sdk import _CONSOLIDATION_SYSTEM_PROMPT, _STRUCTURED_ANSWER_PROMPT


def test_all_structured_answer_examples_are_complete_valid_envelopes() -> None:
    decoder = json.JSONDecoder()
    drafts = []
    for match in re.finditer(r'\{"draft":', _STRUCTURED_ANSWER_PROMPT):
        value, length = decoder.raw_decode(_STRUCTURED_ANSWER_PROMPT[match.start() :])
        assert set(value) == {"draft"}
        drafts.append(
            parse_answer_draft(_STRUCTURED_ANSWER_PROMPT[match.start() : match.start() + length])
        )
    assert [draft.kind for draft in drafts] == [
        "text",
        "source_ids",
        "date",
        "calculation",
        "calculation",
        "calculation",
    ]


@pytest.mark.parametrize("intent", ("reinforce", "correct"))
def test_consolidation_examples_keep_new_proposals_out_of_existing_target_operations(
    intent: str,
) -> None:
    start = _CONSOLIDATION_SYSTEM_PROMPT.index('{"operations":[{"intent":"' + intent + '"')
    example, _length = json.JSONDecoder().raw_decode(_CONSOLIDATION_SYSTEM_PROMPT[start:])
    operation = example["operations"][0]
    properties = response_schema("consolidation")["properties"]
    assert isinstance(properties, Mapping)
    schema = properties["operations"]["items"]
    assert set(operation) == set(schema["required"])
    assert operation["evidence"] == [2]
    assert operation["targets"] == [0]
    assert operation["proposal"] is None and operation["claim"] is None
    assert "Omit a key rather than sending it empty" not in _CONSOLIDATION_SYSTEM_PROMPT
