"""Model examples must obey the same closed envelope as the local parser."""

import json
import re

from mindbridge.kernel.answer_output import parse_answer_draft
from mindbridge.models.openai_sdk import _STRUCTURED_ANSWER_PROMPT


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
