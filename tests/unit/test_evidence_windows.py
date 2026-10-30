"""Query windows must quote exact source ranges within a bounded rendering."""

from itertools import pairwise

import pytest

from mindbridge.exceptions import ValidationError
from mindbridge.kernel.evidence_windows import query_evidence_windows


def test_windows_preserve_unicode_offsets_and_find_a_late_fact() -> None:
    source = (
        "Original booking details.\n" + "无关说明。" * 1000 + "\nCancelled booking; refund GBP 55."
    )
    windows = query_evidence_windows("cancelled booking refund", source, max_chars=240)
    assert windows[0].start == 0
    assert any("refund GBP 55" in window.text for window in windows)
    assert sum(len(window.text) for window in windows) <= 240
    assert all(window.text == source[window.start : window.end] for window in windows)
    assert all(first.end <= second.start for first, second in pairwise(windows))


def test_short_source_is_not_rewritten_or_split() -> None:
    source = "  Exact source, including spacing.\n"
    windows = query_evidence_windows("question", source)
    assert len(windows) == 1
    assert windows[0].text == source
    assert (windows[0].start, windows[0].end) == (0, len(source))
    assert query_evidence_windows("question", "") == ()


@pytest.mark.parametrize("budget", (1, 2, 7, 100))
def test_window_budget_is_honored_even_when_smaller_than_window_count(budget: int) -> None:
    windows = query_evidence_windows("refund", "refund " * 100, max_chars=budget)
    assert sum(len(window.text) for window in windows) <= budget
    assert len(windows) <= 3


def test_unmatched_question_preserves_the_prefix_without_inventing_a_quote() -> None:
    source = "Known source words " * 100
    windows = query_evidence_windows("missing identifier", source, max_chars=80)
    assert len(windows) == 1 and windows[0].text == source[:80]
    with pytest.raises(ValidationError, match="positive"):
        query_evidence_windows("query", source, max_chars=0)
