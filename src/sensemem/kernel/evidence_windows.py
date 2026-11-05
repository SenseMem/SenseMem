"""Exact, query-related source windows; no model calls or storage reads."""

from __future__ import annotations

from dataclasses import dataclass

from sensemem.exceptions import ValidationError
from sensemem.kernel.ranking import lexical_query_terms


@dataclass(frozen=True, slots=True)
class TextWindow:
    """A verbatim, half-open codepoint interval in one source's original text."""

    start: int
    end: int
    text: str


def query_evidence_windows(
    query: str,
    text: str,
    *,
    max_chars: int = 1200,
    max_windows: int = 3,
) -> tuple[TextWindow, ...]:
    """Keep source context and lexical matches without silently keeping only the prefix.

    The first window preserves the source's opening context. Other windows maximize distinct
    query-term coverage, then retain source order. This is a lexical budget policy, not a
    guarantee that a selected window supports an answer or contains every relevant fact.
    """
    if min(max_chars, max_windows) < 1:
        raise ValidationError("evidence window budgets must be positive")
    max_windows = min(max_windows, max_chars)
    if len(text) <= max_chars:
        return (TextWindow(0, len(text), text),) if text else ()
    terms = lexical_query_terms(query)
    if max_windows == 1 or not terms:
        return (TextWindow(0, max_chars, text[:max_chars]),)
    width = max(1, max_chars // max_windows)
    selected = [TextWindow(0, width, text[:width])]
    candidates = []
    for start in range(width, len(text), max(1, width // 2)):
        end = min(len(text), start + width)
        source = text[start:end]
        coverage = len(terms & lexical_query_terms(source))
        if coverage:
            candidates.append((coverage, TextWindow(start, end, source)))
    candidates.sort(key=lambda item: (-item[0], item[1].start))
    for _coverage, window in candidates:
        if all(window.end <= row.start or window.start >= row.end for row in selected):
            selected.append(window)
        if len(selected) == max_windows:
            break
    if len(selected) == 1:
        return (TextWindow(0, max_chars, text[:max_chars]),)
    return tuple(sorted(selected, key=lambda window: window.start))
