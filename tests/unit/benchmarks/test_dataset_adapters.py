"""Regression checks for dataset adapters with nontrivial official release shapes.

Each case pins a shape the pinned releases actually contain and an earlier
version of these loaders rejected or mishandled.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sensemem.benchmarks.download import _patterns
from sensemem.benchmarks.eval_adapters import load_task
from sensemem.benchmarks.memlens import load_memlens
from sensemem.benchmarks.personamem_v3 import (
    load_personamem_v3,
    render_candidates,
    render_event,
)
from sensemem.benchmarks.task_catalog import TASKS, expand


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_longmemeval_catalog_only_accepts_the_current_release() -> None:
    assert expand(("longmemeval-v2",)) == (
        "longmemeval-v2-small",
        "longmemeval-v2-medium",
    )
    for legacy in ("longmemeval", "longmemeval-s", "longmemeval-m", "longmemeval-oracle"):
        with pytest.raises(ValueError, match="unknown task:"):
            expand((legacy,))


def test_memlens_reads_direct_and_data_wrapped_releases_without_changing_root_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "question_id": "q1",
        "question_type": "preference",
        "question": "What is my favorite color?",
        "answer": "Blue.",
        "question_date": "2025/01/15 (Wed) 10:00",
        "haystack_dates": ["2025/01/14 (Tue) 09:00"],
        "haystack_sessions": [[{"role": "user", "content": "It is blue."}]],
    }
    direct_path = _write(tmp_path / "direct.json", [record])
    wrapped = load_memlens(_write(tmp_path / "wrapped.json", {"data": [record]}))
    invalid = _write(tmp_path / "invalid.json", {"questions": [record]})

    with pytest.raises(
        ValueError, match="MEMLENS dataset root must be a list or contain a data list"
    ):
        load_memlens(invalid)

    def fail_stdlib_parse(_payload: object) -> object:
        raise AssertionError("direct MEMLENS lists must use Pydantic's JSON parser")

    monkeypatch.setattr(json, "loads", fail_stdlib_parse)
    assert load_memlens(direct_path) == wrapped


def test_memlens_marks_the_published_answer_sessions_and_rejects_ones_outside_the_haystack(
    tmp_path: Path,
) -> None:
    """`answer_session_ids` is the only gold retrieval label MEMLENS publishes.

    Every released row carries it, as a subset of `haystack_session_ids`; refusal rows carry
    an empty list. Dropping it left the task's retrieval quality reported as unmeasurable.
    """
    record = {
        "question_id": "q1",
        "question_type": "information_extraction",
        "question": "What is my favorite color?",
        "answer": "Blue.",
        "question_date": "2025/01/15 (Wed) 10:00",
        "haystack_dates": ["2025/01/13 (Mon) 09:00", "2025/01/14 (Tue) 09:00"],
        "haystack_session_ids": ["sess_a", "sess_b"],
        "answer_session_ids": ["sess_b"],
        "haystack_sessions": [
            [{"role": "user", "content": "It rained today."}],
            [{"role": "user", "content": "My favorite color is blue."}],
        ],
    }
    (question,) = load_memlens(_write(tmp_path / "labelled.json", [record]))
    assert [session.is_answer_session for session in question.sessions] == [False, True]

    refusal = {**record, "question_id": "q2", "answer_session_ids": []}
    (unlabelled,) = load_memlens(_write(tmp_path / "refusal.json", [refusal]))
    assert not any(session.is_answer_session for session in unlabelled.sessions)

    stray = {**record, "answer_session_ids": ["sess_zzz"]}
    with pytest.raises(ValueError, match="names answer sessions outside its haystack"):
        load_memlens(_write(tmp_path / "stray.json", [stray]))

    # A file without published session IDs keeps loading: the label is checked against the
    # IDs the loader synthesises, not against the absent list.
    synthesised = {k: v for k, v in record.items() if k != "haystack_session_ids"}
    synthesised["answer_session_ids"] = ["session_0001"]
    (derived,) = load_memlens(_write(tmp_path / "synthesised.json", [synthesised]))
    assert [session.is_answer_session for session in derived.sessions] == [False, True]


def _personamem_event(object_id: str, timestamp: int, **extra: object) -> dict[str, object]:
    event: dict[str, object] = {
        "source_object_id": object_id,
        "source_timestamp": timestamp,
        "formatted_timestamp": "03:41, 04/03/2026",
        "source_hashtags": ["#ankara"],
        "source_interaction_type": "explicit_positive",
        "interaction_format": {"app": "Instagram", "action_label": "Posted image"},
        "content": {"caption": "clean lines"},
    }
    event.update(extra)
    return event


def _personamem_query(**overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "query_id": "8:0001:aaa",
        "task_family": "chatbot_response",
        "task_type": "chatbot_personalized_response",
        "query_kind": "user_query",
        "expected_behavior": "personalize",
        "ts": 2_000,
        "user_query": "what should i cook",
        "example_response": "pepper stew",
        "groundtruth_preference": "likes stew",
        "rubric_tags": ["(+) reflect preferences"],
    }
    query.update(overrides)
    return query


def test_personamem_v3_reads_every_released_row_shape(tmp_path: Path) -> None:
    persona = tmp_path / "backend" / "8"
    _write(
        persona / "instagram.json",
        [
            _personamem_event("a", 1_000),
            # A direct-message thread names its parts `sender`/`text`.
            _personamem_event(
                "dm",
                1_500,
                is_dm=True,
                messages=[{"msg_id": "m0", "sender": "friend_5", "text": "your kind of vibe"}],
            ),
        ],
    )
    _write(
        persona / "calendar.json",
        {
            "modifications": [
                {
                    "mod_id": "mod_001",
                    "ts": 1_200,
                    "action": "added",
                    "entry": {
                        "title": "Grant review",
                        "type": "work",
                        # Booked before it starts, so the scheduled window is
                        # not derivable from the modification's own timestamp.
                        "start_ts": 9_000,
                        "end_ts": 16_200,
                        "location": {"city": "Osaka", "country": "Japan"},
                    },
                },
                # An `updated` modification carries no `entry` at all, only the
                # entry ID and a per-field diff.
                {
                    "mod_id": "mod_002",
                    "ts": 1_300,
                    "action": "updated",
                    "entry_id": "cal_006",
                    "diff": {
                        "end_ts": {"from": 16_200, "to": 18_000},
                        "notes": {"from": "", "to": "Ran long."},
                    },
                },
                {
                    "mod_id": "mod_003",
                    "ts": 1_400,
                    "action": "removed",
                    "entry_id": "cal_017",
                    "removal_reason": "canceled: organizer moved it",
                },
            ]
        },
    )
    _write(
        persona / "test.json",
        [
            _personamem_query(),
            # The sycophancy rows publish a bare string where every other
            # family publishes a list.
            _personamem_query(
                query_id="8:0002:syco",
                task_type="over_personalization_sycophancy",
                rubric_tags="sycophancy_resistance",
            ),
            # Cluster-scored rows retain their ordered turns for the threaded runner.
            _personamem_query(
                query_id="8:0003:rep",
                task_type="over_personalization_repetition_chatbot",
                instance_full={
                    "cluster_id": "8:c1d",
                    "queries": [{"ts": 2_000, "user_query": "What should I cook?"}],
                },
            ),
            _personamem_query(
                query_id="8:0004:rank",
                task_type="personalized_recommendation",
                user_query="[system prompt] Recommend feed items.",
                instance_full={
                    "candidates": [
                        {"title": "A", "source_app": "facebook", "_held_out_persona_item": "LEAK"},
                        {"title": "B", "source_app": "threads"},
                    ],
                    "held_out_idx": 0,
                    "hard_negative_idxs": [1],
                },
            ),
        ],
    )

    personas = load_personamem_v3(tmp_path / "backend")

    assert len(personas) == 1
    events = personas[0].events
    assert [event.event_id for event in events] == [
        "instagram:a",
        "calendar:mod_001",
        "calendar:mod_002",
        "calendar:mod_003",
        "instagram:dm",
    ]
    assert events[4].conversation[0].role == "friend_5"
    assert "friend_5: your kind of vibe" in render_event(events[4])

    queries = {query.query_id: query for query in personas[0].queries}
    assert queries["8:0003:rep"].cluster_id == "8:c1d"
    assert queries["8:0003:rep"].cluster_queries[0].user_query == "What should I cook?"
    assert queries["8:0002:syco"].rubric_tags == ("sycophancy_resistance",)
    ranking = queries["8:0004:rank"]
    assert ranking.positive_indexes == (0,)
    assert ranking.negative_indexes == (1,)
    # The slate must never show the scorer-side identity of the held-out item.
    slate = render_candidates(ranking.candidates)
    assert "LEAK" not in slate
    assert "_held_out_persona_item" not in slate
    assert "title='A'" in slate
    # `profile.json` is scorer-side and is never read as memory.
    assert not any(event.app == "Profile" for event in events)


def test_personamem_v3_keeps_what_each_calendar_modification_says(tmp_path: Path) -> None:
    """A calendar memory has to carry the appointment, not just the edit."""
    persona = tmp_path / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "calendar.json",
        {
            "modifications": [
                {
                    "mod_id": "mod_001",
                    "ts": 1_200,
                    "action": "added",
                    "entry": {
                        "entry_id": "cal_006",
                        "title": "Grant review",
                        "type": "work",
                        "start_ts": 9_000,
                        "end_ts": 16_200,
                        "location": {"city": "Osaka", "country": "Japan"},
                    },
                },
                {
                    "mod_id": "mod_002",
                    "ts": 1_300,
                    "action": "updated",
                    "entry_id": "cal_006",
                    "diff": {
                        "end_ts": {"from": 16_200, "to": 18_000},
                        "notes": {"from": "", "to": "Ran long."},
                    },
                },
                {
                    "mod_id": "mod_003",
                    "ts": 1_400,
                    "action": "removed",
                    "entry_id": "cal_006",
                    "removal_reason": "canceled: organizer moved it",
                },
            ]
        },
    )
    _write(persona / "test.json", [_personamem_query()])

    events = {
        event.event_id: event
        for event in load_personamem_v3(tmp_path / "backend")[0].events
        if event.app == "Calendar"
    }
    # Later modifications name only the entry ID; the added appointment must
    # preserve it too so that its updates and cancellation can be linked.
    assert {event.calendar_entry_id for event in events.values()} == {"cal_006"}
    assert all("Calendar entry: cal_006" in render_event(event) for event in events.values())

    added = events["calendar:mod_001"]
    # The appointment's window is a different moment from the edit's, and the
    # duration appears nowhere else, so neither is recoverable from `ts`.
    assert added.occurred_at.isoformat() == "1970-01-01T00:20:00+00:00"
    assert added.scheduled_start is not None
    assert added.scheduled_start.isoformat() == "1970-01-01T02:30:00+00:00"
    assert added.scheduled_end is not None
    assert added.scheduled_end.isoformat() == "1970-01-01T04:30:00+00:00"
    rendered = render_event(added)
    assert "Scheduled: 1970-01-01T02:30:00+00:00 to 1970-01-01T04:30:00+00:00" in rendered
    assert "Location: Osaka, Japan" in rendered
    assert "Type: work" in rendered

    # An `updated` modification has no `entry`; without its diff the memory
    # would be a bare header line with nothing retrievable in it.
    updated = render_event(events["calendar:mod_002"])
    assert "end_ts 1970-01-01T04:30:00+00:00 -> 1970-01-01T05:00:00+00:00" in updated
    assert "notes (none) -> Ran long." in updated
    assert "cal_006" in updated

    assert "canceled: organizer moved it" in render_event(events["calendar:mod_003"])
    # No calendar memory may be a header line on its own.
    assert all(len(render_event(event).splitlines()) > 1 for event in events.values())


def test_download_patterns_cover_extensionless_files_and_pinned_directories() -> None:
    root = Path(".benchmarks")

    def patterns_for(name: str) -> tuple[str, ...]:
        spec = TASKS[name]
        dataset = spec.dataset_path(root)
        release = root / Path(spec.dataset).parts[0]
        return _patterns(spec, dataset, dataset, dataset.relative_to(release).as_posix())

    # A suffix-less file must be offered as a file and as a directory glob.
    spec = replace(TASKS["locomo-refined"], dataset="fixture/annotations")
    dataset = spec.dataset_path(root)
    assert _patterns(spec, dataset, dataset, "annotations") == ("annotations", "annotations/*")
    assert patterns_for("locomo-refined") == ("data/raw/locomo_refined.json",)
    # A directory input may pin exactly the files its loader opens.
    personamem = patterns_for("personamem-v3")
    assert "backend/*/test.json" in personamem
    assert not any("persona.html" in pattern for pattern in personamem)
    assert not any("profile.json" in pattern for pattern in personamem)
    # A directory with no pinned patterns keeps the old both-spellings default.
    assert patterns_for("mem-gallery") == ("data/dialog", "data/dialog/*")


def test_personamem_v3_reads_the_flattened_row_spelling(tmp_path: Path) -> None:
    """Six released rows publish a different spelling of the same record."""
    persona = tmp_path / "backend" / "26"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            {
                "query_id": "26:0043:sensitive_row00_q0",
                "task_family": "over_personalization",
                "task_type": "over_personalization_sensitive_event",
                "ts": 2_000,
                # `query_text`, not `user_query`; no `query_kind` at all;
                # `expected_response_kind` in place of `expected_behavior`;
                # semicolon-joined rubric tags; and the instance as a JSON
                # string rather than an object.
                "query_text": "is that just burnout or something else?",
                "expected_response_kind": "text",
                "rubric_tags": "avoid_overpersonalization;telegraph_avoidance",
                "instance_json": '{"test_id": "row00", "arm": "sensitive_event"}',
            },
            # `user_query` present but null, with the question in the instance.
            {
                **_personamem_query(query_id="26:0044:nulled"),
                "user_query": None,
                "instance_full": {"user_query": "what should i cook"},
            },
            # No question text under any spelling: dropped rather than stored
            # as a memory-less question.
            {
                **_personamem_query(query_id="26:0045:empty"),
                "user_query": None,
                "instance_full": {},
            },
        ],
    )

    queries = {
        query.query_id: query for query in load_personamem_v3(tmp_path / "backend")[0].queries
    }

    assert set(queries) == {"26:0043:sensitive_row00_q0", "26:0044:nulled"}
    flattened = queries["26:0043:sensitive_row00_q0"]
    assert flattened.user_query == "is that just burnout or something else?"
    assert flattened.expected_behavior == "text"
    assert flattened.query_kind == ""
    assert flattened.rubric_tags == ("avoid_overpersonalization", "telegraph_avoidance")
    assert flattened.judge_evidence["arm"] == "sensitive_event"
    assert queries["26:0044:nulled"].user_query == "what should i cook"


@pytest.mark.parametrize(
    ("limit", "offset", "selected"),
    [
        (1, 1, ("8",)),
        (0.5, 1, ("8", "10")),
        (None, 2, ("10", "11")),
        (-1, 2, ("10", "11")),
    ],
)
def test_personamem_v3_selects_personas_before_reading_history(
    tmp_path: Path, limit: int | float | None, offset: int, selected: tuple[str, ...]
) -> None:
    backend = tmp_path / "personamem-v3" / "backend"
    # Personas with no supported queries do not affect the limit's denominator or offset.
    _write(
        backend / "1" / "test.json",
        [],
    )
    for persona_id in ("11", "2", "10", "8"):
        persona = backend / persona_id
        _write(persona / "test.json", [_personamem_query()])
        if persona_id in selected:
            _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
        else:
            # Parsing the history of an unselected persona would fail this evaluation.
            (persona / "instagram.json").write_text("not JSON", encoding="utf-8")

    task = load_task(
        TASKS["personamem-v3"],
        root=tmp_path,
        verify_digest=False,
        limit=limit,
        offset=offset,
    )

    assert tuple(unit.unit_id for unit in task.units) == selected


def test_personamem_v3_ranking_keeps_the_prior_conversation(tmp_path: Path) -> None:
    persona = tmp_path / "personamem-v3" / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            _personamem_query(
                task_type="personalized_recommendation",
                user_query="Which one suits the dinner?",
                prior_conversation=[
                    {"role": "user", "content": "I need a vegetarian dinner."},
                    {"role": "assistant", "content": "How many guests?"},
                    {"role": "user", "content": "Two."},
                ],
                instance_full={"candidates": [{"title": "Stew"}, {"title": "Steak"}]},
            )
        ],
    )

    unit = load_task(TASKS["personamem-v3"], root=tmp_path, verify_digest=False).units[0]
    (question,) = unit.questions
    prompt = "\n".join(str(part) for part in question.content)

    assert "user: I need a vegetarian dinner.\nassistant: How many guests?\nuser: Two." in prompt
    assert "Which one suits the dinner?" in prompt
    assert "title='Stew'" in prompt
    assert "pepper stew" not in prompt
    assert "likes stew" not in prompt


def test_personamem_v3_proactive_prompt_keeps_the_decision_and_evidence_hidden(
    tmp_path: Path,
) -> None:
    persona = tmp_path / "personamem-v3" / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            _personamem_query(
                query_id=f"8:{index}:proactive",
                task_family="proactive_actions",
                task_type="proactive_close_friend_update",
                user_query=f"[scenario label {index}]",
                instance_full={
                    "expected_behavior": behavior,
                    "trigger_evidence": {"private": "SECRET_TRIGGER"},
                    "jitai_card": {"decision_rule": "SECRET_RULE"},
                    "profile": {"name": "SECRET_PROFILE"},
                },
            )
            for index, behavior in enumerate(("act", "restrain"))
        ],
    )

    unit = load_task(TASKS["personamem-v3"], root=tmp_path, verify_digest=False).units[0]
    first, second = unit.questions
    assert first.content == second.content
    prompt = "\n".join(str(part) for part in first.content)
    assert '"should_act": <bool>' in prompt
    assert '"action_class"' in prompt
    assert "SECRET_" not in prompt
    assert "scenario label" not in prompt
    assert "pepper stew" not in prompt
    assert "likes stew" not in prompt
    assert "SECRET_" not in "\n".join(str(part) for item in unit.memories for part in item.content)


def test_personamem_v3_never_records_the_question_as_its_own_reference(tmp_path: Path) -> None:
    """857 released queries publish no `example_response`."""
    from sensemem.benchmarks.eval_adapters import load_task
    from sensemem.benchmarks.task_catalog import TASKS

    persona = tmp_path / "personamem-v3" / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            _personamem_query(query_id="8:0001:gold"),
            {
                **_personamem_query(query_id="8:0002:no-gold"),
                "example_response": "",
                "groundtruth_preference": "Enjoys couple-life content.",
            },
            {
                **_personamem_query(query_id="8:0003:nothing"),
                "example_response": "",
                "groundtruth_preference": "",
            },
        ],
    )

    unit = load_task(TASKS["personamem-v3"], root=tmp_path, verify_digest=False).units[0]
    references = {q.question_id: q.references[0] for q in unit.questions}

    assert references["8:0001:gold"] == "pepper stew"
    # Falls back to label material, never to the prompt itself.
    assert references["8:0002:no-gold"] == "Enjoys couple-life content."
    assert "what should i cook" not in references["8:0003:nothing"]
    assert references["8:0003:nothing"] == "(no reference answer published for this row)"


def test_text_memories_split_passages_the_product_would_reject() -> None:
    """An oversized part is not just dropped -- it voids its whole unit.

    `memory.add` rejects a part over `MAX_TEXT_CHARACTERS`, the runner counts
    an ingest failure, and `_apply_judges` then skips every question in that
    unit. Splitting oversized passages preserves all text within the product limit.
    """
    from sensemem.benchmarks.eval_adapters import _text_memories
    from sensemem.kernel.validation import MAX_TEXT_CHARACTERS

    short = _text_memories("turn", "one short line")
    assert len(short) == 1
    assert short[0].source_id == "turn"
    assert short[0].content == ("one short line",)

    long_turn = "word " * 120_000
    assert len(long_turn) > MAX_TEXT_CHARACTERS
    parts = _text_memories("turn", long_turn, end_seconds=1_775_000_000.0)
    assert len(parts) > 1
    assert all(
        len(text) <= MAX_TEXT_CHARACTERS
        for item in parts
        for text in item.content
        if isinstance(text, str)
    )
    # Split parts stay individually addressable and keep the causal cutoff.
    assert [item.source_id for item in parts[:2]] == ["turn_B0000", "turn_B0001"]
    assert len({item.source_id for item in parts}) == len(parts)
    assert all(item.end_seconds == 1_775_000_000.0 for item in parts)
    # Nothing is dropped.
    rebuilt = "".join(text for item in parts for text in item.content if isinstance(text, str))
    assert rebuilt.replace("\n\n", " ").split() == long_turn.split()


# --- gold evidence: which adapters can carry a source-level retrieval label -----------------


def test_locomo_refined_labels_the_dialogue_turns_that_hold_the_answer(tmp_path: Path) -> None:
    """`evidence` names `dia_id` values, which are this unit's memory source IDs.

    The join is not assumed. An evidence ID that names no stored turn is reported
    under its own key instead of being dropped, so a release whose label vocabulary
    is not the source-ID vocabulary reads as a join failure rather than as recall
    over the handful of IDs that happened to match.
    """
    from sensemem.benchmarks.eval_adapters import load_task

    _write(
        tmp_path / "locomo-refined" / "data" / "raw" / "locomo_refined.json",
        [
            {
                "sample_id": "conv-1",
                "conversation": {
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a beagle."},
                        {"speaker": "Bo", "dia_id": "D1:2", "text": "What did you name her?"},
                        {"speaker": "Ana", "dia_id": "D1:3", "text": "Her name is Juno."},
                    ],
                },
                "qa": [
                    {
                        "question": "What is the dog called?",
                        "answer": ["Juno"],
                        # The release packs several IDs into one string.
                        "evidence": ["D1:1; D1:3"],
                        "category": 1,
                    },
                    {
                        "question": "Which turn is missing?",
                        "answer": ["none"],
                        "evidence": ["D1:2", "D7:9"],
                        "category": 2,
                    },
                    {
                        "question": "Unlabelled question.",
                        "answer": ["none"],
                        "evidence": [],
                        "category": 3,
                    },
                ],
            }
        ],
    )

    unit = load_task(TASKS["locomo-refined"], root=tmp_path, verify_digest=False).units[0]
    stored = {item.source_id for item in unit.memories}
    labels = {question.question_id: question.metadata for question in unit.questions}

    assert stored == {"D1:1", "D1:2", "D1:3"}
    # Exact, source-level gold: the packed string is split and both IDs kept in order.
    assert labels["conv-1#q0000"]["evidence_ids"] == ("D1:1", "D1:3")
    assert labels["conv-1#q0000"]["unresolved_evidence_ids"] == ()
    # `D7:9` names no stored turn, so it is reported rather than silently absorbed.
    assert labels["conv-1#q0001"]["evidence_ids"] == ("D1:2",)
    assert labels["conv-1#q0001"]["unresolved_evidence_ids"] == ("D7:9",)
    # A question the release left unlabelled carries no label, so it is excluded
    # from recall rather than counted as a miss.
    assert labels["conv-1#q0002"]["evidence_ids"] == ()
    # Undated questions are anchored at the conversation's last turn (see the test below).
    assert {question.reference_at for question in unit.questions} == {
        max(item.occurred_at for item in unit.memories if item.occurred_at is not None)
    }


def test_undated_questions_are_anchored_at_the_end_of_their_corpus() -> None:
    """A relative date in a question must resolve the same way in a run and in its replay.

    LoCoMo, ATM-Bench and Mem-Gallery date their memories but not their questions; without an
    anchor `Memory.ask` used the wall clock, so "this year" meant the run's year, not the
    corpus's. Questions that carry their own date keep it.
    """
    from sensemem.benchmarks.eval_adapters import (
        EvalQuestion,
        EvalUnit,
        MemoryItem,
        _with_corpus_reference,
    )

    first = datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    last = datetime(2023, 6, 1, 9, 0, tzinfo=timezone.utc)
    dated = datetime(2024, 1, 1, tzinfo=timezone.utc)
    unit = EvalUnit(
        "conv",
        (
            MemoryItem("t1", ("first",), occurred_at=first),
            MemoryItem("t2", ("second",), occurred_at=first, occurred_end=last),
        ),
        (
            EvalQuestion("undated", ("What happened this year?",), ("nothing",)),
            EvalQuestion("dated", ("And before?",), ("less",), reference_at=dated),
        ),
    )

    anchored = _with_corpus_reference(unit)

    assert anchored.questions[0].reference_at == last
    assert anchored.questions[1].reference_at == dated
    # A corpus without any event time offers no anchor and is left alone.
    untimed = EvalUnit("u", (MemoryItem("t", ("x",)),), (EvalQuestion("q", ("q",), ("a",)),))
    assert _with_corpus_reference(untimed) is untimed


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Today is 2024-05-02. When was dinner last week?", (2024, 5, 2)),
        ("Today is May 2, 2024. When was dinner last week?", (2024, 5, 2)),
        ("Today is Dec 21. What happened yesterday?", (2025, 12, 21)),
        ("When was dinner last week?", (2025, 7, 22)),
    ],
)
def test_question_clock_precedes_corpus_fallback(
    question: str, expected: tuple[int, int, int]
) -> None:
    from sensemem.benchmarks.eval_adapters import (
        EvalQuestion,
        EvalUnit,
        MemoryItem,
        _with_corpus_reference,
    )
    from sensemem.kernel.temporal import query_temporal_range

    latest = datetime(2025, 7, 22, tzinfo=timezone.utc)
    unit = EvalUnit(
        "archive",
        (MemoryItem("last", ("last memory",), occurred_at=latest),),
        (EvalQuestion("q", (question,), ("answer",), source_question=question),),
    )
    anchored = _with_corpus_reference(unit).questions[0]
    assert anchored.reference_at == datetime(*expected, tzinfo=timezone.utc)
    if expected == (2024, 5, 2):
        assert query_temporal_range("last week", anchored.reference_at) == (
            datetime(2024, 4, 22, tzinfo=timezone.utc),
            datetime(2024, 4, 29, tzinfo=timezone.utc),
        )
    # Re-anchoring cannot replace an already resolved question clock.
    assert _with_corpus_reference(_with_corpus_reference(unit)) == _with_corpus_reference(unit)


def test_question_clock_uses_original_text_and_preserves_explicit_dates() -> None:
    from sensemem.benchmarks.eval_adapters import (
        EvalQuestion,
        EvalUnit,
        MemoryItem,
        _with_corpus_reference,
    )

    explicit = datetime(2023, 1, 1, tzinfo=timezone.utc)
    original = "Today is 2024-05-02. When was dinner last week?"
    unit = EvalUnit(
        "archive",
        (MemoryItem("last", ("memory",), occurred_at=datetime(2025, 7, 22, tzinfo=timezone.utc)),),
        (
            EvalQuestion("structured", (original,), ("answer",), reference_at=explicit),
            EvalQuestion(
                "original",
                ("Today is 2099-01-01. Adapter example.", original),
                ("answer",),
                source_question=original,
            ),
        ),
    )
    anchored = _with_corpus_reference(unit)
    assert anchored.questions[0].reference_at == explicit
    assert anchored.questions[1].reference_at == datetime(2024, 5, 2, tzinfo=timezone.utc)


def test_media_only_memory_items_carry_their_source_id_as_metadata_not_text(
    tmp_path: Path,
) -> None:
    """The `[source_id: …]` line is a text memory's label; a clip already has it in metadata.

    As content it became a third retrieval key (the marker text) and made the aggregate key
    differ from the clip's own key, so every 30 s clip was uploaded to the embedder twice. The
    harness reads source ids from metadata (`_evidence`), never from content.
    """
    from sensemem.benchmarks.eval_adapters import MemoryItem
    from sensemem.benchmarks.eval_inputs import _memory_content, _memory_metadata

    clip = tmp_path / "segment.mp4"
    clip.write_bytes(b"video")

    assert _memory_content(MemoryItem("0.mp4-00001", (clip,), 30.0, 60.0)) == (clip,)
    assert _memory_metadata(MemoryItem("0.mp4-00001", (clip,), 30.0, 60.0))["source_id"] == (
        "0.mp4-00001"
    )
    assert _memory_content(MemoryItem("t", ("hello",))) == "[source_id: t]\nhello"
    assert _memory_content(MemoryItem("t", ("hello", clip))) == ("[source_id: t]\nhello", clip)
