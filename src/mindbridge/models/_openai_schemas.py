"""Strict wire shapes; reference and evidence semantics remain locally validated."""

from typing import Literal

SchemaName = Literal[
    "formation", "vision", "consolidation", "recall_plan", "media_selection", "answer"
]


def _object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": items}


def _nullable(schema: dict[str, object]) -> dict[str, object]:
    return {"anyOf": [schema, {"type": "null"}]}


def _proposal(*, witnessed: bool) -> dict[str, object]:
    fields: dict[str, object] = {
        "kind": {
            "type": "string",
            "enum": ["event", "entity", "affect", "state", "relation", "trait"],
        },
        "content": {"type": "string"},
        "confidence": {"type": "number"},
        "subject": {"type": ["string", "null"]},
        "predicate": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"]},
        "valid_from": {"type": ["string", "null"]},
        "valid_until": {"type": ["string", "null"]},
        "spatial": _nullable(
            _object(
                {
                    "frame_id": {"type": "string"},
                    "anchor": {"type": "string", "enum": ["observer", "subject"]},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "orientation_xyzw": _nullable(_array({"type": "number"})),
                    "position_uncertainty_m": {"type": ["number", "null"]},
                }
            )
        ),
        "cue_modality": {
            "type": ["string", "null"],
            "enum": ["text", "image", "video", "audio", None],
        },
        "valence": {"type": ["number", "null"]},
        "arousal": {"type": ["number", "null"]},
    }
    if witnessed:
        fields["evidence_observation_ids"] = _array({"type": "string"})
    variants = []
    for kind in (
        ["event", "entity", "affect", "state", "relation", "trait"]
        if witnessed
        else ["event", "entity", "state", "relation", "trait"]
    ):
        properties = dict(fields)
        properties["kind"] = {"type": "string", "enum": [kind]}
        if kind != "affect":
            for key in ("cue_modality", "valence", "arousal"):
                del properties[key]
        else:
            properties["cue_modality"] = {
                "type": "string",
                "enum": ["text", "image", "video", "audio"],
            }
            properties["valence"] = {"type": "number"}
            properties["arousal"] = {"type": "number"}
        for key in (
            (() if kind == "event" else ("subject",))
            + (("predicate", "value") if kind in {"state", "relation", "trait"} else ())
            + (("value",) if kind == "affect" else ())
        ):
            properties[key] = {"type": "string"}
        variants.append(_object(properties))
    return {"anyOf": variants}


def response_schema(name: SchemaName) -> dict[str, object]:
    """Return a fresh schema so callers cannot mutate another request's contract."""
    if name == "formation":
        return _object(
            {
                "items": _array(
                    _object(
                        {
                            "observation_id": {"type": "string"},
                            "proposals": _array(_proposal(witnessed=True)),
                        }
                    )
                )
            }
        )
    if name == "vision":
        return _object({"descriptions": _array({"type": "string"})})
    if name == "answer":
        return _answer_schema()
    if name == "media_selection":
        return _object(
            {
                "requests": _array(
                    _object(
                        {
                            "media_index": {"type": "integer"},
                            "reason": {
                                "type": "string",
                                "enum": ["detail_missing", "conflicting_evidence"],
                            },
                        }
                    )
                )
            }
        )
    if name == "consolidation":
        return _object(
            {
                "operations": _array(
                    _object(
                        {
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "reinforce",
                                    "consolidate",
                                    "correct",
                                    "forget",
                                    "identify",
                                ],
                            },
                            "rationale": {"type": "string"},
                            "targets": _array({"type": "integer"}),
                            "evidence": _array({"type": "integer"}),
                            "proposal": _nullable(_proposal(witnessed=False)),
                            "claim": _nullable(
                                _object(
                                    {
                                        "identity_id": {"type": "string"},
                                        "name": {"type": ["string", "null"]},
                                        "relationship": {"type": ["string", "null"]},
                                    }
                                )
                            ),
                        }
                    )
                )
            }
        )
    return _object(
        {
            "shape": {
                "type": "string",
                "enum": ["point", "set", "sequence", "entity", "composite"],
            },
            "steps": _array(_recall_steps()),
        }
    )


def _answer_schema() -> dict[str, object]:
    label: dict[str, object] = {"type": "string", "pattern": "^E[1-9][0-9]*$"}
    labels = {**_array(label), "maxItems": 128}
    quote: dict[str, object] = {"type": "string", "minLength": 1, "maxLength": 1_000}
    quoted = {"evidence_label": label, "quote": quote}
    operand = _object(
        {
            **quoted,
            "value": {
                "type": "string",
                "pattern": r"^[+-]?(?:[0-9]{1,64}|[0-9]{1,3}(?:,[0-9]{3}){1,21})(?:\.[0-9]{1,32})?$",
            },
            "unit": {"type": "string", "minLength": 1, "maxLength": 64},
        }
    )
    return _object(
        {
            "draft": {
                "anyOf": [
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["text"]},
                            "answer": {"type": "string", "minLength": 1},
                            "evidence_labels": labels,
                            "abstained": {"type": "boolean"},
                        }
                    ),
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["source_ids"]},
                            "evidence_labels": {**labels, "minItems": 1},
                        }
                    ),
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["date"]},
                            "source_value": {"type": "string", "minLength": 1},
                            "format": {"type": "string", "enum": ["iso8601", "utc_datetime"]},
                            "granularity": {
                                "type": "string",
                                "enum": ["year", "month", "day", "minute", "second"],
                            },
                            "evidence": _object(
                                {
                                    "evidence_label": label,
                                    "quote": _nullable(quote),
                                    "timestamp_field": {
                                        "type": ["string", "null"],
                                        "enum": ["occurred_at", "occurred_end", "created_at", None],
                                    },
                                }
                            ),
                        }
                    ),
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["calculation"]},
                            "operation": {
                                "type": "string",
                                "enum": ["scalar", "sum", "difference"],
                            },
                            "format": {"type": "string", "enum": ["with_unit", "number"]},
                            "operands": {**_array(operand), "minItems": 1, "maxItems": 128},
                        }
                    ),
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["calculation"]},
                            "operation": {"type": "string", "enum": ["count_distinct"]},
                            "members": {
                                **_array(
                                    _object(
                                        {
                                            **quoted,
                                            "key": {
                                                "type": "string",
                                                "minLength": 1,
                                                "maxLength": 200,
                                            },
                                        }
                                    )
                                ),
                                "minItems": 1,
                                "maxItems": 128,
                            },
                        }
                    ),
                    _object(
                        {
                            "kind": {"type": "string", "enum": ["calculation"]},
                            "operation": {"type": "string", "enum": ["duration_days"]},
                            "format": {"type": "string", "enum": ["with_unit", "number"]},
                            "interval_semantics": {
                                "type": "string",
                                "enum": ["half_open", "closed"],
                            },
                            "intervals": {
                                **_array(
                                    _object(
                                        {
                                            **quoted,
                                            "start_date": {"type": "string", "minLength": 1},
                                            "end_date": {"type": "string", "minLength": 1},
                                        }
                                    )
                                ),
                                "minItems": 1,
                                "maxItems": 128,
                            },
                        }
                    ),
                ]
            }
        }
    )


def _recall_steps() -> dict[str, object]:
    time: dict[str, object] = {"anyOf": [{"type": "string"}, _array({"type": ["string", "null"]})]}
    filters = {
        "time": _nullable(time),
        "modality": {"type": ["string", "null"], "enum": ["text", "image", "video", "audio", None]},
        "memory_type": {
            "type": ["string", "null"],
            "enum": ["episodic", "semantic", "procedural", None],
        },
    }
    max_rows = {"type": ["integer", "null"]}
    properties: dict[str, dict[str, object]] = {
        "similar": {"query": {"type": "string"}, "k": {"type": ["integer", "null"]}},
        "match": {
            **filters,
            "terms": _array({"type": "string"}),
            "any_of": {"type": "boolean"},
            "max_rows": max_rows,
        },
        "window": {**filters, "time": time, "max_rows": max_rows},
        "neighbors": {
            "of": {"type": "string"},
            "before": {"type": ["integer", "null"]},
            "after": {"type": ["integer", "null"]},
            "max_rows": max_rows,
        },
        "entity": {"name": {"type": "string"}, "max_rows": max_rows},
    }
    return {
        "anyOf": [
            _object({"op": {"type": "string", "enum": [op]}, **fields})
            for op, fields in properties.items()
        ]
    }
