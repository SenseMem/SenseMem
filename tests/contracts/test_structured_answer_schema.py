"""The optional JSON answer wire schema is closed and names every required field."""

from collections.abc import Mapping

from mindbridge.models._openai_schemas import response_schema


def test_answer_schema_has_a_closed_object_envelope_and_six_closed_variants() -> None:
    schema = response_schema("answer")
    assert schema["type"] == "object"
    assert schema["required"] == ["draft"]
    assert schema["additionalProperties"] is False

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                properties = value["properties"]
                assert isinstance(properties, Mapping)
                assert set(value["required"]) == set(properties)
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    inspect(schema)
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    draft = properties["draft"]
    assert isinstance(draft, Mapping)
    assert len(draft["anyOf"]) == 6


def test_recall_schema_declares_unresolved_gaps_and_visible_keep_aliases() -> None:
    schema = response_schema("recall_plan")
    required = schema["required"]
    assert isinstance(required, list)
    assert set(required) == {"shape", "steps", "unresolved", "keep"}
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    assert properties["keep"]["items"]["pattern"] == "^R[1-9][0-9]*$"
    assert properties["unresolved"]["maxItems"] == 8


def test_answer_schema_separates_source_date_value_and_explicit_presentation_formats() -> None:
    schema = response_schema("answer")
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    draft = properties["draft"]
    assert isinstance(draft, Mapping)
    variants = draft["anyOf"]
    assert isinstance(variants, list)
    formats: list[tuple[str, list[str]]] = []
    for variant in variants:
        fields = variant["properties"]
        if "source_value" in fields:
            assert "value" not in fields
            assert "source_value" in variant["required"]
            assert "format" in variant["required"]
            formats.append(("date", fields["format"]["enum"]))
        elif "operands" in fields or "intervals" in fields:
            assert "format" in variant["required"]
            formats.append(("calculation", fields["format"]["enum"]))
        else:
            assert "format" not in fields
    assert formats == [
        ("date", ["iso8601", "utc_datetime"]),
        ("calculation", ["with_unit", "number"]),
        ("calculation", ["with_unit", "number"]),
    ]
