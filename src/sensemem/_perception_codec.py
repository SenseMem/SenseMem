"""Storage-independent visual-description cache encoding."""

from pydantic import TypeAdapter
from pydantic import ValidationError as SchemaError

from sensemem.exceptions import ModelError, ValidationError
from sensemem.perception import VisualDescription

_VISUAL_DESCRIPTION = TypeAdapter(VisualDescription)


def encode_description(value: VisualDescription) -> str:
    return _VISUAL_DESCRIPTION.dump_json(value).decode("utf-8")


def decode_description(value: str) -> VisualDescription:
    try:
        return _VISUAL_DESCRIPTION.validate_json(value, strict=True)
    except (SchemaError, ValidationError) as error:
        raise ModelError(
            "cached visual description is invalid", reason="response_invalid"
        ) from error
