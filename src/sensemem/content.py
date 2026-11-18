"""Content values and pure operations for the public SDK."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from sensemem._value_validation import (
    _SHA256,
    _media_type,
    _optional_media_type,
    _optional_text,
    _require_matching_media_type,
    _text,
)
from sensemem.exceptions import ValidationError
from sensemem.types import Modality


@dataclass(frozen=True, slots=True)
class Blob:
    """Immutable inline media bytes with an explicit IANA media type."""

    data: bytes = field(repr=False)
    media_type: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValidationError("Blob data must be non-empty bytes")
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        object.__setattr__(self, "name", _optional_text(self.name, "Blob name"))


@dataclass(frozen=True, slots=True)
class AssetRef:
    """A local asset, or an opaque ID for Memory to resolve from SQLite."""

    id: str
    modality: Modality | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    name: str | None = None
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "asset id"))
        modality = self.modality
        if modality is not None and not isinstance(modality, Modality):
            try:
                modality = Modality(modality)
            except (TypeError, ValueError):
                raise ValidationError("asset modality is invalid") from None
        if modality in {Modality.TEXT, Modality.OMNI}:
            raise ValidationError("asset modality must be image, video, or audio")
        media_type = _optional_media_type(self.media_type)
        if modality is not None and media_type is not None:
            _require_matching_media_type(modality, media_type)
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValidationError("asset size_bytes must be a non-negative integer")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValidationError("asset sha256 must be 64 lowercase hexadecimal characters")
        path = self.path
        if path is not None:
            try:
                path = Path(path)
            except TypeError:
                raise ValidationError("asset path must be a filesystem path") from None
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "name", _optional_text(self.name, "asset name"))
        object.__setattr__(self, "path", path)

    @property
    def is_resolved(self) -> bool:
        """Whether this reference carries all metadata required by a model adapter."""
        return (
            self.modality is not None
            and self.media_type is not None
            and self.size_bytes is not None
            and self.sha256 is not None
            and self.path is not None
        )


ContentAtom: TypeAlias = str | Path | Blob | AssetRef


ContentInput: TypeAlias = ContentAtom | Sequence[ContentAtom]


def _stream_content(value: object) -> ContentInput:
    if isinstance(value, str):
        return _text(value, "stream content")
    if isinstance(value, (Path, Blob, AssetRef)):
        return value
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ValidationError("stream content must be text, media, or an ordered sequence of them")
    atoms = tuple(value)
    if not atoms:
        raise ValidationError("stream content must not be empty")
    normalized: list[ContentAtom] = []
    for atom in atoms:
        if isinstance(atom, str):
            normalized.append(_text(atom, "stream text"))
        elif isinstance(atom, (Path, Blob, AssetRef)):
            normalized.append(atom)
        else:
            raise ValidationError("stream content contains an unsupported value")
    return tuple(normalized)


def _assets(value: Sequence[AssetRef], modality: Modality) -> tuple[AssetRef, ...]:
    assets = tuple(value)
    if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
        raise ValidationError("record assets must contain resolved AssetRef values")
    kinds = {asset.modality for asset in assets}
    if modality is Modality.TEXT and assets:
        raise ValidationError("text memories must not contain media assets")
    if modality in {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO} and (
        not assets or kinds != {modality}
    ):
        raise ValidationError("memory modality does not match its assets")
    if modality is Modality.OMNI and len(kinds) < 2:
        raise ValidationError("omni memories must contain at least two media modalities")
    return assets


@dataclass(frozen=True, slots=True)
class ContentPart:
    """One original text/media position, or a separately labelled derived text section."""

    kind: Literal["text", "asset", "audio_transcript", "visual_description", "model_interpretation"]
    value: str

    def __post_init__(self) -> None:
        if self.kind not in {
            "text",
            "asset",
            "audio_transcript",
            "visual_description",
            "model_interpretation",
        }:
            raise ValidationError("content part kind is invalid")
        _text(self.value, "content part value")
