"""Pure normalization and immutable snapshots of caller-supplied content."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sensemem.content import AssetRef, Blob, ContentAtom, ContentInput
from sensemem.exceptions import ValidationError

MAX_CONTENT_PARTS = 128


def content_atoms(content: ContentInput) -> tuple[ContentAtom, ...]:
    if isinstance(content, (str, Path, Blob, AssetRef)):
        return (content,)
    if isinstance(content, bytes) or not isinstance(content, Sequence):
        raise ValidationError("content must be text, media, or an ordered sequence of them")
    atoms = tuple(content)
    if not atoms:
        raise ValidationError("content must not be empty")
    if len(atoms) > MAX_CONTENT_PARTS:
        raise ValidationError(f"content must not exceed {MAX_CONTENT_PARTS} parts")
    if any(not isinstance(atom, (str, Path, Blob, AssetRef)) for atom in atoms):
        raise ValidationError("content contains an unsupported input value")
    return atoms


def snapshot_content(content: ContentInput) -> ContentInput:
    atoms = content_atoms(content)
    if any(isinstance(atom, Path) for atom in atoms):
        raise ValidationError("prefetch paths are mutable; use Blob or AssetRef")
    return atoms[0] if len(atoms) == 1 else atoms
