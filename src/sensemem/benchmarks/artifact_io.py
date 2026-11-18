"""Deterministic JSON encoding and durable replacement of benchmark artifact files."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile


def replace_artifact_files(files: Sequence[tuple[Path, bytes]]) -> None:
    """Stage and fsync artifacts, replace each target atomically, and clean temporary files."""
    temporary: list[tuple[Path, Path]] = []
    try:
        for target, content in files:
            with NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), target))
        for source, target in temporary:
            os.replace(source, target)
    finally:
        for source, _target in temporary:
            source.unlink(missing_ok=True)


def encode_json_bytes(value: object) -> bytes:
    """Encode deterministic UTF-8 JSON followed by one newline."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def encode_jsonl_bytes(values: Iterable[object]) -> bytes:
    """Encode ordered rows using the same canonical JSON representation."""
    return b"".join(encode_json_bytes(value) for value in values)
