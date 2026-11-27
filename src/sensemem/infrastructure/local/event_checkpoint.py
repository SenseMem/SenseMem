"""Host-selected restart checkpoints; final memories remain in the ordinary SQLite store."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import ValidationError as SchemaError

from sensemem.content import Blob
from sensemem.events import EventChunk
from sensemem.exceptions import StorageError, ValidationError
from sensemem.spatial import ObservationContext


class CheckpointBlob(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    digest: str
    media_type: str
    name: str | None
    size_bytes: int


class CheckpointChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    media: tuple[CheckpointBlob, ...]
    occurred_at: datetime
    occurred_end: datetime
    stream_id: str
    context: ObservationContext | None


class CheckpointStream(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stream_id: str
    chunks: tuple[CheckpointChunk, ...]
    last_end: datetime | None
    continuation: str | None
    continuation_chunks: int
    needs_assessment: bool


class CheckpointEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunks: tuple[CheckpointChunk, ...]
    summary: str
    next_part: int


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[5]
    model: str
    recipe: str
    data_dir: str
    max_buffer_bytes: int
    max_chunks: int
    streams: tuple[CheckpointStream, ...]
    events: tuple[CheckpointEvent, ...]
    pending: tuple[CheckpointChunk, ...]

    @field_validator("version", mode="before")
    @classmethod
    def _require_integer_version(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("event checkpoint version must be an integer")
        return value


class EventCheckpoint:
    """Atomically replace a small manifest and write each content-addressed blob once.

    The host owns the path, its permissions, backups, and one-consumer scheduling. These files
    preserve unfinished input; their existence never means a semantic event was completed.
    """

    def __init__(self, path: Path, *, data_dir: Path) -> None:
        self.path = path
        self.data_dir = str(data_dir)
        self._media = path.with_name(f"{path.name}.media")

    def load(self) -> Checkpoint | None:
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StorageError("could not read event checkpoint") from error
        try:
            return Checkpoint.model_validate_json(content)
        except SchemaError:
            raise ValidationError(
                "event checkpoint is invalid or uses an unsupported format"
            ) from None

    def pack(self, chunk: EventChunk) -> CheckpointChunk:
        media = []
        try:
            self._media.mkdir(parents=True, exist_ok=True, mode=0o700)
            for blob in chunk.media:
                digest = hashlib.sha256(blob.data).hexdigest()
                target = self._media / digest
                if not target.exists():
                    _atomic_write(target, blob.data)
                media.append(
                    CheckpointBlob(
                        digest=digest,
                        media_type=blob.media_type,
                        name=blob.name,
                        size_bytes=len(blob.data),
                    )
                )
        except OSError as error:
            raise StorageError("could not save event media") from error
        return CheckpointChunk(
            media=tuple(media),
            occurred_at=chunk.occurred_at,
            occurred_end=chunk.occurred_end,
            stream_id=chunk.stream_id,
            context=chunk.context,
        )

    def unpack(self, chunk: CheckpointChunk) -> EventChunk:
        media = []
        for blob in chunk.media:
            if len(blob.digest) != 64 or any(
                char not in "0123456789abcdef" for char in blob.digest
            ):
                raise ValidationError("event checkpoint media digest is invalid")
            try:
                data = (self._media / blob.digest).read_bytes()
            except OSError as error:
                raise StorageError("event checkpoint media is missing") from error
            if len(data) != blob.size_bytes or hashlib.sha256(data).hexdigest() != blob.digest:
                raise StorageError("event checkpoint media failed integrity verification")
            media.append(Blob(data, blob.media_type, blob.name))
        return EventChunk(
            tuple(media), chunk.occurred_at, chunk.occurred_end, chunk.stream_id, chunk.context
        )

    def verify(self, chunk: CheckpointChunk) -> None:
        """Check durable originals on reopen without retaining their bytes in the runtime."""
        for blob in chunk.media:
            digest = hashlib.sha256()
            size = 0
            try:
                with (self._media / blob.digest).open("rb") as source:
                    while data := source.read(1024 * 1024):
                        digest.update(data)
                        size += len(data)
            except OSError as error:
                raise StorageError("event checkpoint media is missing") from error
            if size != blob.size_bytes or digest.hexdigest() != blob.digest:
                raise StorageError("event checkpoint media failed integrity verification")

    def save(self, checkpoint: Checkpoint) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.path, checkpoint.model_dump_json().encode())
        except OSError as error:
            raise StorageError("could not save event checkpoint") from error

    def prune(self, checkpoint: Checkpoint) -> None:
        """Remove checkpoint media only after the manifest no longer needs it."""
        retained = {
            blob.digest
            for chunks in (
                *(stream.chunks for stream in checkpoint.streams),
                *(event.chunks for event in checkpoint.events),
                checkpoint.pending,
            )
            for chunk in chunks
            for blob in chunk.media
        }
        try:
            if self._media.exists():
                for path in self._media.iterdir():
                    if path.name not in retained:
                        path.unlink()
        except OSError as error:
            raise StorageError("could not prune event checkpoint media") from error


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish bytes only after flush; a crash leaves the previous complete manifest readable."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
