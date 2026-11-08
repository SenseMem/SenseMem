"""Normalize official benchmark releases into one local evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, cast

from sensemem.benchmarks.egotempo import EgoTempoQuestion
from sensemem.benchmarks.mem_gallery import (
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
)
from sensemem.benchmarks.mosaic import MosaicSession
from sensemem.benchmarks.personamem_v3 import PersonaMemQuery
from sensemem.benchmarks.supermemory_vqa import SuperMemoryQuestion
from sensemem.benchmarks.task_catalog import TaskSpec
from sensemem.kernel.temporal import temporal_context

ScoreKind = Literal["choice", "text", "submission"]
Limit: TypeAlias = int | float | None
_T = TypeVar("_T")
# ponytail: eight readers saturate the benchmark NVMe without unbounded file descriptors; expose
# this only if supported storage profiles demonstrate a materially different optimum.
_DIGEST_WORKERS = min(8, os.cpu_count() or 1)
_MEDIA_SUFFIXES = frozenset(
    {
        ".aac",
        ".flac",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".png",
        ".wav",
        ".webm",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One item added to a unit's physically isolated memory."""

    source_id: str
    content: tuple[str | Path, ...]
    start_seconds: float = 0.0
    end_seconds: float | None = None
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.content:
            raise ValueError("memory items need a source ID and content")
        if (
            not math.isfinite(self.start_seconds)
            or self.start_seconds < 0
            or (
                self.end_seconds is not None
                and (not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds)
            )
        ):
            raise ValueError("memory item interval is invalid")
        if self.occurred_at is not None and (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("memory item event time must include a timezone")
        if self.occurred_end is not None and (
            self.occurred_end.tzinfo is None
            or self.occurred_end.utcoffset() is None
            or self.occurred_at is None
            or self.occurred_end <= self.occurred_at
        ):
            raise ValueError("memory item event end must follow a timezone-aware event time")


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One question and the label information kept outside SenseMem."""

    question_id: str
    content: tuple[str | Path, ...]
    references: tuple[str, ...] = ()
    expected_choice: str | None = None
    score_kind: ScoreKind = "text"
    cutoff_seconds: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    reference_at: datetime | None = None
    source_question: str | None = None
    # Set when this task's own query prompt mandates a refusal wording of its own. The product
    # reports `abstained` for its own refusal sentence, which it cannot do for a wording the task
    # substituted; without this the run reports zero refusals while the model refuses.
    refusal: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.content:
            raise ValueError("evaluation questions need an ID and content")
        if self.refusal is not None and not self.refusal.strip():
            raise ValueError("a declared refusal must not be blank")
        if self.score_kind == "choice" and (
            self.expected_choice is None or self.expected_choice not in "ABCDEFGHIJ"
        ):
            raise ValueError("choice questions need an expected A-J option")
        if self.score_kind == "text" and not self.references:
            raise ValueError("text questions need at least one reference")
        if self.cutoff_seconds is not None and (
            not math.isfinite(self.cutoff_seconds) or self.cutoff_seconds < 0
        ):
            raise ValueError("question cutoff must be a non-negative finite number")
        if self.reference_at is not None and (
            self.reference_at.tzinfo is None or self.reference_at.utcoffset() is None
        ):
            raise ValueError("question reference time must include a timezone")
        if self.source_question is not None and not self.source_question.strip():
            raise ValueError("source question must not be blank")


@dataclass(frozen=True, slots=True)
class EvalUnit:
    """One independent corpus that owns one physical ``data_dir``."""

    unit_id: str
    memories: tuple[MemoryItem, ...]
    questions: tuple[EvalQuestion, ...]
    _source_ends: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.unit_id.strip() or not self.questions:
            raise ValueError("evaluation units need an ID and questions")
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("evaluation question IDs must be unique within a unit")
        ends: dict[str, float] = {}
        for memory in self.memories:
            end = math.inf if memory.end_seconds is None else memory.end_seconds
            ends[memory.source_id] = min(ends.get(memory.source_id, math.inf), end)
        object.__setattr__(self, "_source_ends", tuple(sorted(ends.values())))


@dataclass(frozen=True, slots=True)
class LoadedTask:
    """A pinned task ready for the evaluation engine."""

    spec: TaskSpec
    dataset_path: Path
    dataset_sha256: str
    units: tuple[EvalUnit, ...]
    input_sha256: Mapping[str, str] = field(default_factory=dict)
    unavailable_units: Mapping[str, str] = field(default_factory=dict)
    fallback_reference_at: datetime | None = None
    fallback_reference_question_count: int = 0

    def __post_init__(self) -> None:
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if not unit_ids or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("loaded task unit IDs must be non-empty and unique")
        if self.fallback_reference_at is not None and (
            self.fallback_reference_at.tzinfo is None
            or self.fallback_reference_at.utcoffset() is None
        ):
            raise ValueError("fallback reference time must include a timezone")
        if self.fallback_reference_question_count < 0:
            raise ValueError("fallback reference question count must be non-negative")
        if bool(self.fallback_reference_at) != bool(self.fallback_reference_question_count):
            raise ValueError("fallback reference time and question count must be set together")

    @property
    def evaluation_sha256(self) -> str:
        base = self.dataset_sha256
        if self.input_sha256:
            payload = json.dumps(
                dict(sorted(self.input_sha256.items())),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            base = hashlib.sha256(payload.encode()).hexdigest()
        if self.fallback_reference_at is None:
            return base
        payload = json.dumps(
            {
                "base_evaluation_sha256": base,
                "fallback_reference_at": self.fallback_reference_at.isoformat(),
                "fallback_reference_question_count": self.fallback_reference_question_count,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MediaPart:
    content: tuple[str | Path, ...]
    source_id: str
    start_seconds: float = 0.0
    end_seconds: float | None = None


MediaUnit: TypeAlias = tuple[MediaPart, ...]


@dataclass(frozen=True, slots=True)
class MediaTask:
    units: Mapping[str, MediaUnit]
    unavailable_units: Mapping[str, str]
    input_sha256: str


@dataclass(frozen=True, slots=True)
class MediaManifest:
    tasks: Mapping[str, MediaTask]
    # Preserve the source spelling/default omissions for hashing and artifact merging.
    source_json: str


class MediaResolver:
    """Resolve prepared manifest entries first, then exact local media names."""

    def __init__(
        self,
        task_name: str,
        root: Path | None,
        manifest: MediaManifest | None,
        manifest_directory: Path | None,
    ) -> None:
        self.task_name = task_name
        self.root = root
        task = None if manifest is None else manifest.tasks.get(task_name)
        self._manifest = {} if task is None else task.units
        self.unavailable_units = {} if task is None else task.unavailable_units
        self._manifest_directory = manifest_directory
        self._index: dict[str, list[Path]] | None = None

    def parts(
        self,
        unit_id: str,
        source_ids: Sequence[str] = (),
        *,
        relative_paths: Sequence[str] = (),
        allow_all: bool = False,
    ) -> tuple[MemoryItem, ...]:
        declared = self._manifest.get(unit_id)
        if declared is not None:
            return tuple(self._memory_part(part) for part in declared)
        if self.root is None:
            raise FileNotFoundError(
                f"{self.task_name} needs media for {unit_id}; pass --media-manifest or --media-root"
            )
        candidates = tuple(dict.fromkeys((*relative_paths, *source_ids)))
        paths = []
        for value in candidates:
            try:
                paths.append(self._resolve(value))
            except FileNotFoundError:
                if not allow_all:
                    raise
        if not paths and allow_all:
            paths.extend(self._scoped_media(candidates))
        if not paths:
            raise FileNotFoundError(f"{self.task_name} has no media for unit {unit_id}")
        return tuple(MemoryItem(Path(path).stem, (path,)) for path in paths)

    def path(self, value: str, *, exact: bool = False) -> Path:
        if exact:
            if self.root is None:
                raise FileNotFoundError(f"{self.task_name} needs a media root for {value}")
            root = self.root.resolve()
            candidate = (root / value).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return candidate
            raise FileNotFoundError(f"media not found under {self.root}: {value}")
        return self._resolve(value)

    def _memory_part(self, part: MediaPart) -> MemoryItem:
        content = tuple(
            self._manifest_path(item) if isinstance(item, Path) else item for item in part.content
        )
        return MemoryItem(part.source_id, content, part.start_seconds, part.end_seconds)

    def _manifest_path(self, value: Path) -> Path:
        path = value
        if not path.is_absolute():
            if self._manifest_directory is None:
                raise ValueError("relative media manifest paths need a manifest file directory")
            path = self._manifest_directory / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prepared media does not exist: {path}")
        return path

    def _resolve(self, value: str) -> Path:
        if self.root is None:
            raise FileNotFoundError(value)
        supplied = Path(value)
        root = self.root.resolve()
        direct = tuple(
            candidate
            for candidate in ((root / supplied).resolve(), (root / supplied.name).resolve())
            if candidate.is_relative_to(root)
        )
        for candidate in direct:
            if candidate.is_file():
                return candidate
        key = supplied.with_suffix("").as_posix().casefold()
        matches = self._media_index().get(key, [])
        if not matches and supplied.stem.isdigit():
            matches = self._media_index().get(str(int(supplied.stem)), [])
        if not matches:
            raise FileNotFoundError(f"media not found under {self.root}: {value}")
        unique = tuple(dict.fromkeys(matches))
        if len(unique) != 1:
            raise ValueError(f"media name is ambiguous under {self.root}: {value}")
        return unique[0]

    def _media_index(self) -> dict[str, list[Path]]:
        if self._index is not None:
            return self._index
        index: dict[str, list[Path]] = {}
        for path in self._media_files():
            relative = path.relative_to(cast(Path, self.root)).with_suffix("").as_posix().casefold()
            keys = {relative, path.stem.casefold()}
            if path.stem.isdigit():
                keys.add(str(int(path.stem)))
            for key in keys:
                index.setdefault(key, []).append(path)
        self._index = index
        return index

    def _media_files(self) -> tuple[Path, ...]:
        if self.root is None or not self.root.is_dir():
            return ()
        root = self.root.resolve()
        return tuple(
            resolved
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.casefold() in _MEDIA_SUFFIXES
            and (resolved := path.resolve()).is_relative_to(root)
        )

    def _scoped_media(self, candidates: Sequence[str]) -> tuple[Path, ...]:
        files = self._media_files()
        if not candidates:
            return files
        keys = {Path(value).stem.casefold() for value in candidates}
        return tuple(
            path
            for path in files
            if keys.intersection(
                part.casefold() for part in path.relative_to(cast(Path, self.root)).parts
            )
        )


def load_media_manifest(path: Path | None) -> tuple[MediaManifest | None, Path | None]:
    """Load one versioned prepared-media manifest."""
    if path is None:
        return None, None
    return parse_media_manifest(
        json.loads(path.read_text(encoding="utf-8"))
    ), path.expanduser().resolve().parent


def parse_media_manifest(payload: object) -> MediaManifest:
    """Validate prepared-media JSON without reading or resolving any media files."""
    if not isinstance(payload, dict):
        raise ValueError("media manifest must be a JSON object")
    version = payload.get("version")
    if version != 1:
        raise ValueError(f"unsupported media manifest version: {version}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("media manifest tasks must be an object")
    return MediaManifest(
        {name: _parse_media_task(name, task) for name, task in tasks.items()},
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ),
    )


def load_task(
    spec: TaskSpec,
    *,
    root: Path,
    dataset_path: Path | None = None,
    media_root: Path | None = None,
    media_manifest: MediaManifest | None = None,
    manifest_directory: Path | None = None,
    limit: Limit = None,
    offset: int = 0,
    verify_digest: bool = True,
) -> LoadedTask:
    """Load, verify, and normalize one catalog task."""
    dataset = (dataset_path or spec.dataset_path(root)).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"benchmark dataset does not exist: {dataset}")
    missing = tuple(root / path for path in spec.auxiliary if not (root / path).exists())
    if missing:
        raise FileNotFoundError(
            f"{spec.name} auxiliary input does not exist: {', '.join(map(str, missing))}"
        )
    if dataset.is_dir():
        digest, dataset_layout = _directory_input_digests(dataset)
    else:
        digest = dataset_digest(dataset)
        dataset_layout = None
    if verify_digest and spec.digest is not None and digest != spec.digest:
        raise ValueError(
            f"{spec.name} dataset digest mismatch: expected {spec.digest}, found {digest}"
        )
    resolved_media = media_root or spec.media_root(root)
    resolver = MediaResolver(spec.name, resolved_media, media_manifest, manifest_directory)
    if limit is not None and limit != -1 and limit <= 0:
        raise ValueError("limit must be -1, a positive count, or a fraction between zero and one")
    if offset < 0:
        raise ValueError("offset must not be negative")
    loader = _LOADERS[spec.name]
    loaded_units = loader(spec, dataset, resolver, root, limit, offset)
    units = tuple(_with_corpus_reference(unit) for unit in loaded_units)
    if not units or any(not unit.questions for unit in units):
        raise ValueError(f"{spec.name} produced no evaluation questions")
    inputs = {
        "dataset": digest,
        **{path: dataset_digest(root / path) for path in spec.auxiliary},
    }
    if dataset_layout is not None:
        inputs["dataset_layout"] = dataset_layout
    task_manifest = None if media_manifest is None else media_manifest.tasks.get(spec.name)
    if task_manifest is not None and task_manifest.units:
        inputs["media_manifest"] = task_manifest.input_sha256
    inputs["memory"] = _memory_digest(units)
    inputs.update(_question_media_digests(units))
    unavailable_units = _selected_unavailable_units(
        spec,
        dataset,
        resolver.unavailable_units,
        limit,
        offset,
    )
    return LoadedTask(
        spec,
        dataset,
        digest,
        units,
        inputs,
        unavailable_units,
    )


def dataset_digest(path: Path) -> str:
    """Hash one file or the concatenated file digests of a sorted directory tree."""
    if path.is_dir():
        return _directory_input_digests(path)[0]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_digests(paths: Sequence[Path]) -> dict[Path, str]:
    if len(paths) < 2:
        return {path: dataset_digest(path) for path in paths}
    with ThreadPoolExecutor(max_workers=min(_DIGEST_WORKERS, len(paths))) as pool:
        return dict(zip(paths, pool.map(dataset_digest, paths), strict=True))


def _directory_input_digests(path: Path) -> tuple[str, str]:
    files = tuple(item for item in sorted(path.rglob("*")) if item.is_file())
    if not files:
        raise ValueError(f"benchmark dataset contains no files: {path}")
    file_digests = _file_digests(files)
    joined = "".join(file_digests[item] for item in files)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    layout = "".join(
        f"{item.relative_to(path).as_posix()}\0{file_digests[item]}\n" for item in files
    )
    return digest, hashlib.sha256(layout.encode()).hexdigest()


def _question_media_digests(units: Sequence[EvalUnit]) -> dict[str, str]:
    media = tuple(
        (question.question_id, part.resolve())
        for unit in units
        for question in unit.questions
        for part in question.content
        if isinstance(part, Path)
    )
    if not media:
        return {}
    digests = _file_digests(tuple(dict.fromkeys(path for _, path in media)))
    payload = tuple((question_id, digests[path]) for question_id, path in media)
    return {
        "question_media": hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()
    }


def _memory_digest(units: Sequence[EvalUnit]) -> str:
    paths = tuple(
        path.resolve()
        for path in dict.fromkeys(
            content
            for unit in units
            for memory in unit.memories
            for content in memory.content
            if isinstance(content, Path)
        )
    )
    file_digests = _file_digests(paths)

    @cache
    def identity(value: str | Path) -> tuple[str, ...]:
        if isinstance(value, str):
            return ("text", hashlib.sha256(value.encode("utf-8")).hexdigest())
        path = value.resolve()
        return ("file", path.name, file_digests[path])

    digest = hashlib.sha256(b"[")
    for index, unit in enumerate(units):
        if index:
            digest.update(b",")
        payload = (
            unit.unit_id,
            tuple(
                (
                    memory.source_id,
                    memory.start_seconds,
                    memory.end_seconds,
                    (None if memory.occurred_at is None else memory.occurred_at.isoformat()),
                    (None if memory.occurred_end is None else memory.occurred_end.isoformat()),
                    tuple(identity(content) for content in memory.content),
                )
                for memory in unit.memories
            ),
        )
        digest.update(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode())
    digest.update(b"]")
    return digest.hexdigest()


def _locomo(
    spec: TaskSpec,
    dataset: Path,
    _media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.locomo_refined import load_locomo_refined

    units = []
    for conversation in _selected(load_locomo_refined(dataset), limit, offset):
        stored = {turn.dialog_id for turn in conversation.turns}
        memories = tuple(
            MemoryItem(
                turn.dialog_id,
                (
                    f"[{turn.occurred_at.isoformat()}] {turn.speaker} said: {turn.text}"
                    + (
                        f"\n{turn.speaker} shared an image described as: {turn.image_caption}"
                        if turn.image_caption
                        else ""
                    ),
                ),
                occurred_at=turn.occurred_at,
            )
            for turn in conversation.turns
        )
        questions = tuple(
            EvalQuestion(
                question.question_id,
                (_free_text_prompt(question.question),),
                question.reference_answers,
                metadata={
                    "category": question.category,
                    # The release's `evidence` names the dialogue turns that carry the
                    # answer as `dia_id` values, and `dia_id` is this unit's memory
                    # source ID, so gold evidence is exact here rather than derived.
                    # Published IDs are matched against the stored turns rather than
                    # trusted: an ID that names no turn is reported separately, so a
                    # vocabulary mismatch surfaces as a join failure instead of a
                    # believable recall number.
                    "evidence_ids": tuple(
                        value for value in question.evidence_dialog_ids if value in stored
                    ),
                    "unresolved_evidence_ids": tuple(
                        value for value in question.evidence_dialog_ids if value not in stored
                    ),
                },
                source_question=question.question,
            )
            for question in conversation.questions
        )
        units.append(EvalUnit(conversation.sample_id, memories, questions))
    return tuple(units)


def _m3(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.m3_bench import load_m3_bench

    units = []
    for video in _selected(load_m3_bench(dataset), limit, offset):
        if video.video_id in media.unavailable_units:
            continue
        memories = media.parts(
            video.video_id,
            (video.video_id,),
            relative_paths=(video.video_path,),
        )
        questions = tuple(
            EvalQuestion(
                question.question_id,
                _free_text_parts(question.question),
                (question.reference_answer,),
                cutoff_seconds=question.cutoff_seconds,
                metadata={"question_types": question.question_types},
                source_question=question.question,
            )
            for question in video.questions
        )
        _require_timestamped_media(spec.name, video.video_id, memories, questions)
        units.append(EvalUnit(video.video_id, memories, questions))
    return tuple(units)


def _videozerobench(
    _spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.videozerobench import level_queries, load_videozerobench

    grouped: dict[str, list[EvalQuestion]] = {}
    for question in _selected(load_videozerobench(dataset), limit, offset):
        questions = grouped.setdefault(question.video_id, [])
        for level, query in level_queries(question):
            questions.append(
                EvalQuestion(
                    f"{question.question_id}/level-{level}",
                    (query,),
                    (question.answer,),
                    source_question=question.question,
                    metadata={
                        "source_question_id": str(question.question_id),
                        "level": level,
                        "category": question.category,
                        "language": question.language,
                        "evidence_span": question.evidence_span,
                        "annotation_capabilities": question.annotation_capabilities,
                        "evidence_windows": [
                            item.model_dump() for item in question.evidence_windows
                        ],
                        "evidence_boxes": [item.model_dump() for item in question.evidence_boxes],
                    },
                )
            )
    return tuple(
        EvalUnit(video_id, media.parts(video_id, (video_id,)), tuple(questions))
        for video_id, questions in grouped.items()
    )


def _icm_bench(
    _spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.icm_bench import ICM_NO_ASR_CLIPS, clip_position, load_icm_bench

    release = root / "icm-bench"
    rows = [
        json.loads(line)
        for line in (release / "videos/metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    clips = sorted((str(row["video_id"]) for row in rows), key=clip_position)
    if clips != [f"clip_{position:03d}" for position in range(839)]:
        raise ValueError("ICM timeline must contain exactly clip_000 through clip_838")
    transcripts = release / "resources/asr_transcripts"
    expected = {
        f"clip_{position:03d}.srt" for position in range(839) if position not in ICM_NO_ASR_CLIPS
    }
    actual = {path.name for path in transcripts.glob("*.srt") if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"ICM speakerless ASR roster mismatch: missing {sorted(expected - actual)}, "
            f"unexpected {sorted(actual - expected)}"
        )
    questions = _selected(load_icm_bench(dataset), limit, offset)
    memories = []
    for clip in clips:
        position = clip_position(clip)
        transcript = transcripts / f"{clip}.srt"
        # Speaker-labeled transcripts and characters.json are scorer-only resources.
        text = f"Video {clip}"
        if transcript.is_file():
            text += "\nSpeakerless ASR:\n" + transcript.read_text(encoding="utf-8")
        memories.append(
            MemoryItem(
                clip,
                (text, media.path(f"{clip}.mp4")),
                start_seconds=float(position),
                end_seconds=float(position + 1),
            )
        )
    for question in questions:
        if not set(question.evidence_video_ids).issubset(clips):
            raise ValueError("ICM question references absent evidence clips")
        if question.before_clip is not None and question.before_clip not in clips:
            raise ValueError("ICM question cutoff is absent from the timeline")
    return (
        EvalUnit(
            "life-album",
            tuple(memories),
            tuple(
                EvalQuestion(
                    question.question_id,
                    _free_text_parts(question.question),
                    (question.reference_answer,),
                    cutoff_seconds=None
                    if question.before_clip is None
                    else float(clip_position(question.before_clip) + 1),
                    metadata={
                        "category": question.category,
                        "evidence_ids": question.evidence_video_ids,
                        "target_character_ids": question.target_character_ids,
                        "before_clip": question.before_clip,
                    },
                    source_question=question.question,
                )
                for question in questions
            ),
        ),
    )


def _egotempo(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.egotempo import load_egotempo
    from sensemem.benchmarks.prompts import EGOTEMPO_QUERY_PROMPT

    grouped: dict[str, list[EgoTempoQuestion]] = {}
    for question in _selected(load_egotempo(dataset), limit, offset):
        grouped.setdefault(question.clip_id, []).append(question)
    return tuple(
        EvalUnit(
            clip_id,
            media.parts(clip_id, (clip_id,)),
            tuple(
                EvalQuestion(
                    question.question_id,
                    (EGOTEMPO_QUERY_PROMPT.text.format(question=question.question),),
                    (question.reference_answer,),
                    metadata={"question_type": question.question_type},
                    source_question=question.question,
                )
                for question in questions
            ),
        )
        for clip_id, questions in grouped.items()
    )


def _memlens(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.memlens import load_memlens, load_memlens_agent_subset
    from sensemem.benchmarks.prompts import MEMLENS_QUERY_PROMPT

    questions = load_memlens(dataset)
    subset_path = root / spec.auxiliary[0]
    if subset_path.exists():
        wanted = set(load_memlens_agent_subset(subset_path))
        questions = tuple(question for question in questions if question.question_id in wanted)
    images = None if media.root is None else media.root.expanduser().resolve()
    units = []
    for question in _selected(questions, limit, offset):
        memories = tuple(
            MemoryItem(
                turn.turn_id,
                (
                    f"[{session.occurred_at.isoformat()}] {turn.role}: {turn.content}"
                    + "".join(
                        f"\nImage {image.source_file}: {image.caption}"
                        for image in turn.images
                        if image.caption
                    ),
                    *_present_images(images, tuple(image.source_file for image in turn.images)),
                ),
                occurred_at=session.occurred_at,
            )
            for session in question.sessions
            for turn in session.turns
        )
        prompt = MEMLENS_QUERY_PROMPT.text.format(question=question.question)
        # The release labels the answer at session level only, and one memory here is one
        # turn, so the gold label is a group of stored turn IDs per answer session: a session
        # is retrieved when any turn of it is ranked. Refusal rows name no answer session
        # and so carry no gold, which is correct -- there is nothing to retrieve.
        answer_sessions = tuple(
            session for session in question.sessions if session.is_answer_session
        )
        units.append(
            EvalUnit(
                question.question_id,
                memories,
                (
                    EvalQuestion(
                        question.question_id,
                        (prompt,),
                        (question.reference_answer,),
                        metadata={
                            "question_type": question.question_type,
                            "question_subtype": question.question_subtype,
                            "old_answer": question.old_answer,
                            "answer_session_ids": tuple(
                                session.session_id for session in answer_sessions
                            ),
                            "evidence_groups": tuple(
                                tuple(turn.turn_id for turn in session.turns)
                                for session in answer_sessions
                                if session.turns
                            ),
                        },
                        reference_at=question.question_date,
                        source_question=question.question,
                        refusal=MEMLENS_QUERY_PROMPT.refusal,
                    ),
                ),
            )
        )
    return tuple(units)


def _present_images(root: Path | None, names: Sequence[str]) -> tuple[Path, ...]:
    """Return the published image files that are present, ignoring the absent ones.

    MEMLENS hides most of its answers inside the images, but the release ships them
    as a separate archive, so a checkout without them keeps the published captions
    as its only visual evidence instead of failing.
    """
    if root is None:
        return ()
    return tuple(
        path
        for name in names
        if (path := (root / name).resolve()).suffix.casefold() in _MEDIA_SUFFIXES
        and path.is_file()
        and path.is_relative_to(root)
    )


# A fixed, timezone-aware origin for media whose only clock is its own timeline. Any instant
# works as long as it is the same one for every clip in a unit and every replay of the run.
_MEDIA_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _mm_lifelong(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.mm_lifelong import MMLifelongSplit, load_mm_lifelong

    split = cast(MMLifelongSplit, spec.variant)
    questions = _selected(load_mm_lifelong(dataset, split), limit, offset)
    # The split's only clock is the video timeline; the offsets alone reached the store as
    # metadata, so every memory had no `occurred_at` and the answerer resolved "before"/"after"
    # against the ingest wall clock. ponytail: only this adapter anchors offsets; lift it into
    # `_memory_part` once the other media tasks' causal cutoffs are checked against it.
    memories = tuple(
        replace(
            item,
            occurred_at=_MEDIA_EPOCH + timedelta(seconds=item.start_seconds),
            occurred_end=(
                None
                if item.end_seconds is None
                else _MEDIA_EPOCH + timedelta(seconds=item.end_seconds)
            ),
        )
        for item in media.parts(split, allow_all=True)
    )
    return (
        EvalUnit(
            split,
            memories,
            tuple(
                EvalQuestion(
                    str(question.index),
                    (_free_text_prompt(question.question),),
                    (question.reference_answer,),
                    metadata={
                        "question_type": question.question_type,
                        "temporal_certificate": question.temporal_certificate,
                        "reference_intervals": question.reference_intervals,
                    },
                    source_question=question.question,
                )
                for question in questions
            ),
        ),
    )


def _supermemory(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.supermemory_vqa import load_supermemory_vqa

    grouped: dict[int, list[SuperMemoryQuestion]] = {}
    selected = tuple(
        question
        for question in load_supermemory_vqa(dataset)
        if spec.variant != "subject-1" or question.subject == 1
    )
    for question in _selected(selected, limit, offset):
        grouped.setdefault(question.subject, []).append(question)
    units = []
    for subject, raw_questions in sorted(grouped.items()):
        source_ids = tuple(
            dict.fromkeys(
                source for question in raw_questions for source in question.source_video_ids
            )
        )
        unit_id = f"subject-{subject}"
        memories = media.parts(unit_id, source_ids)
        questions = tuple(
            EvalQuestion(
                str(question.question_id),
                (_choice_prompt(question.question, question.choices),),
                expected_choice="ABCD"[question.correct_option_index],
                score_kind="choice",
                cutoff_seconds=question.question_ended_at.timestamp(),
                metadata={
                    "skill": question.skill,
                    "is_answerable": question.is_answerable,
                    "unanswerable_choice": "ABCD"[question.unanswerable_option_index],
                    "choices": question.choices,
                },
            )
            for question in raw_questions
        )
        _require_timestamped_media(spec.name, unit_id, memories, questions)
        units.append(EvalUnit(unit_id, memories, questions))
    return tuple(units)


def _atm(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.atm_bench import (
        atm_capture_time,
        atm_email_block,
        atm_sgm_block,
        load_atm_bench,
        load_atm_emails,
        load_atm_sgm,
    )
    from sensemem.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT

    email_path = root / spec.auxiliary[0]
    memories = [
        MemoryItem(email.email_id, (atm_email_block(email),), occurred_at=email.occurred_at)
        for email in load_atm_emails(email_path)
    ]
    if cast(str, spec.variant).endswith("_sgm"):
        image_path, video_path = (root / item for item in spec.auxiliary[1:])
        for record in (*load_atm_sgm(image_path), *load_atm_sgm(video_path)):
            memories.append(
                MemoryItem(
                    record.media_id,
                    (atm_sgm_block(record),),
                    occurred_at=record.occurred_at,
                )
            )
    else:
        memories.extend(
            replace(item, occurred_at=atm_capture_time(item.source_id))
            for item in media.parts(cast(str, spec.variant), allow_all=True)
        )
    questions = tuple(
        EvalQuestion(
            question.question_id,
            _query_parts(
                ATM_BENCH_QUERY_PROMPT.text,
                question.question,
            ),
            (question.reference_answer,),
            metadata={"qtype": question.qtype, "evidence_ids": question.evidence_ids},
            source_question=question.question,
            refusal=ATM_BENCH_QUERY_PROMPT.refusal,
        )
        for question in _selected(load_atm_bench(dataset), limit, offset)
    )
    return (EvalUnit(cast(str, spec.variant), tuple(memories), questions),)


def _mem_gallery(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.mem_gallery import load_mem_gallery
    from sensemem.benchmarks.prompts import (
        MEM_GALLERY_QUERY_PROMPT,
        mem_gallery_format_constraint,
        mem_gallery_refusal,
    )

    units = []
    for topic in _selected(load_mem_gallery(dataset), limit, offset):
        memories = tuple(
            _gallery_memory(topic.profile.name, session, round_, media)
            for session in topic.sessions
            for round_ in session.rounds
        )
        questions = tuple(
            _gallery_question(
                topic.profile.name,
                question,
                media,
                MEM_GALLERY_QUERY_PROMPT.text,
                mem_gallery_format_constraint(question.point),
                mem_gallery_refusal(question.point),
            )
            for question in topic.questions
        )
        units.append(EvalUnit(topic.topic, memories, questions))
    return tuple(units)


def _gallery_memory(
    profile_name: str,
    session: MemGallerySession,
    round_: MemGalleryRound,
    media: MediaResolver,
) -> MemoryItem:
    text = (
        f"[{session.occurred_at.date().isoformat()}] {profile_name}: {round_.user}"
        f"\nAssistant: {round_.assistant}"
    )
    if round_.image_id is not None:
        text = f"{text}\nImage ID: {round_.image_id}"
    image: Path | None = None
    if round_.image_path:
        image = media.path(round_.image_path)
    return MemoryItem(
        round_.round_id,
        (text,) if image is None else (text, image),
        occurred_at=session.occurred_at,
    )


def _gallery_question(
    profile_name: str,
    question: MemGalleryQuestion,
    media: MediaResolver,
    prompt_template: str,
    format_constraint: str,
    refusal: str | None,
) -> EvalQuestion:
    parts = _split_parts(
        prompt_template.format(
            question=question.question,
            speaker_a=profile_name,
            speaker_b="the assistant",
            format_constraint=format_constraint,
        ).strip()
    )
    image: Path | None = None
    if question.question_image_path:
        image = media.path(question.question_image_path)
    return EvalQuestion(
        question.question_id,
        parts if image is None else (*parts, image),
        (question.reference_answer,),
        metadata={"point": question.point, "clue_ids": question.clue_round_ids},
        source_question=question.question,
        refusal=refusal,
    )


def _query_parts(template: str, question: str, **values: str) -> tuple[str, ...]:
    before, marker, after = template.partition("{question}")
    if not marker:
        raise ValueError("benchmark query template needs a question placeholder")
    return tuple(
        part
        for part in (
            question.strip(),
            before.format(**values).strip(),
            after.format(**values).strip(),
        )
        if part
    )


# SenseMem caps one memory or query part at 65,536 characters
# (`sensemem.kernel.validation.MAX_TEXT_CHARACTERS`). Split long passages into blocks
# well under the cap so each block is a usable retrieval unit.
_TEXT_BLOCK_CHARACTERS = 8_000

# Recorded in place of a reference answer where a release publishes none.
# 857 of PersonaMem-v3's queries -- almost all of them sycophancy rows --
# have no `example_response`, and none of its metrics reads a reference, so
# the slot would otherwise repeat the question back as its own gold answer.
_NO_REFERENCE = "(no reference answer published for this row)"
_MAX_PART_CHARACTERS = 60_000


def _text_blocks(text: str, limit: int) -> tuple[str, ...]:
    """Group a document into blocks of at most `limit` characters.

    Paragraphs are kept whole and packed greedily; a single paragraph longer
    than the limit is cut on the limit, because some records are one unbroken
    table or bibliography with no paragraph structure to key off.
    """
    blocks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        chunk = paragraph.strip()
        if not chunk:
            continue
        while len(chunk) > limit:
            if current:
                blocks.append(current)
                current = ""
            blocks.append(chunk[:limit])
            chunk = chunk[limit:]
        if not current:
            current = chunk
        elif len(current) + len(chunk) + 2 <= limit:
            current = f"{current}\n\n{chunk}"
        else:
            blocks.append(current)
            current = chunk
    if current:
        blocks.append(current)
    return tuple(blocks)


def _text_memories(
    source_id: str,
    text: str,
    *,
    occurred_at: datetime | None = None,
    end_seconds: float | None = None,
) -> tuple[MemoryItem, ...]:
    """Store one passage as memories that each fit the product's part limit.

    An oversized part is not merely dropped. `memory.add` rejects it, the
    runner records an ingest failure for the unit, and `_apply_judges` then
    skips **every** question in that unit -- so one long turn silently voids a
    whole conversation. Text under the block size stays unchanged; only longer
    passages are split.
    """
    blocks = _text_blocks(text, _TEXT_BLOCK_CHARACTERS)
    if len(blocks) == 1:
        return (MemoryItem(source_id, blocks, occurred_at=occurred_at, end_seconds=end_seconds),)
    return tuple(
        MemoryItem(
            f"{source_id}_B{index:04d}",
            (block,),
            occurred_at=occurred_at,
            end_seconds=end_seconds,
        )
        for index, block in enumerate(blocks)
    )


def _ama_bench(
    spec: TaskSpec,
    dataset: Path,
    resolver: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.ama_bench import load_ama_bench

    del spec, resolver, root
    return tuple(
        EvalUnit(
            str(episode.episode_id),
            (
                *_text_memories("task", f"Task: {episode.task}"),
                *(
                    memory
                    for index, step in enumerate(episode.trajectory)
                    for memory in _text_memories(f"step-{index}", step.text())
                ),
            ),
            tuple(
                EvalQuestion(
                    question.question_uuid,
                    (
                        f"## Questions\nQuestion 1: {question.question}\n\n"
                        "## Instructions\nProvide a direct and concise answer.\n\n"
                        "Answer[1]: [your answer here]",
                    ),
                    references=(question.answer,),
                    source_question=question.question,
                    metadata={
                        "episode_id": str(episode.episode_id),
                        "task_description": episode.task,
                        "task_type": episode.task_type,
                        "domain": episode.domain,
                        "qa_type": question.type,
                        "question_index": index,
                    },
                )
                for index, question in enumerate(episode.qa_pairs)
            ),
        )
        for episode in _selected(load_ama_bench(dataset), limit, offset)
    )


def _longmemeval_v2(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.longmemeval_v2 import (
        LongMemEvalV2Question,
        load_longmemeval_v2_haystacks,
        load_longmemeval_v2_questions,
        load_longmemeval_v2_trajectories,
        question_prompt,
    )

    questions = load_longmemeval_v2_questions(dataset)
    haystacks = load_longmemeval_v2_haystacks(
        root / f"longmemeval-v2/haystacks/lme_v2_{spec.variant}.json"
    )
    if set(haystacks) != {question.id for question in questions}:
        raise ValueError("LongMemEval-V2 questions and haystack IDs must agree")
    selected = _selected(questions, limit, offset)
    if not selected:
        return ()
    trajectories = load_longmemeval_v2_trajectories(
        root / "longmemeval-v2/trajectories.jsonl",
        {trajectory_id for question in selected for trajectory_id in haystacks[question.id]},
    )
    # Shared ordered haystacks own one store, just as in the upstream harness.
    # Different domains or orders never share a physical memory directory.
    groups: dict[tuple[str, tuple[str, ...]], list[LongMemEvalV2Question]] = {}
    for question in selected:
        ids = haystacks[question.id]
        if any(trajectories[trajectory_id].domain != question.domain for trajectory_id in ids):
            raise ValueError("LongMemEval-V2 haystack contains a cross-domain trajectory")
        groups.setdefault((question.domain, ids), []).append(question)

    # Medium has 447 ordered haystacks over 1,473 distinct trajectories. Reuse
    # immutable state items instead of copying their observations into every unit.
    trajectory_memories: dict[str, tuple[MemoryItem, ...]] = {}
    for trajectory_id, trajectory in trajectories.items():
        memories: list[MemoryItem] = []
        for state in trajectory.states:
            text = (
                f"Trajectory: {trajectory_id}\nGoal: {trajectory.goal}\n"
                f"Outcome: {trajectory.outcome}\nStart URL: {trajectory.start_url}\n"
                f"State: {state.state_index}\nStep: {state.step}\nURL: {state.url}\n"
                f"Action: {state.action or ''}\nThought: {state.thought or ''}\n"
                f"Observation:\n{state.accessibility_tree}"
            )
            items = _text_memories(f"T{trajectory_id}_S{state.state_index:04d}", text)
            if state.screenshot is not None:
                items = (
                    replace(
                        items[0],
                        content=(*items[0].content, media.path(state.screenshot, exact=True)),
                    ),
                    *items[1:],
                )
            memories.extend(items)
        trajectory_memories[trajectory_id] = tuple(memories)
    units = []
    for (domain, ids), grouped_questions in groups.items():
        eval_questions = tuple(
            EvalQuestion(
                question.id,
                (question_prompt(question),)
                + (() if question.image is None else (media.path(question.image, exact=True),)),
                (question.answer,),
                metadata={
                    "domain": question.domain,
                    "environment": question.environment,
                    "question_type": question.question_type,
                    "eval_function": question.eval_function,
                    "abstention": question.question_type.endswith("-abs"),
                    "haystack_size": len(ids),
                },
                source_question=question.question,
                refusal=r"\boxed{UNKNOWN}",
            )
            for question in grouped_questions
        )
        haystack_id = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        units.append(
            EvalUnit(
                f"{domain}-{haystack_id[:16]}",
                tuple(item for trajectory_id in ids for item in trajectory_memories[trajectory_id]),
                eval_questions,
            )
        )
    return tuple(units)


def _mosaic(
    _spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks._official.mosaic_scoring import answer_system_prompt
    from sensemem.benchmarks.mosaic import load_mosaic

    units = []
    for persona in load_mosaic(dataset, limit=limit, offset=offset):
        persona_id = persona.persona_id
        memories, facts = _mosaic_memories(persona.sessions, media)
        questions = []
        for question in persona.questions:
            required = tuple(
                fact.fact_key
                for fact in question.evidence_plan
                if fact.required_for_answer and fact.available_at_query
            )
            question_sessions = set(question.haystack_session_ids)
            groups = tuple(
                tuple(
                    dict.fromkeys(
                        source_id
                        for session_id, source_id in facts.get(key, ())
                        if session_id in question_sessions
                    )
                )
                for key in required
            )
            questions.append(
                EvalQuestion(
                    question.question_id,
                    (
                        answer_system_prompt(question.question_type)
                        + f"\nQuestion: {question.question}\nAnswer:",
                    ),
                    (question.answer, *question.answer_aliases),
                    metadata={
                        "persona_id": persona_id,
                        "question_type": question.question_type,
                        "answerability": question.answerability,
                        "evidence_groups": tuple(group for group in groups if group),
                        "unresolved_evidence_ids": tuple(
                            key for key, group in zip(required, groups, strict=True) if not group
                        ),
                        "retrieval_granularity": "native_chunk",
                    },
                    reference_at=datetime.strptime(question.question_date, "%Y/%m/%d").replace(
                        tzinfo=timezone.utc
                    ),
                    source_question=question.question,
                    refusal="I don't know.",
                )
            )
        units.append(EvalUnit(persona_id, tuple(memories), tuple(questions)))
    return tuple(units)


def _mosaic_memories(
    sessions: Sequence[MosaicSession],
    media: MediaResolver,
) -> tuple[tuple[MemoryItem, ...], dict[str, list[tuple[str, str]]]]:
    """Build the official native chunks and their scorer-only fact-to-source map."""
    from sensemem.benchmarks.mosaic import mosaic_media_path

    memories = []
    facts: dict[str, list[tuple[str, str]]] = {}
    for session in sessions:
        occurred_at = datetime.strptime(session.timestamp, "%Y/%m/%d (%a) %H:%M").replace(
            tzinfo=timezone.utc
        )
        text_parts = []
        for index, message in enumerate(session.messages):
            attachment = message.multimodal_content
            if attachment is not None:
                source_id = f"{session.session_id}_mm{index}"
                memories.append(
                    MemoryItem(
                        source_id,
                        (
                            *_split_parts(message.content, max_characters=16_000),
                            media.path(mosaic_media_path(attachment.media_path), exact=True),
                        ),
                        occurred_at=occurred_at,
                    )
                )
            else:
                source_id = session.session_id
                if message.content.strip():
                    text_parts.append(f"[{message.role}] {message.content}")
            if message.fact_key:
                facts.setdefault(message.fact_key, []).append((session.session_id, source_id))
        if text_parts:
            # The SDK joins string parts before enforcing its total text bound. Separate
            # records retain all of an oversized source chunk and its original evidence ID.
            memories.extend(
                MemoryItem(
                    session.session_id,
                    (part,),
                    occurred_at=occurred_at,
                )
                for part in _split_parts("\n".join(text_parts), max_characters=16_000)
            )
    return tuple(memories), facts


def _es_memeval(
    _spec: TaskSpec,
    dataset: Path,
    _media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.es_memeval import load_es_memeval
    from sensemem.benchmarks.prompts import ES_MEMEVAL_QA_QUERY_PROMPT

    units = []
    for seeker in _selected(load_es_memeval(dataset), limit, offset):
        # ES-MemEval's published RAG QA baseline indexes one document per
        # dialogue session. Preserve that granularity and its exact rendering:
        # the seeker's name identifies human turns and "supporter" identifies
        # assistant turns, while the date is prepended at retrieval time.
        memories = tuple(
            MemoryItem(
                session.session_id,
                (
                    f"[{session.occurred_at.date().isoformat()}]\n"
                    + "\n".join(
                        f"{seeker.name if turn.role == 'seeker' else 'supporter'}: {turn.content}"
                        for turn in session.turns
                    ),
                ),
                occurred_at=session.occurred_at,
            )
            for session in seeker.sessions
        )
        turn_sessions = {
            turn.source_id: session.session_id
            for session in seeker.sessions
            for turn in session.turns
        }
        questions = tuple(
            EvalQuestion(
                question.question_id,
                (ES_MEMEVAL_QA_QUERY_PROMPT.text.format(question=question.question),),
                (question.answer,),
                metadata={
                    "capability": question.capability,
                    "question_group": question.group_id,
                    # Upstream's session retrieval evaluator marks a session
                    # relevant when any visible turn ID in it occurs in the
                    # question's evidence list. Event annotations are scorer-side
                    # data and never enter memory; a turn label that names no
                    # stored turn is a malformed release ID and is reported the
                    # way every adapter reports one, so `unresolved_gold_evidence_ids`
                    # keeps exposing the vocabulary mismatch it exists for.
                    "evidence_ids": tuple(
                        dict.fromkeys(
                            turn_sessions[value]
                            for value in question.evidence
                            if value in turn_sessions
                        )
                    ),
                    "unresolved_evidence_ids": tuple(
                        value
                        for value in question.evidence
                        if value not in turn_sessions and "event" not in value.casefold()
                    ),
                    "abstention": question.capability == "abstention",
                    "retrieval_granularity": "session",
                },
                source_question=question.question,
                refusal=ES_MEMEVAL_QA_QUERY_PROMPT.refusal,
            )
            for question in seeker.questions
        )
        units.append(EvalUnit(seeker.seeker_id, memories, questions))
    return tuple(units)


def _personamem_v3(
    _spec: TaskSpec,
    dataset: Path,
    _media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from sensemem.benchmarks.personamem_v3 import (
        RANKING_TASK_TYPES,
        load_personamem_v3,
        render_candidates,
        render_event,
    )
    from sensemem.benchmarks.prompts import (
        PERSONAMEM_V3_QUERY_PROMPT,
        PERSONAMEM_V3_RANKING_PROMPT,
    )

    units = []
    for persona in _selected(load_personamem_v3(dataset), limit, offset):
        memories = tuple(
            item
            for event in persona.events
            # `end_seconds` carries the event's moment so the runner's cutoff
            # loop ingests only what happened before each query -- the causal
            # mask the release requires.
            for item in _text_memories(
                event.event_id,
                render_event(event),
                occurred_at=event.occurred_at,
                end_seconds=float(event.timestamp),
            )
        )
        questions = tuple(
            _personamem_question(
                query,
                RANKING_TASK_TYPES,
                render_candidates,
                PERSONAMEM_V3_RANKING_PROMPT.text,
                PERSONAMEM_V3_QUERY_PROMPT.text,
            )
            for query in persona.queries
        )
        units.append(EvalUnit(persona.persona_id, memories, questions))
    return tuple(units)


def _personamem_question(
    query: PersonaMemQuery,
    ranking_task_types: frozenset[str],
    render_candidates: Callable[[Sequence[Mapping[str, object]]], str],
    ranking_template: str,
    query_template: str,
) -> EvalQuestion:
    history = "\n".join(f"{turn.role}: {turn.content}" for turn in query.prior_conversation)
    question = query.user_query if not history else f"{history}\n\n{query.user_query}"
    if query.task_type in ranking_task_types and query.candidates:
        prompt = ranking_template.format(
            query=query.user_query,
            slate=render_candidates(query.candidates),
            count=len(query.candidates),
        )
    else:
        prompt = query_template.format(question=question)
    return EvalQuestion(
        query.query_id,
        _split_parts(prompt),
        (query.example_response or query.groundtruth_preference or _NO_REFERENCE,),
        # Strictly before the query's moment: `_run_unit` ingests memories whose
        # end is `<=` the cutoff, and the release masks events at or after the
        # query. `nextafter` is the exact float below it.
        cutoff_seconds=math.nextafter(float(query.timestamp), -math.inf),
        metadata={
            "task_family": query.task_family,
            "task_type": query.task_type,
            "query_kind": query.query_kind,
            "expected_behavior": query.expected_behavior,
            "rubric_tags": query.rubric_tags,
            "groundtruth_preference": query.groundtruth_preference,
            "distractor_preferences": query.distractor_preferences,
            "candidate_count": len(query.candidates),
            "positive_indexes": query.positive_indexes,
            "negative_indexes": query.negative_indexes,
            "judge_evidence": query.judge_evidence,
        },
        reference_at=query.asked_at,
        source_question=query.user_query[:_MAX_PART_CHARACTERS],
    )


def _split_parts(text: str, *, max_characters: int = _MAX_PART_CHARACTERS) -> tuple[str, ...]:
    """Split one prompt into parts SenseMem will accept.

    Splitting preserves prompts longer than the per-part character limit.
    """
    if len(text) <= max_characters:
        return (text,)
    return tuple(
        text[offset : offset + max_characters] for offset in range(0, len(text), max_characters)
    )


_LOADERS = {
    "ama-bench": _ama_bench,
    "mosaic": _mosaic,
    "icm-bench": _icm_bench,
    "locomo-refined": _locomo,
    "m3-bench-robot": _m3,
    "m3-bench-web": _m3,
    "videozerobench": _videozerobench,
    "egotempo": _egotempo,
    "memlens-32k": _memlens,
    "memlens-64k": _memlens,
    "memlens-128k": _memlens,
    "memlens-256k": _memlens,
    "mm-lifelong-day-test": _mm_lifelong,
    "mm-lifelong-week-test": _mm_lifelong,
    "mm-lifelong-month-train": _mm_lifelong,
    "mm-lifelong-month-val": _mm_lifelong,
    "supermemory-vqa": _supermemory,
    "atm-bench-main": _atm,
    "atm-bench-hard": _atm,
    "atm-bench-main-sgm": _atm,
    "atm-bench-hard-sgm": _atm,
    "mem-gallery": _mem_gallery,
    "longmemeval-v2-small": _longmemeval_v2,
    "longmemeval-v2-medium": _longmemeval_v2,
    "es-memeval-qa": _es_memeval,
    "personamem-v3": _personamem_v3,
}


def _parse_media_task(task_name: str, task: object) -> MediaTask:
    if task_name == "mosaic":
        raise ValueError(
            "MOSAIC does not support --media-manifest; use original media with --media-root"
        )
    if not isinstance(task, dict):
        raise ValueError(f"media manifest task {task_name} must be an object")
    units = task.get("units")
    if not isinstance(units, dict):
        raise ValueError(f"media manifest task {task_name} units must be an object")
    unavailable = task.get("unavailable_units", {})
    if not isinstance(unavailable, dict) or any(
        not isinstance(unit_id, str)
        or not unit_id.strip()
        or not isinstance(reason, str)
        or not reason.strip()
        for unit_id, reason in unavailable.items()
    ):
        raise ValueError(
            f"media manifest task {task_name} unavailable_units must map IDs to reasons"
        )
    parsed = {}
    for unit_id, parts in units.items():
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("media manifest unit IDs must be non-empty text")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"media manifest unit {unit_id} must be a non-empty list")
        parsed[unit_id] = tuple(
            _parse_media_part(unit_id, index, part) for index, part in enumerate(parts)
        )
    encoded = json.dumps(
        units, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return MediaTask(parsed, unavailable, hashlib.sha256(encoded).hexdigest())


def _parse_media_part(unit_id: str, index: int, value: object) -> MediaPart:
    if not isinstance(value, dict):
        raise ValueError(f"media manifest part {unit_id}[{index}] must be an object")
    for name in ("path", "text", "source_id"):
        supplied = value.get(name)
        if supplied is not None and not isinstance(supplied, str):
            raise ValueError(f"media manifest part {unit_id}[{index}] {name} must be text")
    path_value = value.get("path")
    text_value = value.get("text")
    content: list[str | Path] = []
    if isinstance(text_value, str) and text_value.strip():
        content.append(text_value.strip())
    if isinstance(path_value, str) and path_value.strip():
        content.append(Path(path_value).expanduser())
    if not content:
        raise ValueError(f"media manifest part {unit_id}[{index}] needs path or text")
    start = _finite_number(value.get("start_seconds", 0.0), "start_seconds")
    end_value = value.get("end_seconds")
    end = None if end_value is None else _finite_number(end_value, "end_seconds")
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"media manifest part {unit_id}[{index}] has an invalid interval")
    source = value.get("source_id")
    source_id = (
        source.strip() if isinstance(source, str) and source.strip() else f"{unit_id}:{index:05d}"
    )
    return MediaPart(tuple(content), source_id, start, end)


def _selected_unavailable_units(
    spec: TaskSpec,
    dataset: Path,
    unavailable_units: Mapping[str, str],
    limit: Limit,
    offset: int,
) -> Mapping[str, str]:
    if not unavailable_units or spec.name not in {"m3-bench-robot", "m3-bench-web"}:
        return unavailable_units
    from sensemem.benchmarks.m3_bench import load_m3_bench

    selected_ids = {video.video_id for video in _selected(load_m3_bench(dataset), limit, offset)}
    return {
        unit_id: reason for unit_id, reason in unavailable_units.items() if unit_id in selected_ids
    }


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"media manifest {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"media manifest {name} must be a finite number")
    return result


def _with_corpus_reference(unit: EvalUnit) -> EvalUnit:
    """Anchor questions that carry no date of their own at the end of the unit's corpus.

    `Memory.ask` resolves a relative expression such as "last week" or "this year" against
    `reference_at`, and with none given against the wall clock. A dataset that does not date its
    questions therefore retrieved different memories for the same seed depending on when the run
    started, and a replay hours later could not reproduce it. The one instant such a corpus
    defines is the time of its last memory: the question is asked after everything it can
    remember. Structured question dates take precedence; otherwise the product's pure parser
    reads an explicit "Today is ..." anchor from the original question. An omitted year uses
    the corpus year, never the execution clock. Only genuinely undated questions use `latest`.
    """
    if all(question.reference_at is not None for question in unit.questions):
        return unit
    times = [
        moment
        for item in unit.memories
        for moment in (item.occurred_end, item.occurred_at)
        if moment is not None
    ]
    if not times:
        return unit
    latest = max(times)
    return replace(
        unit,
        questions=tuple(
            question
            if question.reference_at is not None
            else replace(
                question,
                reference_at=temporal_context(
                    question.source_question
                    or "\n".join(part for part in question.content if isinstance(part, str)),
                    latest,
                    infer_reference=True,
                )[0],
            )
            for question in unit.questions
        ),
    )


def _selected(values: Sequence[_T], limit: Limit, offset: int) -> Sequence[_T]:
    if limit is None or limit == -1:
        return values[offset:]
    count = max(1, math.ceil(len(values) * limit)) if limit < 1 else int(limit)
    return values[offset : offset + count]


def _require_timestamped_media(
    task_name: str,
    unit_id: str,
    memories: Sequence[MemoryItem],
    questions: Sequence[EvalQuestion],
) -> None:
    if any(question.cutoff_seconds is not None for question in questions) and any(
        memory.end_seconds is None for memory in memories
    ):
        raise ValueError(
            f"{task_name} unit {unit_id} has causal questions; its --media-manifest parts "
            "must declare end_seconds"
        )


def _choice_prompt(question: str, choices: Sequence[str]) -> str:
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip("ABCDEFGHIJ", choices, strict=False)
    )
    labels = ", ".join("ABCDEFGHIJ"[: len(choices)])
    return (
        f"Select the best answer using only the memories. Reply with one letter ({labels}).\n"
        f"Question: {question}\n{options}\nAnswer:"
    )


def _free_text_prompt(question: str) -> str:
    return f"Answer concisely using only the memories.\nQuestion: {question}\nAnswer:"


def _free_text_parts(question: str) -> tuple[str, str]:
    return question, "Answer concisely using only the memories.\nAnswer:"
