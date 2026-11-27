"""Durable benchmark ingestion, checkpoint recovery and ordered chunk release."""

from __future__ import annotations

import asyncio
import json
from collections.abc import (
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from opentelemetry.trace import Tracer

from sensemem import (
    AsyncMemory,
    IndexUnavailableError,
    MemoryCategory,
)
from sensemem._telemetry import (
    SPAN_KIND,
    traced_span,
)
from sensemem.benchmarks.eval_adapters import (
    MemoryItem,
)
from sensemem.benchmarks.eval_config import (
    DEFAULT_INGEST_MODE,
    INGEST_MODES,
)
from sensemem.benchmarks.eval_console import (
    announce,
)
from sensemem.benchmarks.eval_inputs import memory_content, memory_metadata, prefix_end
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    FailureDetail,
    failure_detail,
    raise_if_systemic_embedding,
    restored_failure,
    retry_transient,
)
from sensemem.benchmarks.eval_runtime import SDK_OPERATION_LIMIT, MemoryFactory, model_operation
from sensemem.benchmarks.eval_telemetry import (
    BENCHMARK_ARM,
    BENCHMARK_INGEST_ITEMS,
    BENCHMARK_INGEST_SPAN,
    BENCHMARK_PURPOSE,
    PRODUCT_PURPOSE,
)
from sensemem.benchmarks.isolation import BenchmarkRun


@dataclass(frozen=True, slots=True)
class IngestCheckpoint:
    """Record how much of one unit's causal prefix is durably in its store.

    A store is authoritative for what it holds but says nothing about which benchmark items were
    meant to land there, so a run that is killed mid-corpus leaves nothing that says where to
    start again. The note is written after each committed chunk and never before: one that trails
    the store costs a duplicated chunk, one that leads it silently drops evidence.
    """

    run: BenchmarkRun
    unit_id: str
    digest: str

    async def start(
        self,
        memories: Sequence[MemoryItem],
        cutoffs: Sequence[float | None],
        *,
        memory_factory: MemoryFactory,
        data_dir: Path,
    ) -> tuple[int, list[FailureDetail]]:
        """Return the prefix a resumed run may keep, rebuilding the store when it may not."""
        ingested, failures = self._read()
        # A store ingested past the first pending cutoff would answer that cutoff's questions with
        # memories they must not have seen yet. That leak is silent and scores well, so the store
        # is rebuilt rather than reused.
        if ingested and cutoffs and ingested > prefix_end(memories, cutoffs[0]):
            ingested, failures = 0, []
        # A note is evidence only about a store that is still there. One left over from a deleted
        # or already-rebuilt directory would skip a prefix that nothing ever wrote.
        occupied = _holds_anything(data_dir)
        if ingested and not occupied:
            ingested, failures = 0, []
        if self.run.resume and not ingested:
            if occupied:
                # A store keeps its ownership lock inside its own directory, so deleting one
                # another run still owns would replace that run's DataDirectoryInUseError with
                # two live writers. Opening the store first is how this run learns through the
                # public path that the directory it is about to destroy is nobody else's.
                async with memory_factory(data_dir):
                    pass
                self.run.reset_unit(self.unit_id)
            # The rebuilt store and its note move together. A run killed between the two would
            # otherwise leave the old count describing a directory that no longer holds anything.
            self.record(0, ())
        return ingested, failures

    def record(self, ingested: int, failures: Sequence[FailureDetail]) -> None:
        path = self.run.checkpoint_path(self.unit_id)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "digest": self.digest,
                "ingested": ingested,
                "failures": [detail.json() for detail in failures],
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def _read(self) -> tuple[int, list[FailureDetail]]:
        if not self.run.resume:
            return 0, []
        try:
            document = self.run.checkpoint_path(self.unit_id).read_text(encoding="utf-8")
            payload = json.loads(document)
        except (OSError, ValueError):
            return 0, []
        if not isinstance(payload, Mapping) or payload.get("digest") != self.digest:
            return 0, []
        ingested = payload.get("ingested")
        failures = payload.get("failures")
        if (
            not isinstance(ingested, int)
            or isinstance(ingested, bool)
            or ingested < 0
            or not isinstance(failures, list)
        ):
            return 0, []
        try:
            details = [restored_failure(detail) for detail in failures]
        except (KeyError, TypeError):
            return 0, []
        return ingested, details


def _holds_anything(directory: Path) -> bool:
    """Say whether a unit directory still holds the store a checkpoint claims to describe."""
    return directory.is_dir() and any(directory.iterdir())


def ingest_observer(
    checkpoint: IngestCheckpoint | None,
    ingested: int,
    failures: Sequence[FailureDetail],
    notify: Callable[[int], None],
) -> Callable[[int], None]:
    """Note each committed chunk: a killed run resumes from it, and a live bar advances on it."""

    def observe(done: int) -> None:
        written = ingested + done
        if checkpoint is not None:
            checkpoint.record(written, failures)
        notify(written)

    return observe


@contextmanager
def _durable_write(
    tracer: Tracer | None,
    count: int,
    *,
    arm: str = DEFAULT_ARM,
) -> Iterator[None]:
    """Time one accepted batch through to durable, searchable memory.

    add/add_many return only after SQLite commits, Zvec flushes, and the search-index
    outbox is acknowledged, so this span measures accepted input to durable and searchable
    memory rather than the time until the call was accepted.
    """
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_INGEST_SPAN,
        attributes={
            SPAN_KIND: "stage",
            BENCHMARK_ARM: arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            BENCHMARK_INGEST_ITEMS: count,
        },
    ):
        yield


async def _capture_chunk(
    memory: AsyncMemory,
    chunk: Sequence[MemoryItem],
    *,
    on_failure: Callable[[FailureDetail], None] | None,
    tracer: Tracer | None,
    arm: str = DEFAULT_ARM,
    model_semaphore: asyncio.Semaphore | None = None,
) -> int:
    """Ingest one chunk through capture() + settle(), so both produce real
    sensemem.capture and sensemem.settle spans -- the only way --ingest capture makes
    capture acknowledgement and time-to-searchable measurable by a real run instead of
    unmeasurable.
    """
    failures = 0
    with _durable_write(tracer, len(chunk), arm=arm):
        for item in chunk:
            try:
                await memory.capture(
                    memory_content(item),
                    occurred_at=item.occurred_at,
                    occurred_end=item.occurred_end,
                    metadata=memory_metadata(item),
                    memory_type=MemoryCategory.EPISODIC,
                )
            except IndexUnavailableError:
                raise
            except Exception as error:
                failures += 1
                if on_failure is not None:
                    on_failure(failure_detail(error, source_id=item.source_id))
        # Drain the store's whole capture queue so the span still measures durable-and-searchable
        # wall time, as _ingest_json documents for the add path. Each AsyncMemory here is
        # one evaluation unit's isolated store and cutoffs settle sequentially, so this only ever
        # drains what this unit itself captured.
        while True:
            try:
                settled = await model_operation(model_semaphore, lambda: memory.settle(limit=100))
            except IndexUnavailableError:
                raise
            except Exception as error:
                failures += 1
                if on_failure is not None:
                    on_failure(failure_detail(error))
                continue
            if not settled:
                break
    return failures


async def deliberate_after_ingest(memory: AsyncMemory) -> int:
    """Run the slow loop over what was just ingested, before the cutoff's questions are asked.

    Between ingest and questions rather than after the whole unit: that is where a real
    deployment's loop would have run, and a consolidation applied after the questions could not
    change an answer it was meant to improve.
    """
    report = await memory.consolidate_pending()
    return report.applied


async def ingest(  # noqa: C901 - bisection, systemic-outage abort, and transient retry share one chunk path
    memory: AsyncMemory,
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
    on_failure: Callable[[FailureDetail], None] | None = None,
    on_chunk: Callable[[int], None] | None = None,
    tracer: Tracer | None = None,
    mode: str = DEFAULT_INGEST_MODE,
    arm: str = DEFAULT_ARM,
    prefetch: Callable[[Sequence[MemoryItem]], None] | None = None,
    model_semaphore: asyncio.Semaphore | None = None,
    ingest_concurrency: int = 1,
    run_state: IngestRunState | None = None,
) -> int:
    # An arm that reads no memory never reaches here: _run_unit skips ingestion for it, so the
    # blind control cannot accidentally score a store it was supposed to run without.
    run_state = run_state or IngestRunState()
    if mode not in INGEST_MODES:
        raise ValueError(f"unknown ingest mode {mode!r}; choose from {', '.join(INGEST_MODES)}")

    if ingest_concurrency <= 0 or (ingest_concurrency > 1 and mode != "add"):
        raise ValueError("concurrent ingest requires a positive limit and add mode")

    async def add_chunk(
        chunk: Sequence[MemoryItem],
        failure_callback: Callable[[FailureDetail], None] | None = on_failure,
    ) -> int:
        contents = tuple(memory_content(item) for item in chunk)

        async def add_all() -> None:
            with _durable_write(tracer, len(chunk), arm=arm):
                await memory.add_many(
                    contents,
                    occurred_at=tuple(item.occurred_at for item in chunk),
                    occurred_end=tuple(item.occurred_end for item in chunk),
                    metadata=tuple(memory_metadata(item) for item in chunk),
                    memory_type=MemoryCategory.EPISODIC,
                )

        async def add_first() -> None:
            with _durable_write(tracer, 1, arm=arm):
                await memory.add(
                    contents[0],
                    occurred_at=chunk[0].occurred_at,
                    occurred_end=chunk[0].occurred_end,
                    metadata=memory_metadata(chunk[0]),
                    memory_type=MemoryCategory.EPISODIC,
                )

        try:
            await retry_transient(lambda: model_operation(model_semaphore, add_all))
            return 0
        except IndexUnavailableError:
            raise
        except Exception as error:
            raise_if_systemic_embedding(
                error, "systemic embedding provider failure; item isolation aborted"
            )
            if len(chunk) > 1:
                middle = len(chunk) // 2
                return await add_chunk(chunk[:middle], failure_callback) + await add_chunk(
                    chunk[middle:], failure_callback
                )
        try:
            await retry_transient(lambda: model_operation(model_semaphore, add_first))
        except IndexUnavailableError:
            raise
        except Exception as error:
            raise_if_systemic_embedding(
                error, "systemic embedding provider failure; single-item fallback aborted"
            )
            _announce_first_ingest_failure(error, chunk[0].source_id, run_state)
            if failure_callback is not None:
                failure_callback(failure_detail(error, source_id=chunk[0].source_id))
            return 1
        return 0

    async def capture_chunk(chunk: Sequence[MemoryItem]) -> int:
        return await _capture_chunk(
            memory,
            chunk,
            on_failure=on_failure,
            tracer=tracer,
            arm=arm,
            model_semaphore=model_semaphore,
        )

    chunk_ingest = capture_chunk if mode == "capture" else add_chunk
    buffered_failures: dict[int, list[FailureDetail]] = {}

    async def concurrent_chunk(chunk: Sequence[MemoryItem]) -> int:
        details: list[FailureDetail] = []
        buffered_failures[id(chunk)] = details
        return await add_chunk(chunk, details.append)

    def release_chunk(chunk: Sequence[MemoryItem]) -> None:
        # Later chunks may finish first. Publish only this prefix's failures before checkpointing.
        details = buffered_failures.pop(id(chunk))
        if on_failure is not None:
            for detail in details:
                on_failure(detail)

    return await _ingest_chunks(
        chunk_ingest if ingest_concurrency == 1 else concurrent_chunk,
        items,
        batch_size=batch_size,
        on_chunk=on_chunk,
        prefetch=prefetch,
        ingest_concurrency=ingest_concurrency,
        on_released=None if ingest_concurrency == 1 else release_chunk,
    )


async def _ingest_chunks(  # noqa: C901 - bounded waves drain before releasing ordered receipts
    ingest_chunk: Callable[[Sequence[MemoryItem]], Awaitable[int]],
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
    on_chunk: Callable[[int], None] | None,
    prefetch: Callable[[Sequence[MemoryItem]], None] | None = None,
    ingest_concurrency: int = 1,
    on_released: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> int:
    """Release completed chunk boundaries in input order, after a bounded wave settles.

    The default commits sequentially and prefetches speech for its next chunk. Opt-in waves
    preserve arrival order for overlapping event times; independent times can commit out of
    order. Receipts describe only completed contiguous prefixes.
    """
    failures = 0
    if ingest_concurrency > 1:

        async def after_predecessors(
            chunk: Sequence[MemoryItem], predecessors: Sequence[asyncio.Task[int]]
        ) -> int:
            if predecessors:
                # Fatal predecessors must prevent a later same-time write from overtaking them
                # on resume. Admission to the shared model budget happens inside ingest_chunk.
                await asyncio.gather(*predecessors)
            return await ingest_chunk(chunk)

        width = min(ingest_concurrency, SDK_OPERATION_LIMIT)
        for start in range(0, len(items), batch_size * width):
            offsets = range(start, min(start + batch_size * width, len(items)), batch_size)
            chunks = [items[offset : offset + batch_size] for offset in offsets]
            tasks: list[asyncio.Task[int]] = []
            tails: dict[datetime | None, asyncio.Task[int]] = {}
            unknown_barrier: asyncio.Task[int] | None = None
            for chunk in chunks:
                # Neighbors use occurred_at; interval/latest reads can use occurred_end.
                times = {item.occurred_at for item in chunk} | {
                    item.occurred_end or item.occurred_at for item in chunk
                }
                predecessors = (
                    set(tasks)
                    if None in times
                    else {tails[moment] for moment in times if moment in tails}
                )
                if unknown_barrier is not None:
                    predecessors.add(unknown_barrier)
                task = asyncio.create_task(after_predecessors(chunk, tuple(predecessors)))
                tasks.append(task)
                tails.update(dict.fromkeys(times, task))
                if None in times:
                    unknown_barrier = task
            pending = asyncio.gather(*tasks, return_exceptions=True)
            cancelled = False
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # A cancelled parent must not close Memory while SDK writes are still active.
                    cancelled = True
            for offset, chunk, result in zip(offsets, chunks, pending.result(), strict=True):
                if isinstance(result, BaseException):
                    if cancelled:
                        raise asyncio.CancelledError
                    raise result
                failures += result
                if on_released is not None:
                    on_released(chunk)
                if on_chunk is not None:
                    on_chunk(min(offset + batch_size, len(items)))
            if cancelled:
                raise asyncio.CancelledError
        return failures
    for offset in range(0, len(items), batch_size):
        if prefetch is not None and offset + batch_size < len(items):
            prefetch(items[offset + batch_size : offset + 2 * batch_size])
        # Every item of a chunk is written or counted as failed by the time it returns, so a
        # chunk boundary is the only place a resumable note is exactly true.
        failures += await ingest_chunk(items[offset : offset + batch_size])
        if on_chunk is not None:
            on_chunk(min(offset + batch_size, len(items)))
    return failures


@dataclass(slots=True)
class IngestRunState:
    """Mutable measurements owned by one evaluation run, shared by its isolated units."""

    first_failure_announced: bool = False
    operations_applied: int = 0


def _announce_first_ingest_failure(
    error: BaseException, source_id: str, state: IngestRunState
) -> None:
    """Say once, immediately, that writes are failing.

    Ingest failures are bisected, counted and reported in the final table's unwritten column,
    which is right for a corpus with a few unreadable items and wrong for a misconfiguration that
    fails every write: a run with an embedding dimension the model does not produce looked alive
    for fourteen minutes while every store on the machine stayed empty. One line at the first
    failure carries the error text the summary cannot.
    """
    if state.first_failure_announced:
        return
    state.first_failure_announced = True
    detail = failure_detail(error, source_id=source_id)
    reason = "" if detail.reason is None else f"/{detail.reason}"
    announce(
        f"first ingest failure at source {source_id} ({detail.code}{reason}): {error}"
        " -- further failures are counted in the unwritten column"
    )
