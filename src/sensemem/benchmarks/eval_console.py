"""Console output for eval runs: logging setup, progress bars, announcements, and the summary table."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import Event, RLock, Thread
from typing import Protocol, cast

from tqdm import tqdm

from sensemem.benchmarks.eval_results import DEFAULT_ARM
from sensemem.benchmarks.videozerobench import VIDEOZERO_METRICS

# The harness writes from threads in one process. tqdm's default multiprocessing RLock
# allocates a named semaphore on Python 3.14, which survives abrupt or late shutdown.
tqdm.set_lock(RLock())

# The log namespace this harness is allowed to be verbose about: everything under the
# installed package, derived so a rename cannot leave a stale literal behind.
_ROOT_PACKAGE = __name__.split(".", 1)[0]


class _TqdmHandler(logging.Handler):
    """Preserve progress bars and handler filters, reading redirected stderr per record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # logging must never raise into the run it is reporting on
            self.handleError(record)


class _VerbosityFilter(logging.Filter):
    """Filter dependency records at the handler; their own logger levels bypass the root's."""

    def __init__(self, level: int, third_party: int) -> None:
        super().__init__()
        self._level = level
        self._third_party = third_party

    def filter(self, record: logging.LogRecord) -> bool:
        own = record.name.split(".", 1)[0] == _ROOT_PACKAGE
        return record.levelno >= (self._level if own else self._third_party)


def _configure_logging(verbosity: str) -> None:
    """Configure before importing model dependencies that call `basicConfig` themselves."""
    level: int = logging.getLevelName(verbosity)
    third_party = level if verbosity == "DEBUG" else max(level, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[_TqdmHandler()],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_VerbosityFilter(level, third_party))
    # Modelscope's private, non-propagating handler reads this once at import.
    os.environ.setdefault("MODELSCOPE_LOG_LEVEL", str(third_party))


def _announce(message: str) -> None:
    # `tqdm.write` is how a bar and a message share one stream: it erases the bar, writes the
    # line, and redraws. With no bar live it is a `print` with a flush, so every caller is
    # bar-safe without knowing whether one is running.
    tqdm.write(f"sensemem-bench eval: {message}", file=sys.stderr)


def _ignore_progress(_completed: int, _total: int) -> None:
    pass


class _ProgressReporter(Protocol):
    def __call__(self, completed: int, total: int) -> None: ...

    def set_activity(self, activity: str) -> None: ...


@dataclass(slots=True)
class _CallbackProgressReporter:
    callback: Callable[[int, int], None]

    def __call__(self, completed: int, total: int) -> None:
        self.callback(completed, total)

    def set_activity(self, activity: str) -> None:
        del activity


# How long a non-interactive run may stay silent between progress lines.
_PROGRESS_LOG_SECONDS = 60.0


# A sample can spend minutes rebuilding and ingesting before its answer exists. Redraw the live
# meter while its count is unchanged so elapsed time and the current phase still prove liveness.
_PROGRESS_REFRESH_SECONDS = 1.0


# tqdm's own meter with the bar glyphs removed. A redrawn bar in a log file is one unreadable
# line of carriage returns, but the counts and the ETA are the ones a terminal would have shown,
# so both modes report identical numbers.
_PROGRESS_LOG_FORMAT = (
    "{desc}: {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]"
)


@contextmanager
def _progress(
    stage: str, noun: str, *, total: int, enabled: bool = True
) -> Iterator[_ProgressReporter]:
    """Report progress as a live bar on a terminal and as throttled lines anywhere else."""
    if not enabled or total <= 0:
        yield _CallbackProgressReporter(_ignore_progress)
        return
    if sys.stderr.isatty():
        with tqdm(total=total, desc=stage, unit=noun, file=sys.stderr, leave=False) as bar:
            stopped = Event()

            def refresh() -> None:
                while not stopped.wait(_PROGRESS_REFRESH_SECONDS):
                    bar.refresh()

            @dataclass(slots=True)
            class LiveProgressReporter:
                def __call__(self, completed: int, total: int) -> None:
                    del total
                    bar.update(completed - bar.n)

                def set_activity(self, activity: str) -> None:
                    bar.set_postfix_str(activity, refresh=True)

            refresher = Thread(target=refresh, name="sensemem-bench-progress", daemon=True)
            refresher.start()
            try:
                yield LiveProgressReporter()
            finally:
                stopped.set()
                refresher.join()
        return
    started = time.monotonic()
    last = 0.0

    def report(completed: int, _total: int) -> None:
        # Throttle on elapsed time, not on a fraction of the work: a tenth of a forty-minute task
        # and a tenth of a twenty-second one are not the same amount of silence. The first and
        # the last completion always report. Preparation may also report zero before starting, so
        # a run that stalls on its first source says so at once and the log ends on the final count.
        nonlocal last
        now = time.monotonic()
        if completed not in {0, 1, total} and now - last < _PROGRESS_LOG_SECONDS:
            return
        last = now
        _announce(
            tqdm.format_meter(
                completed,
                total,
                now - started,
                prefix=stage,
                unit=noun,
                bar_format=_PROGRESS_LOG_FORMAT,
            )
        )

    yield _CallbackProgressReporter(report)


@contextmanager
def _deferred_progress(
    stage: str, noun: str, *, enabled: bool = True
) -> Iterator[Callable[[int, int], None]]:
    """Start a progress reporter once a producer discovers its total work count."""
    with ExitStack() as stack:
        expected_total: int | None = None
        report: Callable[[int, int], None] = _ignore_progress

        def advance(completed: int, total: int) -> None:
            nonlocal expected_total, report
            if expected_total is None:
                expected_total = total
                report = stack.enter_context(_progress(stage, noun, total=total, enabled=enabled))
            elif total != expected_total:
                raise ValueError(f"{stage} progress total changed from {expected_total} to {total}")
            report(completed, total)

        yield advance


def _table(results: Mapping[str, object]) -> str:
    tasks = cast(Sequence[Mapping[str, object]], results["tasks"])
    rows = []
    for task in tasks:
        score = cast(Mapping[str, object], task["score"])
        performance = cast(Mapping[str, object], task["performance"])
        duration = cast(Mapping[str, object], performance["duration_seconds"])
        usage = cast(Mapping[str, object], performance["token_usage"])
        controls = cast(Mapping[str, object], task["controls"])
        mean = score.get("mean")
        interval = cast(Sequence[float] | None, score["confidence_interval_95"])
        total_duration = cast(float | None, duration["total"])
        average_duration = cast(float | None, duration["average"])
        total_tokens = cast(int | None, usage["total_tokens"])
        average_tokens = cast(float | None, usage["average_tokens"])
        # When the judge's usage is incomplete the task total is honestly null, but the product's
        # own spend is usually complete; print that with a marker rather than a dash.
        token_marker = ""
        product = cast(Mapping[str, object] | None, usage.get("product"))
        if total_tokens is None and product is not None and product["total_tokens"] is not None:
            total_tokens = cast(int, product["total_tokens"])
            average_tokens = cast(float | None, product["average_tokens"])
            token_marker = "*"
        random_ranker = cast(Mapping[str, object] | None, controls["random_ranker"])
        valid = task.get("score_valid") is not False
        rows.append(
            (
                (
                    str(task["task"])
                    if task.get("arm", DEFAULT_ARM) == DEFAULT_ARM
                    else f"{task['task']} [{task['arm']}]"
                ),
                str(task["primary_metric"]),
                (
                    "INVALID"
                    if not valid
                    else "—"
                    if mean is None
                    else f"{float(cast(float, mean)):.4f}"
                ),
                (
                    "—"
                    if interval is None
                    else f"[{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
                ),
                str(task["question_count"]),
                str(task["error_count"]),
                str(task["ingest_failure_count"]),
                ("—" if total_duration is None else f"{total_duration:.3f}"),
                ("—" if average_duration is None else f"{average_duration * 1_000:.1f}"),
                ("—" if total_tokens is None else f"{total_tokens}{token_marker}"),
                ("—" if average_tokens is None else f"{average_tokens:.1f}{token_marker}"),
                _control_cell(controls.get("recall_at_1")),
                _control_cell(controls.get("recall_at_20")),
                _control_cell(None if random_ranker is None else random_ranker.get("20")),
                ("SELF" if controls.get("is_blind_run") else _control_cell(controls.get("blind"))),
                (
                    "ok"
                    if controls.get("interpretable")
                    else "MISSING " + ",".join(cast(Sequence[str], controls.get("missing", ())))
                ),
            )
        )
    headers = (
        "task",
        "metric",
        "value",
        "95% cluster CI",
        "n",
        "errors",
        # A score reads INVALID whenever writes failed, and until this column existed the table
        # said only "errors 0" beside it: answering had in fact succeeded, on an empty store. One
        # run lost 7 784 writes to a missing transcription dependency and showed nothing.
        "unwritten",
        "total s",
        "avg ms",
        "tokens",
        "tokens/q",
        "R@1",
        "R@20",
        "rand@20",
        "blind",
        "controls",
    )
    widths = tuple(
        max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))
    )
    table = "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (headers, *rows)
    )
    level_lines = []
    for task in tasks:
        if task["task"] != "videozerobench":
            continue
        metrics = cast(Mapping[str, Mapping[str, object]], task["metrics"])
        label = str(task["task"])
        if task.get("arm", DEFAULT_ARM) != DEFAULT_ARM:
            label += f" [{task['arm']}]"
        level_lines.append(f"\n{label}: all levels (0-100)")
        for name in VIDEOZERO_METRICS:
            summary = metrics.get(name)
            if summary is None:
                level_lines.append(f"  {name:<22} NOT SCORED")
                continue
            mean = summary["mean"]
            count = summary["sample_count"]
            value = (
                "INVALID"
                if task.get("score_valid") is False
                else "—"
                if mean is None or not count
                else f"{float(cast(float, mean)):.4f}"
            )
            interval = cast(Sequence[float] | None, summary["confidence_interval_95"])
            confidence = (
                "—" if interval is None else f"[{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
            )
            level_lines.append(f"  {name:<22} {value:<10} 95% CI {confidence}  n={count}")
    return "\n".join((table, *level_lines))


def _control_cell(value: object) -> str:
    """Render one control value, and never let an absent control render as a number."""
    if value is None:
        return "MISSING"
    mean = cast(float | None, cast(Mapping[str, object], value)["mean"])
    if mean is None:
        return "MISSING"
    return f"{mean:.4f}"


def _uninterpretable_tasks(results: Mapping[str, object]) -> tuple[str, ...]:
    """Return the tasks whose mandatory controls are absent."""
    tasks = cast(Sequence[Mapping[str, object]], results["tasks"])
    return tuple(
        str(cast(Mapping[str, object], task["controls"])["reason"])
        for task in tasks
        if not cast(Mapping[str, object], task["controls"])["interpretable"]
    )
