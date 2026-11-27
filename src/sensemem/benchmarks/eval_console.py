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
from typing import Protocol

from tqdm import tqdm

from sensemem.benchmarks.eval_results import DEFAULT_ARM
from sensemem.benchmarks.videozerobench import VIDEOZERO_METRICS

# The log namespace this harness is allowed to be verbose about: everything under the
# installed package, derived so a rename cannot leave a stale literal behind.
_ROOT_PACKAGE = __name__.split(".", 1)[0]
_TQDM_LOCK_CONFIGURED = False


def _ensure_tqdm_lock() -> None:
    """Use a thread-local RLock instead of tqdm's default multiprocessing semaphore."""
    global _TQDM_LOCK_CONFIGURED
    if not _TQDM_LOCK_CONFIGURED:
        tqdm.set_lock(RLock())
        _TQDM_LOCK_CONFIGURED = True


class _TqdmHandler(logging.Handler):
    """Preserve progress bars and handler filters, reading redirected stderr per record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _ensure_tqdm_lock()
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


def configure_logging(verbosity: str) -> None:
    """Configure before importing model dependencies that call basicConfig themselves."""
    _ensure_tqdm_lock()
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


def announce(message: str) -> None:
    # tqdm.write is how a bar and a message share one stream: it erases the bar, writes the
    # line, and redraws. With no bar live it is a print with a flush, so every caller is
    # bar-safe without knowing whether one is running.
    _ensure_tqdm_lock()
    tqdm.write(f"sensemem-bench eval: {message}", file=sys.stderr)


def ignore_progress(_completed: int, _total: int) -> None:
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
        return None


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
def progress_reporter(
    stage: str, noun: str, *, total: int, enabled: bool = True
) -> Iterator[_ProgressReporter]:
    """Report progress as a live bar on a terminal and as throttled lines anywhere else."""
    _ensure_tqdm_lock()
    if not enabled or total <= 0:
        yield _CallbackProgressReporter(ignore_progress)
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
        announce(
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
def deferred_progress(
    stage: str, noun: str, *, enabled: bool = True
) -> Iterator[Callable[[int, int], None]]:
    """Start a progress reporter once a producer discovers its total work count."""
    with ExitStack() as stack:
        expected_total: int | None = None
        report: Callable[[int, int], None] = ignore_progress

        def advance(completed: int, total: int) -> None:
            nonlocal expected_total, report
            if expected_total is None:
                expected_total = total
                report = stack.enter_context(
                    progress_reporter(stage, noun, total=total, enabled=enabled)
                )
            elif total != expected_total:
                raise ValueError(f"{stage} progress total changed from {expected_total} to {total}")
            report(completed, total)

        yield advance


def _expect_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping in evaluation results")
    return value


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _expect_mapping(value)


def _expect_mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected sequence of mappings in evaluation results")
    return tuple(_expect_mapping(item) for item in value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected numeric value in evaluation results")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected integer value in evaluation results")
    return value


def _optional_interval(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise TypeError("expected confidence interval sequence in evaluation results")
    low = _optional_float(value[0])
    high = _optional_float(value[1])
    if low is None or high is None:
        return None
    return (low, high)


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


def render_summary_table(results: Mapping[str, object]) -> str:
    tasks = _expect_mapping_sequence(results["tasks"])
    rows = []
    for task in tasks:
        score = _expect_mapping(task["score"])
        performance = _expect_mapping(task["performance"])
        duration = _expect_mapping(performance["duration_seconds"])
        usage = _expect_mapping(performance["token_usage"])
        controls = _expect_mapping(task["controls"])
        mean = _optional_float(score.get("mean"))
        interval = _optional_interval(score["confidence_interval_95"])
        total_duration = _optional_float(duration["total"])
        average_duration = _optional_float(duration["average"])
        total_tokens = _optional_int(usage["total_tokens"])
        average_tokens = _optional_float(usage["average_tokens"])
        # When the judge's usage is incomplete the task total is honestly null, but the product's
        # own spend is usually complete; print that with a marker rather than a dash.
        token_marker = ""
        product = _optional_mapping(usage.get("product"))
        if total_tokens is None and product is not None and product["total_tokens"] is not None:
            total_tokens = _optional_int(product["total_tokens"])
            average_tokens = _optional_float(product["average_tokens"])
            token_marker = "*"
        random_ranker = _optional_mapping(controls["random_ranker"])
        valid = task.get("score_valid") is not False
        rows.append(
            (
                (
                    str(task["task"])
                    if task.get("arm", DEFAULT_ARM) == DEFAULT_ARM
                    else f"{task['task']} [{task['arm']}]"
                ),
                str(task["primary_metric"]),
                ("INVALID" if not valid else "—" if mean is None else f"{mean:.4f}"),
                ("—" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"),
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
                    else "MISSING " + ",".join(_string_sequence(controls.get("missing", ())))
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
        metrics = _expect_mapping(task["metrics"])
        label = str(task["task"])
        if task.get("arm", DEFAULT_ARM) != DEFAULT_ARM:
            label += f" [{task['arm']}]"
        level_lines.append(f"\n{label}: all levels (0-100)")
        for name in VIDEOZERO_METRICS:
            raw_summary = metrics.get(name)
            if raw_summary is None:
                level_lines.append(f"  {name:<22} NOT SCORED")
                continue
            summary = _expect_mapping(raw_summary)
            mean = _optional_float(summary["mean"])
            count = summary["sample_count"]
            value = (
                "INVALID"
                if task.get("score_valid") is False
                else "—"
                if mean is None or not count
                else f"{mean:.4f}"
            )
            interval = _optional_interval(summary["confidence_interval_95"])
            confidence = "—" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
            level_lines.append(f"  {name:<22} {value:<10} 95% CI {confidence}  n={count}")
    return "\n".join((table, *level_lines))


def _control_cell(value: object) -> str:
    """Render one control value, and never let an absent control render as a number."""
    mapping = _optional_mapping(value)
    if mapping is None:
        return "MISSING"
    mean = _optional_float(mapping.get("mean"))
    if mean is None:
        return "MISSING"
    return f"{mean:.4f}"


def uninterpretable_tasks(results: Mapping[str, object]) -> tuple[str, ...]:
    """Return the tasks whose mandatory controls are absent."""
    tasks = _expect_mapping_sequence(results["tasks"])
    return tuple(
        str(_expect_mapping(task["controls"])["reason"])
        for task in tasks
        if not _expect_mapping(task["controls"])["interpretable"]
    )
