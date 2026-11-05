"""Bounded parallel inference preparation with ordered results and joined cleanup."""

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import TypeVar

_T = TypeVar("_T")


def ordered_calls(calls: Sequence[Callable[[], _T]], *, concurrency: int = 4) -> tuple[_T, ...]:
    if len(calls) < 2:
        return tuple(call() for call in calls)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="sensemem-model") as pool:
        futures: list[Future[_T]] = []
        pending: set[Future[_T]] = set()
        try:
            for call in calls:
                if len(pending) == concurrency:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for completed in done:
                        completed.result()
                future = pool.submit(copy_context().run, call)
                futures.append(future)
                pending.add(future)
        finally:
            wait(futures)
        return tuple(future.result() for future in futures)
