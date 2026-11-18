"""File descriptor admission shared by the local SQLite and Zvec adapters."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from threading import RLock
from types import TracebackType


class ProcessLock:
    """A reentrant allocator lock that a new process never inherits in a held state."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._lock = RLock()

    def __enter__(self) -> None:
        if self._pid != os.getpid():
            self._lock = RLock()
            self._pid = os.getpid()
        self._lock.acquire()

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._lock.release()


FILE_DESCRIPTOR_LOCK = ProcessLock()


def file_descriptor_usage() -> tuple[int, int] | None:
    descriptors = next(
        (path for path in (Path("/proc/self/fd"), Path("/dev/fd")) if path.is_dir()), None
    )
    if os.name != "posix" or descriptors is None:
        return None
    try:
        resource = import_module("resource")
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if int(soft_limit) < 0:
            return None
        return len(os.listdir(descriptors)), int(soft_limit)
    except (AttributeError, ImportError, OSError):
        return None
