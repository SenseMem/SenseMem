"""Immediate process-local admission shared by the runtime and its adapters."""

from __future__ import annotations

import os
from threading import Lock


class Capacity:
    def __init__(self, limit: int, *, parent: Capacity | None = None) -> None:
        self.limit = limit
        self._parent = parent
        self._pid = os.getpid()
        self._lock = Lock()
        self._used = 0

    def acquire(self, amount: int = 1) -> bool:
        if self._pid != os.getpid():
            self._lock = Lock()
            self._used = 0
            self._pid = os.getpid()
        if self._parent is not None and not self._parent.acquire(amount):
            return False
        with self._lock:
            if self._used + amount <= self.limit:
                self._used += amount
                return True
        if self._parent is not None:
            self._parent.release(amount)
        return False

    def release(self, amount: int = 1) -> None:
        if self._pid != os.getpid():
            return
        with self._lock:
            self._used -= amount
        if self._parent is not None:
            self._parent.release(amount)
