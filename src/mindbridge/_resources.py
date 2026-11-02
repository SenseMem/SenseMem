"""Ownership and cleanup for nested resource compositions, independent of the kernel."""

import builtins
from collections.abc import Sequence
from contextlib import suppress
from typing import Protocol, runtime_checkable


class Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class ResourceGroup(Protocol):
    def resources(self) -> Sequence[Closable]: ...


def unique_resources(resources: Sequence[Closable]) -> tuple[Closable, ...]:
    seen: set[int] = set()
    unique: builtins.list[Closable] = []
    pending = list(reversed(resources))
    while pending:
        resource = pending.pop()
        identity = id(resource)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(resource, ResourceGroup):
            pending.extend(reversed(resource.resources()))
            continue
        unique.append(resource)
    return tuple(unique)


def present_resources(*resources: Closable | None) -> tuple[Closable, ...]:
    return tuple(resource for resource in resources if resource is not None)


def close_quietly(*resources: Closable | None) -> None:
    """Close each distinct present resource once, swallowing whatever `close()` raises."""
    for resource in unique_resources(present_resources(*resources)):
        with suppress(Exception):
            resource.close()
