"""Model components must not depend back on the SDK facade that composes them."""

import ast
import inspect
from types import ModuleType

import pytest

from sensemem.models import (
    _openai_completion,
    _openai_media,
    _openai_output,
    _openai_protocol,
    _openai_requests,
    _openai_telemetry,
)


@pytest.mark.parametrize(
    "component",
    [
        _openai_completion,
        _openai_media,
        _openai_output,
        _openai_protocol,
        _openai_requests,
        _openai_telemetry,
    ],
    ids=lambda component: component.__name__.rsplit(".", 1)[-1],
)
def test_components_do_not_import_the_composing_sdk_facade(component: ModuleType) -> None:
    facades = {"openai_embedding", "openai_speech", "openai_completion"}
    modules = {"sensemem.models." + name for name in facades}
    for node in ast.walk(ast.parse(inspect.getsource(component))):
        if isinstance(node, ast.Import):
            assert not modules.intersection(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in modules
            if node.module == "sensemem.models":
                assert not facades.intersection(alias.name for alias in node.names)
