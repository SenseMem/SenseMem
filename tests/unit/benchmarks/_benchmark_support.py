"""Shared helper builders for benchmark unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from sensemem.benchmarks import eval_config as eval_config_module


def make_arguments(**overrides: object) -> eval_config_module.Arguments:
    values: dict[str, object] = {
        "seed": 7,
        "bootstrap_samples": 32,
        "recall_limit": 20,
        "blind": False,
        "offset": 0,
        "limit": None,
    }
    values.update(overrides)
    return cast(eval_config_module.Arguments, SimpleNamespace(**values))
