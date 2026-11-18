"""A capture refresh remains pending when an older settlement finishes or fails."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_fast_capture import CountingEmbedder, CountingFormer

from sensemem import EmbedTask, FormationInput, FormationProposal, Memory, ModelInput, StreamInput


class PausedEmbedder(CountingEmbedder):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self._pause = True

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        if self._pause:
            self._pause = False
            self.started.set()
            assert self.release.wait(10), "capture refresh did not release embedding"
            if self.fail:
                raise RuntimeError("obsolete embedding failed")
        return super().embed(inputs, task)


class PausedFormer(CountingFormer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(10), "capture refresh did not release formation"
            if self.fail:
                raise RuntimeError("obsolete formation failed")
        return tuple(() for _value in inputs)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_refresh_during_embedding_keeps_new_content_and_queue(
    tmp_path: Path, batch_size: int
) -> None:
    embedder = PausedEmbedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        original = memory.capture("source observation")
        if batch_size == 2:
            memory.capture("independent observation")
        with ThreadPoolExecutor(max_workers=1) as pool:
            settlement = pool.submit(memory.settle)
            try:
                assert embedder.started.wait(10)
                refreshed = next(
                    memory.add_stream(
                        [StreamInput("source observation", interpretation="fresh interpretation")],
                        capture=True,
                    )
                )
                pending = memory.pending_captures(memory_ids=(original.id,))
            finally:
                embedder.release.set()
            assert settlement.result(timeout=10) == batch_size - 1

        assert refreshed.id == original.id
        assert memory.get(original.id).content == refreshed.content
        assert memory.pending_captures() == pending
        assert pending[0].attempts == 0 and pending[0].last_error is None
        assert memory.settle() == 1
        assert memory.pending_captures() == ()
        assert memory.get(original.id).content == refreshed.content


def test_refresh_during_a_failing_embedding_does_not_fail_the_new_capture(tmp_path: Path) -> None:
    embedder = PausedEmbedder(fail=True)
    with Memory(tmp_path, embedder=embedder) as memory:
        original = memory.capture("source observation")
        with ThreadPoolExecutor(max_workers=1) as pool:
            settlement = pool.submit(memory.settle)
            try:
                assert embedder.started.wait(10)
                refreshed = next(
                    memory.add_stream(
                        [StreamInput("source observation", interpretation="fresh interpretation")],
                        capture=True,
                    )
                )
                pending = memory.pending_captures()
            finally:
                embedder.release.set()
            assert settlement.result(timeout=10) == 0

        assert memory.get(original.id).content == refreshed.content
        assert memory.pending_captures() == pending
        assert pending[0].attempts == 0 and pending[0].last_error is None
        assert memory.settle() == 1
        assert memory.pending_captures() == ()


@pytest.mark.parametrize("fail", [False, True])
def test_refresh_during_formation_survives_old_completion_or_failure(
    tmp_path: Path, fail: bool
) -> None:
    former = PausedFormer(fail=fail)
    with Memory(tmp_path, embedder=CountingEmbedder(), former=former) as memory:
        original = memory.capture("source observation")
        with ThreadPoolExecutor(max_workers=1) as pool:
            settlement = pool.submit(memory.settle)
            try:
                assert former.started.wait(10)
                refreshed = next(
                    memory.add_stream(
                        [StreamInput("source observation", interpretation="fresh interpretation")],
                        capture=True,
                    )
                )
                pending = memory.pending_captures()
            finally:
                former.release.set()
            assert settlement.result(timeout=10) == 0

        assert memory.get(original.id).content == refreshed.content
        assert memory.pending_captures() == pending
        assert pending[0].attempts == 0 and pending[0].last_error is None
        assert pending[0].awaiting == "enrichment"
        assert memory.settle() == 1
        assert memory.pending_captures() == ()


@pytest.mark.parametrize("batch", [False, True])
def test_add_formation_does_not_complete_a_later_capture_refresh(
    tmp_path: Path, batch: bool
) -> None:
    former = PausedFormer()
    with Memory(tmp_path, embedder=CountingEmbedder(), former=former) as memory:
        with ThreadPoolExecutor(max_workers=1) as pool:
            write = (
                pool.submit(memory.add_many, ("source observation",))
                if batch
                else pool.submit(memory.add, "source observation")
            )
            try:
                assert former.started.wait(10)
                refreshed = next(
                    memory.add_stream(
                        [StreamInput("source observation", interpretation="fresh interpretation")],
                        capture=True,
                    )
                )
                pending = memory.pending_captures()
            finally:
                former.release.set()
            write.result(timeout=10)

        assert memory.get(refreshed.id).content == refreshed.content
        assert memory.pending_captures() == pending
        assert pending[0].awaiting == "enrichment"
        assert memory.settle() == 1
        assert memory.pending_captures() == ()
