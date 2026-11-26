"""Native user speech becomes an attributed observation without waiting for video finality."""

from __future__ import annotations

import io
import json
import struct
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from examples.live import user_audio
from examples.live.user_audio import UserAudioCapture
from sensemem import AsyncMemory, Memory, StreamCommit

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)
SOURCE = "live:user-audio-fixture"


def _event(kind: str, **values: object) -> str:
    return json.dumps({"type": kind, "item_id": "user-turn", **values})


def _finalize(capture: object) -> None:
    # The helper stays local to tests; production consumes the native protocol events.

    assert isinstance(capture, UserAudioCapture)
    capture.observe_event(_event("input_audio_buffer.speech_started", audio_start_ms=50))
    capture.observe_event(_event("input_audio_buffer.speech_stopped", audio_end_ms=250))
    capture.observe_event(
        _event("conversation.item.input_audio_transcription.completed", transcript="钥匙在绿盒子里")
    )


@pytest.mark.asyncio
async def test_stopped_final_asr_captures_exact_original_24k_audio_and_becomes_searchable(
    tmp_path: Path,
) -> None:

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = UserAudioCapture(memory, SOURCE, committed=commits.append, clock=lambda: NOW)
        pcm = struct.pack("<7200h", *range(7200))
        for offset in range(0, len(pcm), 4800):
            capture.append_audio(pcm[offset : offset + 4800], 24_000)
        _finalize(capture)
        await capture.close()
        assert capture.failure is None
        assert len(commits) == 1 and commits[0].pending_settlement
        record = commits[0].record
        assert record.context is not None and record.context.source_id == SOURCE
        assert record.metadata["speaker"] == "user"
        assert "dialogue" not in record.metadata
        speech = record.metadata["speech"]
        assert isinstance(speech, dict)
        assert speech["transcript_alignment"] == "unverified"
        assert record.occurred_at == NOW - timedelta(milliseconds=50)
        assert record.occurred_end == NOW + timedelta(milliseconds=150)
        assert "钥匙在绿盒子里" in record.content
        original = await memory.read_asset(record.id, record.assets[0].id)
        with wave.open(io.BytesIO(original.data), "rb") as audio:
            assert audio.getframerate() == 24_000
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            assert audio.getnframes() == 4800
            assert audio.readframes(4800) == pcm[2400:12000]
        assert len(await memory.pending_captures()) == 1
        await memory.settle()
        assert not await memory.pending_captures()
        assert record.id in {hit.id for hit in (await memory.search("钥匙在绿盒子里")).hits}


@pytest.mark.asyncio
async def test_finality_before_pcm_waits_for_actual_samples_and_deduplicates_item(
    tmp_path: Path,
) -> None:

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = UserAudioCapture(memory, SOURCE, committed=commits.append, clock=lambda: NOW)
        # Server events may reach their observer before the send pump resumes copying PCM.
        capture.observe_event(
            _event(
                "conversation.item.input_audio_transcription.completed", transcript="钥匙在绿盒子里"
            )
        )
        capture.observe_event(_event("input_audio_buffer.speech_stopped", audio_end_ms=250))
        capture.observe_event(_event("input_audio_buffer.speech_started", audio_start_ms=50))
        assert capture.task is None and not (await memory.list()).items
        capture.append_audio(b"\x01\x00" * 2400, 24_000)
        assert capture.task is None
        capture.append_audio(b"\x02\x00" * 3600, 24_000)
        _finalize(capture)
        await capture.close()
        assert len(commits) == 1
        original = await memory.read_asset(commits[0].record.id, commits[0].record.assets[0].id)
        with wave.open(io.BytesIO(original.data), "rb") as audio:
            assert audio.getnframes() == 4800
            assert audio.readframes(4800) == b"\x01\x00" * 1200 + b"\x02\x00" * 3600


@pytest.mark.asyncio
async def test_partial_agent_output_and_unfinished_vad_never_commit_on_disconnect(
    tmp_path: Path,
) -> None:

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = UserAudioCapture(memory, SOURCE, committed=commits.append)
        capture.append_audio(b"\x01\x00" * 8000, 16_000)
        capture.observe_event(_event("input_audio_buffer.speech_started", audio_start_ms=50))
        capture.observe_event(_event("input_audio_buffer.speech_stopped", audio_end_ms=250))
        capture.observe_event(
            _event("speech_to_speech.input_audio_transcription.snapshot", transcript="暂时猜测")
        )
        capture.observe_event(
            _event("conversation.item.input_audio_transcription.delta", delta="暂时猜测")
        )
        capture.observe_event(
            _event("response.output_audio_transcript.done", transcript="Agent说法")
        )
        await capture.close()
        assert commits == [] and not (await memory.list()).items


@pytest.mark.asyncio
async def test_empty_final_or_final_without_complete_vad_cannot_invent_an_interval(
    tmp_path: Path,
) -> None:

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = UserAudioCapture(memory, SOURCE, committed=commits.append)
        capture.append_audio(b"\x01\x00" * 8000, 16_000)
        capture.observe_event(
            _event(
                "conversation.item.input_audio_transcription.completed", transcript="没有时间证据"
            )
        )
        assert capture.task is None
        capture.observe_event(_event("input_audio_buffer.speech_started", audio_start_ms=50))
        capture.observe_event(_event("input_audio_buffer.speech_stopped", audio_end_ms=250))
        # This is a separate silent final, not permission to store a partial transcript.
        capture.observe_event(
            json.dumps(
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "empty",
                    "audio_start_ms": 300,
                }
            )
        )
        capture.observe_event(
            json.dumps(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "empty",
                    "audio_end_ms": 400,
                }
            )
        )
        capture.observe_event(
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "empty",
                    "transcript": " ",
                }
            )
        )
        await capture.close()
        assert len(commits) == 1
        assert "没有时间证据" in commits[0].record.content


@pytest.mark.asyncio
async def test_evicted_original_samples_report_failure_instead_of_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(user_audio, "MAX_AUDIO_BYTES", 32)
    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = user_audio.UserAudioCapture(memory, SOURCE, committed=commits.append)
        capture.append_audio(b"\x01\x00" * 16, 16_000)
        capture.append_audio(b"\x02\x00" * 16, 16_000)
        capture.observe_event(_event("input_audio_buffer.speech_started", audio_start_ms=0))
        capture.observe_event(_event("input_audio_buffer.speech_stopped", audio_end_ms=2))
        capture.observe_event(
            _event("conversation.item.input_audio_transcription.completed", transcript="已丢失原音")
        )
        await capture.close()
        assert capture.failure == "audio_capture_capacity"
        assert commits == [] and not (await memory.list()).items
