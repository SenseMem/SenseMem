"""Public composition of specialist tools, an agent, and temporary pipeline context."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from _feature_support import TinyEmbedder

from sensemem import (
    AgentRunner,
    AgentStep,
    AgentTurn,
    AssetRef,
    AsyncContextSource,
    Blob,
    ContentInput,
    ContextSupplement,
    FaceAnalysis,
    FaceEmbedding,
    Memory,
    Modality,
    ModelInput,
    ToolCall,
    ToolDefinition,
    memory_tools,
    transcription_tool,
)


class LocalASR:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "device-selected-asr"
    transcription_space = "device-selected-asr:recipe-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = 0

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        assert all(asset.path is not None and asset.path.exists() for asset in assets)
        self.calls += 1
        return ("The visitor is Dana.",) * len(assets)

    def close(self) -> None:
        self.closed += 1


class TextEmbedder(TinyEmbedder):
    embedding_capabilities = frozenset({Modality.TEXT})


class TranscribingAgent:
    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id

    def step(
        self, input: ModelInput, *, tools: Sequence[ToolDefinition], history: Sequence[AgentTurn]
    ) -> AgentStep:
        assert input.text == f"Who is visiting? Use recording memory_id={self.memory_id}."
        assert input.assets == ()
        assert "transcribe" in {tool.name for tool in tools}
        if not history:
            return AgentStep(
                calls=(ToolCall("speech", "transcribe", {"memory_id": self.memory_id}),)
            )
        result = history[-1].results[0]
        assert result.error is None and result.output is not None
        assert result.output["model"] == "device-selected-asr"
        return AgentStep(content="The visitor is Dana, according to the recording.")

    def close(self) -> None:
        raise AssertionError("the runner borrows its backend")


async def test_one_asr_supports_pipeline_and_agentic_context_without_new_records(
    tmp_path: Path,
) -> None:
    backend = LocalASR()
    with Memory(tmp_path, embedder=TextEmbedder(), transcriber=backend) as memory:
        record = memory.capture(Blob(b"retained-audio", "audio/wav"))
        assert backend.calls == 0
        assert tuple(item.memory_id for item in memory.pending_captures()) == (record.id,)
        memory.tools.register(transcription_tool(memory, backend))
        direct = memory.call_tool("transcribe", {"memory_id": record.id})
        assert direct.error is None and direct.output is not None
        assert direct.output["space"] == backend.transcription_space
        baseline = tuple(item.id for item in memory.list().items)
        calls = backend.calls
        runner = AgentRunner(TranscribingAgent(record.id), memory.tools)

        async def provide(goal: ContentInput) -> Sequence[ContextSupplement]:
            assert isinstance(goal, str)
            result = await runner.run_async(
                ModelInput(text=f"{goal} Use recording memory_id={record.id}.")
            )
            return (ContextSupplement(result.content, source=f"agent:recording:{record.id}"),)

        source = AsyncContextSource(provide)
        revision = source.submit("Who is visiting?")
        # The ordinary pipeline works while the optional agent is running.
        immediate = memory.compile("Who is visiting?")
        assert immediate.supplements == ()
        update = await source.finalize()
        assert update.revision == revision
        enriched = memory.compile("Who is visiting?", supplements=update.supplements)
        assert enriched.supplements == update.supplements
        assert "Dana" in enriched.render()
        assert backend.calls == calls + 1
        assert tuple(item.id for item in memory.list().items) == baseline
        assert memory.compile("Who is visiting?").supplements == ()
        assert tuple(item.memory_id for item in memory.pending_captures()) == (record.id,)
        assert memory.get(record.id).content == ""
        memory.settle(memory_ids=(record.id,))
        assert backend.calls == calls + 2
        assert "Dana" in memory.get(record.id).content
        assert memory.pending_captures() == ()
    assert backend.closed == 1


class LocalFaces:
    face_capabilities = frozenset({Modality.IMAGE})
    face_model = "device-selected-face"
    face_space = "device-selected-face:2"
    face_analysis_space = "device-selected-face:recipe-v1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face", (1.0, 0.0), (0.1, 0.1, 0.5, 0.5)),))
            for _asset in assets
        )

    def close(self) -> None:
        pass


class IdentityAgent:
    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id

    def step(
        self, input: ModelInput, *, tools: Sequence[ToolDefinition], history: Sequence[AgentTurn]
    ) -> AgentStep:
        del input, tools
        if not history:
            return AgentStep(
                calls=(
                    ToolCall(
                        "face", "perceive_memory", {"memory_id": self.memory_id, "kinds": ["faces"]}
                    ),
                )
            )
        output = history[-1].results[0].output
        assert output is not None
        result = output["result"]
        assert isinstance(result, Mapping)
        if len(history) == 1:
            faces = result["faces"]
            assert isinstance(faces, tuple) and isinstance(faces[0], Mapping)
            return AgentStep(
                calls=(
                    ToolCall("identity", "get_identity", {"identity_id": faces[0]["identity_id"]}),
                )
            )
        assert result["name"] == "Dana"
        return AgentStep(content="This is Dana, according to the registered identity.")

    def close(self) -> None:
        pass


def test_image_understanding_can_call_a_specialist_then_read_registered_identity(
    tmp_path: Path,
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), face_analyzer=LocalFaces()) as memory:
        record = memory.add(Blob(b"retained-image", "image/png"))
        face = memory.perceive(record.id, kinds=frozenset({"faces"})).faces[0]
        memory.register_identity(face.identity_id, "Dana")
        for tool in memory_tools(memory):
            memory.tools.register(tool)
        runner = AgentRunner(IdentityAgent(record.id), memory.tools)
        result = runner.run(ModelInput(text="Who is in this image?"))
        assert result.tool_calls == 2
        assert "Dana" in result.content
        assert all(not item.name.startswith("register") for item in memory.tools.definitions)


def test_custom_stage_can_lease_raw_media_without_creating_a_record(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        with memory.prepare_input(
            ("Read the label", Blob(b"transient-image", "image/png"))
        ) as value:
            assert value.text == "Read the label"
            assert len(value.assets) == 1
            assert value.assets[0].path is not None and value.assets[0].path.exists()
            assert memory.list().items == ()
        assert memory.list().items == ()
