"""Application factories compose external capabilities before any local store opens."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from pydantic import BaseModel, ConfigDict

import sensemem.configuration as configuration
from sensemem import (
    AssetRef,
    Blob,
    Memory,
    MemorySpec,
    Modality,
    PluginConfig,
    PluginFactory,
    Tool,
    ValidationError,
    build_memory_composition,
)
from sensemem._resources import Closable


class ExternalEmbedder(TinyEmbedder):
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class ExternalASR:
    transcription_capabilities = frozenset({Modality.AUDIO})
    transcription_model = "external-asr-test"
    transcription_space = "external-asr-test:v1"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls = 0
        self.closed = 0

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        self.calls += 1
        return (self.transcript,) * len(assets)

    def close(self) -> None:
        self.closed += 1


class LookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str


class LookupOutput(BaseModel):
    definition: str | None


class LookupOptions(BaseModel):
    entries: dict[str, str]


@pytest.mark.parametrize("typed", [False, True])
def test_registered_embedder_asr_and_tool_factories_receive_options_and_reach_the_sdk(
    tmp_path: Path, typed: bool
) -> None:
    embedder = ExternalEmbedder()
    asr = ExternalASR("The spare key is in the workshop")
    received: dict[str, Mapping[str, object]] = {}

    def build_embedder(options: Mapping[str, object]) -> ExternalEmbedder:
        received["embedder"] = dict(options)
        return embedder

    def build_asr(options: Mapping[str, object]) -> ExternalASR:
        received["asr"] = dict(options)
        return asr

    def build_lookup(options: Mapping[str, object]) -> Tool[LookupInput, LookupOutput]:
        received["lookup"] = dict(options)
        values = LookupOptions.model_validate(options)

        def lookup(value: LookupInput) -> LookupOutput:
            return LookupOutput(definition=values.entries.get(value.term))

        return Tool("lookup", "Look up application knowledge", LookupInput, LookupOutput, lookup)

    factories: dict[str, PluginFactory] = {
        "application_encoder": build_embedder,
        "application_asr": build_asr,
        "application_lookup": build_lookup,
    }
    config: dict[str, object] = {
        "data_dir": str(tmp_path / "memory"),
        "embedding": {"factory": "application_encoder", "options": {"checkpoint": "encoder-a"}},
        "speech": {"factory": "application_asr", "options": {"language": "en"}},
        "tools": [
            {
                "factory": "application_lookup",
                "options": {"entries": {"episodic": "An observed event"}},
            }
        ],
        "settings": {"minimum_relevance": 0, "ambiguity_margin": 0},
    }
    declared = MemorySpec.model_validate(config) if typed else config
    with Memory.from_config(declared, factories=factories) as memory:
        record = memory.add(Blob(b"external recorded audio", "audio/wav"))
        assert "spare key" in record.content
        assert memory.search("spare key").hits[0].id == record.id
        assert memory.call_tool("lookup", {"term": "episodic"}).output == {
            "definition": "An observed event"
        }
        assert asr.calls == 1
        assert memory.capabilities.transcription_space == ExternalASR.transcription_space
    assert received == {
        "embedder": {"checkpoint": "encoder-a"},
        "asr": {"language": "en"},
        "lookup": {"entries": {"episodic": "An observed event"}},
    }
    assert embedder.closed == asr.closed == 1


def test_unknown_registered_factory_fails_before_opening_storage(tmp_path: Path) -> None:
    data_dir = tmp_path / "unopened"
    with pytest.raises(ValidationError, match="unregistered plugin factory"):
        Memory.from_config(
            {
                "data_dir": str(data_dir),
                "embedding": {"factory": "package.module.factory"},
            },
            factories={},
        )
    assert not data_dir.exists()


@pytest.mark.parametrize("slot", ["speech", "tool"])
def test_incompatible_factory_result_and_previously_created_plugins_are_closed(
    tmp_path: Path, slot: str
) -> None:
    class WrongCapability:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    embedder = ExternalEmbedder()
    wrong = WrongCapability()
    config: dict[str, object] = {
        "data_dir": str(tmp_path / "unopened"),
        "embedding": {"factory": "encoder"},
    }
    if slot == "tool":
        config["tools"] = [{"factory": "wrong"}]
    else:
        config[slot] = {"factory": "wrong"}
    with pytest.raises(ValidationError, match="incompatible capability"):
        Memory.from_config(
            config,
            factories={"encoder": lambda _options: embedder, "wrong": lambda _options: wrong},
        )
    assert embedder.closed == wrong.closed == 1
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("failure", ["raises", "wrong_protocol", "duplicate"])
def test_a_later_tool_factory_failure_closes_shared_resources_only_once(
    tmp_path: Path, failure: str
) -> None:
    embedder = ExternalEmbedder()

    def lookup(value: LookupInput) -> LookupOutput:
        return LookupOutput(definition=value.term)

    tool = Tool(
        "lookup", "Look up knowledge", LookupInput, LookupOutput, lookup, resources=(embedder,)
    )

    def fail(_options: Mapping[str, object]) -> Closable:
        if failure == "raises":
            raise RuntimeError("factory failed")
        if failure == "wrong_protocol":
            return embedder
        return tool

    expected_error = RuntimeError if failure == "raises" else ValidationError
    with pytest.raises(expected_error):
        build_memory_composition(
            {
                "data_dir": str(tmp_path / "unopened"),
                "embedding": {"factory": "encoder"},
                "tools": [{"factory": "lookup"}, {"factory": "fail"}],
            },
            factories={
                "encoder": lambda _options: embedder,
                "lookup": lambda _options: tool,
                "fail": fail,
            },
        )
    assert embedder.closed == 1
    assert tool.closed
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("missing", [False, True])
def test_native_config_supports_registered_tools_and_cleans_up_unknown_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    embedder = ExternalEmbedder()

    def build_embedding(_config: configuration.EmbeddingProviderConfig) -> ExternalEmbedder:
        return embedder

    monkeypatch.setattr(configuration, "_build_embedding", build_embedding)
    received: list[Mapping[str, object]] = []

    def build_tool(options: Mapping[str, object]) -> Tool[LookupInput, LookupOutput]:
        received.append(dict(options))
        values = LookupOptions.model_validate(options)

        def lookup(value: LookupInput) -> LookupOutput:
            return LookupOutput(definition=values.entries.get(value.term))

        return Tool(
            "lookup", "Look up knowledge", LookupInput, LookupOutput, lookup, resources=(embedder,)
        )

    config = MemorySpec.model_validate(
        {
            "data_dir": str(tmp_path / "memory"),
            "embedding": {"provider": "openai"},
            "tools": [{"factory": "lookup", "options": {"entries": {"key": "workshop"}}}],
        }
    )
    if missing:
        with pytest.raises(ValidationError, match="unregistered plugin factory"):
            Memory.from_config(config)
        assert received == []
        assert not (tmp_path / "memory").exists()
    else:
        with Memory.from_config(config, factories={"lookup": build_tool}) as memory:
            assert memory.call_tool("lookup", {"term": "key"}).output == {"definition": "workshop"}
        assert received == [{"entries": {"key": "workshop"}}]
    assert embedder.closed == 1


def test_composition_builds_owned_plugins_without_opening_a_store(tmp_path: Path) -> None:
    embedder = ExternalEmbedder()
    config = MemorySpec(
        data_dir=tmp_path / "unopened",
        embedding=PluginConfig(factory="encoder", options={"revision": "fixed"}),
    )
    composition = build_memory_composition(
        spec=config, factories={"encoder": lambda _options: embedder}
    )
    assert composition.plugins.embedder is embedder
    assert not composition.data_dir.exists()
    composition.close()
    assert embedder.closed == 1


@pytest.mark.parametrize("invalid", ["declaration", "settings"])
def test_invalid_memory_assembly_closes_the_factory_resource_once(
    tmp_path: Path, invalid: str
) -> None:
    class InvalidEmbedder(ExternalEmbedder):
        embedding_dimension = 0 if invalid == "declaration" else 4

    embedder = InvalidEmbedder()
    config: dict[str, object] = {
        "data_dir": str(tmp_path / "unopened"),
        "embedding": {"factory": "encoder"},
    }
    if invalid == "settings":
        config["settings"] = {"evidence_expansion_max_rows": 1_000_000}
    with pytest.raises(ValidationError):
        Memory.from_config(
            config,
            factories={"encoder": lambda _options: embedder},
        )
    assert embedder.closed == 1
    assert not (tmp_path / "unopened").exists()
