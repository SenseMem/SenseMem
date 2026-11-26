"""Small packaging contracts for the supported local product."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import cast

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from sensemem.models.funasr import _FUNASR_REVISION

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "sensemem"
DOCUMENT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = cast(dict[str, object], DOCUMENT["project"])
DEPENDENCIES = cast(list[str], PROJECT["dependencies"])
EXTRAS = cast(dict[str, list[str]], PROJECT["optional-dependencies"])
SCRIPTS = cast(dict[str, str], PROJECT["scripts"])


def test_base_import_does_not_load_optional_protocols() -> None:
    code = (
        "import json, sys; import sensemem; "
        "blocked = {'cv2', 'fastapi', 'mcp', 'openai', 'sentence_transformers', 'torch', "
        "'transformers', 'uvicorn', 'vllm'}; "
        "print(json.dumps(sorted(blocked.intersection(sys.modules))))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == []


def test_full_modal_contract_is_exported_from_the_package_root() -> None:
    package = importlib.import_module("sensemem")
    names = {
        "ASRPartial",
        "AcousticBoundary",
        "Blob",
        "AssetRef",
        "AsyncAudioStream",
        "AsyncOmniPrefetch",
        "AudioBoundary",
        "AudioStreamPacket",
        "AsyncVisionStream",
        "ContentAtom",
        "ContentInput",
        "IndexQuantization",
        "MemoryCategory",
        "MemorySettings",
        "MemoryComposition",
        "MemorySpec",
        "SenseMemError",
        "MemoryPlugins",
        "Modality",
        "PrefetchResult",
        "RetrievalCandidateTrace",
        "RetrievalRejection",
        "RetrievalTrace",
        "SceneBoundary",
        "SearchResult",
        "EmbeddingBackend",
        "FaceBackend",
        "FaceObservation",
        "GenerationBackend",
        "ModelInput",
        "OpenAICompletion",
        "OpenAIEmbedder",
        "OpenAITranscriber",
        "OpenCVFaceAnalyzer",
        "PCMChunk",
        "EmbedTask",
        "SentenceTransformersEmbedder",
        "IdentityNotFoundError",
        "StreamInput",
        "StreamingGenerationBackend",
        "TranscriptionBackend",
        "VADPacket",
        "VisionBoundary",
        "VisionDescriptionBackend",
        "VisionFrame",
        "VisionPartial",
        "VisionStreamPacket",
        "build_memory_composition",
    }

    assert names <= set(package.__all__)
    assert all(hasattr(package, name) for name in names)


def test_product_does_not_address_benchmarks() -> None:
    benchmark_root = SOURCE / "benchmarks"
    violations = {
        str(path.relative_to(SOURCE)): references
        for path in SOURCE.rglob("*.py")
        if not path.is_relative_to(benchmark_root) and (references := _benchmark_references(path))
    }
    assert violations == {}


def test_shared_components_do_not_address_runtime_implementations() -> None:
    runtime_modules = {"__init__.py", "memory.py", "configuration.py", "cli.py", "streams.py"}
    violations = {
        path.name: sorted(references)
        for path in SOURCE.glob("*.py")
        if path.name not in runtime_modules
        if (
            references := {
                name
                for name in _imports(path).names
                if name.startswith(("sensemem.kernel", "sensemem.infrastructure", "sensemem.api"))
            }
        )
    }
    assert violations == {}, "shared values and policies must be independent of runtime wiring"


def test_api_adapters_do_not_address_local_storage() -> None:
    violations = {
        str(path.relative_to(SOURCE)): sorted(references)
        for path in (SOURCE / "api").rglob("*.py")
        if (
            references := {
                name for name in _imports(path).names if name.startswith("sensemem.infrastructure")
            }
        )
    }
    assert violations == {}, "protocol adapters must call the SDK for persistence"


def test_api_components_do_not_address_application_routing() -> None:
    violations = {
        str(path.relative_to(SOURCE)): sorted(references)
        for path in (SOURCE / "api").rglob("*.py")
        if path.name not in {"__init__.py", "app.py"}
        if (
            references := {
                name
                for name in _imports(path).names
                if name == "sensemem.api.app" or name.startswith("sensemem.api.app.")
            }
        )
    }
    assert violations == {}, "API contracts and transports must be independent of app assembly"


def test_benchmark_components_do_not_address_the_evaluation_runner() -> None:
    violations = {
        str(path.relative_to(SOURCE)): sorted(references)
        for path in (SOURCE / "benchmarks").rglob("*.py")
        if path.name not in {"cli.py", "eval.py"}
        if (
            references := {
                name
                for name in _imports(path).names
                if name == "sensemem.benchmarks.eval"
                or name.startswith("sensemem.benchmarks.eval.")
            }
        )
    }
    assert violations == {}, "benchmark components must not depend back on the evaluation runner"


def test_model_request_components_do_not_address_response_parsers() -> None:
    parsers = ("sensemem.models._answer_output", "sensemem.models._openai_output")
    violations = {
        path.name: sorted(references)
        for path in (SOURCE / "models").glob("_openai_*.py")
        if path.name in {"_openai_requests.py", "_openai_protocol.py", "_openai_media.py"}
        if (
            references := {
                name
                for name in _imports(path).names
                if any(name == parser or name.startswith(parser + ".") for parser in parsers)
            }
        )
    }
    assert violations == {}, "request construction must use evidence contracts rather than parsers"


def test_local_infrastructure_does_not_address_model_clients() -> None:
    violations = {
        str(path.relative_to(SOURCE)): sorted(references)
        for path in (SOURCE / "infrastructure").rglob("*.py")
        if (
            references := {
                name for name in _imports(path).names if name.startswith("sensemem.models")
            }
        )
    }
    assert violations == {}, "local adapters must depend on perception values, not model modules"


def test_imported_distributions_are_declared() -> None:
    """Require direct imports to be declared; isolated CI installs check transitive dependencies."""
    declared = {
        canonicalize_name(_name(item))
        for item in [*DEPENDENCIES, *(item for items in EXTRAS.values() for item in items)]
    }
    provided = _installed_roots()
    undeclared = {
        f"{path.relative_to(SOURCE)}:{line}": root
        for path in sorted(SOURCE.rglob("*.py"))
        for root, line in sorted(_imports(path).third_party)
        if not provided.get(root, frozenset()) & declared
    }
    assert undeclared == {}, (
        f"imported but declared in no extra: {undeclared}; add each distribution to the "
        "narrowest optional extra that reaches the import"
    )


def _installed_roots() -> dict[str, frozenset[str]]:
    """Map each importable root to the installed distributions that provide it.

    importlib.metadata.packages_distributions reads only top_level.txt before Python
    3.12, and modern wheels omit that file, so on 3.11 it reports almost nothing and
    the declaration check above would call every third-party import undeclared. Reading the
    recorded files instead is what 3.12 does internally and behaves the same on every
    supported interpreter.
    """
    roots: dict[str, set[str]] = {}
    for installed in distributions():
        name = installed.metadata["Name"]
        if not name:
            continue
        canonical = canonicalize_name(name)
        for recorded in installed.files or ():
            root = recorded.parts[0]
            if root.startswith((".", "_distutils_hack")) or root.endswith(".dist-info"):
                continue
            if root.endswith(".py"):
                root = root[: -len(".py")]
            elif "." in root:
                continue
            roots.setdefault(root, set()).add(canonical)
    return {root: frozenset(names) for root, names in roots.items()}


def _benchmark_references(path: Path) -> set[str]:
    return {
        name
        for name in _imports(path).names
        if name == "benchmarks" or name.startswith(("benchmarks.", "sensemem.benchmarks"))
    }


@dataclass(frozen=True, slots=True)
class _Imports:
    """Module names a file references, and the third-party roots among them with line numbers."""

    names: frozenset[str]
    third_party: frozenset[tuple[str, int]]


_FIRST_PARTY = frozenset({"sensemem", *sys.stdlib_module_names})


def _imports(path: Path) -> _Imports:
    """Collect static and deferred imports; relative imports name no distribution."""
    names: set[str] = set()
    third_party: set[tuple[str, int]] = set()

    def add(name: str, line: int, *, resolvable: bool) -> None:
        names.add(name)
        root = name.split(".", 1)[0]
        if resolvable and root and root not in _FIRST_PARTY:
            third_party.add((root, line))

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, node.lineno, resolvable=True)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            add(module, node.lineno, resolvable=not node.level)
            for alias in node.names:
                add(f"{module}.{alias.name}".lstrip("."), node.lineno, resolvable=False)
        elif isinstance(node, ast.Call) and (loaded := _loader_argument(node)) is not None:
            add(loaded, node.lineno, resolvable=True)
    return _Imports(frozenset(names), frozenset(third_party))


def _loader_argument(node: ast.Call) -> str | None:
    """Return the module named by a single-argument import_module or find_spec call."""
    called = node.func
    name = called.attr if isinstance(called, ast.Attribute) else None
    name = called.id if isinstance(called, ast.Name) else name
    if name not in {"import_module", "find_spec"} or len(node.args) != 1:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def test_console_script_targets_exist() -> None:
    assert set(SCRIPTS) == {"sensemem", "sensemem-bench"}
    installed = distribution("sensemem")
    assert installed.metadata["Name"] == PROJECT["name"]
    assert installed.version == PROJECT["version"]
    assert {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    } == SCRIPTS
    for target in SCRIPTS.values():
        module_name, attribute = target.split(":", 1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_all_extra_contains_each_optional_dependency() -> None:
    assert set(EXTRAS["all"]) == {
        item for extra, items in EXTRAS.items() if extra != "all" for item in items
    }


def test_openai_extra_installs_the_native_realtime_transport() -> None:
    requirement = next(item for item in EXTRAS["openai"] if _name(item) == "openai")
    assert "realtime" in Requirement(requirement).extras


def test_funasr_runtime_and_cache_identity_use_the_same_immutable_source() -> None:
    assert len(_FUNASR_REVISION) == 40
    int(_FUNASR_REVISION, 16)
    for extra in ("local", "all"):
        requirement = next(Requirement(item) for item in EXTRAS[extra] if _name(item) == "funasr")
        assert requirement.url == (
            f"git+https://github.com/modelscope/FunASR.git@{_FUNASR_REVISION}"
        )


def _name(requirement: str) -> str:
    return Requirement(requirement).name.lower()
