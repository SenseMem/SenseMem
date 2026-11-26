# Deployment

Run SenseMem inside your application, expose it over HTTP, or connect an agent host through MCP. All topologies use one live `Memory` owner per physical `data_dir`. Independent memory domains use separate directories.

This page chooses and configures a topology. See [architecture](architecture.md) for invariants, [operations](operations.md) for runbooks, and [troubleshooting](troubleshooting.md) when startup or traffic fails.

## Choose a topology

| Need | Topology | Constraint |
| --- | --- | --- |
| One Python application owns memory | Embed `Memory` in that process | Keep one instance open and close it during shutdown. |
| Other processes need HTTP | Expose the owner with `create_app()` | Run one ASGI worker for that directory and supply network controls outside SenseMem. |
| One agent host needs tools | Run `build_mcp_server()` in the owner process | Prefer stdio; secure SSE or streamable HTTP as a network service. |
| Independent workers need independent state | Give each worker a distinct directory | The directories are separate memory domains; they do not synchronize. |

Do not place a Python owner, REST process, and MCP process over the same directory. Put the desired adapters around one constructed `Memory`, or have other processes call the existing REST owner.

## Deployment checklist

Before starting the owner, verify:

- Python 3.11 through 3.14 and compatible 64-bit wheels exist for Zvec 0.8.1+ (within 0.8.x) and every selected backend on the target platform.
- The data directory is on one local, durable filesystem supporting SQLite WAL, file locking, atomic rename, and durable writes.
- Python links SQLite 3.35 or newer with the JSON1 extension (compiled in by default from 3.38, required by `consolidation_candidates`).
- Only the service account can read the directory, model credentials, and backups.
- Disk capacity covers SQLite, original media, the derived index, and temporary rebuild or compaction space; RAM, accelerator memory, and file-descriptor limits cover model inference and the memory-mapped index.
- The embedding, transcription, face, and index recipes match any existing store.
- A shutdown hook calls `Memory.close()` and closes caller-owned provider clients.

Network filesystems and shared-volume multi-writer deployments are unsupported. On POSIX, SenseMem sets the top-level directory to `0700` and its SQLite, lock, staging, and asset files to restrictive modes. The operator owns parent permissions, service accounts, disk encryption, backup access, and retention.

## Embedded Python

Install only the optional surfaces used by the application:

```bash
uv add "sensemem[local,openai]"
```

```python
from openai import OpenAI

from sensemem import Memory, OpenAICompletion, SentenceTransformersEmbedder

with OpenAI(timeout=30.0, max_retries=3) as client:
    with Memory(
        "/var/lib/sensemem/assistant",
        embedder=SentenceTransformersEmbedder(model_id="google/embeddinggemma-2"),
        answerer=OpenAICompletion(client=client),
    ) as memory:
        memory.add("Process-owned memory")
```

Keep one `Memory` open for a long-running process. `Memory.close()` closes supplied backend objects; `OpenAICompletion.close()` leaves caller-owned OpenAI clients open. Construct `Memory` after any process fork, never before it.

For OpenAI generation slots, configure `response_format="json_schema"` when supported or `response_format="json_object"` for JSON-only endpoints. Pass the SDK base URL without appending `/chat/completions`, keep credentials in environment lookup or a secret manager, and retain provider request IDs with local trace IDs for diagnosis. Local storage does not imply local inference.

## EmbeddingGemma 2 model server

Run vLLM separately from the SenseMem owner in its own environment with a version supporting EmbeddingGemma 2 pooling. Start the pooling server with the checkpoint's bundled chat template:

```bash
vllm serve google/embeddinggemma-2 \
  --runner pooling \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 127.0.0.1 \
  --port 8000
```

Keep text, image, audio, and video enabled for full multimodal support; the 8,192-token limit is shared across text and media per input. Do not substitute a generative Gemma chat template or rename the served model: SenseMem selects its retrieval recipe from the exact model name.

Use `--dtype float32` when BF16 is unavailable; do not use FP16. For text-only serving, pass `--limit-mm-per-prompt image=0,audio=0,video=0` and declare `"modalities": ["text"]` in SenseMem.

Install `sensemem[openai]` in the application environment and follow the [EmbeddingGemma 2 vLLM configuration](configuration/models.md#embeddinggemma-2-through-vllm). Multiple SenseMem owners can share one model server while each keeps its own physical `data_dir`.

## REST owner

Install the server surface with the selected model extras:

```bash
uv add "sensemem[local,openai,server]"
```

Keep the owner and caller-owned client open around the ASGI server in `my_application.py`:

```python
import uvicorn
from openai import OpenAI

from sensemem import Memory, OpenAICompletion, SentenceTransformersEmbedder
from sensemem.api import create_app


def main() -> None:
    with OpenAI(timeout=30.0, max_retries=3) as client:
        with Memory(
            "/var/lib/sensemem/assistant",
            embedder=SentenceTransformersEmbedder(
                model_id="google/embeddinggemma-2",
            ),
            answerer=OpenAICompletion(client=client),
        ) as memory:
            uvicorn.run(create_app(memory=memory), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
```

Run the single-process owner so context managers close memory before the provider client on exit:

```bash
uv run python my_application.py
```

`create_app()` does not own or close either resource. For an importable ASGI app, run the same cleanup in the host lifespan and run one worker. The adapter adds no authentication, authorization, TLS, rate limiting, quota, or access audit log: bind to a trusted interface or place it behind an authenticating gateway or middleware. Keep `/healthz` protected consistently; it reports liveness and declared capabilities, not model readiness. The app publishes fourteen always-on routes under `/v1`, or twenty-three with `identity_operations=True` and `embodied_operations=True`.

Do not retry a timed-out request blindly: adds are content-idempotent, but SQLite commit can precede a transport timeout or index failure.

## MCP owner

Install the MCP surface and pass it the same process-owned instance:

```bash
uv add "sensemem[local,mcp]"
```

```python
from sensemem import Memory
from sensemem.api.mcp import build_mcp_server

with Memory.from_config(
    {
        "data_dir": "/var/lib/sensemem/assistant",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "google/embeddinggemma-2",
        },
    }
) as memory:
    build_mcp_server(memory).run("stdio")
```

`build_mcp_server()` does not close `memory`. It snapshots `memory.capabilities` once while building the server and publishes it as server instructions. A stdio server inherits the host user's filesystem, environment, and credentials; if exposed over SSE or streamable HTTP, the host must add authentication, authorization, TLS, request limits, and rate limiting. The server publishes thirteen tools, and MCP error subjects may include owner-local paths.

## Media and model placement

Python accepts regular local `Path` values, inline `Blob` bytes, and `AssetRef` values from the same directory. REST and MCP accept inline bytes or stored asset IDs, never server paths or HTTP(S) URLs; fetch remote media in application code so redirects, SSRF controls, credentials, and download limits stay in the host's network policy.

After a successful add, SenseMem owns an immutable copy under `assets/` and the original source file is no longer required. Keep SQLite and `assets/` together through backup and restore.

Model placement is explicit: local adapters accept or delegate device selection, and custom backends may run locally or remotely. SenseMem does not schedule GPUs or move inference between edge and cloud.
