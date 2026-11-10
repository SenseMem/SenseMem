# Quick start

Give an agent its first persistent memory. This guide stores a fact, retrieves it, and reopens the
same store. It then adds grounded answers, event time, and an image observation.

The first example runs locally without a database service or model API key.

## 1. Install SenseMem

In an existing uv project:

```bash
uv add "sensemem[local]"
```

With pip:

```bash
pip install "sensemem[local]"
```

Use `uv init` first if you are starting in an empty directory. For a repository checkout, follow
[Contributing](../CONTRIBUTING.md#setup). Runtime requirements and other integrations are listed
in [installation](configuration.md#1-install-only-the-surfaces-you-use).

## 2. Store and retrieve a memory

Save this as `quickstart.py`:

```python
from sensemem import Memory

config = {
    "data_dir": "./data/sensemem-quickstart",
    "embedding": {
        "provider": "sentence-transformers",
        "model": "google/embeddinggemma-2",
    },
}

with Memory.from_config(config) as memory:
    stored = memory.add("The spare key is in the blue toolbox.")
    hits = memory.search("Where is the spare key?", limit=3).hits
    print(hits[0].content)
```

Run it:

```bash
uv run python quickstart.py
```

The first run downloads the embedding model. The output is:

```text
The spare key is in the blue toolbox.
```

`add()` returns a durable, searchable record. `search()` returns ranked evidence in
`SearchResult.hits`. The context manager closes model and storage resources when the block ends.

### Reopen the same memory

Append this second block to the script:

```python
with Memory.from_config(config) as memory:
    recovered = memory.get(stored.id)
    print(recovered.content)
```

It prints the same fact from the persisted store. Repeating an add with identical canonical content,
metadata, time, type, and context returns the existing record. SQLite retains records and
embeddings, and a missing search index rebuilds from those embeddings.

In your application, handle empty `hits` as a normal outcome. The default relevance gate excludes
weak candidates; `limit=1` also enables the top-two ambiguity check. Use
[retrieval diagnostics](troubleshooting.md#search-returns-no-hits) to tune a larger corpus.

## 3. Add grounded answers

A generation backend lets `ask()` turn retrieved evidence into an answer. Install the integration
alongside the local embedder and set its credential:

```bash
uv add "sensemem[local,openai]"
export OPENAI_API_KEY="replace-with-your-key"
```

Append this to `quickstart.py`:

```python
answer_config = {
    **config,
    "generation": {"provider": "openai", "model": "gpt-5-mini-2025-08-07"},
}

with Memory.from_config(answer_config) as memory:
    result = memory.ask("Where is the spare key?")

    if result.abstained:
        print("More evidence needed:", result.abstention_reason)
    else:
        print(result.answer)
        print("Evidence:", [hit.id for hit in result.hits])
```

`AnswerResult` keeps the answer and its evidence together. Your application can display the
supporting records and handle insufficient evidence explicitly. With this configuration, storage
and embedding stay local; the question and retrieved evidence go to the generation endpoint.

To build your own agent prompt instead, use [context compilation](context-compilation.md).
It produces a structured, budgeted bundle without a generation model.

## 4. Remember an event

Attach a memory role and event time when the application needs to recall what happened when.
The following snippets reuse `config` from step 2.

```python
from datetime import datetime, timezone

from sensemem import MemoryType

with Memory.from_config(config) as memory:
    memory.add(
        "Maya placed the spare key in the toolbox during the workshop.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
    )
    hits = memory.search(
        "What happened during the workshop?",
        memory_type=MemoryType.EPISODIC,
    ).hits
```

Use semantic memories for facts, episodic memories for events, and procedural memories for
instructions. [Memory types, time, and decay](memory-types-time-and-decay.md) covers event
intervals, historical validity, spatial scope, and ranking.

## 5. Add an image to the same observation

Place an image at `./workshop.jpg`, then add text and media together:

```python
from pathlib import Path

with Memory.from_config(config) as memory:
    record = memory.add(
        ("Maya is showing the blue toolbox.", Path("./workshop.jpg")),
    )
    print(record.modality, record.assets)
```

The ordered text and image share one memory ID, and SenseMem retains an immutable copy of the
original image. The same content contract accepts audio, video, inline `Blob` bytes, and existing
`AssetRef` values when the configured backends support them.

## Prepare your application

Keep one live owner per physical `data_dir`; use distinct directories for independent memory
domains. Keep the embedding space, dimension, and precision compatible when reopening a store.
For remote media, download and validate it in the host before passing bytes or a local path.

Choose model backends in [configuration](configuration.md), and use the
[deployment guide](deployment.md) to select process and network boundaries.

## Where to go next

| Goal | Guide |
| --- | --- |
| Choose capabilities for an application | [Product capabilities](product-capabilities.md) |
| Understand records, evidence, and model routing | [Core concepts](concepts.md) |
| Configure models and optional formation | [Configuration](configuration.md) |
| Use the complete API | [SDK reference](api/python-sdk.md) |
| Capture live audio, vision, or interaction | [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) |
| Connect another process or an agent host | [REST](api/rest.md) · [MCP](api/mcp.md) · [CLI](api/cli.md) |
| Evaluate your setup | [Benchmarking](benchmarking.md) |
