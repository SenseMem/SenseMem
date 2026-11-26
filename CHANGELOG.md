# Changelog

Release changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The [documentation index](docs/README.md) describes the current contracts.

## Unreleased

Changes for version `0.3.0`.

### Runtime and stored data

- Install `sensemem` and update imports, commands (`sensemem`, `sensemem-bench`), exceptions
  (`SenseMemError`), and benchmark variables (`SENSEMEM_*`). Python 3.11–3.14 and Zvec
  0.8.1 or later within the 0.8.x series are supported. Linux x86-64 local model extras use CUDA 13
  PyTorch builds. See [deployment upgrades](docs/deployment.md#upgrading-to-sensemem).
- The embedded runtime uses one owner per physical data directory. Replace the former external
  storage services and logical tenant/request scoping with separate physical memory domains.
- SQLite requires schema 25, complete operation-log payloads, and compatible recipe identities.
  Use the original runtime to read older directories and replay sources into a fresh one.
  Check persisted model and ingestion-stage identities before reopening. The `complete-v2` and
  `independent-v2` evidence projections require replay from older recipes. Changing quantization
  within the current index recipe rebuilds Zvec from stored FP32 vectors. See
  [representation changes](docs/operations.md#changing-the-runtime-or-stored-representations).
- `AsyncEventStream.append()` durably accepts media before `process()` assesses it. Checkpoint
  v5 retains disk references and enforces `max_pending_bytes`. Finish earlier checkpoints and
  outstanding work tied to an older segmenter recipe with their original runtime, or replay
  retained source input. See [continuous events](docs/omni-streaming-and-interaction-memory.md#automatic-event-segmentation).
- The remote EmbeddingGemma 2 `messages` recipe uses `special-tokens-v3`. Local EmbeddingGemma 2
  uses PyAV audio decoding and defaults to BF16 on CUDA GPUs with native support. Replay sources
  into a fresh directory when changing embedding spaces. Remote vLLM reranking uses `/score`
  with `queries` and `documents`. See [model setup](docs/configuration/models.md).

### SDK and model behavior

- Use `MemoryCategory`, `RecordKind`, `RecordSemantics`, `MemorySettings`, `MemorySpec`,
  `QueryRepresentationPlan`, and `RecallProgram` in place of `MemoryType`, `MemoryKind`,
  `MemoryContext`, `MemoryConfig`, `SenseMemConfig`, `QueryPlan`, and `RecallPlan`.
  Wire fields continue to use `memory_type`. Search, perception, and consolidation return
  typed results. See the [SDK reference](docs/api/python-sdk.md).
- Pass runtime policy as `settings=`. `MemorySpec` resolves each backend from a bundled provider
  or a host factory, replacing `RegisteredMemoryConfig` and the nested `plugins` shape.
  `build_memory_composition()` replaces `resolve_memory_config()` and returns owned adapters.
  See [construction](docs/api/python-sdk.md#construction).
- Replace `OpenAIModels` with `OpenAIEmbedder`, `OpenAITranscriber`, and `OpenAICompletion`.
  They each accept one client. Keep caller-owned clients open unless ownership is transferred.
  Recipe descriptions use the `classes` mapping. Completion defaults to strict JSON Schema and
  `store=False`. Configure `json_object` explicitly for JSON-only servers. See
  [provider configuration](docs/configuration/providers.md).
- Use `AgentRunner.run_async()` in place of `arun()`, and
  `ContextProof.structurally_independent_of()` in place of `independent_of()`.
  `EvidenceGeneration` prepares evidence inside its answer methods. Return
  `ContextSelection(anchor_ids, diversify=False)` from custom selectors and accept typed
  `query_failure` in custom consolidators. See [computation plugins](docs/api/python-policies.md)
  and [backend protocols](docs/api/python-backends.md).
- Automatic formation and consolidation proposals use `MODEL_INFERENCE`. Explicit host
  confirmation uses typed `apply(CONSOLIDATE)`. Formation sends up to eight history rows and
  6,000 content characters by default. Set `formation_history_max_rows: 0` to restrict inputs
  to the current batch. See [formation history](docs/configuration/lifecycle.md#bounded-formation-history).
- Automatic answer reinforcement defaults to `False`. Host-authored identity merges replace
  implicit face/voice co-occurrence linking and the `identity_link_min_assets` and `link_identities`
  controls. See [memory operations](docs/api/python-memory.md) and [identity operations](docs/api/python-identity.md).
- Compiled context includes temporary supplements, text-only projections, and delivery proof
  scoring metadata across SDK, REST, and MCP. Projections do not satisfy complete source proofs.
  See [context compilation](docs/context-compilation.md).

### Evaluation

- Evaluation artifact schema 25 uses `sensemem` arm names and `sensemem_version`. Text-only
  projection evidence is tracked separately from complete source coverage and original-media
  delivery. Retain older artifacts as records of their runs and start new runs with current inputs.
- Check the [supported tasks](docs/benchmarks/tasks.md) before rerunning an evaluation. The
  `longmemeval-v2-small` and `longmemeval-v2-medium` agent-trajectory tasks use different
  protocols from the retired `longmemeval-s` dialogue task.
