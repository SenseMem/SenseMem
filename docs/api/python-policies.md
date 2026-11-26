# Computation plugins and processing stages

Replace computation through public protocols while the kernel continues to own source identity,
SQLite transactions, index recovery, evidence closure, consent, and final budget admission.
Import every value below from `sensemem`.

| Plugin | Constructor keyword | Result |
| --- | --- | --- |
| `QueryPolicy` | `query_policy` | A `QueryRepresentationPlan` selecting rewritten text, embedding inputs and lexical routes |
| `CandidateSource` | `candidate_source` | A `CandidateBatch` containing index or local memory references |
| `RankingPolicy` | `ranking_policy` | One `RankingScore` per supplied candidate |
| `ContextSelector` | `context_selector` | An ordered subset of supplied context anchor IDs |
| `IngestionStage` | `ingestion_stages` | A `StageResult` containing derived interpretation |

The same keywords are available on `MemoryPlugins`; `AsyncMemory.from_plugins()` constructs
the same runtime. Registered factories can construct these roles alongside bundled models;
see [plugin configuration](../configuration/plugins.md).

## Query preparation

`QueryPolicy.plan(QueryRequest)` receives the leased query, retrieval mode, supported complete
and focused representations, and available descriptions. Return a `QueryRepresentationPlan` with the query
text, required `embedding_inputs`, optional `optional_embedding_inputs`, and `lexical_queries`.
The policy may rewrite text and choose representations; the kernel executes embedding calls.
Required embeddings fail the operation on error. Optional embedding failures retain other routes.

`DefaultQueryPolicy` embeds the supported aggregate and bounded atomic keys across the ordered
input, including text after media. Lexical retrieval includes the full text alongside the focused
query and available visual descriptions. Lexical mode performs no embedding, and dense mode
adds no lexical routes. Custom plans obey those same mode constraints. Media references must
match the leased query exactly, and each selected input must satisfy the configured encoder's
joint modality and asset-count capabilities. Invalid plans fail with `ModelError` at
`stage="query.plan"`.

```python
from sensemem import QueryRepresentationPlan, QueryRequest


class GlossaryQuery:
    def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
        text = request.focused_text.replace("spare", "backup")
        return QueryRepresentationPlan(text=text, lexical_queries=(text,))

    def close(self) -> None:
        pass
```

This example is a lexical query policy. Use it with lexical retrieval or a candidate source
whose `requires_embeddings` is `False`. The latter makes query preparation lexical even when
the configured retrieval mode also supports dense ranking.

## Candidate discovery

`CandidateSource.search(CandidateRequest, CandidateReadPort)` receives prepared routes and the
current candidate-window limit. Its `requires_embeddings` declaration is fixed at assembly.
`DefaultCandidateSource` uses the read port for native dense and lexical discovery, including
memory-type routes and preferred temporal ranges. The read port runs route batches with the
runtime's bounded parallelism. A `CandidateQuery` may narrow type or occurrence bounds; the
kernel fixes instance identity, embedding space, place and identity scope, and route depth.

Return `CandidateBatch` with native `CandidateHit` values or external `MemoryCandidate` values
referring to existing local memory IDs. Unknown, deleted, expired and out-of-scope records are
dropped during authoritative SQLite hydration. A source cannot inject record content or media.
It can combine `DefaultCandidateSource().search(request, reader)` with external references in
one batch; no separate registry or composite-source type is required.

```python
from sensemem import CandidateBatch, CandidateReadPort, CandidateRequest, MemoryCandidate


class ApplicationShortlist:
    requires_embeddings = False

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
        return CandidateBatch(
            memories=(MemoryCandidate(memory_id=self.memory_id, relevance=1.0),),
            exhausted=True,
        )

    def close(self) -> None:
        pass
```

Each pass permits at most `request.limit` external references and
`request.max_routes * request.limit` native hits. `exhausted` describes discovery routes, not
complete coverage of a recall plan. The kernel retains adaptive candidate widening, SQLite
hydration, confidence and relevance gates, media leases and final ranking. Invalid batches and
read requests raise `ValidationError`. Unexpected source failures use
`ModelError(stage="candidates", reason="model_failed")` without exposing exception bodies.

## Retrieval scoring

`RankingPolicy.rank(request: RankingRequest)` receives a query string, reference time,
retrieval mode, and immutable `RankingCandidate` snapshots. Each candidate includes a public
`RankingMemory`, dense relevance, lexical relevance and coverage, and history factors
for reinforcement, temporal proximity, and retention. Candidates have already passed authoritative
SQLite hydration and request scope checks. Candidate asset references contain metadata with
`path=None`; scoring does not open or validate files that final selection may discard. Media
inference belongs in the leased `RerankingBackend` path. Original content is untrusted evidence.
Snapshots also expose `updated_at`, `last_accessed_at`, and `access_count` so custom policies can
compute their own history factors instead of depending on the default factors.

Return one `RankingScore(memory_id, score, relevance)` for every supplied candidate. Both numeric
values must be finite and between zero and one. `score` proposes the final ranking score before
observation confidence. `relevance` proposes query relevance before the kernel applies temporal
proximity, observation confidence, the minimum-relevance threshold, and the ambiguity check.
The optional `lexical_relevance` and `lexical_bonus` fields describe the policy's lexical
contribution in retrieval traces. History factors in those traces describe the supplied inputs;
a custom policy decides which factors affect its score.

Missing, repeated, or invented memory IDs fail with `ModelError(reason="response_invalid",
stage="ranking")`. Execution failures use `model_failed` without exposing exception bodies.
The policy cannot introduce records or change their content. Final ordering, result limits,
asset leases, and optional model reranking remain in the retrieval plane.

```python
from sensemem import DefaultRankingPolicy, RankingRequest, RankingScore


class DenseRanking:
    def rank(self, request: RankingRequest) -> tuple[RankingScore, ...]:
        return tuple(
            RankingScore(
                candidate.memory.id,
                candidate.dense_relevance,
                candidate.dense_relevance,
            )
            for candidate in request.candidates
        )

    def close(self) -> None:
        pass


default = DefaultRankingPolicy()
custom = DenseRanking()
```

The built-in `DefaultRankingPolicy` follows this same protocol and retains the existing bounded
dense/lexical fusion, reinforcement, temporal proximity, and decay behavior. A custom scorer
changes retrieval policy and requires task-specific evaluation before deployment. The same
retrieval plane supplies `search()`, `ask()`, and `compile()`.

## Context selection

`ContextSelector.select(request: ContextSelectionRequest)` returns `ContextSelection`. Its
`anchor_ids` are unique IDs in admission priority order and may be empty. `diversify=True` asks
the compiler to spread successful admissions across sections before filling the remaining budget. The request supplies the goal,
reference time, remaining budget, reserved characters, and `ContextCandidate` values describing
the authorized hits, section costs, and complete required evidence. Selectors are deterministic
computations over this request; inference belongs in a backend or processing stage.

```python
from sensemem import ContextSelection, ContextSelectionRequest


class TaskSelection:
    def select(self, request: ContextSelectionRequest) -> ContextSelection:
        return ContextSelection(
            tuple(
                candidate.hit.id
                for candidate in request.candidates
                if candidate.hit.metadata.get("task") == request.goal
            )
        )

    def close(self) -> None:
        pass
```

The compiler maps IDs back to its own records and applies complete evidence closures, conflict
dependencies, consent, and item, character, and media budgets. Unknown or duplicate IDs raise
`ValidationError`. Deliberately excluded anchors remain distinguishable from budget omissions.
Conflict detection retains the full authorized candidate set, even when a selector omits one
side of a conflict. A selected anchor can therefore bring required evidence whose ID the selector
did not separately select. Selection priority does not rewrite retrieval scores or their public
display order.

`DefaultContextSelector` retains rank-first selection and affordable section diversity. Selectors only choose priorities and diversity; a single compiler admission ledger owns
full records, excerpts, projections, proofs and item, character and media accounting. Both
the default and custom implementations use that admission. Optional evidence
annotations can change costs, so the compiler may request selection again within one compilation.

## Ordered ingestion stages

An `IngestionStage` declares `name` and `space`, implements `process(ModelInput) -> StageResult`,
and supplies `close()`. Names must be unique within a composition. `space` identifies the complete
recipe, including model revision, prompts, and processing settings when applicable.

```python
from sensemem import ModelInput, StageResult


class SceneTags:
    name = "scene-tags"
    space = "application-scene-tags:v1"

    def process(self, input: ModelInput) -> StageResult:
        if "workshop" in input.text.lower():
            return StageResult("Interpretation: this observation concerns the workshop.")
        return StageResult()

    def close(self) -> None:
        pass
```

Stages run in declared order before built-in perception and embedding. Each sees preceding
stage interpretations and leased source assets. The kernel appends nonempty results as labelled
`model_interpretation` parts, preserving raw text, asset identity, metadata, and source IDs.
These interpretations do not become transcripts, visual observations, identity claims, consent,
or host instructions. A stage cannot by itself make an unsupported media input embeddable;
source-backed media fallback still requires the matching transcription or vision backend.

`add()`, `add_many()`, and streamed additions share these stages. `capture()` commits the raw
source without running them; `settle()` runs them before making it searchable. Duplicate adds
reuse existing derivations. Results are cached within one operation so an internal identity
retry does not repeat the same external computation. Failed computations leave additions
uncommitted and deferred captures retryable.

The ordered name/space list is stored in SQLite metadata as `processing.ingestion_recipe`.
Changing a nonempty store's stage composition, order, or recipe, including removing stages,
requires replaying original sources into a fresh directory. The marker is required even for
an empty stage list; a populated directory without it must also be replayed. This prevents silently mixing
incompatible derived content under the same stable source IDs. Ranking and context selection
are request-local, as are query policies and candidate sources; they can change after reopening
without re-ingestion.

## Ownership

Composition is fixed for one `Memory` lifetime. Plugins must be thread-safe. Shared objects and
nested resources close once, including objects serving several model, strategy, stage, or tool
roles. Assembly failures clean up resources already constructed. Custom Python code remains
trusted host code; the typed protocols grant no additional storage authority.
