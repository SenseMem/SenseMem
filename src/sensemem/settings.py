"""Value-only settings for one local memory instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from sensemem.control_values import RetentionPolicy
from sensemem.types import IndexQuantization, RetrievalMode

MAX_FORMATION_HISTORY_ROWS = 128
MAX_RECALL_QUERIES = 32

_StrictBool = Annotated[bool, Field(strict=True)]
_UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1)]
_PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorySettings:
    """Value-only local policy kept separate from runtime capability objects."""

    # Reuse configured SpeechBackend analysis in the index without another model call.
    # Plain TranscriptionBackend output reaches the index through transcript derivation.
    index_speech: _StrictBool = True
    # Experimental support projection, pinned for the lifetime of the physical store.
    independent_evidence: _StrictBool = False
    # Bounded causal context lets formation reuse existing claims; zero explicitly disables it.
    formation_history_max_rows: Annotated[
        int, Field(strict=True, ge=0, le=MAX_FORMATION_HISTORY_ROWS)
    ] = 8
    formation_history_budget_chars: _PositiveInt = 6_000
    index_quantization: IndexQuantization = IndexQuantization.NONE
    # A stable instance policy for diagnostics and lexical fallback. It selects candidate routes;
    # it does not introduce a request-local visibility scope.
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    # Gate dense/lexical relevance, temporal proximity, and observation confidence. Zero admits
    # everything. Reinforcement and retention affect ranking only, so SearchHit.score may be lower.
    minimum_relevance: _UnitInterval = 0.10
    ambiguity_margin: _UnitInterval = 0.01
    # Supplemental character budget after the mandatory limit hits, with text-equivalent media
    # charges. None disables the ranked tail; planning and linked expansion have separate controls.
    # This does not cap provider tokens or shrink the mandatory window. Larger budgets increase
    # answer cost; the default's text-corpus evaluation does not establish media-corpus quality.
    evidence_budget_chars: _PositiveInt | None = 24_000
    # Whether ask plans bounded reads before answering. Off by default: it costs
    # one extra generation call per question, and it only pays where similarity has no way to
    # express the question -- a count, a list of "all", an adjacency, everything about a person.
    # With it off, ask runs exactly the one search it always has.
    recall_planning: _StrictBool = False
    # Experimental query execution for similar steps, bounded across all rounds of one ask.
    # Default recall programs reuse the original question's ranking.
    recall_query_expansion: _StrictBool = False
    # One shared search budget across recall steps and abstention-triggered answer retries.
    recall_max_queries: Annotated[int, Field(strict=True, ge=1, le=MAX_RECALL_QUERIES)] = 6
    # Describe query images transiently for text retrieval, retaining their native image keys.
    query_vision: _StrictBool = False
    # How many characters of evidence a set, sequence or entity plan may add to the ranked
    # window. A point question keeps limit (and evidence_budget_chars); a question that asks
    # for every matching record needs more than a window of that width, and this is what bounds
    # what it adds. It never trims the window itself -- that is what the question would have been
    # answered from with no plan at all. Only read when recall_planning is on.
    recall_set_budget_chars: _PositiveInt = 30_000
    # Additional matched rows when recall_planning is on. Rows beyond the cap are reported as
    # omitted, making the matched set incomplete. The mandatory ranked window is outside this cap.
    recall_set_max_rows: _PositiveInt = 60
    # How many answer attempts one ask may spend. Only an eligible abstention triggers
    # another plan: under best-effort, or under strict with query expansion enabled.
    # 1 disables follow-up rounds. Only read when recall_planning is on.
    recall_rounds: _PositiveInt = 2
    # Add records linked by capture, identity, place, lineage, or evidence without model calls.
    # Expansion preserves the ranked window and drops nonselective edges. The capture edge scans
    # memory_semantics; the other edges are indexed.
    evidence_expansion: _StrictBool = False
    # How many linked records one expansion may add, whatever evidence_budget_chars leaves room
    # for. The same reasoning as recall_set_max_rows, at the smaller size a window extension
    # wants rather than a set read: this is the tail of a window, not the answer's shape. Media
    # rows among them stop earlier still, at twice the ask's own limit -- the same cap a recall
    # program's matched media rows hit, because a linked clip costs face and speech recognition
    # before the answer call and again on every replan, and a capture on a photo corpus is a whole
    # day of them.
    evidence_expansion_max_rows: _PositiveInt = 24
    decay_half_life_days: _PositiveFloat | None = None
    # A model seeing a source does not establish that it used it correctly. Hosts may opt into
    # this heuristic, or explicitly reinforce records after observing useful task outcomes.
    reinforce_on_answer: _StrictBool = False
    # Conservative voice threshold; not calibrated for this max-over-20-exemplars matcher.
    # The pinned CAM++ recipe's 0.31 is a pairwise threshold and cannot be reused for max-over-N.
    # Higher values fragment identities; lower values can merge different people. Changes require
    # task-specific calibration. Margin compares competing identities and does not calibrate this.
    speaker_similarity: _UnitInterval = 0.78
    # Minimum separation between the best two candidate identities.
    speaker_margin: _UnitInterval = 0.05
    # SFace cosine threshold; max-over-N exemplar calibration remains necessary.
    face_similarity: _UnitInterval = 0.363
    face_margin: _UnitInterval = 0.05
    # The PRESSURE trigger's whole definition: how many active records this instance is meant to
    # hold. None -- the default -- means the host has declared no budget, and no amount of
    # growth is evidence that consolidating or forgetting anything is useful, so the trigger
    # derives nothing. A count rather than bytes on disk because that is the quantity the
    # deliberation acts on (records to consolidate or forget) and the one a host can reason about
    # without knowing how media is stored.
    memory_budget_records: _PositiveInt | None = None
    # How far back consolidation_candidates looks for repeated recall failures. Two or more
    # near-equal queries that returned nothing inside this window are one QUERY_FAILURE
    # candidate; one failure is not a signal, and a failure from last month is not this week's
    # gap. One hour matches an interactive session.
    query_failure_window_seconds: _PositiveFloat = 3600.0
    # How many recall failures the store keeps at all. The table is a bounded signal buffer, not
    # a query log: the oldest rows falling out is the retention policy.
    query_failure_history: _PositiveInt = 512
    # What apply_retention() is allowed to delete. Declarative configuration spells it as its
    # own retention section rather than a field in here, because every other setting shapes
    # what recall returns and can be changed back, and this one deletes. Empty by default: no
    # age is declared, so nothing ages out.
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
