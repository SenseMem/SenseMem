"""Control values values and pure operations for the public SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from sensemem._value_validation import _memory_ids, _optional_text, _require_aware, _text
from sensemem.exceptions import ValidationError
from sensemem.identity import ConsentClaim, IdentityChange, IdentityClaim, IdentityProfile
from sensemem.records import MemoryRecord
from sensemem.semantics import FormationProposal
from sensemem.spatial import RetrievalScope
from sensemem.types import MemoryCategory


class MemoryIntent(str, Enum):
    """One memory-management operation the agentic control plane may propose."""

    REINFORCE = "reinforce"
    CONSOLIDATE = "consolidate"
    CORRECT = "correct"
    FORGET = "forget"
    IDENTIFY = "identify"
    # Host-initiated only, for the same reason MERGE is host-initiated only: consent is a
    # statement the data subject makes, so a model that could propose one could manufacture it.
    CONSENT = "consent"
    # Host-confirmed only. Repeated co-occurrence does not establish who spoke; model-proposed
    # MERGE operations are unauthorized.
    MERGE = "merge"


_IDENTITY_INTENTS = frozenset(
    {MemoryIntent.MERGE, MemoryIntent.CORRECT, MemoryIntent.FORGET},
)


class MemoryTrigger(str, Enum):
    """Why one bounded memory-management deliberation ran."""

    MANUAL = "manual"
    EVIDENCE = "evidence"
    FEEDBACK = "feedback"
    CONTRADICTION = "contradiction"
    QUERY_FAILURE = "query_failure"
    PRESSURE = "pressure"
    IDLE = "idle"


class MemoryOutcome(str, Enum):
    """What later evidence said about one already-applied control-plane operation.

    Recorded after the fact by record_outcome(), which is what makes the slow loop's quality
    measurable: consolidation precision and contradiction recovery are CONFIRMED rates over
    CONSOLIDATE and CORRECT rows, and false retirement is the REFUTED rate over rows that
    forgot something. The kernel never sets it and never reads it back into a decision.
    """

    CONFIRMED = "confirmed"
    REFUTED = "refuted"


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryOperation:
    """One proposed memory operation whose required fields follow from its intent."""

    intent: MemoryIntent
    evidence_ids: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    proposal: FormationProposal | None = None
    claim: IdentityClaim | None = None
    # Set exactly by CONSENT, which asserts how one person may be processed rather than
    # moving any person or any memory.
    consent: ConsentClaim | None = None
    # Set exactly by the three identity-lifecycle intents: a MERGE, and a CORRECT or
    # FORGET that names a person instead of memories.
    identity: IdentityChange | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - intent-specific boundary validation
        if not isinstance(self.intent, MemoryIntent):
            try:
                object.__setattr__(self, "intent", MemoryIntent(self.intent))
            except (TypeError, ValueError):
                raise ValidationError("memory operation intent is invalid") from None
        object.__setattr__(self, "evidence_ids", _memory_ids(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "target_ids", _memory_ids(self.target_ids, "target_ids"))
        object.__setattr__(
            self,
            "rationale",
            _optional_text(self.rationale, "operation rationale"),
        )
        if self.proposal is not None and not isinstance(self.proposal, FormationProposal):
            raise ValidationError("memory operation proposal is invalid")
        if self.claim is not None and not isinstance(self.claim, IdentityClaim):
            raise ValidationError("memory operation claim is invalid")
        if self.consent is not None and not isinstance(self.consent, ConsentClaim):
            raise ValidationError("memory operation consent is invalid")
        if self.intent is MemoryIntent.CONSENT:
            # A consent statement names its subject and nothing else: no memory is evidence for
            # it, no memory is its target, and it moves no identity.
            if (
                self.consent is None
                or self.claim is not None
                or self.identity is not None
                or self.target_ids
                or self.evidence_ids
                or self.proposal is not None
            ):
                raise ValidationError("consent names one identity and one state, and nothing else")
            return
        if self.consent is not None:
            raise ValidationError(f"{self.intent.value} must not carry consent")
        if self.identity is not None:
            if not isinstance(self.identity, IdentityChange):
                raise ValidationError("memory operation identity is invalid")
            if self.intent not in _IDENTITY_INTENTS:
                raise ValidationError(f"{self.intent.value} must not carry an identity")
        if self.intent is MemoryIntent.MERGE:
            # One absorbed identity, no memory: which naming assertions and exemplars moved is
            # an effect the log row records, not something a proposal states.
            if (
                self.identity is None
                or len(self.identity.moved_ids) != 1
                or self.target_ids
                or self.evidence_ids
                or self.proposal is not None
            ):
                raise ValidationError(
                    "merge names one surviving and one absorbed identity, and no memory"
                )
            return
        if self.intent is MemoryIntent.IDENTIFY:
            # A host naming somebody cites nothing, so an empty evidence set is well formed here
            # and it is the kernel that requires cited evidence of an agent's proposal.
            if self.claim is None or self.target_ids or self.proposal is not None:
                raise ValidationError(
                    "identify requires a claim and cites evidence, and names no target"
                )
            return
        if self.claim is not None:
            raise ValidationError(f"{self.intent.value} must not carry a claim")
        if self.intent is MemoryIntent.CONSOLIDATE:
            # target_ids on a consolidation is consolidation forgetting: the sources to retire
            # from ordinary recall once the derived record exists. The kernel enforces the
            # containment rule (a target must be one of this proposal's own evidence IDs); the
            # value type only knows that naming a target requires evidence to name it from.
            if self.proposal is None or not self.evidence_ids:
                raise ValidationError("consolidate requires a proposal and cited evidence")
        elif self.proposal is not None:
            raise ValidationError(f"{self.intent.value} must not carry a proposal")
        elif self.intent is MemoryIntent.REINFORCE:
            if len(self.target_ids) != 1 or not self.evidence_ids:
                raise ValidationError("reinforce requires exactly one target and cited evidence")
        elif self.identity is not None:
            # A split or an erasure names the person, not a record. Naming both would leave the
            # log row claiming two unrelated effects for one operation.
            if self.target_ids or self.evidence_ids:
                raise ValidationError(
                    f"{self.intent.value} names either memories or one identity, not both"
                )
        elif not self.target_ids or (self.intent is MemoryIntent.FORGET and self.evidence_ids):
            raise ValidationError(
                f"{self.intent.value} requires targets; only correct may cite counterevidence"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryOperationRecord:
    """One applied control-plane operation as the append-only log holds it."""

    operation_id: int
    operation: MemoryOperation
    trigger: MemoryTrigger
    applied_at: datetime
    model_id: str | None = None
    recipe: str | None = None
    created_ids: tuple[str, ...] = ()
    changed_ids: tuple[str, ...] = ()
    # Records this operation moved out of ordinary recall. On a FORGET row that is cognitive
    # forgetting; on a CONSOLIDATE row it is consolidation forgetting, the sources the derived
    # record replaced. rollback() clears exactly these.
    forgotten_ids: tuple[str, ...] = ()
    # (memory_id, version) pairs the kernel's own lineage rule superseded while applying this
    # operation: the records a new STATE or user-stated TRAIT replaced in its lineage, which
    # the backend never named and may never have been shown. rollback() restores exactly these.
    superseded: tuple[tuple[str, int], ...] = ()
    rolled_back_at: datetime | None = None
    # Post-hoc, written by record_outcome() and never by the kernel: what later evidence said
    # about this operation. None means nobody has judged it yet, which is not the same as
    # unrefuted.
    outcome: MemoryOutcome | None = None
    outcome_note: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.operation_id, bool)
            or not isinstance(self.operation_id, int)
            or self.operation_id <= 0
        ):
            raise ValidationError("operation_id must be a positive integer")
        if not isinstance(self.operation, MemoryOperation):
            raise ValidationError("operation must be a MemoryOperation")
        if not isinstance(self.trigger, MemoryTrigger):
            raise ValidationError("operation trigger is invalid")
        _require_aware(self.applied_at, "applied_at")
        _require_aware(self.rolled_back_at, "rolled_back_at")
        for name in ("model_id", "recipe"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"operation {name}"),
            )
        object.__setattr__(self, "created_ids", _memory_ids(self.created_ids, "created_ids"))
        object.__setattr__(self, "changed_ids", _memory_ids(self.changed_ids, "changed_ids"))
        object.__setattr__(self, "forgotten_ids", _memory_ids(self.forgotten_ids, "forgotten_ids"))
        superseded = tuple(self.superseded)
        for memory_id, version in superseded:
            _memory_ids((memory_id,), "superseded")
            if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
                raise ValidationError("superseded version must be a positive integer")
        object.__setattr__(self, "superseded", superseded)
        if self.outcome is not None and not isinstance(self.outcome, MemoryOutcome):
            raise ValidationError("operation outcome is invalid")
        object.__setattr__(
            self,
            "outcome_note",
            _optional_text(self.outcome_note, "operation outcome_note"),
        )
        if self.outcome is None and self.outcome_note is not None:
            raise ValidationError("an outcome note requires an outcome")


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryFailure:
    """The text and retrieval constraints of a failed memory request."""

    query: str
    operation: Literal["search", "compile", "ask"] = "search"
    memory_types: tuple[MemoryCategory, ...] | None = None
    scope: RetrievalScope | None = None
    reference_at: datetime | None = None
    occurred_from: datetime | None = None
    occurred_until: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "failed query"))
        if not isinstance(self.operation, str) or self.operation not in {
            "search",
            "compile",
            "ask",
        }:
            raise ValidationError("failed query operation is invalid")
        if self.memory_types is not None:
            values = tuple(self.memory_types)
            if any(not isinstance(value, MemoryCategory) for value in values):
                raise ValidationError(
                    "failed query memory_types must contain MemoryCategory values"
                )
            object.__setattr__(self, "memory_types", tuple(dict.fromkeys(values)))
        if self.scope is not None and not isinstance(self.scope, RetrievalScope):
            raise ValidationError("failed query scope must be a RetrievalScope")
        for name in ("reference_at", "occurred_from", "occurred_until"):
            _require_aware(getattr(self, name), f"failed query {name}")
        if (
            self.occurred_from is not None
            and self.occurred_until is not None
            and self.occurred_from >= self.occurred_until
        ):
            raise ValidationError("failed query occurrence bounds must increase")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidationCandidate:
    """One piece of deliberation work the store's own state says is due.

    memory_ids is the set to hand straight to consolidate(evidence_ids=...). evidence_count
    is what the trigger counted: new evidence links for EVIDENCE, distinct conflicting values
    for CONTRADICTION, recorded confirmations for FEEDBACK, recorded recall failures for
    QUERY_FAILURE, and the record count over budget for PRESSURE. An IDLE row counts one:
    the operator opened the window, so the count is the lineage, not a measured signal.
    """

    trigger: MemoryTrigger
    memory_ids: tuple[str, ...]
    evidence_count: int
    query_failure: QueryFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trigger, MemoryTrigger):
            raise ValidationError("candidate trigger is invalid")
        object.__setattr__(self, "memory_ids", _memory_ids(self.memory_ids, "memory_ids"))
        if not self.memory_ids and self.query_failure is None:
            raise ValidationError("a consolidation candidate must name at least one memory")
        if self.query_failure is not None and (
            self.trigger is not MemoryTrigger.QUERY_FAILURE
            or not isinstance(self.query_failure, QueryFailure)
        ):
            raise ValidationError("query failure context requires its matching trigger")
        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(self.evidence_count, int)
            or self.evidence_count <= 0
        ):
            raise ValidationError("evidence_count must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidationReport:
    """What a bounded consolidation run applied and which proposals the kernel refused.

    weighed is how many evidence records the backend was actually shown. Zero means the pass
    reached no backend at all. rounds counts executed passes, skipped counts inactive
    candidate sets, and model_calls counts backend round trips. Explicit evidence runs one
    pass; automatic candidate runs aggregate the same values and operation records.
    """

    operations: tuple[MemoryOperationRecord, ...] = ()
    rejected: tuple[tuple[MemoryOperation, str], ...] = ()
    weighed: int = 0
    rounds: int = 0
    skipped: int = 0
    model_calls: int = 0
    unresolved_queries: tuple[QueryFailure, ...] = ()

    @property
    def applied(self) -> int:
        return len(self.operations)

    def __post_init__(self) -> None:
        for name in ("weighed", "rounds", "skipped", "model_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"consolidation {name} must be a non-negative integer")
        operations = tuple(self.operations)
        if any(not isinstance(value, MemoryOperationRecord) for value in operations):
            raise ValidationError("consolidation operations are invalid")
        try:
            rejected = tuple((value, reason) for value, reason in self.rejected)
        except (TypeError, ValueError):
            raise ValidationError("consolidation rejections are invalid") from None
        if any(
            not isinstance(value, MemoryOperation)
            or not isinstance(reason, str)
            or not reason.strip()
            for value, reason in rejected
        ):
            raise ValidationError("consolidation rejections are invalid")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "rejected", rejected)
        unresolved = tuple(self.unresolved_queries)
        if any(not isinstance(value, QueryFailure) for value in unresolved):
            raise ValidationError("consolidation unresolved queries are invalid")
        object.__setattr__(self, "unresolved_queries", unresolved)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportBundle:
    """Everything one physical memory holds about one data subject, in transferable form.

    The right of access, answered from the same authoritative rows every other read uses.
    records carries every version, including the retired and the cognitively forgotten ones,
    because a subject asking what is held about them is owed the history rather than the
    projection; each record's context.evidence_ids is the evidence link that record stands on
    and each record's assets are its media by identity, size, and digest. No media bytes are
    ever inlined: an export names the assets, and copying them is the host's decision.
    """

    exported_at: datetime
    identity_id: str | None
    identities: tuple[IdentityProfile, ...]
    records: tuple[MemoryRecord, ...]
    operations: tuple[MemoryOperationRecord, ...]

    def __post_init__(self) -> None:
        _require_aware(self.exported_at, "exported_at")
        object.__setattr__(
            self,
            "identity_id",
            _optional_text(self.identity_id, "identity_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetentionPolicy:
    """Declarative maximum ages after which stored material is physically deleted.

    Every field is None by default, and None means "no policy", not "keep for zero days":
    physical deletion is the one lifecycle step SenseMem never takes because a clock ticked
    and nobody said so. Ages are in days and are measured from the material's own recorded
    time, not from the last time it was read, so a policy cannot be reset by traffic.
    """

    # Media assets older than this, and every memory that still references them, are deleted.
    # Media dominates storage growth, so this is the cost knob as well as the privacy one.
    media_days: float | None = None
    # Records cognitively forgotten longer ago than this are physically deleted. Cognitive
    # forgetting is reversible by design; this is what eventually makes it final.
    forgotten_days: float | None = None
    # Capture-queue rows enqueued longer ago than this that have failed at least once are
    # abandoned. The memory itself survives -- only the promise to keep retrying is dropped.
    capture_failure_days: float | None = None

    def __post_init__(self) -> None:
        for name in ("media_days", "forgotten_days", "capture_failure_days"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValidationError(f"retention {name} must be a positive number of days")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class RetentionReport:
    """What one retention pass deleted, or would have deleted under dry_run.

    Identifiers rather than content: delete() leaves no operation-log row, so this report is
    the only account of a physical deletion, and it has to be safe to keep after the content it
    names is unrecoverable.
    """

    dry_run: bool
    # Memories deleted because they referenced media older than media_days.
    media_memory_ids: tuple[str, ...] = ()
    # Memories deleted because they had been cognitively forgotten longer than forgotten_days.
    forgotten_memory_ids: tuple[str, ...] = ()
    # Additional derived records deleted because every complete support clause was lost.
    cascade_memory_ids: tuple[str, ...] = ()
    # Media descriptors and their bytes removed because no memory referenced them any more.
    asset_ids: tuple[str, ...] = ()
    # Capture-queue rows abandoned; the memories themselves stay.
    capture_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dry_run, bool):
            raise ValidationError("dry_run must be a boolean")
        for name in (
            "media_memory_ids",
            "forgotten_memory_ids",
            "cascade_memory_ids",
            "asset_ids",
            "capture_memory_ids",
        ):
            object.__setattr__(self, name, _memory_ids(getattr(self, name), name))

    @property
    def deleted(self) -> int:
        """How many memories this pass removed, which is what a policy is judged on."""
        return len(
            {
                *self.media_memory_ids,
                *self.forgotten_memory_ids,
                *self.cascade_memory_ids,
            }
        )
