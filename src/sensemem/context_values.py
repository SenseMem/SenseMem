"""Context values values and pure operations for the public SDK."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Container, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, TypeAlias

from sensemem._naming_conditions import experience_footprint
from sensemem._validation import positive_int, unit_interval
from sensemem._value_validation import (
    _SHA256,
    _memory_ids,
    _optional_text,
    _require_aware,
    _require_interval,
    _text,
)
from sensemem.content import AssetRef, ContentPart
from sensemem.exceptions import ValidationError
from sensemem.prompts import context as context_prompts
from sensemem.records import SearchHit
from sensemem.semantics import RecordSemantics
from sensemem.types import MemoryCategory, RecordKind


class ContextUnknownKind(str, Enum):
    """Why a compiled bundle is missing something the request implied it might contain."""

    SCOPE_EMPTY = "scope_empty"
    SECTION_EMPTY = "section_empty"
    BUDGET_EXCLUDED = "budget_excluded"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"
    MODALITY_UNSUPPORTED = "modality_unsupported"
    STAGE_SKIPPED = "stage_skipped"
    CONSENT_WITHHELD = "consent_withheld"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"


class ContextSymbolNamespace(str, Enum):
    """The durable identifier family one request-local context symbol represents."""

    MEMORY = "memory"
    IDENTITY = "identity"


class ContextSymbolCoverage(str, Enum):
    """How much of a symbol's memory source is present in this presentation."""

    FULL = "full"
    PARTIAL = "partial"
    TEXT_ONLY = "text_only"
    REFERENCE_ONLY = "reference_only"


class ContextSymbolRole(str, Enum):
    """How a stable identifier is used by one compact context presentation."""

    HIT = "hit"
    EXCERPT = "excerpt"
    PROJECTION = "projection"
    EVIDENCE = "evidence"
    AFFECT_EVENT = "affect_event"
    AFFECT_FOCUS = "affect_focus"
    AFFECT_OBJECT = "affect_object"
    ACTOR = "actor"
    OBSERVED_IN = "observed_in"
    NAMING_ASSERTION = "naming_assertion"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class ContextSupplement:
    """Caller-supplied temporary evidence, never an instruction or a stored memory."""

    content: str
    source: str
    observed_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _text(self.content, "supplement content"))
        object.__setattr__(self, "source", _text(self.source, "supplement source"))
        _require_aware(self.observed_at, "supplement observed_at")
        _require_aware(self.expires_at, "supplement expires_at")
        if (
            self.observed_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.observed_at
        ):
            raise ValidationError("supplement expires_at must follow observed_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBudget:
    """The size, type, confidence, and freshness bounds one context compilation may spend.

    max_chars bounds the rendered evidence, the same quantity ContextBundle.chars reports:
    the four-line header, each section heading, each memory's rendered line with its - [id]
    frame and confidence suffix, a text-equivalent price per grounded media part, and each
    NamedActor or ProvisionalActor line. A bundle that carries no compilation diagnostics
    therefore satisfies len(bundle.render()) <= max_chars; only the ## Conflicts,
    ## Unknowns, and Omitted: blocks are appended outside it, because they explain the
    bundle rather than ground it and suppressing them to fit a budget would make a thin bundle
    look like an empty store. An actor line is priced and fit in after the ranked hits it
    depends on, the same as any other line the compiler grounds -- dropped, not given away for
    free, when what is left of max_chars cannot buy it. Because the header is inside the bound
    and is written whether or not anything else fits, a max_chars the header alone overruns has
    no bundle that satisfies it: compilation refuses it with ValidationError naming the floor,
    rather than returning a bundle over the limit it was given. The floor moves with the goal,
    which the header repeats.

    max_media_items bounds grounded media parts instead of their price: 0 compiles a
    text-only bundle, and None lets max_chars alone decide. The default max_chars buys the
    most expensive single grounded part -- a video part is priced at 12 000 characters -- and
    still leaves room for its record text and a cheaper second part, so a default compilation
    can reach a media memory at all.
    """

    max_chars: int = 16000
    max_items: int = 24
    max_media_items: int | None = None
    memory_types: frozenset[MemoryCategory] | None = None
    min_confidence: float = 0.0
    freshness: timedelta | None = None
    # A deadline, not a timeout: the compiler checks the clock between stages and stops adding
    # optional ones. It never interrupts work already in flight, so an exceeded deadline is
    # reported on the bundle rather than raised.
    max_latency_ms: int | None = None
    # Full provenance remains the default; selected delivery carries explicit certificates.
    selected_proofs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.selected_proofs, bool):
            raise ValidationError("budget selected_proofs must be a boolean")
        object.__setattr__(self, "max_chars", positive_int(self.max_chars, "budget max_chars"))
        object.__setattr__(self, "max_items", positive_int(self.max_items, "budget max_items"))
        if self.max_media_items is not None:
            if isinstance(self.max_media_items, bool) or not isinstance(self.max_media_items, int):
                raise ValidationError("budget max_media_items must be a non-negative integer")
            if self.max_media_items < 0:
                raise ValidationError("budget max_media_items must be a non-negative integer")
        if self.max_latency_ms is not None:
            object.__setattr__(
                self,
                "max_latency_ms",
                positive_int(self.max_latency_ms, "budget max_latency_ms"),
            )
        if self.memory_types is not None:
            memory_types = frozenset(self.memory_types)
            if not memory_types or any(
                not isinstance(value, MemoryCategory) for value in memory_types
            ):
                raise ValidationError("budget memory_types must name at least one MemoryCategory")
            object.__setattr__(self, "memory_types", memory_types)
        object.__setattr__(
            self,
            "min_confidence",
            unit_interval(self.min_confidence, "budget min_confidence"),
        )
        if self.freshness is not None and (
            not isinstance(self.freshness, timedelta) or self.freshness <= timedelta(0)
        ):
            raise ValidationError("budget freshness must be a positive timedelta")

    def document(self) -> dict[str, object]:
        """Return the JSON-ready bound document every surface reads and writes.

        freshness is a timedelta; JSON carries the same bound as freshness_seconds, which is
        the one rename between this projection and the dataclass.
        """
        return {
            "selected_proofs": self.selected_proofs,
            "max_chars": self.max_chars,
            "max_items": self.max_items,
            "max_media_items": self.max_media_items,
            "memory_types": (
                None
                if self.memory_types is None
                else sorted(value.value for value in self.memory_types)
            ),
            "min_confidence": self.min_confidence,
            "freshness_seconds": (
                None if self.freshness is None else self.freshness.total_seconds()
            ),
            "max_latency_ms": self.max_latency_ms,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextConflict:
    """Two or more candidates in one lineage that disagree on the same value.

    The compiler reports a disagreement and never resolves it. values and memory_ids are
    aligned: each value is paired with the highest-ranked candidate that asserts it, whether or
    not the budget included that memory; render() marks the ones it did not.
    """

    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 2 or len(self.values) != len(self.memory_ids):
            raise ValidationError("a conflict pairs at least two values with one memory each")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProvisionalActor:
    """One recognized person in the compiled evidence whom no visible assertion names.

    A stranger in the room is not a stranger missing from context. This carries the identity
    and the included memories that observed them, and nothing else: there is no name to carry,
    which is the whole point. It is not a hit and occupies no item slot, but its rendered line
    is charged against ContextBudget.max_chars like every other line the compiler grounds.
    """

    identity_id: str
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if not self.memory_ids:
            raise ValidationError("a provisional actor names at least one observing memory")
        object.__setattr__(
            self,
            "memory_ids",
            tuple(_text(value, "memory id") for value in self.memory_ids),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedActor:
    """One person a currently visible naming assertion names, reached through evidence other
    than the assertion itself.

    A memory carries an identity edge two ways -- its own semantic assertion may be bound to
    an identity, or its media asset may have recognized one -- and either can point to
    somebody already named without that naming assertion ranking into the bundle. This
    reports the name anyway. memory_ids are the included evidence memories whose identity
    edge resolved here, never the naming assertion's own memory: that memory renders as its
    own actors hit when it is itself included, so reporting it again here would list the same
    person twice. naming_assertion_id is that assertion's memory id, kept for provenance
    even though it may not be in the bundle. Like a ProvisionalActor, its rendered line is
    charged against ContextBudget.max_chars.
    """

    identity_id: str
    name: str
    memory_ids: tuple[str, ...]
    naming_assertion_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        if not self.memory_ids:
            raise ValidationError("a named actor names at least one observing memory")
        object.__setattr__(
            self,
            "memory_ids",
            tuple(_text(value, "memory id") for value in self.memory_ids),
        )
        object.__setattr__(
            self,
            "naming_assertion_id",
            _optional_text(self.naming_assertion_id, "naming_assertion_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextUnknown:
    """One thing the request implied a bundle might contain that this bundle does not.

    Every unknown is a deterministic statement about the compilation itself -- a scope that
    matched nothing, a requested type nothing was included for, evidence the budget could not
    buy, a modality this composition cannot search, a stage a deadline skipped. The compiler
    calls no model to produce one, so an unknown is never a guess about the world.
    """

    kind: ContextUnknownKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ContextUnknownKind):
            raise ValidationError("unknown kind is invalid")
        object.__setattr__(self, "detail", _text(self.detail, "unknown detail"))


@dataclass(frozen=True, slots=True)
class AffectCue(SearchHit):
    """One affect hit in a compiled bundle, carrying the evidence the cue hangs on.

    An affect entry asserts something about how somebody felt, so a reader has to be able to
    tell a model's inference from a person's own statement. context.basis, confidence,
    cue_modality, valence, and arousal already carry that, and render() prints all five
    on the line rather than leaving them to a caller who may never look.

    The observations the cue cites are its own context.evidence_ids, which every transport
    already carries. event_ids are the active EVENT records formed from at least one of
    those same observations: **co-occurrence inside one capture, never an attributed cause**.
    Nothing here says the event caused the feeling, or even that the cue is about the event;
    both were derived from the same thing somebody observed. Only the IDs are carried, so the
    hop grounds nothing and buys no item slot: no event is fetched, get() reads an event's text
    when a caller decides it wants it, and an event the ranking already bought appears in
    episodes on its own merit. The IDs the line does print are charged like any other rendered
    characters, so ContextBundle.chars still bounds render(). At most eight events are
    carried, so one cue's hop is bounded however many events one capture formed.
    """

    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # slots=True rebuilds the class, so the zero-argument super() cell would name the
        # pre-rebuild class. The base validation is called explicitly for that reason.
        SearchHit.__post_init__(self)
        object.__setattr__(
            self,
            "event_ids",
            tuple(dict.fromkeys(_text(value, "affect cue event_ids") for value in self.event_ids)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TextSpanPiece:
    """One exact source range in a contextual text embedding input."""

    role: Literal["context", "body"]
    start_codepoint: int
    end_codepoint: int
    source_text: str
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"context", "body"}:
            raise ValidationError("text span role must be context or body")
        if (
            isinstance(self.start_codepoint, bool)
            or not isinstance(self.start_codepoint, int)
            or isinstance(self.end_codepoint, bool)
            or not isinstance(self.end_codepoint, int)
            or self.start_codepoint < 0
            or self.end_codepoint <= self.start_codepoint
        ):
            raise ValidationError("text span offsets must be a non-empty half-open interval")
        if not isinstance(self.source_text, str) or not self.source_text:
            raise ValidationError("text span source_text must not be empty")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValidationError("text span sha256 must be 64 lowercase hexadecimal characters")
        if hashlib.sha256(self.source_text.encode("utf-8")).hexdigest() != self.sha256:
            raise ValidationError("text span source_text does not match its sha256")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextSpanSelector:
    """Digest-bound exact source pieces which produced one text embedding input."""

    parent_content_sha256: str
    embedding_input_sha256: str
    recipe_version: str
    pieces: tuple[TextSpanPiece, ...]

    def __post_init__(self) -> None:
        for name in ("parent_content_sha256", "embedding_input_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"text selector {name} must be a SHA-256 identifier")
        object.__setattr__(
            self,
            "recipe_version",
            _text(self.recipe_version, "text selector recipe_version"),
        )
        if not self.pieces or any(not isinstance(piece, TextSpanPiece) for piece in self.pieces):
            raise ValidationError("text selector pieces must contain TextSpanPiece values")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextExcerpt:
    """One compiler-selected partial rendering of an authoritative raw observation."""

    source_memory_id: str
    matched_index_id: str
    content: str
    selector: TextSpanSelector
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: RecordSemantics | None = None
    place_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_memory_id", _text(self.source_memory_id, "source_memory_id")
        )
        object.__setattr__(
            self, "matched_index_id", _text(self.matched_index_id, "matched_index_id")
        )
        if not isinstance(self.content, str) or not self.content:
            raise ValidationError("context excerpt content must not be empty")
        if not isinstance(self.selector, TextSpanSelector):
            raise ValidationError("context excerpt selector must be a TextSpanSelector")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, int | float)
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValidationError("context excerpt score must be between zero and one")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        if not isinstance(self.memory_type, MemoryCategory):
            raise ValidationError("context excerpt memory_type is invalid")
        if self.context is not None and not isinstance(self.context, RecordSemantics):
            raise ValidationError("context excerpt context is invalid")
        object.__setattr__(self, "place_id", _optional_text(self.place_id, "place_id"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextProjection:
    """Complete ordered text sections of a raw source whose native media is omitted.

    Asset parts preserve positions, but their references deliberately carry no readable path.
    Derived text keeps its role and cannot stand in for the omitted original evidence.
    """

    source_memory_id: str
    parts: tuple[ContentPart, ...]
    omitted_assets: tuple[AssetRef, ...]
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: RecordSemantics | None = None
    place_id: str | None = None
    coverage: Literal["text_only"] = "text_only"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_memory_id", _text(self.source_memory_id, "source_memory_id")
        )
        parts = tuple(self.parts)
        assets = tuple(self.omitted_assets)
        if not parts or any(not isinstance(part, ContentPart) for part in parts):
            raise ValidationError("context projection parts must contain ContentPart values")
        if not any(part.kind != "asset" for part in parts):
            raise ValidationError("context projection requires text sections")
        if not assets or any(not isinstance(asset, AssetRef) for asset in assets):
            raise ValidationError("context projection omitted_assets must contain AssetRef values")
        if any(asset.path is not None for asset in assets):
            raise ValidationError("omitted assets cannot carry readable paths")
        asset_ids = {asset.id for asset in assets}
        if len(asset_ids) != len(assets) or asset_ids != {
            part.value for part in parts if part.kind == "asset"
        }:
            raise ValidationError("context projection asset positions must match omitted assets")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "omitted_assets", assets)
        object.__setattr__(self, "score", unit_interval(self.score, "context projection score"))
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        if not isinstance(self.memory_type, MemoryCategory):
            raise ValidationError("context projection memory_type is invalid")
        if self.context is not None and (
            not isinstance(self.context, RecordSemantics)
            or self.context.kind is not RecordKind.OBSERVATION
            or self.context.evidence_ids
        ):
            raise ValidationError("context projections require a raw observation")
        object.__setattr__(self, "place_id", _optional_text(self.place_id, "place_id"))
        if self.coverage != "text_only":
            raise ValidationError("context projection coverage must be text_only")


@dataclass(frozen=True, slots=True)
class ContextSymbol:
    """One request-local alias and its presentation coverage."""

    symbol: str
    namespace: ContextSymbolNamespace
    stable_id: str
    coverage: ContextSymbolCoverage
    roles: tuple[ContextSymbolRole, ...]
    selector: TextSpanSelector | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, ContextSymbolNamespace):
            raise ValidationError("context symbol namespace is invalid")
        expected_prefix = "m" if self.namespace is ContextSymbolNamespace.MEMORY else "i"
        if not re.fullmatch(rf"{expected_prefix}[1-9][0-9]*", self.symbol):
            raise ValidationError("context symbol is invalid for its namespace")
        object.__setattr__(self, "stable_id", _text(self.stable_id, "context symbol stable_id"))
        if not isinstance(self.coverage, ContextSymbolCoverage):
            raise ValidationError("context symbol coverage is invalid")
        if not self.roles or any(not isinstance(role, ContextSymbolRole) for role in self.roles):
            raise ValidationError("context symbol roles are invalid")
        object.__setattr__(self, "roles", tuple(dict.fromkeys(self.roles)))
        _validate_context_symbol_coverage(self)


def _validate_context_symbol_coverage(symbol: ContextSymbol) -> None:
    if symbol.namespace is ContextSymbolNamespace.IDENTITY:
        if symbol.coverage is not ContextSymbolCoverage.REFERENCE_ONLY:
            raise ValidationError("identity symbols must have reference-only coverage")
        if symbol.selector is not None:
            raise ValidationError("identity symbols cannot carry a text selector")
    elif symbol.coverage is ContextSymbolCoverage.PARTIAL:
        if not isinstance(symbol.selector, TextSpanSelector):
            raise ValidationError("partial memory symbols require a text selector")
    elif symbol.selector is not None:
        raise ValidationError("only partial memory symbols can carry a text selector")


@dataclass(frozen=True, slots=True)
class ContextCitation:
    """A decoded memory citation with the exact presentation's delivery limits.

    Constructing this value does not prove that its source or selector was emitted by SenseMem;
    callers validate model aliases through ContextPresentation.resolve_citation().
    """

    memory_id: str
    coverage: ContextSymbolCoverage
    selector: TextSpanSelector | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _text(self.memory_id, "context citation memory_id"))
        if not isinstance(self.coverage, ContextSymbolCoverage) or self.coverage not in {
            ContextSymbolCoverage.FULL,
            ContextSymbolCoverage.PARTIAL,
            ContextSymbolCoverage.TEXT_ONLY,
        }:
            raise ValidationError("context citation coverage must be full or partial or text_only")
        if self.coverage is ContextSymbolCoverage.PARTIAL:
            if not isinstance(self.selector, TextSpanSelector):
                raise ValidationError("partial context citations require a text selector")
        elif self.selector is not None:
            raise ValidationError("only partial context citations can carry a text selector")


@dataclass(frozen=True, slots=True)
class ContextPresentation:
    """Frozen request-local text and typed aliases needed to interpret citations.

    Symbols have no meaning outside this object. The text and symbol table together are
    reversible; the compact text by itself is not an ID-preserving rewrite of arbitrary goals or
    stored content.
    """

    text: str
    symbols: tuple[ContextSymbol, ...]
    chars: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "context presentation text"))
        if self.chars != len(self.text):
            raise ValidationError("context presentation chars must equal the text length")
        if any(not isinstance(symbol, ContextSymbol) for symbol in self.symbols):
            raise ValidationError("context presentation symbols are invalid")
        keys = tuple((symbol.namespace, symbol.symbol) for symbol in self.symbols)
        stable = tuple((symbol.namespace, symbol.stable_id) for symbol in self.symbols)
        if len(set(keys)) != len(keys) or len(set(stable)) != len(stable):
            raise ValidationError("context presentation symbols must be unique")

    def resolve(self, symbol: str, *, namespace: ContextSymbolNamespace) -> str:
        """Decode one alias without asserting delivery, eligibility, or evidentiary support."""
        if not isinstance(namespace, ContextSymbolNamespace):
            raise ValidationError("context symbol namespace is invalid")
        for entry in self.symbols:
            if entry.namespace is namespace and entry.symbol == symbol:
                return entry.stable_id
        raise ValidationError("context symbol is unknown in this presentation")

    def resolve_citation(self, symbol: str) -> ContextCitation:
        """Decode an eligible memory citation, rejecting identities and reference-only IDs."""
        for entry in self.symbols:
            if entry.symbol != symbol:
                continue
            if entry.namespace is not ContextSymbolNamespace.MEMORY:
                raise ValidationError("identity symbols cannot be memory citations")
            if entry.coverage is ContextSymbolCoverage.REFERENCE_ONLY:
                raise ValidationError("reference-only memory symbols cannot be citations")
            return ContextCitation(
                memory_id=entry.stable_id,
                coverage=entry.coverage,
                selector=entry.selector,
            )
        raise ValidationError("context symbol is unknown in this presentation")


ContextProofScoring: TypeAlias = Literal["capture_groups", "independent_pair"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextProofNode:
    """One complete AND assessment, or a capture leaf, in a delivery certificate.

    Assessments come from SQLite, not from the flattened evidence_ids union. Confidence
    is the assessment's score; a leaf's capture_id identifies shared capture provenance.
    source_group_id retains the store's group across unselected OR alternatives.
    """

    memory_id: str
    confidence: float
    sources: tuple[str, ...] = ()
    capture_id: str | None = None
    source_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _text(self.memory_id, "proof memory_id"))
        object.__setattr__(self, "sources", _memory_ids(self.sources, "proof sources"))
        object.__setattr__(self, "confidence", unit_interval(self.confidence, "proof confidence"))
        object.__setattr__(self, "capture_id", _optional_text(self.capture_id, "proof capture_id"))
        object.__setattr__(
            self,
            "source_group_id",
            _optional_text(self.source_group_id, "proof source_group_id"),
        )
        if bool(self.sources) == (self.capture_id is not None) or self.confidence == 0.0:
            raise ValidationError(
                "a proof node needs positive confidence and either sources or a capture"
            )
        if (
            self.capture_id is not None
            and self.source_group_id is not None
            and self.source_group_id != self.capture_id
        ):
            raise ValidationError("a capture leaf's source_group_id must match its capture_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextProof:
    """An acyclic, complete witness for one included assertion, separate from its audit union."""

    anchor_id: str
    nodes: tuple[ContextProofNode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", _text(self.anchor_id, "proof anchor_id"))
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(node, ContextProofNode) for node in nodes):
            raise ValidationError("proof nodes must contain ContextProofNode values")
        by_id = {node.memory_id: node for node in nodes}
        if len(by_id) != len(nodes) or self.anchor_id not in by_id:
            raise ValidationError("proof nodes must be unique and contain the anchor")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(memory_id: str) -> None:
            if memory_id in visiting or memory_id not in by_id:
                raise ValidationError("a proof cannot contain a cycle or a missing AND member")
            if memory_id in visited:
                return
            visiting.add(memory_id)
            for source in by_id[memory_id].sources:
                visit(source)
            visiting.remove(memory_id)
            visited.add(memory_id)

        # Iteration in the producer is bounded; reject excessive user-supplied recursion here.
        if len(nodes) > 256:
            raise ValidationError("a proof cannot exceed 256 nodes")
        visit(self.anchor_id)
        if len(visited) != len(nodes):
            raise ValidationError("proof nodes must all be reachable from the anchor")
        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda node: node.memory_id)))

    @property
    def confidence(self) -> float:
        return min(node.confidence for node in self.nodes)

    @property
    def footprint(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            ("capture", node.capture_id)
            if node.capture_id is not None
            else ("node", node.memory_id)
            for node in self.nodes
            if node.memory_id != self.anchor_id or node.capture_id is not None
        )

    def structurally_independent_of(self, other: ContextProof) -> bool:
        """Compare source sets and complete footprints without applying evidence policy."""
        if self.anchor_id != other.anchor_id:
            return False
        left = next(node.sources for node in self.nodes if node.memory_id == self.anchor_id)
        right = next(node.sources for node in other.nodes if node.memory_id == other.anchor_id)
        return set(left) != set(right) and self.footprint.isdisjoint(other.footprint)

    def _experience_footprint(self, conditions: Container[str]) -> frozenset[tuple[str, str]]:
        return experience_footprint(
            self.anchor_id,
            {node.memory_id: (node.sources, node.capture_id) for node in self.nodes},
            conditions,
        )

    def render(self, *, symbols: Sequence[ContextSymbol] = ()) -> str:
        def identifier(memory_id: str) -> str:
            return (
                _symbol(memory_id, ContextSymbolNamespace.MEMORY, symbols) if symbols else memory_id
            )

        return context_prompts.proof_line(
            identifier(self.anchor_id),
            tuple(
                (
                    identifier(node.memory_id),
                    tuple(identifier(source) for source in node.sources),
                    None if node.capture_id is None else identifier(node.capture_id),
                    node.confidence,
                    None if node.source_group_id is None else identifier(node.source_group_id),
                )
                for node in self.nodes
            ),
        )


def _symbol(
    stable_id: str,
    namespace: ContextSymbolNamespace,
    symbols: Sequence[ContextSymbol],
) -> str:
    for entry in symbols:
        if entry.namespace is namespace and entry.stable_id == stable_id:
            return entry.symbol
    raise RuntimeError("compact context formatter omitted a structural identifier")
