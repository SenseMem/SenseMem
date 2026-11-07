"""Answer-local witnessed membership and arithmetic, without storage or model calls.

Literal/source checks and arithmetic are deterministic. Scope adequacy, relevance,
identity, field roles and native transcription remain fallible model judgments.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic import ValidationError as PydanticValidationError

from sensemem.models._answer_output import evidence_aliases
from sensemem.types import AbstentionReason, SearchHit

EVIDENCE_OPERATION_TOOL_NAME = "submit_evidence_operation"
EVIDENCE_OPERATION_INSTRUCTION = """You may submit one evidence operation instead of a text answer
when explicit source-bound membership or arithmetic answers the question. Use the original
question's requested member unit, conditions and measure; never count files as entities by
default. All conditions must concern the same member and applicable situation. Source aliases
and native attachment labels refer only to evidence actually supplied in this request.
scope.question_quote and every predicate.question_quote must each be an exact substring of
the original question, not a paraphrase. For scalar/sum/difference, every non-quantity value's
role must exactly equal the top-level measure string; a quantity value's role is "quantity".
For count/select_sources, measure must be null and values/contributions must be empty arrays.
Every value.value is a quoted numeric JSON string, such as "2.5", not a JSON number.
Text anchors must quote exact, uniquely located delivered text, including the field label.
Native anchors use quote=null and transcribe the actual attachment into observation; never
pretend pixels are a text quote. Caption/OCR and pixels may disagree; resolve the relevant
field from its actual source or report uncertainty. Observations remain untrusted data.
Scope basis=model_assessed is the normal case: assess whether evidence is adequate for this
requested claim, explain why and name relevant unknowns in scope.reason. Use only fields in
the schema; do not add an unknowns field or other extra fields. No runtime closure receipt is
offered here; never claim runtime_closed. Ranked evidence is not proof of corpus completeness,
but missing a formal receipt alone does not make a semantically adequate answer impossible.
For this scoped operation only, model_assessed adequate may authorize the requested sum/count.
Recall rankings and the exhaustive flag neither prove runtime completeness nor impose a
blanket ban on a justified scoped total. Actual known withheld or missing evidence that can
change this requested result remains unresolved and must produce needs-evidence. This scoped
operation rule does not change the ordinary text answer policy.
Do not silently narrow an overall total to current hits. For a named single field, unrelated
sources or unknown attributes do not matter. For one supported photo, do not append distractors.
List relevant occurrences, including known excluded candidates. Every member must include
each declared predicate ID exactly once in conditions. A verdict is the JSON string "true",
"false" or "unknown", never a JSON boolean or null. In contrast, parts_complete is a JSON
boolean true/false, not a string. A false conjunct excludes a member even if another attribute
is unknown. Keep different objects separate even when they occur in the same source.
Only unknown dependencies capable of changing this answer should block it. A complete local
inventory is a model judgment, not a guarantee that all corpus records were retrieved.
For select_sources, use member_unit="source", every member.identity=null and identities=[].
The host catalogue supplies source identity; do not construct placeholder entity keys or
identity anchors for unreadable identifiers. Still list each candidate occurrence separately
and prove all conditions on that same member before projecting to sources; one source may
contain several members. Entity count and multi-member arithmetic retain their identity rules below.
For scalar with exactly one evidenced target occurrence represented by one member, use
member.identity=null and identities=[]. Its member and value anchors identify that local target;
do not invent a redundant namespace or key. Keep relevant excluded or unresolved candidates.
If a scalar target has repeated recorded occurrences, keep them separate and use supported
identity keys or same/different relations when needed to reconcile them; do not collapse
possible targets or change the requested member unit merely to avoid identity uncertainty.
Entity identity requires a namespaced source-bound key or an explicit same/different relation.
Both identity.namespace and identity.value must occur literally in that identity's source
anchor: copy them from its text quote or native transcription; do not invent a namespace.
Absence of a same relation does not prove different; same title/amount/time alone is not identity.
For each included entity choose one measure basis: whole, a complete disjoint partition, or
explicit integer quantity times per_unit. Never add TOTAL and its parts, count a duplicate
notice as a payment, or assign an order total to one item. A quantity and per-unit value must
belong to the same member. No implicit unit/currency conversion, guessed sign, date rewrite,
or arbitrary formula. Value role and its literal label must be bound to the same source.
For difference, contributions are ordered: first contribution minus second. The member list
order does not set subtraction order. Different key namespaces may identify the same real
entity; only compare unequal keys automatically within one namespace, otherwise supply an
explicit same/different relation or leave identity unknown.
Use empty arrays for unused lists and explicit null for every unused required nullable field,
including quote/attachment/observation, member identity, value unit/part_key and measure; do not
omit those fields. An honestly unresolved requested fact is needs-evidence,
not zero/false. Keep a relevant unresolved member, with an unknown condition when appropriate;
omit an unavailable numeric fact and its contribution rather than inventing a value. Mark an
unresolved scope.status as uncertain/incomplete and inventory.status as partial/unknown, as
applicable. If the question
explicitly requests this function and a supported operation applies, submit even when a
required fact is unresolved: the executor can return needs-evidence. Do not replace that
submission with ordinary UNKNOWN. Otherwise an ordinary no-call answer remains valid.
Submit the same operation envelope for unresolved facts: needs_evidence is an executor
outcome, not an operation name or an extra status/answer/missing field in the arguments.
A resolved empty set may count zero. If this operation cannot express the question, give the
ordinary text answer instead of manufacturing a witness.

Synthetic protocol example only: all facts below are invented. Never copy its source, values
or answer into a real proposal; use only the actual question and supplied evidence.
Assume the question is: "What is the TOTAL on the sample invoice?"
Assume the only supplied source E1 has text: "SAMPLE INVOICE\nTOTAL 7.25 EUR"
The following complete arguments submit that hypothetical scalar:
<synthetic_scalar_arguments>
{"operation":"scalar","member_unit":"source","quantifier":"one","measure":"total",
"format":"with_unit","scope":{"kind":"single_source","basis":"model_assessed",
"status":"adequate","question_quote":"TOTAL on the sample invoice","evidence_labels":["E1"],
"reason":"The named invoice and its TOTAL are supplied."},"predicates":[],
"inventory":{"status":"complete","anchors":[{"evidence_label":"E1","quote":"TOTAL 7.25 EUR",
"attachment":null,"observation":null}]},"members":[{"id":"invoice",
"anchor":{"evidence_label":"E1","quote":"TOTAL 7.25 EUR","attachment":null,"observation":null},
"conditions":[],"identity":null}],"identities":[],"values":[{"id":"total","member_id":"invoice",
"role":"total","label":"TOTAL","value":"7.25","unit":"EUR","extent":"whole","part_key":null,
"anchor":{"evidence_label":"E1","quote":"TOTAL 7.25 EUR","attachment":null,"observation":null}}],
"contributions":[{"member_id":"invoice","basis":"whole","value_ids":["total"],"parts_complete":true}]}
</synthetic_scalar_arguments>

Another invented example: assume the question is:
"What is the exact combined mass of these two vessels?"
E1 text is "VESSEL C1 MASS 2 kg"; E2 text is "VESSEL C2 MASS UNREADABLE".
The unresolved vessel remains a member; only the known mass has a value and contribution.
This complete sum proposal produces needs-evidence, not a claimed total of 2 kg:
<synthetic_unknown_arguments>
{"operation":"sum","member_unit":"entity","quantifier":"exact","measure":"mass",
"format":"with_unit","scope":{"kind":"explicit_source_set","basis":"model_assessed",
"status":"incomplete","question_quote":"combined mass of these two vessels",
"evidence_labels":["E1","E2"],"reason":"Both named vessels are supplied, but C2 mass is unreadable."},
"predicates":[],"inventory":{"status":"complete","anchors":[
{"evidence_label":"E1","quote":"VESSEL C1 MASS 2 kg","attachment":null,"observation":null},
{"evidence_label":"E2","quote":"VESSEL C2 MASS UNREADABLE","attachment":null,"observation":null}]},
"members":[{"id":"c1","anchor":{"evidence_label":"E1","quote":"VESSEL C1 MASS 2 kg",
"attachment":null,"observation":null},"conditions":[],"identity":{"namespace":"VESSEL","value":"C1",
"anchor":{"evidence_label":"E1","quote":"VESSEL C1 MASS 2 kg","attachment":null,"observation":null}}},
{"id":"c2","anchor":{"evidence_label":"E2","quote":"VESSEL C2 MASS UNREADABLE",
"attachment":null,"observation":null},"conditions":[],"identity":{"namespace":"VESSEL","value":"C2",
"anchor":{"evidence_label":"E2","quote":"VESSEL C2 MASS UNREADABLE","attachment":null,"observation":null}}}],
"identities":[],"values":[{"id":"mass_c1","member_id":"c1","role":"mass","label":"MASS",
"value":"2","unit":"kg","extent":"whole","part_key":null,"anchor":{"evidence_label":"E1",
"quote":"VESSEL C1 MASS 2 kg","attachment":null,"observation":null}}],
"contributions":[{"member_id":"c1","basis":"whole","value_ids":["mass_c1"],"parts_complete":true}]}
</synthetic_unknown_arguments>

Another invented example: assume the question is:
"Which supplied source shows the same toy car that is red and parked?"
E1 text is "Toy car A is red and parked."
E2 text is "Toy car B is red and moving. Toy car C is blue and parked."
Host source IDs are sample-a and sample-b.
Only sample-a matches: the red car and the parked car in E2 are different objects. Each
occurrence below accounts for both predicates, using JSON string verdicts:
<synthetic_and_arguments>
{"operation":"select_sources","member_unit":"source","quantifier":"exact","measure":null,
"format":"source_ids","scope":{"kind":"explicit_source_set","basis":"model_assessed",
"status":"adequate","question_quote":"same toy car that is red and parked",
"evidence_labels":["E1","E2"],"reason":"Both supplied scene inventories identify each toy car and both attributes."},
"predicates":[{"id":"red","question_quote":"red"},{"id":"parked","question_quote":"parked"}],
"inventory":{"status":"complete","anchors":[
{"evidence_label":"E1","quote":"Toy car A is red and parked.","attachment":null,"observation":null},
{"evidence_label":"E2","quote":"Toy car B is red and moving. Toy car C is blue and parked.",
"attachment":null,"observation":null}]},"members":[
{"id":"car_a","anchor":{"evidence_label":"E1","quote":"Toy car A is red and parked.",
"attachment":null,"observation":null},"conditions":[
{"predicate_id":"red","verdict":"true","anchors":[{"evidence_label":"E1",
"quote":"Toy car A is red and parked.","attachment":null,"observation":null}]},
{"predicate_id":"parked","verdict":"true","anchors":[{"evidence_label":"E1",
"quote":"Toy car A is red and parked.","attachment":null,"observation":null}]}],"identity":null},
{"id":"car_b","anchor":{"evidence_label":"E2","quote":"Toy car B is red and moving.",
"attachment":null,"observation":null},"conditions":[
{"predicate_id":"red","verdict":"true","anchors":[{"evidence_label":"E2",
"quote":"Toy car B is red and moving.","attachment":null,"observation":null}]},
{"predicate_id":"parked","verdict":"false","anchors":[{"evidence_label":"E2",
"quote":"Toy car B is red and moving.","attachment":null,"observation":null}]}],"identity":null},
{"id":"car_c","anchor":{"evidence_label":"E2","quote":"Toy car C is blue and parked.",
"attachment":null,"observation":null},"conditions":[
{"predicate_id":"red","verdict":"false","anchors":[{"evidence_label":"E2",
"quote":"Toy car C is blue and parked.","attachment":null,"observation":null}]},
{"predicate_id":"parked","verdict":"true","anchors":[{"evidence_label":"E2",
"quote":"Toy car C is blue and parked.","attachment":null,"observation":null}]}],"identity":null}],
"identities":[],"values":[],"contributions":[]}
</synthetic_and_arguments>
"""

_Label = Annotated[str, StringConstraints(pattern=r"^E[1-9][0-9]*$")]
_Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
_Text = Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
_Number = Annotated[
    str,
    StringConstraints(
        pattern=r"^[+-]?(?:[0-9]{1,64}|[0-9]{1,3}(?:,[0-9]{3}){1,21})(?:\.[0-9]{1,32})?$"
    ),
]


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    """A bounded schema diagnostic containing only allowlisted codes and field names."""

    error_type: str
    location: tuple[str, ...]


class InvalidEvidenceOperation(ValueError):
    """A structurally invalid or incorrectly bound proposal, with a safe static message."""

    def __init__(
        self,
        message: str,
        *,
        category: Literal["schema", "invalid_evidence"] = "invalid_evidence",
        schema_issues: tuple[SchemaIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.schema_issues = schema_issues


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceAnchor(_Value):
    evidence_label: _Label
    quote: _Text | None
    attachment: _Text | None
    observation: _Text | None

    @model_validator(mode="after")
    def _channel(self) -> Self:
        if self.quote is not None:
            if self.attachment is not None or self.observation is not None:
                raise ValueError("a text anchor has no native fields")
        elif self.attachment is None or self.observation is None:
            raise ValueError("a native anchor needs an attachment and observation")
        return self


class ScopeAssessment(_Value):
    kind: Literal["single_source", "explicit_source_set", "open_corpus"]
    basis: Literal["model_assessed", "runtime_closed"]
    status: Literal["adequate", "uncertain", "incomplete"]
    question_quote: _Text
    evidence_labels: Annotated[tuple[_Label, ...], Field(max_length=32)]
    reason: _Text


class Predicate(_Value):
    id: _Id
    question_quote: _Text


class PredicateVerdict(_Value):
    predicate_id: _Id
    verdict: Literal["true", "false", "unknown"]
    anchors: Annotated[tuple[EvidenceAnchor, ...], Field(max_length=8)]


class IdentityKey(_Value):
    namespace: _Text
    value: _Text
    anchor: EvidenceAnchor


class OperationMember(_Value):
    id: _Id
    anchor: EvidenceAnchor
    conditions: Annotated[tuple[PredicateVerdict, ...], Field(max_length=8)]
    identity: IdentityKey | None


class IdentityRelation(_Value):
    left: _Id
    right: _Id
    relation: Literal["same", "different", "unknown"]
    anchors: Annotated[tuple[EvidenceAnchor, ...], Field(max_length=8)]


class MemberInventory(_Value):
    status: Literal["complete", "partial", "unknown"]
    anchors: Annotated[tuple[EvidenceAnchor, ...], Field(max_length=32)]


class OperationValue(_Value):
    id: _Id
    member_id: _Id
    role: _Text
    label: _Text
    value: _Number
    unit: _Text | None
    extent: Literal["whole", "part", "quantity", "per_unit"]
    part_key: _Text | None
    anchor: EvidenceAnchor


class MemberContribution(_Value):
    member_id: _Id
    basis: Literal["whole", "partition", "quantity_per_unit"]
    value_ids: Annotated[tuple[_Id, ...], Field(min_length=1, max_length=32)]
    parts_complete: bool


class EvidenceOperation(_Value):
    operation: Literal["select_sources", "count", "scalar", "sum", "difference"]
    member_unit: Literal["source", "entity"]
    quantifier: Literal["exact", "confirmed_subset", "one"]
    measure: _Text | None
    format: Literal["source_ids", "number", "with_unit"]
    scope: ScopeAssessment
    predicates: Annotated[tuple[Predicate, ...], Field(max_length=8)]
    inventory: MemberInventory
    members: Annotated[tuple[OperationMember, ...], Field(max_length=32)]
    identities: Annotated[tuple[IdentityRelation, ...], Field(max_length=64)]
    values: Annotated[tuple[OperationValue, ...], Field(max_length=64)]
    contributions: Annotated[tuple[MemberContribution, ...], Field(max_length=32)]


EVIDENCE_OPERATION_PARAMETERS: dict[str, object] = EvidenceOperation.model_json_schema()
_SCHEMA_ERROR_TYPES = frozenset(
    {
        "json_invalid",
        "json_type",
        "model_type",
        "dict_type",
        "tuple_type",
        "string_type",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "bool_type",
        "literal_error",
        "none_required",
        "missing",
        "extra_forbidden",
        "too_long",
        "too_short",
        "value_error",
    }
)
_SCHEMA_FIELD_NAMES = frozenset(
    name
    for model in (
        EvidenceAnchor,
        ScopeAssessment,
        Predicate,
        PredicateVerdict,
        IdentityKey,
        OperationMember,
        IdentityRelation,
        MemberInventory,
        OperationValue,
        MemberContribution,
        EvidenceOperation,
    )
    for name in model.model_fields
)


@dataclass(frozen=True, slots=True)
class EvidenceCatalogue:
    """Actual fitted evidence and successfully supplied native labels, owned by the adapter."""

    hits: tuple[SearchHit, ...]
    visible_media: Mapping[str, frozenset[str]]
    source_metadata_key: str


@dataclass(frozen=True, slots=True)
class EvidenceOperationOutcome:
    status: Literal["executed", "needs_evidence"]
    answer: str | None
    hits: tuple[SearchHit, ...]
    reason: AbstentionReason | None
    missing: tuple[str, ...]
    scope_assurance: Literal["model_assessed", "runtime_closed"]


def make_evidence_catalogue(
    hits: Sequence[SearchHit],
    *,
    visible_media: Mapping[str, frozenset[str]],
    source_metadata_key: str = "source_id",
) -> EvidenceCatalogue:
    """Bind an adapter's post-fitting source/attachment catalogue without exposing asset IDs."""
    by_id: dict[str, SearchHit] = {}
    for hit in hits:
        by_id.setdefault(hit.id, hit)
    unique = tuple(by_id.values())
    labels = set(evidence_aliases(unique).values())
    if not source_metadata_key.strip() or set(visible_media) - labels:
        raise InvalidEvidenceOperation("evidence catalogue has invalid source bindings")
    for attachments in visible_media.values():
        for attachment in attachments:
            if not attachment.strip():
                raise InvalidEvidenceOperation("evidence catalogue has invalid attachment bindings")
    return EvidenceCatalogue(
        unique,
        MappingProxyType({label: frozenset(items) for label, items in visible_media.items()}),
        source_metadata_key,
    )


def _invalid(message: str) -> InvalidEvidenceOperation:
    return InvalidEvidenceOperation(message)


def _anchor_text(
    anchor: EvidenceAnchor, catalogue: EvidenceCatalogue, visible: Mapping[str, SearchHit]
) -> str:
    hit = visible.get(anchor.evidence_label)
    if hit is None:
        raise _invalid("operation references evidence outside the fitted context")
    if anchor.quote is not None:
        start = hit.content.find(anchor.quote)
        if start < 0 or hit.content.find(anchor.quote, start + 1) >= 0:
            raise _invalid("operation requires a uniquely located delivered text quote")
        return anchor.quote
    if anchor.attachment not in catalogue.visible_media.get(anchor.evidence_label, frozenset()):
        raise _invalid("operation native anchor was not supplied for its source")
    assert anchor.observation is not None
    return anchor.observation


def _token(value: str, text: str) -> bool:
    return (
        re.search(r"(?<![\w.,+\-\u2212])" + re.escape(value) + r"(?![\w]|[.,][0-9])", text)
        is not None
    )


def _number(
    fact: OperationValue, catalogue: EvidenceCatalogue, visible: Mapping[str, SearchHit]
) -> tuple[Decimal, tuple[str, str, int]]:
    text = _anchor_text(fact.anchor, catalogue, visible)
    if fact.label not in text:
        raise _invalid("operation value label is absent from its anchor")
    # The original neighbours prevent a shortened quote from removing a sign or decimal digit.
    whole = text
    if fact.anchor.quote is not None:
        whole = visible[fact.anchor.evidence_label].content
    unit = "" if fact.unit is None else fact.unit
    pattern = (
        r"(?<![\w.,+\-\u2212])(?:"
        + re.escape(unit)
        + r"\s*)?(?<![0-9.,+\-\u2212])(?P<value>"
        + re.escape(fact.value)
        + r")(?![0-9]|[.,][0-9])(?:\s*"
        + re.escape(unit)
        + r")?(?!\w|[.,][0-9])"
    )
    matches = tuple(re.finditer(pattern, text))
    if len(matches) != 1:
        raise _invalid("operation requires one complete numeric literal")
    match = matches[0]
    offset = 0 if fact.anchor.quote is None else whole.index(text)
    begin, end = match.span("value")
    if (offset + begin, offset + end) not in {
        item.span("value") for item in re.finditer(pattern, whole)
    }:
        raise _invalid("operation value omits part of its source number")
    if fact.extent == "quantity":
        value = Decimal(fact.value.replace(",", ""))
        if fact.unit is not None or value < 0 or value != value.to_integral_value():
            raise _invalid("operation quantity must be an explicit nonnegative integer")
    else:
        if fact.unit is None:
            raise _invalid("operation numeric measure requires a literal unit")
        before, after = text[:begin].rstrip(), text[end:].lstrip()
        if not (
            (
                after.startswith(unit)
                and (
                    len(after) == len(unit)
                    or not (after[len(unit)].isalnum() or after[len(unit)] == "_")
                )
            )
            or (
                before.endswith(unit)
                and (
                    len(before) == len(unit)
                    or not (before[-len(unit) - 1].isalnum() or before[-len(unit) - 1] == "_")
                )
            )
        ):
            raise _invalid("operation unit is not attached to its value")
        value = Decimal(fact.value.replace(",", ""))
    if (fact.extent == "part") != (fact.part_key is not None):
        raise _invalid("operation part identity does not match its numeric extent")
    location = (
        (fact.anchor.evidence_label, "", offset + begin)
        if fact.anchor.quote is not None
        else (fact.anchor.attachment or "", text, begin)
    )
    return value, location


def _source_id(hit: SearchHit, catalogue: EvidenceCatalogue) -> str:
    source = hit.metadata.get(catalogue.source_metadata_key)
    if not isinstance(source, str) or not source.strip():
        raise _invalid("operation requires host-selected application source metadata")
    return source


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if Decimal(text) == 0 else text


def execute_evidence_operation(
    arguments: str, catalogue: EvidenceCatalogue, *, question: str
) -> EvidenceOperationOutcome:
    """Validate one native proposal and execute only its supported requested claim."""
    try:
        proposal = EvidenceOperation.model_validate_json(arguments)
    except PydanticValidationError as error:
        issues = tuple(
            SchemaIssue(
                issue["type"] if issue["type"] in _SCHEMA_ERROR_TYPES else "other",
                tuple(
                    "item"
                    if isinstance(segment, int)
                    else segment
                    if segment in _SCHEMA_FIELD_NAMES
                    and (issue["type"] != "extra_forbidden" or index < len(issue["loc"]) - 1)
                    else "other"
                    for index, segment in enumerate(issue["loc"])
                )
                or ("root",),
            )
            for issue in error.errors(
                include_input=False, include_context=False, include_url=False
            )[:16]
        )
    else:
        return _execute(proposal, catalogue, question=question)
    # Raise outside the handler so the rejected provider payload is not retained as
    # the new exception's implicit Pydantic exception context.
    raise InvalidEvidenceOperation(
        "native evidence operation arguments were invalid",
        category="schema",
        schema_issues=issues,
    ) from None


def _execute(  # noqa: C901 - evaluate the requested operation's dependency footprint together
    proposal: EvidenceOperation, catalogue: EvidenceCatalogue, *, question: str
) -> EvidenceOperationOutcome:
    bound = _bind(proposal, catalogue, question=question)
    roots, distinct = _identity_groups(proposal, bound, catalogue)
    grouped: dict[str, list[str]] = {}
    for identifier, root in roots.items():
        grouped.setdefault(root, []).append(identifier)
    included: list[str] = []
    unknown: list[str] = []
    for root, members in grouped.items():
        states = {bound.states[member] for member in members}
        if "true" in states:
            included.append(root)
        elif "unknown" in states:
            unknown.append(root)
    missing: list[str] = []
    if proposal.scope.status != "adequate":
        missing.append("scope_insufficient")
    if (
        proposal.quantifier == "exact" or (proposal.quantifier == "one" and not included)
    ) and proposal.operation != "scalar":
        if proposal.inventory.status != "complete":
            missing.append("inventory_incomplete")
        if unknown:
            missing.append("membership_unknown")
    chosen = included[:1] if proposal.quantifier == "one" else included
    if proposal.operation == "difference" and len(proposal.contributions) == 2:
        ordered = [roots[item.member_id] for item in proposal.contributions]
        if set(ordered) == set(chosen):
            chosen = ordered
    answer: str | None
    if proposal.operation in {"select_sources", "count"}:
        answer = _membership_answer(proposal, chosen, grouped, bound, catalogue)
        if proposal.operation == "count" and _identity_unknown(chosen, distinct):
            missing.append("identity_unknown")
    else:
        answer, numeric_missing, totals = _numeric_answer(proposal, chosen, roots, bound)
        missing.extend(numeric_missing)
        if (
            proposal.operation == "sum"
            and any(totals.values())
            and _identity_unknown(chosen, distinct)
        ):
            missing.append("identity_unknown")
        if proposal.operation == "scalar" and unknown:
            missing.append("membership_unknown")
    if not catalogue.hits:
        missing.append("no_evidence")
    hits = tuple(
        hit for label, hit in bound.visible.items() if label in proposal.scope.evidence_labels
    )
    if missing:
        return EvidenceOperationOutcome(
            "needs_evidence",
            None
            if not chosen
            or answer is None
            or set(missing)
            & {"identity_unknown", "conflicting_values", "unit_conflict", "target_ambiguous"}
            else _partial_answer(proposal, answer),
            hits,
            AbstentionReason.NO_EVIDENCE
            if not catalogue.hits
            else AbstentionReason.INSUFFICIENT_EVIDENCE,
            tuple(dict.fromkeys(missing)),
            "model_assessed",
        )
    if answer is None:
        raise _invalid("operation did not produce a supported answer")
    return EvidenceOperationOutcome("executed", answer, hits, None, (), "model_assessed")


@dataclass(frozen=True, slots=True)
class _BoundOperation:
    visible: Mapping[str, SearchHit]
    members: Mapping[str, OperationMember]
    facts: Mapping[str, OperationValue]
    numbers: Mapping[str, Decimal]
    locations: Mapping[str, tuple[str, str, int]]
    states: Mapping[str, str]


def _unique(values: Sequence[str], message: str) -> None:
    if len(set(values)) != len(values):
        raise _invalid(message)


def _bind(  # noqa: C901 - validate one model-owned envelope once before typed execution
    proposal: EvidenceOperation, catalogue: EvidenceCatalogue, *, question: str
) -> _BoundOperation:
    aliases = evidence_aliases(catalogue.hits)
    visible = {aliases[hit.id]: hit for hit in catalogue.hits}
    scope = proposal.scope
    if scope.basis != "model_assessed":
        raise _invalid("operation cannot manufacture a runtime scope receipt")
    if scope.question_quote not in question:
        raise _invalid("operation scope is not anchored to the question")
    _unique(scope.evidence_labels, "operation repeats a scope source")
    if set(scope.evidence_labels) - visible.keys():
        raise _invalid("operation scope references evidence outside the fitted context")
    if scope.kind == "single_source" and len(scope.evidence_labels) != 1:
        raise _invalid("operation single-source scope requires one fitted source")
    _unique([item.id for item in proposal.predicates], "operation repeats a predicate")
    predicates = {item.id for item in proposal.predicates}
    if any(item.question_quote not in question for item in proposal.predicates):
        raise _invalid("operation condition is not anchored to the question")
    _unique([item.id for item in proposal.members], "operation repeats a member")
    members = {item.id: item for item in proposal.members}
    states: dict[str, str] = {}
    covered: set[str] = set()
    for anchor in proposal.inventory.anchors:
        _anchor_text(anchor, catalogue, visible)
        covered.add(anchor.evidence_label)
    for member in proposal.members:
        label = member.anchor.evidence_label
        _anchor_text(member.anchor, catalogue, visible)
        covered.add(label)
        if label not in scope.evidence_labels:
            raise _invalid("operation member lies outside its requested source scope")
        _unique(
            [item.predicate_id for item in member.conditions],
            "operation repeats a member condition",
        )
        if {item.predicate_id for item in member.conditions} != predicates:
            raise _invalid("operation member does not account for every condition")
        for condition in member.conditions:
            if condition.verdict != "unknown" and not condition.anchors:
                raise _invalid("operation condition requires source-bound support")
            for anchor in condition.anchors:
                _anchor_text(anchor, catalogue, visible)
                if anchor.evidence_label != label:
                    raise _invalid("operation conditions mix different occurrences")
        verdicts = {item.verdict for item in member.conditions}
        states[member.id] = (
            "false" if "false" in verdicts else "unknown" if "unknown" in verdicts else "true"
        )
        if member.identity is not None:
            key = member.identity
            text = _anchor_text(key.anchor, catalogue, visible)
            if key.anchor.evidence_label != label or not all(
                _token(value, text) for value in (key.namespace, key.value)
            ):
                raise _invalid("operation identity key is not supported by its member")
    if covered - set(scope.evidence_labels):
        raise _invalid("operation inventory lies outside its source scope")
    if proposal.inventory.status == "complete" and set(scope.evidence_labels) - covered:
        raise _invalid("operation inventory omits a declared scope source")
    _unique([item.id for item in proposal.values], "operation repeats a numeric fact")
    facts = {item.id: item for item in proposal.values}
    numbers: dict[str, Decimal] = {}
    locations: dict[str, tuple[str, str, int]] = {}
    for fact in proposal.values:
        fact_member = members.get(fact.member_id)
        if fact_member is None or fact.anchor.evidence_label != fact_member.anchor.evidence_label:
            raise _invalid("operation value is not bound to its member source")
        if fact.role != ("quantity" if fact.extent == "quantity" else proposal.measure):
            raise _invalid("operation value uses a different requested measure")
        numbers[fact.id], locations[fact.id] = _number(fact, catalogue, visible)
        if fact.part_key is not None and not _token(
            fact.part_key, _anchor_text(fact.anchor, catalogue, visible)
        ):
            raise _invalid("operation partition key is not supported by its source")
    for contribution in proposal.contributions:
        if contribution.member_id not in members or set(contribution.value_ids) - facts.keys():
            raise _invalid("operation contribution references an unknown member or value")
        _unique(contribution.value_ids, "operation contribution repeats a value")
    selection = proposal.operation == "select_sources"
    numeric = proposal.operation in {"scalar", "sum", "difference"}
    if selection != (proposal.format == "source_ids"):
        raise _invalid("operation format does not match its requested projection")
    if numeric != (proposal.measure is not None):
        raise _invalid("operation measure does not match its requested operation")
    if not numeric and (proposal.values or proposal.contributions):
        raise _invalid("operation membership projection cannot execute numeric contributions")
    return _BoundOperation(visible, members, facts, numbers, locations, states)


def _identity_groups(  # noqa: C901 - the same/different graph is one identity boundary
    proposal: EvidenceOperation, bound: _BoundOperation, catalogue: EvidenceCatalogue
) -> tuple[dict[str, str], set[frozenset[str]]]:
    parents = {member: member for member in bound.members}

    def root(member: str) -> str:
        while parents[member] != member:
            member = parents[member]
        return member

    def merge(left: str, right: str) -> None:
        parents[root(right)] = root(left)

    keys: dict[tuple[str, str], str] = {}
    if proposal.member_unit == "source":
        for member in proposal.members:
            source = _source_id(bound.visible[member.anchor.evidence_label], catalogue)
            key = ("source", source)
            if key in keys:
                merge(keys[key], member.id)
            keys[key] = member.id
    else:
        for member in proposal.members:
            if member.identity is not None:
                key = (member.identity.namespace, member.identity.value)
                if key in keys:
                    merge(keys[key], member.id)
                keys[key] = member.id
    for relation in proposal.identities:
        if relation.left not in parents or relation.right not in parents:
            raise _invalid("operation identity relation references an unknown member")
        covered = {
            anchor.evidence_label
            for anchor in relation.anchors
            if _anchor_text(anchor, catalogue, bound.visible)
        }
        labels = {
            bound.members[item].anchor.evidence_label for item in (relation.left, relation.right)
        }
        if relation.relation != "unknown" and not labels <= covered:
            raise _invalid("operation identity relation lacks both source witnesses")
        if relation.relation == "same" and proposal.member_unit == "entity":
            merge(relation.left, relation.right)
    roots = {member: root(member) for member in parents}
    distinct: set[frozenset[str]] = set()
    keyed = list(keys.items())
    for index, (left_key, left) in enumerate(keyed):
        for right_key, right in keyed[index + 1 :]:
            if left_key[0] == right_key[0] and left_key[1] != right_key[1]:
                if roots[left] == roots[right]:
                    raise _invalid("operation identity key conflicts with a same relation")
                distinct.add(frozenset((roots[left], roots[right])))
    for relation in proposal.identities:
        if relation.relation == "different":
            if roots[relation.left] == roots[relation.right]:
                raise _invalid("operation identity relation contradicts its equivalence group")
            distinct.add(frozenset((roots[relation.left], roots[relation.right])))
    return roots, distinct


def _identity_unknown(chosen: Sequence[str], distinct: set[frozenset[str]]) -> bool:
    return any(
        frozenset((left, right)) not in distinct
        for index, left in enumerate(chosen)
        for right in chosen[index + 1 :]
    )


def _membership_answer(
    proposal: EvidenceOperation,
    chosen: Sequence[str],
    grouped: Mapping[str, list[str]],
    bound: _BoundOperation,
    catalogue: EvidenceCatalogue,
) -> str:
    if proposal.operation == "count":
        return str(len(chosen))
    sources: list[str] = []
    for group in chosen:
        for member in grouped[group]:
            if bound.states[member] != "true":
                continue
            label = bound.members[member].anchor.evidence_label
            source = _source_id(bound.visible[label], catalogue)
            if source not in sources:
                sources.append(source)
    if proposal.quantifier == "one":
        sources = sources[:1]
    if sources:
        return ", ".join(sources)
    return (
        "No confirmed matching sources."
        if proposal.quantifier == "confirmed_subset"
        else "No matching sources."
    )


def _contribution(
    contribution: MemberContribution, bound: _BoundOperation, roots: Mapping[str, str]
) -> tuple[Decimal | None, str | None]:
    facts = [bound.facts[item] for item in contribution.value_ids]
    _unique(
        [str(bound.locations[fact.id]) for fact in facts],
        "operation contribution repeats one source numeric occurrence",
    )
    if any(roots[fact.member_id] != roots[contribution.member_id] for fact in facts):
        raise _invalid("operation contribution combines different members")
    extents = [fact.extent for fact in facts]
    if contribution.basis == "whole":
        if len(facts) != 1 or extents != ["whole"]:
            raise _invalid("operation whole basis requires exactly one whole value")
    elif contribution.basis == "partition":
        if set(extents) != {"part"} or len({fact.part_key for fact in facts}) != len(facts):
            raise _invalid("operation partition requires distinct supported part identities")
        if not contribution.parts_complete:
            return None, None
    elif len(facts) != 2 or set(extents) != {"quantity", "per_unit"}:
        raise _invalid("operation product requires one quantity and one per-unit value")
    units = {fact.unit for fact in facts if fact.extent != "quantity"}
    if len(units) != 1:
        return None, None
    unit = next(iter(units))
    if contribution.basis == "quantity_per_unit":
        return bound.numbers[facts[0].id] * bound.numbers[facts[1].id], unit
    return sum((bound.numbers[fact.id] for fact in facts), Decimal(0)), unit


def _numeric_answer(  # noqa: C901 - operation-dependent numeric dependency evaluation
    proposal: EvidenceOperation,
    chosen: Sequence[str],
    roots: Mapping[str, str],
    bound: _BoundOperation,
) -> tuple[str | None, list[str], dict[str, Decimal]]:
    missing: list[str] = []
    contributions: dict[str, MemberContribution] = {}
    for contribution in proposal.contributions:
        root = roots[contribution.member_id]
        if root in contributions:
            raise _invalid("operation repeats an entity contribution or mixes accounting bases")
        contributions[root] = contribution
    if set(contributions) - set(chosen):
        raise _invalid("operation contributes a member outside its selected membership")
    expected = (
        1 if proposal.operation == "scalar" else 2 if proposal.operation == "difference" else None
    )
    if expected is not None and len(chosen) != expected:
        missing.append("target_ambiguous")
    totals: dict[str, Decimal] = {}
    units: set[str] = set()
    used_measure_locations: set[tuple[str, str, int]] = set()
    observations: dict[tuple[str, str, str | None], set[tuple[Decimal, str | None]]] = {}
    for fact in proposal.values:
        root = roots[fact.member_id]
        if root in chosen:
            key = (root, fact.extent, fact.part_key)
            observations.setdefault(key, set()).add((bound.numbers[fact.id], fact.unit))
    if any(len(values) > 1 for values in observations.values()):
        missing.append("conflicting_values")
    with localcontext() as context:
        context.prec = 256
        for root in chosen:
            selected_contribution = contributions.get(root)
            if selected_contribution is None:
                missing.append("value_missing")
                continue
            measure_locations = {
                bound.locations[identifier]
                for identifier in selected_contribution.value_ids
                if bound.facts[identifier].extent != "quantity"
            }
            if used_measure_locations & measure_locations:
                raise _invalid("operation repeats one measure occurrence across different members")
            used_measure_locations.update(measure_locations)
            value, unit = _contribution(selected_contribution, bound, roots)
            if value is None or unit is None:
                missing.append("measure_incomplete")
                continue
            totals[root] = value
            units.add(unit)
        if len(units) > 1:
            missing.append("unit_conflict")
        if len(totals) != len(chosen) or len(units) != 1:
            return None, list(dict.fromkeys(missing or ["value_missing"])), totals
        if not totals:
            # A count has a natural zero; an empty monetary/physical sum has no supplied unit.
            return None, ["unit_missing"], totals
        if proposal.operation == "difference":
            if len(chosen) != 2:
                return None, list(dict.fromkeys(missing)), totals
            value = totals[chosen[0]] - totals[chosen[1]]
        else:
            value = sum(totals.values(), Decimal(0))
    text = _decimal_text(value)
    if proposal.format == "with_unit":
        text += " " + next(iter(units))
    return text, list(dict.fromkeys(missing)), totals


def _partial_answer(proposal: EvidenceOperation, answer: str) -> str:
    label = "matching sources" if proposal.operation == "select_sources" else "result"
    return f"Supported {label} in the supplied evidence: {answer}. The requested full answer is unresolved."
