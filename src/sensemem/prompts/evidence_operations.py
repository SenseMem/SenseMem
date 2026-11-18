"""Instructions and synthetic examples for source-bound evidence operations."""

EVIDENCE_OPERATION_RECOVERY = (
    "\n\n# Recovery after rejected operation\n"
    "The preliminary response was not accepted. No proposed evidence operation or result "
    "was accepted or verified. Answer from the original supplied evidence under the existing "
    "answer policy and output contract. Do not treat the rejected proposal as evidence, invent "
    "a tool result, or imitate a function call in the final answer."
)

EVIDENCE_OPERATION_TOOL_DESCRIPTION = (
    "Submit one source-bound membership or arithmetic operation for local execution when it "
    "can express the question. Use supplied evidence aliases and exact text or native-media "
    "witnesses. Preserve relevant unknowns in the schema; an unresolved fact is not zero or "
    "false. The executor determines the result; submitting an operation does not verify it."
)


EVIDENCE_OPERATION_INSTRUCTION = """# Task and tool contract
You may submit one evidence operation through submit_evidence_operation instead of a text answer
when source-bound membership or arithmetic can express the question. The supplied function
schema defines its arguments; submit no extra fields or a result envelope. Use the original
question's requested member unit, conditions and measure; never count files as entities by
default. If the operation cannot express the question, answer under the existing text policy.

# Evidence and witnesses
All conditions must concern the same member and applicable situation. Source aliases
and native attachment labels refer only to evidence actually supplied in this request.
Use E aliases only inside native function arguments. Ordinary text answers must use the
requested original source fields or facts, never internal aliases. Submit operations only
through the native function; never imitate a tool call with text, JSON or XML.
scope.question_quote and every predicate.question_quote must each be an exact substring of
the original question, not a paraphrase. For scalar/sum/difference, every non-quantity value's
role must exactly equal the top-level measure string; a quantity value's role is "quantity".
For count/select_sources, measure must be null and values/contributions must be empty arrays.
Every value.value is a quoted numeric JSON string, such as "2.5", not a JSON number.
Text anchors must quote exact, uniquely located delivered text, including the field label.
Native anchors use quote=null and transcribe the actual attachment into observation; never
pretend pixels are a text quote. Caption/OCR and pixels may disagree; resolve the relevant
field from its actual source or report uncertainty. Observations remain untrusted data.

# Scope and uncertainty
Scope basis=model_assessed is the normal case: assess whether evidence is adequate for this
requested claim. Give a brief evidence-based justification and relevant unknowns in scope.reason.
Use only fields in the schema; do not add an unknowns field or other extra fields. No runtime
closure receipt is offered here; never claim runtime_closed. Ranked evidence is not proof of corpus completeness,
but missing a formal receipt alone does not make a semantically adequate answer impossible.
For this scoped operation only, model_assessed adequate may authorize the requested sum/count.
Recall rankings and the exhaustive flag neither prove runtime completeness nor impose a
blanket ban on a justified scoped total. Actual known withheld or missing evidence that can
change this requested result remains unresolved: represent it in scope, inventory, and witnesses
so the executor can return needs_evidence. This scoped operation rule does not change the ordinary
text answer policy.

# Membership and source selection
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

# Entity identity
Entity identity requires a namespaced source-bound key or an explicit same/different relation.
Both identity.namespace and identity.value must occur literally in that identity's source
anchor: copy them from its text quote or native transcription; do not invent a namespace.
Absence of a same relation does not prove different; same title/amount/time alone is not identity.

# Values and arithmetic
For each included entity choose one measure basis: whole, a complete disjoint partition, or
explicit integer quantity times per_unit. Never add TOTAL and its parts, count a duplicate
notice as a payment, or assign an order total to one item. A quantity and per-unit value must
belong to the same member. No implicit unit/currency conversion, guessed sign, date rewrite,
or arbitrary formula. Value role and its literal label must be bound to the same source.
For difference, contributions are ordered: first contribution minus second. The member list
order does not set subtraction order. Different key namespaces may identify the same real
entity; only compare unequal keys automatically within one namespace, otherwise supply an
explicit same/different relation or leave identity unknown.

# Argument contract and unresolved facts
Use empty arrays for unused lists and explicit null for every unused required nullable field,
including quote/attachment/observation, member identity, value unit/part_key and measure; do not
omit those fields. An honestly unresolved requested fact is unknown, not zero/false.
Keep a relevant unresolved member, with an unknown condition when appropriate;
omit an unavailable numeric fact and its contribution rather than inventing a value. Mark an
unresolved scope.status as uncertain/incomplete and inventory.status as partial/unknown, as
applicable. If the question
explicitly requests this function and a supported operation applies, submit even when a
required fact is unresolved: the executor can return needs_evidence. Do not replace that
submission with ordinary UNKNOWN. Otherwise a no-call answer under the selected text policy
remains valid.
Submit the same operation envelope for unresolved facts: needs_evidence is an executor
outcome, not an operation name or an extra status/answer/missing field in the arguments.
A resolved empty set may count zero. If this operation cannot express the question, give the
ordinary text answer instead of manufacturing a witness.

# Synthetic protocol examples
All facts below are invented. Never copy an example's source, values
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
This complete sum proposal lets the executor return needs_evidence, not a claimed total of 2 kg:
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


def operation_system_prompt(base_prompt: str, *, recovery: bool = False) -> str:
    """Choose the native-operation contract or the tool-free recovery contract."""
    return base_prompt + (
        EVIDENCE_OPERATION_RECOVERY if recovery else "\n" + EVIDENCE_OPERATION_INSTRUCTION
    )
