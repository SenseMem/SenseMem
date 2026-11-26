# Typed memory and evidence

Turn observations into entities, events, changing states, and relationships while keeping their
supporting evidence. Typed memory separates what was captured from what was inferred and
maintains confidence, visibility, and history as evidence changes.

Use [memory types and time](memory-types-time-and-decay.md) for ordinary roles and filters.
Enable a formation backend with [formation configuration](configuration/lifecycle.md#automatic-memory-formation).

## Add an observation

The snippets assume an open `memory`. Caller input uses `ObservationContext`:

```python
from sensemem import EvidenceBasis, ObservationContext

source = memory.add(
    "The mug is on the kitchen table.",
    context=ObservationContext(
        basis=EvidenceBasis.OBSERVATION,
        source_id="camera-1:frame-42",
        confidence=0.94,
        place_id="kitchen",
    ),
)
```

The persisted record exposes `RecordSemantics` and `MemoryRecord.place_id`. Sources are
`OBSERVATION` records; `add()` cannot directly write a derived kind.

## Derived kinds

| Kind | Memory type | Meaning |
| --- | --- | --- |
| `ENTITY` | Semantic | Person, object, or concept |
| `EVENT` | Episodic | Event assertion |
| `STATE` | Semantic | Value with world validity |
| `RELATION` | Semantic | Relation that may accumulate values |
| `AFFECT` | Episodic | Situated affect cue with a source modality |
| `TRAIT` | Semantic | Longer-term characteristic |
| `RESPONSE_POLICY` | Procedural | Host-authorized, feedback-grounded response guidance |

A formation backend can propose the first six kinds; the host authorizes response policies through
`apply()`. Source commit precedes formation. A model error leaves that observation durable and
the unfinished formation retryable. The kernel checks proposals, assigns IDs, versions claims,
and commits evidence-linked records without rewriting the source.

An affect cue must name a modality present in its source. Inferred traits and inferred names need
two independent capture groups before active retrieval; user-stated claims are visible immediately.
A capture group is `ObservationContext.source_id`, falling back to the raw record ID.
Text/audio observations of the same capture, and all their derived cues, count as one group.
Dependent confidence and visibility recompute transactionally when support changes.

An affect's object and eliciting event use separate typed `RELATION` records. `affect_target`
and `affect_elicitor` require an `AffectLink` identifying the cue, role, object reference, and
`affirmed`/`denied` polarity. Targets may use a recognized identity or entity/event record;
elicitors require an event record. The focal cue and a record-valued object enter the relation's
required evidence closure, without becoming independent corroboration. A missing link expresses
unknown attribution, and a denied link rejects only its particular cue/role/object association.
Reporting witnesses retain original source IDs, reporter names, and exact quotations. See
[affective memory](affective-memory.md#emotional-targets-and-elicitors).

## Support clauses

```text
support(claim) = (A AND B) OR C
```

One explicit formation/consolidation evidence tuple is one joint assessment. Losing A or B
withdraws that entire clause; C remains an independent alternative. Several cited records in one
assessment cannot corroborate that assessment itself.

| Producer | Evidence interpretation |
| --- | --- |
| `FormationProposal.evidence_ids=None` | Singleton primary input |
| Explicit formation tuple | Complete conjunction including the primary input |
| One `CONSOLIDATE` operation | Complete cited set as one conjunction |
| Separate operations for the same stable claim | Alternative clauses |
| `REINFORCE` and general context evidence | Independent singleton alternatives |

Formation tuples reject empty, duplicate, foreign, or out-of-batch IDs before derived commits.
`RecordSemantics.evidence_ids` is the ordered audit union of complete active clauses. A joint
proposal contributes one assessment; the default confidence is the greater of the joint-assessment
maximum and the capture-group singleton noisy-OR projection. These are operational support scores,
not calibrated truth probabilities or proof of semantic entailment.

For a `MODEL_INFERENCE` claim bound to a stable identity, a host-confirmed naming assertion
(`ENTITY`, predicate `identity`, `USER_STATEMENT`, matching `identity_id`) is a condition of
attribution, not an observation of that claim. It stays in every complete AND assessment and
its audit union, and retracting it withdraws those assessments. It does not supply an experience
vote or make distinct observations dependent merely because both use that name. Naming alone
supplies no experience count or confidence. This exception does not apply to inferred names,
other identities, an `ENTITY` target, or host-confirmed claims.

The default projection excludes these conditions only when counting each complete clause:
one remaining member uses its existing capture group; multiple remaining members retain the
joint-assessment maximum. Other joint clauses and the all-group singleton noisy-OR stay unchanged.
The independent projection instead compares complete observation paths. Naming ancestors that
also appear through an observation or summary path still count as shared evidence.

A non-observation whose basis is neither `USER_STATEMENT` nor `RESPONSE_FEEDBACK` needs an active
grounded clause to remain visible. Observations and host assertions are roots. Deleting evidence
evaluates the affected reverse-dependency region to a grounded fixed point: unsupported cycles
cannot keep one another alive, while a cycle supported by an independent root can survive.

Clause retirement, reactivation, confidence changes, and rollback append transaction intervals.
Historical reads retain the support that existed then. Rollback restores the exact predecessor,
preserves unrelated alternatives, and never turns a joint clause into a smaller one. Physical
deletion cannot be undone by clause history.

Default [context compilation](context-compilation.md#evidence-closed-selection) delivers the
complete audit union and can conservatively refuse cycles even when another alternative is grounded.
Selected proofs are a separate delivery opt-in.

## Formation failures and inherited context

Individually ungrounded proposals, including missing cue modality, incompatible frame, and
conflicts within one source, are refused independently. The first accepted conflicting proposal
stands. Refusals increment `sensemem.formation.refused_proposals`, complete the recipe, and are
not automatically retried. Invalid batch structure/evidence declarations and model, embedding, or
commit failures retain their errors and can leave a committed source pending settlement.

Formation inherits event/valid time and metric pose from its primary source. Place and metadata
are retained only when every cited source agrees on each complete value. Entities, relations,
inferred traits, and response policies shared across formation inputs must also agree with the
existing record. Deletion/rollback recomputes live-source agreement. Derived records carry text
and evidence links, not their own media assets.

## State replacement

Functional state lineages use kind, normalized subject, predicate, and spatial frame/anchor.
For a claim bound to a stable identity, that identity replaces its display name in the claim ID,
so renaming a person does not split an otherwise identical trait or relation. ENTITY naming
assertions still distinguish names and preserve their separate naming history.
Formation and consolidation bind a claim only through its cited, visible, current `ENTITY`
naming assertion. A matching registry display name grants no binding. Citing names for two
different identities rejects the proposal; two people with the same display name keep separate
claims. The naming assertion remains a required member of that assessment's evidence clause.
Original text that explicitly names a label can support an unbound claim without citing a
historical name, including when history contains a stable identity with that display name.
The claim keeps `identity_id=None` and a separate claim key; it does not join that person's
profile or gain a naming dependency. Transcripts, descriptions, model interpretations, and
pronouns do not grant this exception. A proposal that does cite the historical name keeps that
dependency; competing cited identities are still refused. Sharing a naming record does not
create independent support. Retaining an unbound report does not satisfy a requirement to
bind its subject to a registered identity.
A later `STATE` replaces overlapping validity only when its proposal's authority and evidence
are sufficient:

1. `USER_STATEMENT` and `RESPONSE_FEEDBACK` outrank `MODEL_INFERENCE`.
2. Within one authority tier, new confidence must meet the standing evidence confidence.
3. A higher-authority claim can replace an inference with a lower scalar score.

Automatic formation and consolidation proposals must use `MODEL_INFERENCE`. Marking a source
`ObservationContext(basis=EvidenceBasis.USER_STATEMENT)` preserves the source's provenance;
it does not confirm the model's interpretation of its subject, attribution, polarity, or value.
Copying an exact quote cannot establish those meanings. The bundled wire schema accepts neither
`statement_quote` nor a model-selected `basis`, and the kernel also rejects elevated authority
from custom automatic backends.

The host can confirm a specific typed claim through `apply()`:

```python
from sensemem import FormationProposal, MemoryIntent, RecordKind, MemoryOperation

source = memory.add("Rava says she has been patient for years.")
confirmation = memory.apply(
    MemoryOperation(
        intent=MemoryIntent.CONSOLIDATE,
        evidence_ids=(source.id,),
        proposal=FormationProposal(
            kind=RecordKind.TRAIT,
            basis=EvidenceBasis.USER_STATEMENT,
            content="Rava is patient.",
            subject="Rava",
            predicate="disposition",
            value="patient",
        ),
    )
)
```

This explicit host assertion can make a trait immediately visible. An automatically extracted
current `STATE` can also be visible after one source, with inference authority; a later model
inference cannot replace a stronger host-confirmed value merely by quoting its own source.
Host assertions are roots: withdrawing the supporting source does not revoke the host's
confirmation. Use `rollback(confirmation.operation_id)` or an explicit correction to withdraw it.

A weaker alternative remains supported conflict; filtering it out cannot make compilation
assert the other value alone. Use explicit correction and replacement for a wrong inference.

Accepted replacement preserves unaffected validity segments, including backfill and A → B → A.
Independent assertions in one SQLite batch remain conflicting; separate transactions are ordered
even with equal clocks. Evidence deletion replays the same comparison transitively. Explicit
corrections remain withdrawn until reasserted or rolled back. Reinforcement adds support without
creating a new validity interval or revision order.

User-stated traits also have one standing value per lineage. Relations and model-inferred traits
accumulate values. Registered scalar predicate aliases share claim keys across languages, while
unknown predicates keep literal keys; subject identity, kind, and spatial scope still separate claims.
Reviewed aliases use the same stored claim ID as their canonical predicate, so independent
sources accumulate support on one claim rather than creating separate hidden copies.

Explicit `CORRECT` withdraws the complete evidence assessments that depend on the corrected
claim. If an intermediate claim loses eligibility, withdrawal continues to its dependents;
independent surviving assessments remain. This includes a summary retained as evidence after
consolidation. Only the host can correct such a retained target. Rollback restores the withdrawn
assessments. A historical state interval or ordinary consolidation retirement does not itself
withdraw its evidence.

The bundled former uses `STATE` for a current single-valued preference and keeps the same
kind/predicate when its value changes. It uses the reviewed `preferred_drink`, `home_city`, and
`current_location` keys for those properties; words such as “current,” “only,” and “changed”
do not create a separate property. Predicate registry `sensemem-predicates-v2` additionally maps
the observed `current_preferred_drink` and `only_current_preferred_drink` aliases to
`preferred_drink`. Other unknown strings remain literal, and different kinds remain separate.
Unsupported on-disk predicate provenance is refused rather than silently reinterpreted.

## Naming a person

`register_identity()` creates an identity-bound `ENTITY` naming assertion with
`USER_STATEMENT` basis. It is searchable immediately and needs no former. Subject is the name,
value is the relationship, and place/metadata are not inherited. All names of one identity share
a lineage, so renaming retires the earlier assertion.

The registry name and relationship and the indexed transcript text project the current visible
assertion. When naming or rollback requires a speech refresh, SQLite evaluates and rolls back the
candidate mutation to collect preparation inputs. The kernel prepares refreshed text and embeds it
outside SQLite. The final transaction repeats the mutation, revalidates the inputs, and commits
the names, evidence, operation log, text, vectors, and index outbox together. See
[write consistency](architecture.md#write-consistency).

Naming logs a manual `IDENTIFY`. Rollback restores its predecessor and projections and withdraws
complete downstream assessments that depend on the retracted naming assertion. Repeating a standing
name changes nothing. Model-proposed identification, including
typed `speaker_names` returned by a vision model, instead carries `MODEL_INFERENCE`, the producing
model and recipe, and the original source-memory IDs. It requires two independent groups before
the name is projected; `IdentityProfile.confirmed` remains false until the host registers a name.
Removing its support retracts the inferred projection without changing a host-confirmed name. See
[the identity API](api/python-identity.md) for recognition, consent, linking, and erasure.

## Experimental independent evidence

`settings.independent_evidence: true`, or `MemorySettings(independent_evidence=True)`, selects an
experimental support projection. It is off by default and pins `independent-v2` instead of
`complete-v2` in `evidence.projection_recipe`. Version 2 separates matching host naming conditions
from experience support in both policies. Earlier projection recipes are refused without migration;
Evidence projection policy is separate from the SQLite schema. Changing policy requires a fresh
directory and replay; mismatches or missing markers on nonempty stores raise `StorageError`. Open
never rewrites historical evidence.

Proofs trace AND/OR ancestry through original captures and intermediate assertions. Two assessments
corroborate only when their observation paths share neither captures nor intermediate claims;
matching host naming conditions retain their complete proof but supply no observation path. Alternatives of
one assessment still count once. Capture-disjointness alone does not establish statistical independence.

Support count saturates at two. Proof confidence is its minimum path confidence; reported confidence
is the maximum single proof or eligible-pair noisy-OR. Count and confidence can have different
witnesses. Extra assessments beyond two do not automatically raise confidence. User-stated derived
records retain their ancestry in this calculation.

Per projected memory, computation loads at most 256 records and 4096 accepted clause-member rows
plus a truncation sentinel, keeps up to 64 proofs per node, and limits clause/join/pair work to
4096 steps. Only complete grounded proofs contribute. Bounds warn and produce a conservative
lower estimate; adding evidence can change the retained subset, so monotonicity is not promised
at those bounds. A whole dependent closure can cost more than one projection.

Evidence changes reproject dependents in the same transaction. Withdrawal preserves conjunctions
and alternatives and makes no model request. Formation/apply/rollback may re-embed renamed speech
inside that transaction; embedding failure rolls it back. Deletion prepares a batch from an exact
rollback-only preview before its final commit.

Use the [model quality gates](model-quality.md) and [benchmark harness](benchmarking.md) to evaluate
a model configuration against the tasks your application needs.
