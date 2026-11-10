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

The persisted record exposes `MemoryContext` and `MemoryRecord.place_id`. Sources are
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
`MemoryContext.evidence_ids` is the ordered audit union of complete active clauses. A joint
proposal contributes one assessment; the default confidence is the greater of the joint-assessment
maximum and the capture-group singleton noisy-OR projection. These are operational support scores,
not calibrated truth probabilities or proof of semantic entailment.

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
A later `STATE` replaces overlapping validity only when its proposal's authority and evidence
are sufficient:

1. `USER_STATEMENT` and `RESPONSE_FEEDBACK` outrank `MODEL_INFERENCE`.
2. Within one authority tier, new confidence must meet the standing evidence confidence.
3. A higher-authority claim can replace an inference with a lower scalar score.

A raw user statement does not promote a model proposal's basis. A weaker alternative remains
supported conflict; filtering it out cannot make compilation assert the other value alone.
Use explicit correction and replacement for a wrong inference.

Accepted replacement preserves unaffected validity segments, including backfill and A → B → A.
Independent assertions in one SQLite batch remain conflicting; separate transactions are ordered
even with equal clocks. Evidence deletion replays the same comparison transitively. Explicit
corrections remain withdrawn until reasserted or rolled back. Reinforcement adds support without
creating a new validity interval or revision order.

User-stated traits also have one standing value per lineage. Relations and model-inferred traits
accumulate values. Registered scalar predicate aliases share claim keys across languages, while
unknown predicates keep literal keys; subject identity, kind, and spatial scope still separate claims.

## Naming a person

`register_identity()` creates an identity-bound `ENTITY` naming assertion with
`USER_STATEMENT` basis. It is searchable immediately and needs no former. Subject is the name,
value is the relationship, and place/metadata are not inherited. All names of one identity share
a lineage, so renaming retires the earlier assertion.

The registry name/relationship and indexed transcript text project the current visible assertion
in one commit. Naming logs a manual `IDENTIFY`; rollback restores its predecessor and both
projections. Repeating a standing name changes nothing. Model-proposed identification instead
needs observed identity evidence and two independent groups. See
[the identity API](api/python-identity.md) for recognition, consent, linking, and erasure.

## Experimental independent evidence

`settings.independent_evidence: true`, or `MemoryConfig(independent_evidence=True)`, selects an
experimental support projection. It is off by default and pins `independent-v1` instead of
`complete-v1` in `evidence.projection_recipe`. Changing policy requires a fresh directory and
replay; mismatches or missing markers on nonempty stores raise `StorageError`. Open never rewrites
historical evidence.

Proofs trace AND/OR ancestry through original captures and intermediate assertions. Two assessments
corroborate only when their proofs share neither captures nor intermediate claims; alternatives of
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
