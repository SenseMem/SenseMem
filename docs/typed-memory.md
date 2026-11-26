# Typed memory and evidence

Turn observations into entities, events, changing states, and relationships while keeping their supporting evidence. Typed memory separates what was captured from what was inferred and maintains confidence, visibility, and history as evidence changes.

Use [memory types and time](memory-types-time-and-decay.md) for ordinary roles and filters. Enable a formation backend with [formation configuration](configuration/lifecycle.md#automatic-memory-formation).

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

The persisted record exposes `RecordSemantics` and `MemoryRecord.place_id`. Sources are `OBSERVATION` records; `add()` cannot directly write a derived kind.

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

A formation backend can propose the first six kinds; the host authorizes response policies through `apply()`. Source commit precedes formation, so a model error leaves the observation durable and pending retry via `settle()`. The kernel validates proposals, assigns IDs, versions claims, and commits evidence-linked records without rewriting the source.

An affect cue must name a modality present in its source. Inferred traits and inferred names require two independent capture groups before active retrieval; user-stated claims are visible immediately. A capture group is `ObservationContext.source_id`, falling back to the raw record ID. Dependent confidence and visibility recompute transactionally when support changes.

Emotional targets (`affect_target`) and eliciting events (`affect_elicitor`) use separate `RELATION` records carrying an `AffectLink` (`affect_id`, role, `object_ref`, `affirmed`/`denied` polarity) and optional `AffectWitness` quotations; see [affective memory](affective-memory.md#emotional-targets-and-elicitors).

## Support clauses

```text
support(claim) = (A AND B) OR C
```

One explicit formation or consolidation evidence tuple is one joint assessment (conjunction): losing A or B withdraws that clause, while C remains an independent alternative.

| Producer | Evidence interpretation |
| --- | --- |
| `FormationProposal.evidence_ids=None` | Singleton primary input |
| Explicit formation tuple | Complete conjunction including the primary input |
| One `CONSOLIDATE` operation | Complete cited set as one conjunction |
| Separate operations for the same stable claim | Alternative clauses |
| `REINFORCE` and general context evidence | Independent singleton alternatives |

Formation tuples reject empty, duplicate, foreign, or out-of-batch IDs before derived commits. `RecordSemantics.evidence_ids` is the ordered audit union of complete active clauses. Default confidence is the greater of the joint-assessment maximum and the capture-group singleton noisy-OR projection.

For a `MODEL_INFERENCE` claim bound to a stable identity, a host-confirmed naming assertion (`ENTITY`, predicate `identity`, `USER_STATEMENT`, matching `identity_id`) is an attribution condition rather than an experience observation: it stays in the AND assessment and retracting it withdraws dependent assessments, while not counting as an independent experience vote.

A non-observation whose basis is neither `USER_STATEMENT` nor `RESPONSE_FEEDBACK` needs an active grounded clause to remain visible. Deleting evidence evaluates the reverse-dependency region to a grounded fixed point so unsupported cycles cannot keep one another alive. Clause retirement, reactivation, confidence changes, and rollback append transaction intervals so historical `known_at` reads retain past support.

Default [context compilation](context-compilation.md#evidence-closed-selection) delivers the complete audit union; selected delivery certificates are a separate opt-in.

## Formation failures and inherited context

Individually ungrounded proposals (missing cue modality, incompatible frame, or intra-source conflict) are refused independently, incrementing `sensemem.formation.refused_proposals` while valid proposals and the source commit. Invalid batch envelopes and model, embedding, or commit failures raise errors and leave the committed source pending settlement.

Formation inherits event/valid time and metric pose from its primary source. Place and metadata are retained only when every cited source agrees on each complete value. Derived records carry text and evidence links, not their own media assets.

## State replacement

Functional state lineages key on kind, normalized subject (or bound `identity_id`), predicate, and spatial frame/anchor. Formation and consolidation bind a claim to an identity only through a cited, visible, current `ENTITY` naming assertion. Explicit names in original text can still support an unbound claim (`identity_id=None`) with a separate lineage key without joining that identity's profile.

A later `STATE` replaces overlapping validity when its authority and evidence qualify:

1. `USER_STATEMENT` and `RESPONSE_FEEDBACK` outrank `MODEL_INFERENCE`.
2. Within one authority tier, new confidence must meet the standing evidence confidence.
3. A higher-authority claim can replace an inference with a lower scalar score.

Automatic formation and consolidation proposals always carry `MODEL_INFERENCE` basis, even when the source observation is marked `USER_STATEMENT`. Hosts explicitly confirm a typed claim through `apply()`:

```python
from sensemem import EvidenceBasis, FormationProposal, MemoryIntent, MemoryOperation, RecordKind

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

Host assertions are roots: withdrawing the supporting source does not revoke a host confirmation; use `rollback(confirmation.operation_id)` or an explicit correction to withdraw it. A weaker alternative remains a supported conflict rather than being hidden.

Accepted replacements preserve unaffected validity segments (including backfill and A → B → A). Independent assertions in one SQLite batch remain conflicting; separate transactions are ordered even with equal clocks. User-stated traits have one standing value per lineage, while relations and model-inferred traits accumulate values. Reviewed scalar predicate aliases (`preferred_drink`, `home_city`, `current_location`, and their reviewed variants) share canonical claim keys so independent sources corroborate one claim; unknown predicates keep literal keys.

Explicit `CORRECT` withdraws complete evidence assessments depending on the corrected claim and cascades to dependents that lose eligibility; rollback restores them.

## Naming a person

`register_identity()` creates an identity-bound `ENTITY` naming assertion with `USER_STATEMENT` basis, searchable immediately and logged as a reversible `IDENTIFY` operation. Registry name/relationship and indexed transcript text update atomically in one commit; rollback restores the predecessor and withdraws dependent assessments. Model-proposed names (including typed `speaker_names` from vision analysis) carry `MODEL_INFERENCE`, require two independent groups before projection, and leave `IdentityProfile.confirmed` false until host registration. See [the identity API](api/python-identity.md).

## Experimental independent evidence

`MemorySettings(independent_evidence=True)` (or `settings.independent_evidence: true`) enables an opt-in AND/OR path-independence support projection pinned to the physical directory. Changing evidence projection policy requires replaying sources into a fresh directory.

Proofs trace AND/OR ancestry through captures and intermediate claims; two assessments corroborate only when their observation paths share neither captures nor intermediate claims (matching host naming conditions remain required premises without supplying an observation path). Support count saturates at two, proof confidence is the minimum path confidence, and reported confidence is the maximum single proof or eligible-pair noisy-OR. Computation is bounded per projected memory (256 records, 4096 clause-member rows, 64 proofs per node, 4096 work steps) and reprojects dependents in the same SQLite transaction.
