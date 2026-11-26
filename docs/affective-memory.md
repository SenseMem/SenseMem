# Affective memory

Keep interaction context without turning one moment into a permanent profile. SenseMem records
affect as a sourced, timed assessment and requires independent support before an inferred trait
becomes active in recall. Original observations, model estimates, and a person's own statements
remain distinct.

This page explains representation, formation, visibility, and response-policy authority. The
[SDK reference](api/python-sdk.md) defines the values, [memory lifecycle guide](memory-types-time-and-decay.md)
defines visibility and withdrawal, and [context compilation](context-compilation.md) defines delivery.

## Representation

| Information | Representation | Meaning |
| --- | --- | --- |
| Original words, audio, or frames | `OBSERVATION` | Source evidence retained independently of an interpretation. |
| Situated affect assessment | `AFFECT` | An episodic record with optional valence, arousal, cue modality, confidence, and validity. |
| Co-occurring event | `EVENT` | An event assertion linked to its source observation. |
| Emotional target or elicitor | `RELATION` with `affect_link` | A separate, sourced affirmation or denial about one particular affect cue. |
| Longer-horizon characteristic | `TRAIT` | A semantic claim with additional support requirements for model inferences. |
| Response guidance | `RESPONSE_POLICY` | Procedural guidance authorized by a host operation with response-feedback provenance. |

For example, “the voice rose,” “the person may be anxious,” and “the person says they were
nervous” make different claims. Storing one does not overwrite the others. `AFFECT` records do not
automatically supersede each other or become standing state assertions.

`FormationProposal` accepts `valence` in `[-1, 1]`, `arousal` in `[0, 1]`, a single atomic
`cue_modality`, confidence, subject, and validity bounds. A cue cannot use `omni` or name a modality
absent from its source. Text-only input therefore cannot support a claim about vocal tone.
Confidence is an operational assessment, not a calibrated probability.

## Producing affect records

The bundled OpenAI-compatible `FormationBackend` can propose affect and trait records after a
source observation commits. Formation is disabled unless configured. Its native modalities default
to text; source-labelled captions and transcripts can also provide a text route.

The bundled adapter can send original audio to an audio-capable generation endpoint when its
formation modalities include audio. A custom formation adapter can provide dedicated prosody,
facial-expression, or acoustic-affect analysis. Declare the modalities actually analyzed and
evaluate the estimates on representative inputs. Formation modality declarations control routing.
See [formation configuration](configuration/lifecycle.md#automatic-memory-formation).

Keep one estimate per cue channel when the channels disagree. The kernel excludes `AFFECT` from
functional conflict reconciliation, so two simultaneous cues remain separate records. Use an
explicit host correction or deletion to withdraw an incorrect inference while retaining its source.
A backend with insufficient evidence can return no proposal.

The evidence basis records whether a claim is an observation, user statement, model inference, or
response feedback. `AffectWitness(source_id, reporter, quote)` preserves an exact original-text
report and its reporter. A statement about somebody else's feelings remains a reported judgment;
it does not become that person's own statement. Automatic affect and attribution formation stays
`MODEL_INFERENCE`, including when the source is host-marked `USER_STATEMENT`: a copied quote and
a model-written reporter name cannot verify who spoke about whom. The host can explicitly assert
`USER_STATEMENT` through `apply(CONSOLIDATE)` with a host-marked source, exact witness, and reporter
matching the affect subject. This is the host's assertion of self-report status. Reporting
witnesses preserve the original words without granting that authority themselves.

## Emotional targets and elicitors

An affect cue answers how somebody seemed or said they felt. A separate relation can assert
what the feeling concerns (`target`) or which event elicited it (`elicitor`). These roles differ:
somebody may be angry at a colleague because a meeting was cancelled. A target can refer to a
stable recognized identity or a particular entity/event record; an elicitor references an event.

`AffectLink` names the focal `affect_id`, a role, an `AffectObjectRef`, and polarity `affirmed` or
`denied`. Its relation predicate is `affect_target` or `affect_elicitor`. An explicit denial
preserves a particular rejected explanation. An absent relation leaves attribution unknown;
it does not assert that the feeling has no object or cause. Multiple elicitors may coexist.

Formation can propose a cue, event, and relations in one batch. Give a proposal a batch-local
`local_id`, then refer to it as `proposal:<local_id>` inside the link. The kernel resolves those
references before storage and requires an active cue and grounded object. Stored relations
carry stable record or identity references. The focal cue and a record-valued object are required
premises, so deleting their support also withdraws dependent attribution. They do not count as
new independent observations.

Configured formation examines positive, negative, mixed, and low-arousal inputs alike. It can
return no proposal when the available observation and bounded history do not establish the
target or elicitor. Later observations can supply an explanation or deny an earlier one;
recording order alone does not make a competing explanation true.

A later attribution keeps the focal cue's occurrence time, identity, place, and spatial context.
Its recording time and reporting witness identify the later explanation. Moving to another
location before clarifying an earlier feeling therefore does not move that feeling's attribution
to the reporting location or hide a denial from the original episode's place filter.

## Trait visibility

A model-inferred `TRAIT` stays hidden from active retrieval until two independent evidence groups
support the normalized claim. A trusted `USER_STATEMENT` trait is visible immediately.

All automatic state and trait extraction remains `MODEL_INFERENCE`, even when its source is
host-marked `USER_STATEMENT`. An original quote does not confirm the model's attribution or
interpretation. The host can explicitly confirm a typed trait through `apply(CONSOLIDATE)` with
`FormationProposal(basis=EvidenceBasis.USER_STATEMENT, ...)`. Current single-valued preferences
are `STATE` records and can be visible after one observation without becoming host-confirmed
traits. See [state replacement](typed-memory.md#state-replacement) for authority and history rules.

Capture identity determines independence: an observation uses its `ObservationContext.source_id`,
or its record ID when no source ID was supplied. Derived records inherit the groups of their
sources. Several cues from one capture count as one group. Jointly cited sources are one support
assessment; listing more source IDs does not create independent corroboration.

Evidence changes recompute confidence and visibility. `get()` and `list()` can still expose a
hidden trait with `context.visible=False`; ordinary retrieval excludes it. The
[support rules](typed-memory.md) define clause withdrawal
and the optional independent-evidence projection.

## Compiled context

`compile()` places affect and traits in separate sections. An `AffectCue` carries the hit's stored
context and `event_ids` for eligible events sharing source evidence. Rendered affect lines show
basis, confidence, cue modality, valence, arousal, and bounded source/event identifiers when present.

Shared evidence establishes co-occurrence, not cause or emotional target. An event ID does not
mean that the event caused the affect, nor does it deliver the event's content. Event enrichment
is optional and can be skipped after the compilation deadline. See
[affect cues](context-compilation.md#affect-cues) for rendering and limits.

Ordinary relevance, time, scope, and evidence rules determine affect retrieval. Applications can
use those records for their own trajectory or per-person analysis.

Typed attribution relations appear as independent lines in `relationships`. Each line retains its
role, polarity, basis, and any reporting witnesses. Before delivery, the compiler reads active
affirmations and denials for that exact cue/role/object group from SQLite, including peers that
vector ranking omitted. It delivers their required evidence together within the normal budget.
An incomplete read, unavailable peer evidence, or a passed deadline withholds the attribution and
reports an unknown while an independently deliverable cue remains available. Compilation makes
no model call to infer an explanation. See [attribution delivery](context-compilation.md#attribution-delivery).

## Response-policy authority

Inferred affect does not authorize a change in how an agent should behave toward somebody.
Formation and consolidation backends cannot create `RESPONSE_POLICY` records, regardless of the
basis their proposals claim. Only a host-authored `apply()` operation can authorize one; the
stored record carries `response_feedback`.

Model-originated writes also cannot modify an active response policy or its transitive support
graph through reinforcement, correction, forgetting, source replacement, or implicit lineage
supersession. SQLite checks that boundary in the write transaction. Host administration retains
its documented authority to change or withdraw the policy and its evidence.

## Adapter considerations

Declare the modalities the adapter actually analyzes. A text route over a transcript does not
analyze acoustic cues. The bundled FunASR transcriber strips `<|...|>` tags from transcription;
an affect adapter must return typed proposals rather than rely on emotion tags surviving that
text filter.

The kernel refuses an ungrounded affect proposal individually and counts it in
`sensemem.formation.refused_proposals`; other valid proposals and the source can remain durable.
A malformed batch envelope still fails the call. Preserve model and recipe identity whenever
changing the analysis or prompt.

## Evaluation and privacy

Evaluate an affect adapter on held-out, representative, appropriately authorized inputs. Useful
measurements include valence/arousal error, label accuracy, calibration, abstention, false trait
promotion, correction behavior, and downstream task quality against a text-only baseline.
Report the model revision, source population, modality, annotation basis, and costs.

The [synthetic affect and preference gate](model-quality.md#affect-and-preference-checks) checks
source attribution, text/image disagreement, trait support, preference replacement, and injected
instructions. Real-model qualification is not yet complete: recorded runs include missed cues,
inconsistent trait promotion, and stale preferences. Kernel lifecycle tests establish the storage
and authority contracts, not recognition accuracy.

Typed target/elicitor and reporting-witness checks establish reference, lifecycle, and delivery
contracts. They do not qualify a model's attribution accuracy, distinguish actual psychological
causes, or establish acoustic-affect quality. The earlier synthetic gate does not qualify the
expanded formation recipe; see [attribution qualification](model-quality.md#attribution-qualification).

The host controls recording disclosure, consent, remote inference, retention, and access. Consent
to transcription does not by itself authorize every derived analysis. SenseMem's identity-consent
mechanism is not a complete per-modality consent workflow. Affect estimates must not be presented
as clinical diagnoses or verified psychological traits.

New analysis capabilities follow the [plugin admission rule](architecture.md#admission-rule).
They use the same authoritative records, evidence links, and lifecycle as other derived memory.
See [Security](../SECURITY.md) for model-data and deletion boundaries.
