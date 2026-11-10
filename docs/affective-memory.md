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

Use a custom formation adapter for prosody, facial-expression, or acoustic-affect analysis.
Declare the modalities it analyzes and evaluate its estimates on representative inputs.
Formation modality declarations control routing.
See [formation configuration](configuration/lifecycle.md#automatic-memory-formation).

Keep one estimate per cue channel when the channels disagree. The kernel excludes `AFFECT` from
functional conflict reconciliation, so two simultaneous cues remain separate records. Use an
explicit host correction or deletion to withdraw an incorrect inference while retaining its source.
A backend with insufficient evidence can return no proposal.

The evidence basis records whether a claim is an observation, user statement, model inference, or
response feedback. It has no separate value for another person's judgment about how somebody
seemed. Applications must not present such a judgment as the subject's own statement.

## Trait visibility

A model-inferred `TRAIT` stays hidden from active retrieval until two independent evidence groups
support the normalized claim. A trusted `USER_STATEMENT` trait is visible immediately.

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

The host controls recording disclosure, consent, remote inference, retention, and access. Consent
to transcription does not by itself authorize every derived analysis. SenseMem's identity-consent
mechanism is not a complete per-modality consent workflow. Affect estimates must not be presented
as clinical diagnoses or verified psychological traits.

New analysis capabilities follow the [plugin admission rule](plugin-architecture.md#admission-rule).
They use the same authoritative records, evidence links, and lifecycle as other derived memory.
See [Security](../SECURITY.md) for model-data and deletion boundaries.
