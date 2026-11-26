# Affective memory

Keep interaction context without turning one moment into a permanent profile. SenseMem records affect as a sourced, timed assessment and requires independent support before an inferred trait becomes active in recall. Original observations, model estimates, and a person's own statements remain distinct.

This page explains representation, formation, visibility, and response-policy authority. The [SDK reference](api/python-sdk.md) defines the values, the [memory lifecycle guide](memory-types-time-and-decay.md) defines visibility and withdrawal, and [context compilation](context-compilation.md) defines delivery.

## Representation

| Information | Representation | Meaning |
| --- | --- | --- |
| Original words, audio, or frames | `OBSERVATION` | Source evidence retained independently of an interpretation. |
| Situated affect assessment | `AFFECT` | An episodic record with optional valence, arousal, cue modality, confidence, and validity. |
| Co-occurring event | `EVENT` | An event assertion linked to its source observation. |
| Emotional target or elicitor | `RELATION` with `affect_link` | A separate, sourced affirmation or denial about one particular affect cue. |
| Longer-horizon characteristic | `TRAIT` | A semantic claim with additional support requirements for model inferences. |
| Response guidance | `RESPONSE_POLICY` | Procedural guidance authorized by a host operation with response-feedback provenance. |

For example, "the voice rose," "the person may be anxious," and "the person says they were nervous" make different claims. Storing one does not overwrite the others, and `AFFECT` records do not automatically supersede each other or become standing state assertions.

`FormationProposal` accepts `valence` in `[-1, 1]`, `arousal` in `[0, 1]`, a single atomic `cue_modality`, confidence, subject, and validity bounds. A cue cannot use `omni` or name a modality absent from its source, so text-only input cannot support a claim about vocal tone.

## Producing affect records

The bundled OpenAI-compatible `FormationBackend` can propose affect and trait records after a source observation commits. Formation is disabled unless configured. Its native modalities default to text; source-labelled captions and transcripts can also provide a text route.

The bundled adapter can send original audio to an audio-capable generation endpoint when its formation modalities include audio. A custom formation adapter can provide dedicated prosody, facial-expression, or acoustic-affect analysis. Formation modality declarations control routing; see [formation configuration](configuration/lifecycle.md#automatic-memory-formation).

Keep one estimate per cue channel when channels disagree. The kernel excludes `AFFECT` from functional conflict reconciliation, so simultaneous cues remain separate records. Use an explicit host correction or deletion to withdraw an incorrect inference while retaining its source.

The evidence basis records whether a claim is an observation, user statement, model inference, or response feedback. `AffectWitness(source_id, reporter, quote)` preserves an exact original-text report and its reporter. Automatic affect and attribution formation always stays `MODEL_INFERENCE`, including when the source is host-marked `USER_STATEMENT`. The host can explicitly assert `USER_STATEMENT` through `apply(CONSOLIDATE)` with a host-marked source, exact witness, and reporter matching the affect subject.

## Emotional targets and elicitors

An affect cue records how someone seemed or said they felt; a separate relation records what the feeling concerns (`target`) or which event elicited it (`elicitor`). A target can refer to a recognized identity or an entity/event record; an elicitor references an event.

`AffectLink` names the focal `affect_id`, a role, an `AffectObjectRef`, and polarity `affirmed` or `denied` (relation predicate `affect_target` or `affect_elicitor`). An explicit denial preserves a rejected explanation, an absent relation leaves attribution unknown, and multiple elicitors may coexist.

Formation can propose a cue, event, and relations in one batch by giving a proposal a batch-local `local_id` and referring to it as `proposal:<local_id>`. The kernel resolves those references before storage and requires an active cue and grounded object. The focal cue and any record-valued object are required premises without counting as new independent observations.

A later attribution keeps the focal cue's occurrence time, identity, place, and spatial context while its recording time and reporting witness identify the later explanation. Moving to another location before clarifying an earlier feeling does not move that attribution or hide a denial from the original episode's place filter.

## Trait visibility

A model-inferred `TRAIT` stays hidden from active retrieval until two independent evidence groups support the normalized claim. A host-confirmed `USER_STATEMENT` trait is visible immediately.

Automatic state and trait extraction always remains `MODEL_INFERENCE`. The host can confirm a typed trait explicitly through `apply(CONSOLIDATE)` with `FormationProposal(basis=EvidenceBasis.USER_STATEMENT, ...)`. Current single-valued preferences are `STATE` records and can be visible after one observation; see [state replacement](typed-memory.md#state-replacement).

Capture identity determines independence: an observation uses `ObservationContext.source_id`, or its record ID when unset. Derived records inherit the groups of their sources, so several cues from one capture count as one group. Evidence changes recompute confidence and visibility; `get()` and `list()` still expose a hidden trait with `context.visible=False`, while ordinary retrieval excludes it.

## Compiled context

`compile()` places affect and traits in separate sections. An `AffectCue` carries the hit's stored context and `event_ids` for eligible events sharing source evidence. Shared evidence establishes co-occurrence, not causation; see [affect cues](context-compilation.md#affect-cues).

Typed attribution relations appear as independent lines in `relationships`. Before delivery, the compiler reads active affirmations and denials for that exact cue/role/object group from SQLite and delivers their required evidence together within the normal budget. An incomplete read, unavailable peer evidence, or passed deadline withholds the attribution and reports an unknown while leaving an independently deliverable cue available; see [attribution delivery](context-compilation.md#attribution-delivery).

## Response-policy authority

Inferred affect does not authorize a change in how an agent behaves toward someone. Formation and consolidation backends cannot create `RESPONSE_POLICY` records regardless of claimed basis; only a host-authored `apply()` operation with `response_feedback` provenance can authorize one.

Model-originated writes cannot modify an active response policy or its transitive support graph through reinforcement, correction, forgetting, source replacement, or lineage supersession. Host administration retains authority to change or withdraw the policy and its evidence.

## Adapter considerations

Declare the modalities the adapter actually analyzes. The bundled FunASR transcriber strips `<|...|>` tags from transcription, so an affect adapter must return typed proposals rather than rely on emotion tags surviving text filtering.

The kernel refuses an ungrounded affect proposal individually and increments `sensemem.formation.refused_proposals` while other valid proposals and the source remain durable. Preserve model and recipe identity whenever changing analysis logic or prompts.

## Privacy and governance

The host controls recording disclosure, consent, remote inference, retention, and access. Consent to transcription does not by itself authorize every derived analysis, and affect estimates should not be presented as clinical diagnoses or verified psychological traits.

New analysis capabilities compose through the [plugin factory and protocol boundaries](configuration/plugins.md) and share the same authoritative records, evidence links, and lifecycle as other derived memory. See [Security](../SECURITY.md) for model-data and deletion boundaries and the [model quality gate](model-quality.md#check-affective-and-attribution-behavior) for affective integration tests.
