"""Identity governance: naming, consent, linking, unlinking, and erasure."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from typing import cast

from opentelemetry.trace import Tracer

from sensemem._telemetry import (
    IDENTITY_EVIDENCE_ASSETS,
    IDENTITY_EVIDENCE_REQUIRED,
    IDENTITY_LINKED,
    IDENTITY_NAMES_BOUND,
    IDENTITY_NAMES_REFUSED,
)
from sensemem.control import IDENTITY_LINK_RECIPE, dump_operation, operation_key
from sensemem.exceptions import (
    IdentityNotFoundError,
    MemoryNotFoundError,
    ModelError,
    StorageError,
    ValidationError,
)
from sensemem.infrastructure.local.store import (
    IdentityLink,
    StaleOperationError,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
)
from sensemem.kernel.content import prepare_memory
from sensemem.kernel.contracts import Backends
from sensemem.kernel.derived import (
    has_indexed_speech,
    speaker_labels,
)
from sensemem.kernel.embedding import Embedding
from sensemem.kernel.execution import Execution
from sensemem.kernel.formation import (
    CONSENT_RECIPE,
    NAMING_RECIPE,
    Formation,
    consent_proposal,
    formation_context,
    formation_memory_id,
    formation_memory_type,
    naming_proposal,
)
from sensemem.kernel.hydration import operation_record
from sensemem.kernel.lifecycle import Lifecycle, OperationAssets
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.projection import Projection
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.kernel.settings import Settings
from sensemem.kernel.speech import Speech
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import (
    validated_identifier,
    validated_identity_name,
)
from sensemem.kernel.vision import Vision
from sensemem.models.base import SpeechBackend
from sensemem.perception import VisualDescription
from sensemem.types import (
    ConsentClaim,
    ConsentState,
    EvidenceBasis,
    IdentityChange,
    IdentityClaim,
    IdentityErasure,
    IdentityProfile,
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryTrigger,
    Modality,
    PerceptionCoverage,
    PerceptionKind,
    PerceptionResult,
    PerceptionSkipReason,
)

_LOGGER = logging.getLogger(__name__)


def _perception_coverage(
    assets: Sequence[StoredAsset],
    *,
    kind: PerceptionKind,
    kinds: frozenset[PerceptionKind] | None,
    modalities: set[Modality],
    available: bool,
    supported: frozenset[Modality],
) -> tuple[PerceptionCoverage, ...]:
    """Describe the execution scope independently of whether recognition found a person."""
    coverage = []
    for asset in assets:
        reason: PerceptionSkipReason | None = None
        if kinds is not None and kind not in kinds:
            reason = "not_requested"
        elif Modality(asset.modality) not in modalities:
            reason = "unsupported_modality"
        elif not available:
            reason = "backend_not_configured"
        elif Modality(asset.modality) not in supported:
            reason = "unsupported_modality"
        coverage.append(PerceptionCoverage(kind=kind, asset_id=asset.asset_id, reason=reason))
    return tuple(coverage)


class Identities(Traced):
    """Naming, consent, linking, unlinking, and erasing the people memories are about."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        lifecycle: Lifecycle,
        materializer: Materializer,
        speech: Speech,
        embedding: Embedding,
        projection: Projection,
        vision: Vision,
        formation: Formation,
        settings: Settings,
        execution: Execution,
    ) -> None:
        super().__init__(tracer)
        self._formation_lock = storage.formation_lock
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._lifecycle = lifecycle
        self._materializer = materializer
        self._speech = speech
        self._embedding = embedding
        self._projection = projection
        self._vision = vision
        self._formation = formation
        self._settings = settings
        self._execution = execution

    def perceive_memory(
        self, memory_id: str, *, kinds: frozenset[PerceptionKind] | None = None
    ) -> PerceptionResult:
        if kinds is not None and (
            not isinstance(kinds, frozenset) or not kinds or not kinds <= {"speech", "faces"}
        ):
            raise ValidationError("kinds must be a non-empty frozenset of speech and/or faces")
        selected: frozenset[PerceptionKind] = kinds if kinds is not None else frozenset()
        if kinds is None:
            if isinstance(self._backends.transcriber, SpeechBackend):
                selected |= {"speech"}
            if self._backends.face_analyzer is not None:
                selected |= {"faces"}
        with (
            self._trace("sensemem.perceive", kind="operation"),
            self._lifecycle.operation(model_work=True) as operation,
        ):
            normalized_id = validated_identifier(memory_id, "memory_id")
            with self._write_lock, translate_storage_errors("read perception memory"):
                memory = self._store.records.read_memory(normalized_id)
                if memory is None:
                    raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
                assets = tuple({asset.asset_id: asset for asset in memory.assets}.values())
                self._lifecycle.lease_assets(assets, operation.leased)
            coverage: list[PerceptionCoverage] = []
            for kind, modalities, available, supported, missing in (
                (
                    "speech",
                    {Modality.AUDIO, Modality.VIDEO},
                    isinstance(self._backends.transcriber, SpeechBackend),
                    self._backends.transcription_capabilities,
                    "configured transcription backend does not provide speaker recognition",
                ),
                (
                    "faces",
                    {Modality.IMAGE, Modality.VIDEO},
                    self._backends.face_analyzer is not None,
                    self._backends.face_capabilities,
                    "no face backend is configured",
                ),
            ):
                kind = cast(PerceptionKind, kind)
                coverage.extend(
                    _perception_coverage(
                        assets,
                        kind=kind,
                        kinds=kinds,
                        modalities=modalities,
                        available=available,
                        supported=supported,
                    )
                )
                if kind not in selected:
                    continue
                relevant = {Modality(asset.modality) for asset in assets} & modalities
                if relevant and not available:
                    raise ModelError(missing, reason="backend_not_configured")
                unsupported = relevant - supported
                if unsupported and kinds is not None:
                    names = ", ".join(sorted(value.value for value in unsupported))
                    raise ModelError(
                        f"configured {kind} backend does not support: {names}",
                        reason="unsupported_modality",
                    )
            operation.persisted.update(asset.asset_id for asset in assets)
            self.perceive(assets, operation, kinds=selected)
            return PerceptionResult(
                coverage=tuple(coverage),
                speech=tuple(
                    segment
                    for asset in assets
                    for segment in operation.speech_segments.get(asset.asset_id, ())
                ),
                faces=tuple(
                    face
                    for asset in assets
                    for face in operation.face_observations.get(asset.asset_id, ())
                ),
            )

    def identity(self, identity_id: str) -> IdentityProfile | None:
        with (
            self._trace("sensemem.identity", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            requested_id = validated_identifier(identity_id, "identity_id")
            with translate_storage_errors("read identity profile"):
                return self._store.identities.identity_profile(requested_id)

    def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None:
        requested_id = validated_identifier(identity_id, "identity_id")
        if not isinstance(state, ConsentState):
            raise ValidationError("state must be a ConsentState")
        claim = ConsentClaim(identity_id=requested_id, state=state, note=note)
        with (
            self._trace("sensemem.record_consent", kind="operation"),
            self._lifecycle.operation(model_work=True) as operation,
            # Serialize identity policy changes; formation takes the write lock only after
            # preparing model output, in the same lock order as consolidation.
            self._formation_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                known = normalized_id is not None and (
                    self._store.identities.identity_memory_ids(normalized_id) is not None
                )
            if normalized_id is None or not known:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            return self._assert_consent(
                replace(claim, identity_id=normalized_id),
                operation=operation,
            )

    def consent(self, identity_id: str) -> ConsentState | None:
        with (
            self._trace("sensemem.consent", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            requested_id = validated_identifier(identity_id, "identity_id")
            with translate_storage_errors("read identity consent"):
                return self._store.identities.identity_consent(requested_id)

    def _assert_consent(
        self,
        claim: ConsentClaim,
        *,
        operation: OperationAssets,
    ) -> MemoryOperationRecord | None:
        """Commit one consent statement as a bound STATE assertion with its own log row.

        Structurally the twin of `_assert_identity_name`: the assertion is the record, the
        projection reads it, and the log row is what makes it auditable and reversible. A STATE
        rather than an ENTITY kind because that is what it is -- how this person may be
        processed, right now -- and because it keeps consent out of the naming lineage the
        registry projects.
        """
        proposal = consent_proposal(claim)
        now = datetime.now(timezone.utc)
        context = replace(
            formation_context(
                None,
                proposal,
                model_id=None,
                recipe=CONSENT_RECIPE,
                recorded_at=now,
                identity_id=claim.identity_id,
            ),
            # No validity interval, unlike an ordinary STATE. A state of the world holds over a
            # period and a later one splits it; a consent statement simply stands until the
            # subject makes another, so the lineage rule retires the old one outright instead of
            # carrying a past-covering version that would go on projecting beside the new one.
            valid_from=None,
        )
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, operation),
                occurred_at=None,
                occurred_end=None,
                metadata=None,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                claim.identity_id,
                proposal,
                recipe=CONSENT_RECIPE,
                context=context,
            ),
            context=context,
        )
        proposed = MemoryOperation(
            intent=MemoryIntent.CONSENT,
            consent=claim,
            rationale="the data subject stated how they may be processed",
        )
        key = operation_key(proposed, recipe=CONSENT_RECIPE)
        with translate_storage_errors("check a consent operation"):
            if self._store.control.read_operations(operation_key=key):
                return None
        logged = self._formation.commit(
            ((prepared, None, proposal.confidence),),
            (),
            assets=operation,
            completed_at=now,
            recipe=CONSENT_RECIPE,
            operation=StoredOperation(
                operation_key=key,
                intent=proposed.intent.value,
                trigger=MemoryTrigger.MANUAL.value,
                recipe=CONSENT_RECIPE,
                operation_json=dump_operation(proposed),
                applied_at=now,
            ),
        )
        return None if logged is None else operation_record(logged)

    def unlink_identity(self, alias_id: str) -> str | None:
        with (
            self._trace("sensemem.unlink_identity", kind="operation"),
            self._lifecycle.operation(model_work=True) as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: a split must not land while a consolidate apply
            # pass is in progress, and taking the two locks in a different order here would
            # deadlock against that path.
            self._formation_lock,
            self._write_lock,
        ):
            requested_id = validated_identifier(alias_id, "alias_id")
            memories, embeddings = self._unlinked_speaker_index(requested_id, operation)
            with translate_storage_errors("unlink identity"):
                restored = self._store.identities.unlink_identity(
                    requested_id,
                    memories=memories,
                    embeddings=embeddings,
                    operation=self._identity_split_operation(requested_id),
                )
            self._projection.drain()
            return restored

    def _identity_split_operation(self, alias_id: str) -> StoredOperation | None:
        """Build the CORRECT log row one split commits with, or None when there is no merge.

        An ID that is not a merge alias resolves to itself, and the store refuses the split
        anyway, so there is nothing to log and no row is built.
        """
        with translate_storage_errors("read identity alias"):
            survivor = self._store.identities.resolve_identity_id(alias_id)
        if survivor is None or survivor == alias_id:
            return None
        proposed = MemoryOperation(
            intent=MemoryIntent.CORRECT,
            identity=IdentityChange(identity_id=survivor, moved_ids=(alias_id,)),
            rationale="the host split this face-and-voice merge",
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            trigger=MemoryTrigger.MANUAL.value,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def _unlinked_speaker_index(
        self,
        alias_id: str,
        operation: OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild the indexed speech text that a pending unlink is about to invalidate.

        Reversing a merge moves every segment of the contributed modality back, so text
        indexed under the surviving identity's name would go on quoting a name that is no
        longer the speaker's, and answer to it in search. Only a voice contribution matters
        here: a face name is projected at answer time and never written into stored content.
        """
        with translate_storage_errors("read identity memories"):
            if self._store.identities.identity_alias_modality(alias_id) != "voice":
                return (), ()
            target_id = self._store.identities.resolve_identity_id(alias_id)
        if target_id is None:
            return (), ()
        # Every one of the target's segments moves to the restored identity, and a merge keeps
        # one profile which the target keeps, so no naming assertion is bound to the identity
        # coming back: its projection is a nameless speaker. Asking the store would resolve the
        # alias straight back to the target and answer with the wrong person's name.
        return self.reindex_speech(
            alias_id,
            speaker_name=None,
            operation=operation,
            previous_speaker_id=target_id,
        )

    def deleted_naming_index(
        self,
        memory_id: str,
        operation: OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild indexed speech text for the projection a pending delete will leave behind.

        Deleting the record that names somebody is the ordinary delete path applied to an
        extraordinary record. The name it projected has to stop answering to search in the same
        commit that removes the assertion, so the text is rebuilt from the assertion that will
        still be visible afterwards -- usually none, which reindexes the speaker as nameless.
        Deleting a memory an assertion cites as its only evidence takes the assertion with it,
        so the store answers for those too rather than only for the assertion itself.
        """
        with translate_storage_errors("read identity naming assertion"):
            removed, projection = self._store.identities.naming_projection_after_delete(memory_id)
        factory = self._formation.naming_projection_factory(operation)
        if factory is not None:
            with translate_storage_errors("read affected speech memories"):
                memory_ids = sorted(
                    {
                        memory_id
                        for identity_id, _projected in projection
                        for memory_id in self._store.identities.speaker_memory_ids(identity_id)
                        or ()
                    }
                    - set(removed)
                )
                affected = self._store.records.read_memories(memory_ids)
            # All previewed names travel together: two people may share one
            # document, which must be embedded once with both resulting names.
            return self._embedding.refresh_naming_memories(
                dict(projection), affected, operation=operation
            )
        memories: tuple[StoredMemory, ...] = ()
        embeddings: tuple[StoredEmbedding, ...] = ()
        for identity_id, projected in projection:
            reindexed, revectored = self.reindex_speech(
                identity_id,
                speaker_name=projected,
                operation=operation,
            )
            # The deleted record can be one of the speaker's own memories. Handing it back as a
            # replacement would ask the same transaction to reindex a row it is removing.
            memories += tuple(memory for memory in reindexed if memory.memory_id not in removed)
            embeddings += tuple(
                embedding for embedding in revectored if embedding.memory_id not in removed
            )
        return memories, embeddings

    def projected_identity(self, identity_id: str) -> tuple[str | None, str | None]:
        with translate_storage_errors("read identity naming assertion"):
            return self._store.identities.projected_identity_name(identity_id)

    def reindex_speech(
        self,
        identity_id: str,
        *,
        speaker_name: str | None,
        operation: OperationAssets,
        previous_speaker_id: str | None = None,
        expected: list[StoredMemory] | None = None,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild every indexed memory whose speech projection names one identity.

        This always runs beside a projection recompute, so the text a search matches and the
        name `identities` reports are written from the same assertion in one commit.
        """
        with self._write_lock, translate_storage_errors("read identity memories"):
            memory_ids = self._store.identities.speaker_memory_ids(
                previous_speaker_id or identity_id
            )
            if not memory_ids:
                return (), ()
            stored = self._store.records.read_memories(memory_ids)
            indexed = tuple(memory for memory in stored if has_indexed_speech(memory))
            self._lifecycle.lease_assets(
                tuple(asset for memory in indexed for asset in memory.assets), operation.leased
            )
        if expected is not None:
            expected.extend(indexed)
        if not indexed:
            return (), ()
        return self._embedding.refresh_speaker_memories(
            indexed,
            speaker_id=identity_id,
            speaker_name=speaker_name,
            previous_speaker_id=previous_speaker_id,
            update_operation=previous_speaker_id is None,
            operation=operation,
        )

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        requested_id = validated_identifier(identity_id, "identity_id")
        normalized_name = validated_identity_name(name)
        normalized_relationship = (
            None
            if relationship is None
            else validated_identity_name(relationship, "identity relationship")
        )
        with (
            self._trace("sensemem.register_identity", kind="operation"),
            self._lifecycle.operation(model_work=True) as operation,
            # Naming shares the policy lock with consolidation. Its model preparation runs
            # before formation acquires the write lock for the atomic commit.
            self._formation_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                if normalized_id is None:
                    identity_exists = False
                    memory_ids = None
                else:
                    identity_exists = (
                        self._store.identities.identity_memory_ids(normalized_id) is not None
                    )
                    speaker_memory_ids = self._store.identities.speaker_memory_ids(normalized_id)
                    memory_ids = () if speaker_memory_ids is None else speaker_memory_ids
            if not identity_exists:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            assert normalized_id is not None
            assert memory_ids is not None
            # `_assert_identity_name` prepares the affected speech documents before it commits,
            # then writes the assertion, registry projection, documents, vectors, and log in one
            # SQLite transaction.
            self._assert_identity_name(
                normalized_id,
                normalized_name,
                relationship=(
                    normalized_relationship
                    if normalized_relationship is not None
                    else self._recorded_relationship(normalized_id)
                ),
                operation=operation,
            )

    def _recorded_relationship(self, identity_id: str) -> str | None:
        _name, relationship = self.projected_identity(identity_id)
        return relationship

    def _assert_identity_name(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None,
        operation: OperationAssets,
    ) -> str:
        """Record naming one person as a typed ENTITY claim bound to that identity.

        This is what makes a name retrievable knowledge instead of a label on a row, and it is
        the versioned thing `identities.name` projects. It rests on the host's authority rather
        than on a model, so it needs neither a former nor a consolidator to be configured, and
        it carries no evidence: nothing was observed, somebody said so.

        It is logged like any other control-plane operation, so naming is auditable through
        `operations()` and reversible through `rollback()`. The logged intent is `IDENTIFY` and
        the claim travels on the operation itself, so `operations()` never has to pass an
        identity ID off as a memory ID in `evidence_ids`. A host cites nothing: nothing was
        observed, somebody said so, so the evidence set is empty.

        Re-asserting the same name and relationship is idempotent, because the derived memory ID
        is a function of the claim; a different name supersedes through the shared lineage.
        """
        proposal = naming_proposal(name, relationship, basis=EvidenceBasis.USER_STATEMENT)
        now = datetime.now(timezone.utc)
        context = formation_context(
            None,
            proposal,
            model_id=None,
            recipe=NAMING_RECIPE,
            recorded_at=now,
            identity_id=identity_id,
        )
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, operation),
                occurred_at=None,
                occurred_end=None,
                metadata=None,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                identity_id,
                proposal,
                recipe=NAMING_RECIPE,
                context=context,
            ),
            context=context,
        )
        proposed = MemoryOperation(
            intent=MemoryIntent.IDENTIFY,
            claim=IdentityClaim(identity_id=identity_id, name=name, relationship=relationship),
            rationale="the host registered this name",
        )
        base_key = operation_key(proposed, recipe=NAMING_RECIPE)
        with translate_storage_errors("check a naming operation"):
            key = self._store.control.naming_operation_key(base_key, prepared.memory_id)
        # `_register_identity` holds `_formation_lock` for this whole call (the same lock
        # `add()`'s automatic formation holds across its own `_commit_formation`, see
        # `_form_sources`), so this IDENTIFY commit cannot land while a consolidate apply pass
        # is in progress either.
        for attempt in range(4):
            expected: list[StoredMemory] = []
            try:
                self._formation.commit(
                    ((prepared, None, proposal.confidence),),
                    (),
                    assets=operation,
                    completed_at=now,
                    recipe=NAMING_RECIPE,
                    # Re-asserting a standing name changes nothing, so it logs nothing: the operation
                    # key is already active and the log's unique index would refuse it anyway.
                    operation=(
                        None
                        if key is None
                        else StoredOperation(
                            operation_key=key,
                            intent=proposed.intent.value,
                            trigger=MemoryTrigger.MANUAL.value,
                            recipe=NAMING_RECIPE,
                            operation_json=dump_operation(proposed),
                            applied_at=now,
                        )
                    ),
                    projection_identity_id=identity_id,
                    projection_factory=partial(
                        self.reindex_speech,
                        identity_id,
                        speaker_name=name,
                        operation=operation,
                        expected=expected,
                    ),
                    projection_sources=expected,
                )
            except StaleOperationError:
                if attempt == 3:
                    raise StorageError(
                        "identity projection is busy; retry the update",
                        reason="capacity_exceeded",
                        stage="admission",
                    ) from None
            else:
                break
        return prepared.memory_id

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        requested_id = validated_identifier(identity_id, "identity_id")
        with (
            self._trace("sensemem.forget_identity", kind="operation"),
            self._lifecycle.operation(model_work=True) as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: an irreversible erasure must not land while a
            # consolidate apply pass is in progress, and taking the two locks in a different
            # order here would deadlock against that path.
            self._formation_lock,
            self._write_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                known = normalized_id is not None and (
                    self._store.identities.identity_memory_ids(normalized_id) is not None
                )
            if normalized_id is None or not known:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            # `speaker_name=None` is the projection of a person with no naming assertion left,
            # which is what erasure makes them. The rebuilt documents are handed to the store so
            # the erasure and the reindex commit in one transaction: a crash between them would
            # leave the name searchable for a person whose template was already gone.
            memories, embeddings = self.reindex_speech(
                normalized_id,
                speaker_name=None,
                operation=operation,
            )
            with translate_storage_errors("forget identity"):
                erased = self._store.identities.forget_identity(
                    normalized_id,
                    memories=memories,
                    embeddings=embeddings,
                    operation=self._identity_erasure_operation(normalized_id),
                )
            if erased is None:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            erasure, orphaned = erased
            self._lifecycle.queue_asset_cleanup(orphaned)
            self._projection.drain()
            return erasure

    def _identity_erasure_operation(self, identity_id: str) -> StoredOperation:
        """Build the irreversible log row one identity erasure commits with.

        Erasure is physical forgetting of a person, so the row is audit history and nothing
        else: it carries the identity, its aliases, and the naming assertions the erasure
        deleted -- ids and counts, never content -- and `rollback()` refuses it. That is the
        marker that separates this from the cognitive `FORGET` a row with `target_ids` records.
        """
        with translate_storage_errors("read identity aliases"):
            members = self._store.identities.identity_equivalence_class(identity_id)
        proposed = MemoryOperation(
            intent=MemoryIntent.FORGET,
            identity=IdentityChange(
                identity_id=identity_id,
                moved_ids=() if members is None else members[1:],
            ),
            rationale="the data subject asked to be erased",
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            trigger=MemoryTrigger.MANUAL.value,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def bind_speaker_names(self, operation: OperationAssets) -> None:
        """Register the staged names, now that the memory carrying the voices is readable.

        `register_identity` in all but name -- it lands the same `IDENTIFY` assertion through the
        same commit, so it is auditable through `operations()` and reversible through
        `rollback()` -- but it resolves and checks the person itself rather than raising on one
        who has since been merged away, and it holds the existing operation instead of nesting a
        second one inside this write.

        A different standing name is never overwritten. The name came from a model reading a
        transcript, and one mishearing would rewrite every document that person appears in; the
        host's own `register_identity` stays the only thing that can replace a name. Nothing here
        links identities either: `identity_link_min_assets` governs who is the same person, and a
        name is not evidence about that.
        """
        bindings = dict(operation.speaker_names)
        operation.speaker_names.clear()
        bound = 0
        refused = 0
        for identity_id, name in bindings.items():
            with self._formation_lock:
                with translate_storage_errors("read identity profile"):
                    profile = self._store.identities.identity_profile(identity_id)
                try:
                    proposed = None if profile is None else validated_identity_name(name)
                except ValidationError:
                    proposed = None
                if profile is None or proposed is None:
                    # A merged/deleted identity or invalid name named nobody.
                    operation.speaker_names_refused += 1
                    continue
                if profile.name == proposed:
                    continue
                if profile.name is not None:
                    refused += 1
                    _LOGGER.warning(
                        "a distilled name conflicts with an existing name; "
                        "keeping the registered name"
                    )
                    continue
                self._assert_identity_name(
                    profile.identity_id,
                    proposed,
                    relationship=profile.relationship,
                    operation=operation,
                )
            bound += 1
        refused += operation.speaker_names_refused
        operation.speaker_names_refused = 0
        if bound or refused:
            with self._trace("sensemem.identity.names", kind="stage") as span:
                span.set_attribute(IDENTITY_NAMES_BOUND, bound)
                span.set_attribute(IDENTITY_NAMES_REFUSED, refused)

    def stage_speaker_names(
        self,
        descriptions: Mapping[str, VisualDescription],
        operation: OperationAssets,
    ) -> None:
        """Collect the names a caption's facts stated for this write's own diarised speakers.

        A stated name is the one distillation that is more than indexed text: it re-keys the
        person in every document that mentions them, now and for every later clip that resolves
        to the same voice, which is what carries an identity across clips at all.

        Only a label this asset's own recognizer produced can be resolved, so a fact naming a
        speaker who is not in this clip is dropped rather than guessed at -- counted into
        `operation.speaker_names_refused`, the same drop tally `_bind_speaker_names` adds to and
        reports, so a fact naming a label nobody produced is not simply invisible. Cached
        descriptions are read too: re-ingesting a corpus must reach the same named store as the
        first ingest, and re-asserting a standing name is a no-op.
        """
        for asset_id, description in descriptions.items():
            segments = operation.speech_segments.get(asset_id)
            if not segments:
                continue
            identities = {label: identity for identity, label in speaker_labels(segments).items()}
            for proposed in description.speaker_names:
                identity_id = identities.get(proposed.speaker_label)
                if identity_id is None:
                    operation.speaker_names_refused += 1
                    continue
                operation.speaker_names.setdefault(identity_id, proposed.name)

    def known_speaker_names(self, asset_id: str, operation: OperationAssets) -> dict[str, str]:
        """Map this asset's `speaker_N` labels to whatever name is already known for them.

        "Known now" is a name this very batch's own facts just staged (`operation.speaker_names`)
        or a name an already-registered identity carries on its recognized segments; a name a
        later write asserts is not, which is the write-time/retrieval-time split documented on
        `project_fact_labels`.
        """
        segments = operation.speech_segments.get(asset_id)
        if not segments:
            return {}
        names: dict[str, str] = {}
        for identity_id, label in speaker_labels(segments).items():
            name = operation.speaker_names.get(identity_id)
            if name is None:
                name = next(
                    (
                        segment.speaker_name
                        for segment in segments
                        if segment.speaker_id == identity_id and segment.speaker_name
                    ),
                    None,
                )
            if name:
                names[label] = name
        return names

    def perceive(
        self,
        assets: Sequence[StoredAsset],
        operation: OperationAssets,
        *,
        link: bool = True,
        kinds: frozenset[PerceptionKind] = frozenset({"speech", "faces"}),
    ) -> None:
        """Recognize independent voice/face inputs concurrently, then authorize identity links."""
        speech = self._speech.answer_speech_assets(assets) if "speech" in kinds else ()
        faces = self._vision.answer_face_assets(assets) if "faces" in kinds else ()
        if speech and faces:
            with self._execution.overlap_speech(
                self._speech.recognize, speech, operation
            ) as pending:
                self._vision.recognize(faces, operation)
                pending.result()
        elif speech:
            self._speech.recognize(speech, operation)
        elif faces:
            self._vision.recognize(faces, operation)
        if link:
            with self._formation_lock:
                for asset in faces:
                    self._link_asset_identity(asset.asset_id, operation)

    def _identity_link_is_corroborated(
        self,
        speaker_id: str,
        face_id: str,
        asset_id: str,
    ) -> bool:
        """Record this asset's voice-and-face co-occurrence and report whether it is enough.

        One asset's co-occurrence is not evidence that one person produced both. Egocentric
        capture is the adversarial case: the wearer speaks while a different person's face
        fills the frame, so a single clip would bind the listener's face to the wearer's voice
        and nothing downstream could tell.

        Counting assets raises the price of that mistake but does not prevent it. A wearer
        talks to the same person across many clips, so the wrong pair accumulates as fast as a
        genuine speaker's, and measurement on synthetic egocentric traffic confirms it: at the
        default of two assets the wearer still binds to an interlocutor's face under every
        ingestion order tried. What the count does buy is that a face seen once is never
        bound, and `_link_asset_identity` keeps the damage to that one bind by refusing to let
        an identity holding both modalities absorb anything further.
        """
        observed = self._store.identities.record_identity_link_evidence(
            speaker_id, face_id, asset_id
        )
        corroborated = observed >= self._settings.identity_link_min_assets
        with self._trace("sensemem.identity.link", kind="stage") as span:
            span.set_attribute(IDENTITY_EVIDENCE_ASSETS, observed)
            span.set_attribute(IDENTITY_EVIDENCE_REQUIRED, self._settings.identity_link_min_assets)
            span.set_attribute(IDENTITY_LINKED, corroborated)
        return corroborated

    def _identity_link_operation(self, plan: IdentityLink) -> StoredOperation:
        """Build the MERGE log row one corroborated cross-modal bind commits with."""
        proposed = MemoryOperation(
            intent=MemoryIntent.MERGE,
            identity=IdentityChange(
                identity_id=plan.target_id,
                moved_ids=(plan.source_id,),
            ),
            rationale=(
                f"one voice and one face co-occurred in at least "
                f"{self._settings.identity_link_min_assets} assets"
            ),
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            # Independent evidence accumulated until it corroborated the pair, which is exactly
            # what this trigger names -- no clock and no host request is involved.
            trigger=MemoryTrigger.EVIDENCE.value,
            model_id=self._backends.face_model,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def _link_asset_identity(self, asset_id: str, operation: OperationAssets) -> None:
        for attempt in range(4):
            try:
                self._link_asset_identity_once(asset_id, operation)
            except StaleOperationError:
                if attempt == 3:
                    raise StorageError(
                        "identity projection is busy; retry the update",
                        reason="capacity_exceeded",
                        stage="admission",
                    ) from None
            else:
                return

    def _link_asset_identity_once(self, asset_id: str, operation: OperationAssets) -> None:
        speaker_ids = {
            segment.speaker_id
            for segment in operation.speech_segments.get(asset_id, ())
            if segment.speaker_id is not None
        }
        face_ids = {
            observation.identity_id for observation in operation.face_observations.get(asset_id, ())
        }
        if len(speaker_ids) != 1 or len(face_ids) != 1:
            return
        speaker_id, face_id = next(iter(speaker_ids)), next(iter(face_ids))
        # Withheld or withdrawn consent stops the merge on either side. Fusing a voice and a
        # face is a new claim about a person -- that these two templates are one human -- so it
        # is exactly the processing a refusal refuses, and unlike enrolment it is not something
        # answering a question needs.
        with translate_storage_errors("read identity consent"):
            restrained = self._store.identities.restrained_identities()
        if {speaker_id, face_id} & restrained:
            return
        if speaker_id == face_id:
            # Already one identity. Recording this would store an identity co-occurring with
            # itself, once per asset forever, and no such pair can ever yield a plan.
            return
        if not self._identity_link_is_corroborated(speaker_id, face_id, asset_id):
            return
        # Only a voice-only and a face-only identity may fuse here. Letting a fragment rejoin
        # an identity that already holds its modality is what turns one wrong cross-modal bind
        # into a cascade: once a wearer's voice owns one interlocutor's face, that identity
        # holds both modalities, and every later fragment (the interlocutor's own voice, then
        # the next interlocutor's face) is a fragment rejoining it. Measured on synthetic
        # egocentric traffic (one off-camera wearer, three interlocutors, random ingestion
        # order), permitting it collapsed all four people into one identity every time;
        # refusing it capped the damage at the single unavoidable first bind and raised
        # correct merges from 0/3 to 2/3. `LocalStore` still offers the wider merge to a
        # caller that has established the claim some other way.
        plan = self._store.identities.identity_link_plan(speaker_id, face_id)
        if plan is None:
            return
        expected: list[StoredMemory] | None = None
        memories: tuple[StoredMemory, ...] = ()
        embeddings: tuple[StoredEmbedding, ...] = ()
        if self._settings.index_speech and plan.source_id == speaker_id:
            expected = []
            memories, embeddings = self.reindex_speech(
                plan.target_id,
                speaker_name=plan.name,
                previous_speaker_id=plan.source_id,
                operation=operation,
                expected=expected,
            )
        with self._write_lock, translate_storage_errors("link recognized identities"):
            linked = self._store.identities.link_identities(
                speaker_id,
                face_id,
                expected=plan,
                memories=memories,
                embeddings=embeddings,
                projection_sources=expected,
                operation=self._identity_link_operation(plan),
            )
            if linked is None:
                return
            self._projection.drain()
        linked_ids = {plan.target_id, plan.source_id}
        for cached_asset, segments in tuple(operation.speech_segments.items()):
            operation.speech_segments[cached_asset] = tuple(
                replace(
                    segment,
                    speaker_id=plan.target_id,
                    speaker_name=plan.name,
                )
                if segment.speaker_id in linked_ids
                else segment
                for segment in segments
            )
        for cached_asset, observations in tuple(operation.face_observations.items()):
            operation.face_observations[cached_asset] = tuple(
                replace(
                    observation,
                    identity_id=plan.target_id,
                    identity_name=plan.name,
                )
                if observation.identity_id in linked_ids
                else observation
                for observation in observations
            )
