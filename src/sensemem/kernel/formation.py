"""Formation: grounding typed proposals in committed observations.

Formation runs the formation backend and commits accepted proposals with evidence, versions,
and vectors in one store transaction. Shared proposal rules live in formation_rules.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from threading import Condition

from opentelemetry.trace import Tracer

from sensemem._telemetry import mark_model_requests, record_formation_refusals
from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.infrastructure.local.store._naming_projection import NamingPreparationRequired
from sensemem.infrastructure.local.store.errors import ProtectedMemoryError, StaleOperationError
from sensemem.infrastructure.local.store.rows import StoredEmbedding, StoredMemory, StoredOperation
from sensemem.kernel.affect import affect_witness_refusal, resolve_affect_proposal
from sensemem.kernel.content import (
    PreparedMemory,
    decode_metadata,
    embedding_row_id,
    observation_from_record,
    prepare_memory,
)
from sensemem.kernel.contracts import Backends
from sensemem.kernel.derived import prepared_from_stored
from sensemem.kernel.embedding import DOCUMENT_TASK, Embedding
from sensemem.kernel.formation_identity import claim_identity
from sensemem.kernel.formation_identity import history_identity_refusal as _history_identity_refusal
from sensemem.kernel.formation_rules import (
    agreed_inheritance,
    formation_context,
    formation_evidence_active,
    formation_memory_id,
    formation_memory_type,
    formation_refusal,
)
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import OperationAssets
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.projection import Projection
from sensemem.kernel.routing import route_input
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.kernel.settings import ResolvedSettings
from sensemem.kernel.temporal import intervals_overlap
from sensemem.kernel.tracing import Traced
from sensemem.models.base import EmbedTask, FormationInput, ModelInput
from sensemem.records import MemoryRecord
from sensemem.semantics import FormationProposal, RecordSemantics
from sensemem.types import RecordKind

_EMPTY_METADATA_JSON = "{}"


def _cited_sources(pairs: Sequence[tuple[PreparedMemory, str | None, float]]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source_id
            for prepared, _source, _confidence in pairs
            if isinstance(prepared.context, RecordSemantics)
            for source_id in prepared.context.evidence_ids
        )
    )


def _proposal_sources(
    proposal: FormationProposal,
    source: FormationInput,
    batch: Mapping[str, FormationInput],
    records: Mapping[str, MemoryRecord],
) -> tuple[str, ...] | None:
    evidence = (source.memory_id,) if proposal.evidence_ids is None else proposal.evidence_ids
    history_ids = {record.id for record in source.history}
    if source.memory_id not in evidence or any(
        item not in batch and item not in history_ids for item in evidence
    ):
        raise ModelError(
            "formation proposal cited an observation outside its batch",
            reason="response_invalid",
            stage="form",
        )
    return evidence if all(formation_evidence_active(records[item]) for item in evidence) else None


class RejectedOperation(Exception):
    """Internal signal that kernel policy refused one proposed operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _grounded_formation_pairs(
    pairs: Sequence[tuple[PreparedMemory, str | None, float]],
) -> tuple[tuple[tuple[PreparedMemory, str | None, float], ...], int]:
    """Keep the first of each conflicting proposal from one source; return the rest as refused.

    Two proposals of one record that disagree, and two contradictory states in one lineage, are
    grounding faults inside a single model response -- not damage to the envelope. The source is
    already committed when formation runs, so failing the write would report a stored observation
    as unwritten and every retry would re-run the model and fail identically.
    """
    kept: list[tuple[PreparedMemory, str | None, float]] = []
    refused = 0
    seen: dict[tuple[str, str | None], PreparedMemory] = {}
    states: dict[tuple[str | None, str], list[RecordSemantics]] = {}
    for pair in pairs:
        prepared, source_id, _confidence = pair
        context = prepared.context
        if isinstance(context, RecordSemantics) and prepared.memory_id in context.evidence_ids:
            # A source-independent ID can restate a historical claim it also cites. Its whole
            # conjunction is circular; removing only the self premise would invent support.
            refused += 1
            continue
        if seen.setdefault((prepared.memory_id, source_id), prepared) != prepared:
            refused += 1
            continue
        if isinstance(context, RecordSemantics) and context.kind is RecordKind.STATE:
            lineage = states.setdefault(
                (source_id, context.lineage_id or prepared.memory_id),
                [],
            )
            if any(
                standing.value != context.value
                and intervals_overlap(
                    standing.valid_from,
                    standing.valid_until,
                    context.valid_from,
                    context.valid_until,
                )
                for standing in lineage
            ):
                refused += 1
                continue
            lineage.append(context)
        kept.append(pair)
    return tuple(kept), refused


def _agreed_columns(
    values: Sequence[PreparedMemory],
    stored: StoredMemory | None,
) -> PreparedMemory:
    """Return the first proposal of one derived record carrying only agreed inherited columns.

    Same rule as agreed_inheritance, applied where the sources arrive one write at a time: the
    place and the metadata survive only while every proposal of this record -- and the row already
    written for it -- says the same thing.
    """
    places = {value.place_id for value in values}
    tags = {value.metadata_json for value in values}
    if stored is not None:
        places.add(stored.place_id)
        tags.add(stored.metadata_json)
    return replace(
        values[0],
        place_id=values[0].place_id if len(places) == 1 else None,
        metadata_json=values[0].metadata_json if len(tags) == 1 else _EMPTY_METADATA_JSON,
    )


class Formation(Traced):
    """Run the formation backend over committed sources and commit accepted proposals."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        settings: ResolvedSettings,
        backends: Backends,
        materializer: Materializer,
        embedding: Embedding,
        projection: Projection,
        hydrator: Hydrator,
    ) -> None:
        super().__init__(tracer)
        self._formation_lock = storage.formation_lock
        self._store = storage.store
        self._settings = settings
        self._write_lock = storage.write_lock
        self._backends = backends
        self._materializer = materializer
        self._embedding = embedding
        self._projection = projection
        self._hydrator = hydrator
        self._forming: set[str] = set()
        self._formation_claims = Condition()

    def _history_for(
        self, source: MemoryRecord, excluded: Sequence[str]
    ) -> tuple[tuple[MemoryRecord, ...], bool]:
        limit = self._settings.formation_history_max_rows
        if limit == 0:
            return (), False
        with translate_storage_errors("read formation history"):
            stored = self._store.records.formation_history(
                known_before=source.created_at, exclude_ids=excluded, limit=limit + 1
            )
        history: list[MemoryRecord] = []
        remaining = self._settings.formation_history_budget
        truncated = len(stored) > limit
        for record in stored[:limit]:
            if len(record.content) > remaining:
                # Do not hide a recent correction to make room for a shorter old assertion.
                truncated = True
                break
            hydrated = self._hydrator.memory_record(record)
            history.append(hydrated)
            remaining -= len(record.content)
        return tuple(reversed(history)), truncated

    def _formation_inputs(self, sources: Sequence[MemoryRecord]) -> tuple[FormationInput, ...]:
        excluded = tuple(source.id for source in sources)
        result = []
        for source in sources:
            history, truncated = self._history_for(source, excluded)
            content = ModelInput(text=source.content, assets=source.assets, parts=source.parts)
            if content.modalities - self._backends.formation_capabilities:
                with translate_storage_errors("read formation representation"):
                    stored = self._store.records.read_memory(source.id)
                if stored is not None:
                    try:
                        content = route_input(
                            prepared_from_stored(stored).content,
                            content,
                            self._backends.formation_capabilities,
                            "formation",
                        )
                    except ModelError as error:
                        if error.reason != "unsupported_modality":
                            raise
            result.append(
                FormationInput(
                    memory_id=source.id,
                    content=content,
                    context=observation_from_record(source),
                    history=history,
                    history_truncated=truncated,
                    occurred_at=source.occurred_at,
                    occurred_end=source.occurred_end,
                    created_at=source.created_at,
                )
            )
        return tuple(result)

    def form_sources(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
    ) -> None:
        if self._backends.former is None or not sources:
            return
        claimed = {source.id for source in sources}
        with self._formation_claims:
            while claimed & self._forming:
                self._formation_claims.wait()
            self._forming.update(claimed)
        try:
            self._form_sources_claimed(sources, operation=operation)
        finally:
            with self._formation_claims:
                self._forming.difference_update(claimed)
                self._formation_claims.notify_all()

    def _form_sources_claimed(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
    ) -> None:
        assert self._backends.former is not None
        pending = tuple({source.id: source for source in sources}.values())
        pending = tuple(
            source
            for source in pending
            if not self._store.semantics.formation_completed(
                source.id, self._backends.formation_space
            )
        )
        if not pending:
            return
        all_inputs = self._formation_inputs(pending)
        inputs = tuple(
            value
            for value in all_inputs
            if value.content.modalities <= self._backends.formation_capabilities
        )
        if not inputs:
            return
        try:
            with self._model_trace(
                "formation",
                "form",
                model=self._backends.formation_model,
                batch_size=len(inputs),
                modalities=(modality for value in inputs for modality in value.content.modalities),
            ):
                mark_model_requests(1)
                proposals_by_source = self._backends.former.form(inputs)
        except SenseMemError:
            raise
        except Exception as error:
            raise ModelError(
                "automatic memory formation failed",
                reason="model_failed",
                stage="form",
            ) from error
        if (
            not isinstance(proposals_by_source, tuple)
            or len(proposals_by_source) != len(inputs)
            or any(
                not isinstance(values, tuple)
                or any(not isinstance(value, FormationProposal) for value in values)
                for values in proposals_by_source
            )
        ):
            raise ModelError(
                "formation backend returned an invalid batch",
                reason="response_invalid",
                stage="form",
            )
        with self._formation_lock:
            self._apply_proposals(inputs, proposals_by_source, pending, operation=operation)

    def _apply_proposals(  # noqa: C901 - two-stage grounding shares one formation transaction
        self,
        inputs: Sequence[FormationInput],
        proposals_by_source: tuple[tuple[FormationProposal, ...], ...],
        pending: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
    ) -> None:
        proposals_by_id: dict[str, tuple[FormationProposal, ...]] = {
            value.memory_id: proposals
            for value, proposals in zip(inputs, proposals_by_source, strict=True)
        }

        now = datetime.now(timezone.utc)
        pairs: list[tuple[PreparedMemory, str | None, float]] = []
        inputs_by_id = {value.memory_id: value for value in inputs}
        formed_sources = tuple(source for source in pending if source.id in inputs_by_id)
        batch_records = {source.id: source for source in formed_sources}
        refused = 0
        local_prepared: dict[tuple[str, str], PreparedMemory] = {}
        for source in formed_sources:
            # One ID can have different states at different observations' causal cutoffs. Never
            # let another target's background overwrite the snapshot this proposal actually saw.
            evidence_by_id = ChainMap(
                batch_records,
                {record.id: record for record in inputs_by_id[source.id].history},
            )
            proposals = proposals_by_id[source.id]
            # Application data and the symbolic place travel with the knowledge formed from them:
            # a host that filters recall by metadata expects the tag on the observation to hold
            # for what was learned from it, and place_id is a hard filter, so knowledge formed
            # from an observation in a room has to stand in that room too -- otherwise a
            # place-scoped question sees the raw observation and none of the entities, states, or
            # relations formed from it. One source agrees with itself; a record several sources
            # share keeps only what they all say, which _commit_formation settles.
            for proposal in proposals:
                if proposal.affect_link is not None:
                    continue
                if (
                    proposal.local_id is not None
                    and sum(item.local_id == proposal.local_id for item in proposals) != 1
                ):
                    refused += 1
                    continue
                evidence_ids = _proposal_sources(
                    proposal, inputs_by_id[source.id], inputs_by_id, evidence_by_id
                )
                if evidence_ids is None or (
                    self._settings.formation_history_max_rows
                    and _history_identity_refusal(
                        proposal,
                        inputs_by_id[source.id],
                        tuple(evidence_by_id[item] for item in evidence_ids),
                    )
                ):
                    refused += 1
                    continue
                # The source is already committed; drop a misgrounded proposal without failing
                # its write. The batch envelope is validated before any proposal is applied.
                refusal = formation_refusal(proposal, inputs_by_id[source.id])
                if refusal is not None:
                    refused += 1
                    continue
                proposal = replace(proposal, evidence_ids=evidence_ids)
                if affect_witness_refusal(proposal, evidence_by_id):
                    refused += 1
                    continue
                cited = tuple(evidence_by_id[item] for item in evidence_ids)
                try:
                    identity_id = claim_identity(proposal, cited)
                except ValidationError:
                    refused += 1
                    continue
                place_id, metadata = agreed_inheritance(cited)
                prepared = prepare_memory(
                    self._materializer.materialize_content(proposal.content, operation),
                    occurred_at=source.occurred_at,
                    occurred_end=source.occurred_end,
                    metadata=metadata,
                    memory_type=formation_memory_type(proposal.kind),
                )
                context = formation_context(
                    source,
                    proposal,
                    model_id=self._backends.formation_model,
                    recipe=self._backends.formation_space,
                    recorded_at=now,
                    identity_id=identity_id,
                )
                context = replace(context, evidence_ids=evidence_ids)
                prepared = replace(
                    prepared,
                    memory_id=formation_memory_id(
                        source.id,
                        proposal,
                        recipe=self._backends.formation_space,
                        context=context,
                    ),
                    context=context,
                    place_id=place_id,
                )
                pairs.append((prepared, source.id, proposal.confidence))
                if proposal.local_id is not None:
                    local_prepared[(source.id, proposal.local_id)] = prepared
        ordinary, conflicting = _grounded_formation_pairs(pairs)
        pairs = list(ordinary)
        accepted = {
            (source_id, prepared.memory_id): prepared for prepared, source_id, _ in ordinary
        }
        for source in formed_sources:
            offered = inputs_by_id[source.id]
            new_records: dict[str, MemoryRecord] = {}
            local_ids: dict[str, str] = {}
            for (source_id, local_id), prepared in local_prepared.items():
                if (
                    source_id != source.id
                    or accepted.get((source_id, prepared.memory_id)) != prepared
                ):
                    continue
                local_ids[f"proposal:{local_id}"] = prepared.memory_id
                assert isinstance(prepared.context, RecordSemantics)
                new_records[prepared.memory_id] = MemoryRecord(
                    id=prepared.memory_id,
                    content=prepared.content.text,
                    created_at=now,
                    occurred_at=prepared.occurred_at,
                    occurred_end=prepared.occurred_end,
                    metadata=decode_metadata(prepared.metadata_json),
                    memory_type=prepared.memory_type,
                    context=prepared.context,
                    place_id=prepared.place_id,
                    parts=prepared.content.parts,
                )
            evidence_by_id = ChainMap(
                batch_records, {record.id: record for record in offered.history}, new_records
            )
            for proposal in proposals_by_id[source.id]:
                if proposal.affect_link is None:
                    continue
                evidence_ids = _proposal_sources(proposal, offered, inputs_by_id, evidence_by_id)
                if evidence_ids is None or formation_refusal(proposal, offered) is not None:
                    refused += 1
                    continue
                resolved = resolve_affect_proposal(
                    replace(proposal, evidence_ids=evidence_ids), evidence_by_id, local_ids
                )
                if resolved is None or affect_witness_refusal(resolved, evidence_by_id):
                    refused += 1
                    continue
                proposal = resolved
                assert proposal.evidence_ids is not None
                cited = tuple(evidence_by_id[item] for item in proposal.evidence_ids)
                if _history_identity_refusal(proposal, offered, cited):
                    refused += 1
                    continue
                link = proposal.affect_link
                assert link is not None
                focal = evidence_by_id[link.affect_id]
                assert focal.context is not None
                _, metadata = agreed_inheritance(cited)
                context = replace(
                    formation_context(
                        source,
                        proposal,
                        model_id=self._backends.formation_model,
                        recipe=self._backends.formation_space,
                        recorded_at=now,
                        identity_id=focal.context.identity_id,
                    ),
                    valid_from=proposal.valid_from,
                    valid_until=proposal.valid_until,
                    spatial=focal.context.spatial,
                    evidence_ids=proposal.evidence_ids,
                )
                prepared = replace(
                    prepare_memory(
                        self._materializer.materialize_content(proposal.content, operation),
                        occurred_at=focal.occurred_at,
                        occurred_end=focal.occurred_end,
                        metadata=metadata,
                        memory_type=formation_memory_type(proposal.kind),
                    ),
                    memory_id=formation_memory_id(
                        source.id, proposal, recipe=self._backends.formation_space, context=context
                    ),
                    context=context,
                    place_id=focal.place_id,
                )
                pairs.append((prepared, source.id, proposal.confidence))
        grounded, link_conflicts = _grounded_formation_pairs(pairs)
        record_formation_refusals(refused + conflicting + link_conflicts)
        self._commit_targets(grounded, formed_sources, operation=operation, completed_at=now)

    def _commit_targets(
        self,
        grounded: Sequence[tuple[PreparedMemory, str | None, float]],
        targets: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
        completed_at: datetime,
    ) -> None:
        remaining = tuple(grounded)
        try:
            while True:
                try:
                    self.commit(
                        remaining,
                        targets,
                        assets=operation,
                        completed_at=completed_at,
                        joint_evidence_clauses=True,
                        require_active=tuple(source.id for source in targets),
                        require_visible=tuple(
                            memory_id
                            for memory_id in _cited_sources(remaining)
                            if memory_id not in {pair[0].memory_id for pair in remaining}
                        ),
                        preserve_response_policies=True,
                    )
                except ProtectedMemoryError as error:
                    # The failed transaction committed nothing. Keep valid siblings together
                    # so independent assertions retain their original batch lineage boundary.
                    accepted = tuple(
                        pair for pair in remaining if pair[0].memory_id not in error.memory_ids
                    )
                    assert len(accepted) < len(remaining)
                    record_formation_refusals(len(remaining) - len(accepted))
                    remaining = accepted
                else:
                    return
        except StaleOperationError as error:
            raise ModelError(
                "formation evidence changed before commit; the observation remains stored",
                reason="response_invalid",
                stage="form",
            ) from error

    def commit(
        self,
        pairs: Sequence[tuple[PreparedMemory, str | None, float]],
        sources: Sequence[MemoryRecord],
        *,
        assets: OperationAssets,
        completed_at: datetime,
        recipe: str | None = None,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
        require_visible: Sequence[str] = (),
        preserve_response_policies: bool = False,
        projection_identity_id: str | None = None,
        joint_evidence_clauses: bool = False,
    ) -> StoredOperation | None:
        """Embed and commit derived records; return the log row when one was requested.

        A None source ID states that the record rests on no other memory, which is what an
        asserted claim such as a naming assertion is: it cites nothing and needs no evidence row.
        """
        recipe = self._backends.formation_space if recipe is None else recipe
        ordered_pairs = tuple(sorted(pairs, key=lambda value: (value[0].memory_id, value[1] or "")))
        by_id: dict[str, list[PreparedMemory]] = {}
        for prepared, _source, _score in ordered_pairs:
            by_id.setdefault(prepared.memory_id, []).append(prepared)
        with translate_storage_errors("check formed memories"):
            existing = self._store.records.read_memories(tuple(sorted(by_id)))
        existing_by_id = {value.memory_id: value for value in existing}
        unique = tuple(
            _agreed_columns(values, existing_by_id.get(memory_id))
            for memory_id, values in sorted(by_id.items())
        )
        # A record whose ID excludes its source -- an entity, a relation, an inferred trait -- is
        # written once and every later source only adds evidence to it, so what the first source
        # said would otherwise stand for evidence that disagrees. The columns it no longer agrees
        # on are cleared on the stored row inside the same transaction; nothing is re-embedded,
        # because neither column reaches the index.
        narrowed = tuple(
            (value.memory_id, value.place_id, value.metadata_json)
            for value in unique
            if (row := existing_by_id.get(value.memory_id)) is not None
            and (value.place_id, value.metadata_json) != (row.place_id, row.metadata_json)
        )
        missing = tuple(value for value in unique if value.memory_id not in existing_by_id)
        parts = tuple(
            (memory, object_part, model_input)
            for memory in missing
            for object_part, model_input in enumerate(
                self._embedding.build_retrieval_inputs(memory.content)
            )
        )
        vectors = self._embedding.embed(
            tuple(model_input for _memory, _part, model_input in parts),
            task=EmbedTask.DOCUMENT,
        )
        stored_list: list[StoredMemory] = []
        for memory in unique:
            assert isinstance(memory.context, RecordSemantics)
            # Existing records travel too: what they skip is the embedding, not the write. The
            # store decides whether restating a claim it already holds is a no-op or a new
            # version, which is how re-asserting a retired one -- a name changed back, a
            # registration repeated after rollback -- reaches the lineage at all.
            stored_list.append(
                StoredMemory(
                    memory_id=memory.memory_id,
                    content=memory.content.text,
                    modality=memory.content.modality.value,
                    memory_type=memory.memory_type.value,
                    assets=(),
                    metadata_json=memory.metadata_json,
                    occurred_at=memory.occurred_at,
                    occurred_end=memory.occurred_end,
                    created_at=completed_at,
                    updated_at=completed_at,
                    place_id=memory.place_id,
                    context=memory.context,
                )
            )
        stored = tuple(stored_list)
        embeddings = tuple(
            StoredEmbedding(
                embedding_id=embedding_row_id(memory.memory_id, object_part),
                memory_id=memory.memory_id,
                values=vector,
                model_id=self._backends.embedding_model,
                space_id=self._backends.space_id,
                task=DOCUMENT_TASK,
                created_at=completed_at,
                object_part=object_part,
                normalized=True,
            )
            for (memory, object_part, _model_input), vector in zip(parts, vectors, strict=True)
        )
        if operation is not None:
            created = tuple(value.memory_id for value in missing)
            operation = replace(
                operation,
                created_ids=created,
                changed_ids=tuple(
                    memory_id
                    for memory_id in dict.fromkeys(
                        prepared.memory_id for prepared, _source, _score in ordered_pairs
                    )
                    if memory_id not in set(created)
                ),
            )
        commit_naming = partial(
            self._store.semantics.apply_formation,
            stored,
            embeddings,
            evidence_clauses=(
                ()
                if not joint_evidence_clauses
                else tuple(
                    dict.fromkeys(
                        (
                            prepared.memory_id,
                            prepared.context.evidence_ids,
                            confidence,
                        )
                        for prepared, _source_id, confidence in ordered_pairs
                        if isinstance(prepared.context, RecordSemantics)
                        and prepared.context.evidence_ids
                    )
                )
            ),
            evidence=tuple(
                dict.fromkeys(
                    (prepared.memory_id, evidence_id, confidence)
                    for prepared, _source_id, confidence in ordered_pairs
                    for evidence_id in (
                        ()
                        if not isinstance(prepared.context, RecordSemantics)
                        else prepared.context.evidence_ids
                    )
                )
            ),
            source_memory_ids=tuple(source.id for source in sources),
            narrowed=narrowed,
            recipe=recipe,
            completed_at=completed_at,
            operation=operation,
            forget_ids=forget_ids,
            require_active=require_active,
            require_unretired=require_unretired,
            require_visible=require_visible,
            preserve_response_policies=preserve_response_policies,
            projection_identity_id=projection_identity_id,
        )
        try:
            with self._write_lock, translate_storage_errors("commit automatic memory formation"):
                applied = commit_naming()
        except NamingPreparationRequired as pending_naming:
            prepared_naming = self._embedding.prepare_naming_documents(
                pending_naming.documents, assets
            )
            with self._write_lock, translate_storage_errors("commit automatic memory formation"):
                applied = commit_naming(naming_projection=prepared_naming)
        with self._write_lock:
            self._projection.drain()
        if operation is None:
            return None
        if not applied:
            # A concurrent duplicate won the key inside the transaction. Its row is in the log
            # under the same key, but it is not this call's operation, so report the duplicate.
            return None
        with translate_storage_errors("read a memory operation"):
            logged = self._store.control.read_operations(operation_key=operation.operation_key)
        return logged[0] if logged else None
