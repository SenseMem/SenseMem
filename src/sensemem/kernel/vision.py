"""Visual descriptions and face recognition."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from opentelemetry.trace import Tracer

from sensemem._telemetry import (
    VISION_BATCHES_FAILED,
    mark_model_requests,
)
from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.infrastructure.local.store import (
    StoredAsset,
)
from sensemem.kernel.content import PreparedContent, asset_content
from sensemem.kernel.contracts import Backends
from sensemem.kernel.derived import (
    decode_description,
    encode_description,
    face_identity_text,
    has_stream_description,
    speaker_prose,
    truncated_describe_context,
    validated_descriptions,
)
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import Lifecycle, OperationAssets
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.routing import has_derived_text
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.kernel.settings import Settings
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import MAX_TEXT_CHARACTERS
from sensemem.models.base import (
    FaceAnalysis,
    ModelInput,
    VisionAnalysisBackend,
)
from sensemem.perception import VisualDescription
from sensemem.types import (
    AssetRef,
    Blob,
    Modality,
)

_LOGGER = logging.getLogger(__name__)


class Vision(Traced):
    """Visual descriptions and face recognition."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer

    def pending_descriptions(
        self,
        contents: Sequence[PreparedContent],
        operation: OperationAssets,
    ) -> dict[str, VisualDescription]:
        """Batch uncached visual descriptions for the write path.

        Reuse descriptions from the current vision space so repeated ingestion preserves the
        indexed text and avoids repeated model calls. Stage fresh values on the operation until
        asset rows exist. Query descriptions use their own request-local cache.
        """
        if self._backends.vision_describer is None:
            return {}
        assets = tuple(
            {
                asset.asset_id: asset
                for content in contents
                for asset in content.assets
                if Modality(asset.modality) in self._backends.vision_capabilities
                # Either derived section standing in the document means this asset is described:
                # a caption that was all facts leaves no description marker behind, and asking
                # for it again would buy the same text twice and append a duplicate section.
                and not has_stream_description(content.text, (asset,))
            }.values()
        )
        if not assets:
            return {}
        with translate_storage_errors("read cached visual descriptions"):
            stored_captions = self._store.media.read_visual_descriptions(
                tuple(asset.asset_id for asset in assets),
                space_id=self._backends.vision_space,
            )
        cached = {
            identifier: decode_description(value) for identifier, value in stored_captions.items()
        }
        cached.update(operation.descriptions)
        pending = tuple(asset for asset in assets if asset.asset_id not in cached)
        if not pending:
            # No request is made, so the vision span and its token counters never open: a run that
            # re-ingests a described corpus reports zero vision cost because it paid none.
            return dict(cached)
        try:
            descriptions = self._described_batch(pending, operation)
        except ModelError as error:
            pending_ids = {asset.asset_id for asset in pending}
            required = any(
                Modality(asset.modality) not in self._backends.embedding_capabilities
                and not (operation.transcripts.get(asset.asset_id) or asset.transcript)
                and not has_derived_text(content.text, asset.asset_id, Modality(asset.modality))
                for content in contents
                for asset in content.assets
                if asset.asset_id in pending_ids
            )
            if required:
                raise
            _LOGGER.warning(
                "vision description failed for %d asset(s); storing them without a caption: %s",
                len(pending),
                error.reason,
            )
            # Native media or an existing transcript still supplies a usable representation.
            # Leave failed captions uncached so later ingestion can try again.
            return dict(cached)
        fresh = dict(zip((asset.asset_id for asset in pending), descriptions, strict=True))
        operation.descriptions.update(fresh)
        return {**cached, **fresh}

    def query_descriptions(
        self,
        content: PreparedContent,
        operation: OperationAssets,
        *,
        required: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        """Describe transient query anchors; preserve errors needed for modality conversion."""
        if self._backends.vision_describer is None:
            return ()
        images = tuple(
            {
                asset.asset_id: asset
                for asset in content.assets
                if Modality(asset.modality) in self._backends.vision_capabilities
                and not has_stream_description(content.text, (asset,))
            }.values()
        )
        pending = tuple(
            asset for asset in images if asset.asset_id not in operation.query_descriptions
        )
        if pending:
            descriptions: tuple[VisualDescription | ModelError, ...]
            try:
                descriptions = self._vision_descriptions(
                    tuple(self._hydrator.model_input(asset_content(asset)) for asset in pending)
                )
            except ModelError as error:
                _LOGGER.warning(
                    "query image description failed for %d asset(s): %s",
                    len(pending),
                    error.reason,
                )
                descriptions = (error,) * len(pending)
            # Kept separately from write-path descriptions: query pixels and model text must
            # never create a cache row, a named identity, or a durable memory.
            operation.query_descriptions.update(
                zip((asset.asset_id for asset in pending), descriptions, strict=True)
            )
        anchors: dict[str, None] = {}
        for asset in images:
            description = operation.query_descriptions[asset.asset_id]
            if isinstance(description, ModelError):
                if asset.asset_id in required:
                    raise description
                continue
            anchors[description.content] = None
            # The bundled caption recipe transcribes visible text on this labelled line.
            # Keep its exact words searchable without all the scene's other terms.
            for line in description.text.splitlines():
                label, separator, value = line.partition(":")
                if separator and label.strip().casefold() == "text" and value.strip():
                    anchors[value.strip()] = None
        return tuple(anchors)

    def _described_batch(
        self,
        pending: Sequence[StoredAsset],
        operation: OperationAssets,
    ) -> tuple[VisualDescription, ...]:
        """Describe one batch; the adapter owns its request retry budget."""
        inputs = tuple(
            self._hydrator.model_input(
                replace(
                    asset_content(asset),
                    # Whatever this write already knows the clip says, under the same `speaker_N`
                    # labels the index projection prints. A durable fact about a person is in the
                    # words, not the pixels: four stills say two people are at a table, and only
                    # the dialogue says which one is called Lily. Empty for an image and for any
                    # composition with no speech backend, which then describes pixels as before.
                    # Capped: an unbounded transcript would let one long clip's words dominate the
                    # token budget every visual in it pays.
                    text=truncated_describe_context(
                        speaker_prose(
                            asset.asset_id,
                            operation.speech_segments.get(asset.asset_id, ()),
                        )
                        or ""
                    ),
                )
            )
            for asset in pending
        )
        return self._vision_descriptions(inputs)

    def persist_descriptions(self, operation: OperationAssets) -> None:
        """Cache captions derived in this operation, once their assets are stored.

        Captions are derived before the `media_assets` rows exist -- they can rescue media the
        embedder cannot take, so they have to precede the write -- which is why they are staged on
        the operation and written here instead of where they are computed. An asset that did not
        reach storage is skipped rather than failing the write: the caption is derived convenience
        and its foreign key is the asset.
        """
        if not operation.descriptions:
            return
        pending = {
            asset_id: encode_description(description)
            for asset_id, description in operation.descriptions.items()
            if asset_id in operation.persisted and description.content.strip()
        }
        operation.descriptions.clear()
        if not pending:
            return
        with (
            self._trace("sensemem.storage.write", kind="stage"),
            self._write_lock,
            translate_storage_errors("cache visual descriptions"),
        ):
            self._store.media.write_visual_descriptions(
                pending,
                model_id=self._backends.vision_model,
                space_id=self._backends.vision_space,
            )

    def describe(self, images: Sequence[Blob]) -> tuple[str, ...]:
        """Caption live frames through the cache the write path fills, not around it.

        A scene the stream describes and a later `add()` of the same frame must not pay two
        calls: the endpoint returns a different caption for the same image on every call, so the
        second would also change what the memory's document says. The caption this derives is
        cached when the memory carrying the frame is written -- the row's foreign key is the
        asset, which does not exist until then.
        """
        if self._backends.vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        if not images:
            raise ValidationError("vision description requires at least one image")
        with self._lifecycle.operation(model_work=True) as operation:
            prepared = tuple(self._materializer.prepare(image, operation) for image in images)
            # One blob in, one asset out, so the cached captions line up with the inputs.
            asset_ids = tuple(content.assets[0].asset_id for content in prepared)
            with translate_storage_errors("read cached visual descriptions"):
                described = {
                    identifier: decode_description(value)
                    for identifier, value in self._store.media.read_visual_descriptions(
                        asset_ids, space_id=self._backends.vision_space
                    ).items()
                }
            pending = tuple(
                {
                    asset_id: index
                    for index, asset_id in enumerate(asset_ids)
                    if asset_id not in described
                }.items()
            )
            if pending:
                fresh = self._vision_descriptions(
                    tuple(self._hydrator.model_input(prepared[index]) for _id, index in pending)
                )
                described.update(
                    zip((asset_id for asset_id, _index in pending), fresh, strict=True)
                )
            return tuple(described[asset_id].content for asset_id in asset_ids)

    def _vision_descriptions(self, inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]:
        """Validate one adapter call and account for its final failure."""
        if self._backends.vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        inputs = tuple(inputs)
        # Media only: an input's text is the context this write derived for the visual, and a
        # describer's capability set is narrowed to image and video at construction, so counting
        # text as required would refuse every described clip that arrived with a transcript.
        unsupported = frozenset(
            asset.modality
            for value in inputs
            for asset in value.assets
            if asset.modality is not None
            and asset.modality not in self._backends.vision_capabilities
        )
        if unsupported:
            names = ", ".join(sorted(modality.value for modality in unsupported))
            raise ModelError(
                f"configured vision model does not support: {names}",
                reason="unsupported_modality",
            )
        with self._model_trace(
            "vision",
            "vision.description",
            model=self._backends.vision_model,
            batch_size=len(inputs),
            modalities=(modality for value in inputs for modality in value.modalities),
        ) as span:
            mark_model_requests(1)
            # Validated inside the span so that a reply the model did return but that cannot be
            # used is counted as a failed batch too, and so that the count lands on a span the
            # evaluation telemetry aggregates -- it reads counters only from model spans.
            try:
                backend = self._backends.vision_describer
                if isinstance(backend, VisionAnalysisBackend):
                    return validated_descriptions(backend.describe_analysis(inputs), inputs)
                raw = backend.describe(inputs)
                values = tuple(
                    VisualDescription(value) if isinstance(value, str) and value.strip() else value
                    for value in raw
                )
                return validated_descriptions(values, inputs)
            except Exception:
                span.set_attribute(VISION_BATCHES_FAILED, 1)
                raise

    def recognize(
        self,
        assets: Sequence[StoredAsset],
        operation: OperationAssets,
    ) -> None:
        if self._backends.face_analyzer is None:
            raise ModelError("no face backend is configured", reason="backend_not_configured")
        face_assets = tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"}
                and asset.asset_id not in operation.face_observations
            }.values()
        )
        missing = []
        with (
            self._trace("sensemem.storage.lookup", kind="stage"),
            translate_storage_errors("read cached face recognition"),
        ):
            for asset in face_assets:
                observations = self._store.media.read_faces(
                    asset.asset_id,
                    space_id=self._backends.face_analysis_space,
                )
                if observations is None:
                    missing.append(asset)
                else:
                    operation.face_observations[asset.asset_id] = observations
        if missing:
            analyses = self._analyze_faces(
                tuple(self._hydrator.asset_ref(asset) for asset in missing)
            )
            with (
                self._trace("sensemem.storage.write", kind="stage"),
                self._write_lock,
                translate_storage_errors("persist face recognition"),
            ):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.media.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    # Deliberately no preferred_identity. That shortcut adopted the asset's
                    # lone voice identity for its lone face with no corroboration at all, a
                    # second cross-modal door that bypassed the evidence gate in
                    # _link_asset_identity. Cross-modal binding now has exactly one entrance.
                    operation.face_observations[asset.asset_id] = self._store.media.write_faces(
                        asset.asset_id,
                        analysis,
                        model_id=self._backends.face_model,
                        space_id=self._backends.face_space,
                        analysis_space_id=self._backends.face_analysis_space,
                        minimum_similarity=self._settings.face_similarity,
                        minimum_margin=self._settings.face_margin,
                    )
        analyzed = {asset.asset_id for asset in missing}
        for asset in face_assets:
            self._trace_identity_yield(
                "sensemem.identity.faces",
                tuple(
                    (observation.identity_id, observation.identity_score)
                    for observation in operation.face_observations[asset.asset_id]
                ),
                cached=asset.asset_id not in analyzed,
            )

    def _analyze_faces(
        self,
        assets: Sequence[AssetRef],
    ) -> tuple[FaceAnalysis, ...]:
        if self._backends.face_analyzer is None:
            raise ModelError("no face backend is configured", reason="backend_not_configured")
        with self._model_trace(
            "face",
            "face_recognition",
            model=self._backends.face_model,
            batch_size=len(assets),
            modalities=(cast(Modality, asset.modality) for asset in assets),
        ):
            try:
                analyses = self._backends.face_analyzer.analyze(assets)
            except SenseMemError:
                raise
            except Exception as error:
                raise ModelError("failed to analyze face input", reason="model_failed") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, FaceAnalysis) for analysis in analyses
            ):
                raise ModelError("face model returned invalid output", reason="response_invalid")
            return tuple(analyses)

    def answer_face_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if self._backends.face_analyzer is None:
            return ()
        supported = {modality.value for modality in self._backends.face_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"} and asset.modality in supported
            }.values()
        )

    def with_face_identities(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> PreparedContent:
        face_assets = self.answer_face_assets(prepared.assets)
        text = face_identity_text(prepared.text, face_assets, operation.face_observations)
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ModelError(
                "face identity evidence exceeded the supported text length",
                reason="payload_too_large",
            )
        return replace(prepared, text=text)
