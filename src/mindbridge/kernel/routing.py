"""Pure native/derived input selection shared by every model consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from mindbridge.exceptions import ModelError
from mindbridge.kernel.content import PreparedContent
from mindbridge.kernel.derived import (
    derived_text,
    has_stream_description,
)
from mindbridge.kernel.validation import MAX_TEXT_CHARACTERS
from mindbridge.models.base import ModelInput
from mindbridge.perception import VisualDescription
from mindbridge.types import Modality


def has_derived_text(text: str, asset_id: str, modality: Modality) -> bool:
    """A representation belongs to one asset, never to an entire mixed observation."""
    kinds: list[str] = []
    if modality in {Modality.AUDIO, Modality.VIDEO}:
        kinds.extend(("transcript", "speech identities"))
    if modality in {Modality.IMAGE, Modality.VIDEO}:
        kinds.extend(("visual description", "facts"))
    markers = {f"[{kind}:{asset_id}]" for kind in kinds}
    return any(
        marker in markers and bool(body.strip())
        for section in text.split("\n\n")
        for marker, _, body in (section.partition("\n"),)
    )


def derived_modalities(values: Sequence[ModelInput]) -> frozenset[Modality]:
    """Report modalities with a source-backed representation for every supplied asset."""
    available: set[Modality] = set()
    missing: set[Modality] = set()
    for value in values:
        for asset in value.assets:
            modality = asset.modality
            if modality is not None:
                available.add(modality)
                if not has_derived_text(value.text, asset.id, modality):
                    missing.add(modality)
    return frozenset(available - missing)


def with_descriptions(
    prepared: PreparedContent, descriptions: Mapping[str, VisualDescription | ModelError]
) -> PreparedContent:
    """Attach bounded, source-labelled descriptions without changing source identity."""
    text = prepared.text
    parts = list(prepared.canonical_parts)
    described = prepared.visual_description
    for asset in prepared.assets:
        description = descriptions.get(asset.asset_id)
        if (
            description is None
            or isinstance(description, ModelError)
            or has_stream_description(text, (asset,))
        ):
            continue
        section = f"[visual description:{asset.asset_id}]\n{description.content}"
        candidate = "\n\n".join(value for value in (text, section) if value)
        if len(candidate) <= MAX_TEXT_CHARACTERS:
            text = candidate
            parts.append(("visual_description", section))
            described = True
    return replace(prepared, text=text, canonical_parts=tuple(parts), visual_description=described)


def route_input(
    prepared: PreparedContent,
    value: ModelInput,
    supported: frozenset[Modality],
    operation: str,
) -> ModelInput:
    """Keep native assets, substituting only source-backed text for unsupported assets."""
    missing = value.modalities - supported
    if not missing:
        return value
    if (
        missing == {Modality.TEXT}
        and (prepared.audio_transcript or prepared.visual_description)
        and value.assets
    ):
        return ModelInput(assets=value.assets)
    unsupported = []
    for asset in prepared.assets:
        modality = Modality(asset.modality)
        if modality in supported:
            continue
        transcript = bool(asset.transcript and asset.transcript.strip())
        if Modality.TEXT not in supported or not (
            transcript or has_derived_text(prepared.text, asset.asset_id, modality)
        ):
            unsupported.append(modality)
    if Modality.TEXT in missing:
        unsupported.append(Modality.TEXT)
    if unsupported:
        names = ", ".join(sorted({modality.value for modality in unsupported}))
        raise ModelError(
            f"configured {operation} model does not support: {names}",
            reason="unsupported_modality",
        )
    text = derived_text(value.text, prepared.assets)
    assets = tuple(asset for asset in value.assets if asset.modality in supported)
    if not text and not assets:
        raise ModelError(
            f"configured {operation} model has no usable representation",
            reason="unsupported_modality",
        )
    return ModelInput(text=text, assets=assets)
