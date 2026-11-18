"""Pure validation of current store identities and index settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from sensemem.exceptions import StorageError, ValidationError
from sensemem.types import IndexQuantization

_INDEX_RECIPE_PREFIX = (
    "zvec-0.8:hnsw-cosine-m50-efc500:fts-bigram-fused:grouped-range:context-keys-v12"
)

STORE_METADATA_KEYS = {
    "model": "embedding.model_id",
    "space": "embedding.space_id",
    "transcription": "transcription.space_id",
    "face": "face.space_id",
    "face_analysis": "face.analysis_space_id",
    "dimension": "embedding.dimension",
    "index": "index.recipe",
}
EVIDENCE_RECIPE_KEY = "evidence.projection_recipe"
INGESTION_RECIPE_KEY = "processing.ingestion_recipe"


def plan_ingestion_recipe(
    stages: tuple[tuple[str, str], ...], *, previous: str | None, has_records: bool
) -> dict[str, str]:
    """Require populated stores to retain their exact ordered ingestion recipe."""
    recipe = json.dumps(
        [{"name": name, "space": space} for name, space in stages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if previous == recipe:
        return {}
    if has_records:
        raise StorageError(
            "ingestion stage recipes changed; replay sources into a fresh data directory",
            reason="schema_unsupported",
            stage="open",
        )
    return {INGESTION_RECIPE_KEY: recipe}


def validated_index_quantization(value: object) -> IndexQuantization:
    if not isinstance(value, IndexQuantization):
        raise ValidationError("index_quantization must be an IndexQuantization value")
    return value


def index_recipe_for(quantization: IndexQuantization) -> str:
    return f"{_INDEX_RECIPE_PREFIX}:quantization-{quantization.value}"


@dataclass(frozen=True, slots=True)
class MetadataPlan:
    updates: Mapping[str, str]
    rebuild_index: bool


def plan_metadata(
    expected: Mapping[str, str],
    stored: Mapping[str, str | None],
    *,
    has_records: bool,
    initializable: frozenset[str],
) -> MetadataPlan:
    """Validate every marker before returning changes that recovery may commit."""
    updates: dict[str, str] = {}
    rebuild_index = False
    for key, value in expected.items():
        previous = stored[key]
        if previous == value:
            continue
        elif previous is None:
            updates[key] = value
        elif key == STORE_METADATA_KEYS["model"]:
            # The declared space and dimension determine vector compatibility. Model names
            # are diagnostics and may change when compatible encoders are composed or reordered.
            updates[key] = value
        elif key == STORE_METADATA_KEYS["index"] and previous in {
            index_recipe_for(mode) for mode in IndexQuantization
        }:
            # Quantization is a current instance setting over the same authoritative vectors.
            rebuild_index = True
            updates[key] = value
        else:
            raise StorageError(
                f"local store metadata mismatch for {key}: expected {value!r}, found {previous!r}"
            )
    missing = tuple(key for key in expected if stored[key] is None and key not in initializable)
    if has_records and missing:
        raise StorageError(
            f"local store metadata is missing required markers: {', '.join(missing)}"
        )
    return MetadataPlan(updates, rebuild_index)
