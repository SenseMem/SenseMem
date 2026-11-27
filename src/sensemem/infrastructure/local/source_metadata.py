"""Integrity-checked image metadata and offline map enrichment, independent of generation."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence

from opentelemetry import trace

from sensemem.content import AssetRef
from sensemem.exceptions import ModelError
from sensemem.infrastructure.local.gazetteer import OfflineGazetteer
from sensemem.infrastructure.local.media_metadata import image_capture_metadata
from sensemem.types import Modality


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class GazetteerEnricher:
    def __init__(self, gazetteer: OfflineGazetteer) -> None:
        self._gazetteer = gazetteer

    def enrich(self, assets: Sequence[AssetRef]) -> dict[str, dict[str, object]]:
        capture = {}
        for asset in assets:
            if asset.modality is not Modality.IMAGE or asset.id in capture:
                continue
            if asset.path is None:
                raise ModelError("source asset is unresolved", reason="asset_unavailable")
            try:
                with asset.path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError:
                raise ModelError(
                    "source asset is unavailable", reason="asset_unavailable"
                ) from None
            if digest != asset.sha256:
                raise ModelError("source asset changed", reason="asset_changed")
            fields = image_capture_metadata(asset.path)
            if fields:
                capture[asset.id] = fields
        return enrich_metadata(capture, assets, self._gazetteer)

    def close(self) -> None:
        pass


def enrich_metadata(
    capture: Mapping[str, dict[str, object]],
    assets: Sequence[AssetRef],
    gazetteer: OfflineGazetteer,
) -> dict[str, dict[str, object]]:
    """Append whole untrusted candidates, charging every actual metadata occurrence."""
    result = {identifier: dict(fields) for identifier, fields in capture.items()}
    occurrences = Counter(asset.id for asset in assets)
    spent = available = admitted = 0
    for identifier, fields in capture.items():
        gps = fields.get("gps")
        if not isinstance(gps, Mapping):
            continue
        response = gazetteer.nearby(gps["latitude"], gps["longitude"])
        raw_candidates = response["candidates"]
        candidates = (
            [entry for entry in raw_candidates if isinstance(entry, dict)]
            if isinstance(raw_candidates, list)
            else []
        )
        available += len(candidates)
        if not candidates:
            continue
        raw_sources = response.get("sources")
        sources = (
            [entry for entry in raw_sources if isinstance(entry, Mapping)]
            if isinstance(raw_sources, list)
            else None
        )
        if sources is None:
            source = response["source"]
            assert isinstance(source, Mapping)
            raw_header = source.get("header", {})
            header = raw_header if isinstance(raw_header, Mapping) else {}
            source_fields: dict[str, object] = {
                "map_snapshot_date": source["snapshot_date"],
                "map_data_timestamp": header.get("osmosis_replication_timestamp"),
            }
        else:
            source_fields = {"map_sources": []}
        enrichment = {
            "untrusted": True,
            **source_fields,
            "index_sha256": gazetteer.index_sha256,
            "attribution": response["attribution"],
            "license_url": response["license_url"],
            "geometry_recipe": response["geometry_recipe"],
            "radius_m": response["radius_m"],
            "candidate_limit": response["limit"],
            "interpretation": "Nearby or covering map geometry is not the photographed object's identity or a mentioned place. GPS error and map changes since capture remain possible; explicit address tags describe the map feature only.",
            "photo_identity_confidence": "unknown",
            "global_complete": False,
        }
        chosen: list[dict[str, object]] = []
        chosen_payload: dict[str, object] | None = None
        cost = 0
        for candidate in candidates:
            trial = {**enrichment, "candidates": [*chosen, candidate]}
            if sources is not None:
                source_indices = sorted(
                    {
                        source_index
                        for value in [*chosen, candidate]
                        if isinstance(source_index := value["source_index"], int)
                    }
                )
                trial["map_sources"] = [
                    {
                        "source_index": index,
                        "url": sources[index]["url"],
                        "sha256": sources[index]["sha256"],
                        "snapshot_date": sources[index]["snapshot_date"],
                        "osmosis_replication_timestamp": (
                            source_header.get("osmosis_replication_timestamp")
                            if isinstance(
                                source_header := sources[index].get("header", {}), Mapping
                            )
                            else None
                        ),
                    }
                    for index in source_indices
                ]
            added = (len(_json_text({"offline_osm_candidates": trial})) - 1) * occurrences[
                identifier
            ]
            if spent + added > gazetteer.budget_chars:
                break
            chosen.append(candidate)
            chosen_payload = trial
            cost = added
        if chosen_payload is not None:
            result[identifier]["offline_osm_candidates"] = chosen_payload
            spent += cost
            admitted += len(chosen)
    span = trace.get_current_span()
    span.set_attribute("sensemem.grounding.gazetteer.index_sha256", gazetteer.index_sha256)
    span.set_attribute("sensemem.grounding.gazetteer.added_chars", spent)
    span.set_attribute("sensemem.grounding.gazetteer.available_candidates", available)
    span.set_attribute("sensemem.grounding.gazetteer.omitted_candidates", available - admitted)
    return result
