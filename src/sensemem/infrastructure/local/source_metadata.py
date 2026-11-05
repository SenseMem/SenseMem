"""Integrity-checked image metadata and offline map enrichment, independent of generation."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from opentelemetry import trace

from sensemem.exceptions import ModelError
from sensemem.infrastructure.local.gazetteer import OfflineGazetteer
from sensemem.models._media import image_capture_metadata
from sensemem.types import AssetRef, Modality


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
        candidates = cast(list[dict[str, object]], response["candidates"])
        available += len(candidates)
        if not candidates:
            continue
        sources = cast(list[Mapping[str, object]] | None, response.get("sources"))
        if sources is None:
            source = cast(Mapping[str, object], response["source"])
            header = cast(Mapping[str, object], source.get("header", {}))
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
                trial["map_sources"] = [
                    {
                        "source_index": index,
                        "url": sources[index]["url"],
                        "sha256": sources[index]["sha256"],
                        "snapshot_date": sources[index]["snapshot_date"],
                        "osmosis_replication_timestamp": cast(
                            Mapping[str, object], sources[index].get("header", {})
                        ).get("osmosis_replication_timestamp"),
                    }
                    for index in sorted(
                        {cast(int, value["source_index"]) for value in [*chosen, candidate]}
                    )
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
