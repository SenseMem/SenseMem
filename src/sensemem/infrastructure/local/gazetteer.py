"""Optional read-only OSM geometry lookup, independent of a memory store."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import date
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from sensemem.exceptions import StorageError, ValidationError

DEFAULT_GAZETTEER_RADIUS_M = 500
DEFAULT_GAZETTEER_LIMIT = 8
DEFAULT_GAZETTEER_BUDGET_CHARS = 16_000
_INDEX_RECIPE = "offline-osm-named-geometries-v1-aeqd-wgs84-step0005"
_MULTI_INDEX_RECIPE = "offline-osm-named-geometries-v2-aeqd-wgs84-step0005"
_ATTRIBUTION = "© OpenStreetMap contributors; ODbL 1.0"
_LICENSE_URL = "https://www.openstreetmap.org/copyright"
_MAX_PREFILTER_ROWS = 10_000
_MAX_GEOMETRY_BYTES = 32 * 1024 * 1024
_NAME_KEYS = ("name:en", "name", "official_name", "alt_name")
_GeometryRow = tuple[str, int, str, str, bytes] | tuple[str, int, str, str, bytes, int]


class _GazetteerLookup(Protocol):
    def __call__(self, latitude: float, longitude: float) -> dict[str, object]: ...


def _number(value: object, name: str, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not low <= value <= high
        or not math.isfinite(value)
    ):
        raise ValidationError(f"gazetteer {name} must be finite and between {low} and {high}")
    return float(value)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid mapping")
    result: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            raise ValueError("invalid mapping")
        result[key] = entry
    return result


def _validate_source(value: object) -> None:
    source = _mapping(value)
    if (
        not isinstance(source["url"], str)
        or not source["url"].startswith("https://")
        or not isinstance(source["sha256"], str)
        or re.fullmatch("[0-9a-f]{64}", source["sha256"]) is None
        or not isinstance(source["snapshot_date"], str)
    ):
        raise ValueError("unsupported index provenance")
    date.fromisoformat(source["snapshot_date"])
    bounds = source["extract_bbox_lonlat"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("invalid extract bounds")
    west, south, east, north = (
        _number(
            value, "extract bound", -180 if index % 2 == 0 else -90, 180 if index % 2 == 0 else 90
        )
        for index, value in enumerate(bounds)
    )
    if west > east or south > north:
        raise ValueError("invalid extract bounds")
    _mapping(source.get("header", {}))


def _metadata(connection: sqlite3.Connection) -> dict[str, object]:
    metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value FROM metadata")
    }
    recipe = metadata["recipe"]
    if (
        recipe not in {_INDEX_RECIPE, _MULTI_INDEX_RECIPE}
        or metadata["attribution"] != _ATTRIBUTION
        or metadata["license_url"] != _LICENSE_URL
        or metadata["geometry_segment_max_degrees"] != 0.0005
        or metadata["complete_world_or_region_map"] is not False
    ):
        raise ValueError("unsupported index provenance")
    multiple = recipe == _MULTI_INDEX_RECIPE
    expected = {
        "metadata": ("key", "value"),
        "features": ("id", "osm_type", "osm_id", "kind", "tags_json", "geometry_wkb")
        + (("source_index",) if multiple else ()),
        "bounds": ("id", "min_lon", "max_lon", "min_lat", "max_lat"),
    }
    for table, columns in expected.items():
        if tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})")) != columns:
            raise ValueError("unsupported index schema")
    if multiple:
        sources = metadata["sources"]
        if "source" in metadata or not isinstance(sources, list) or len(sources) < 2:
            raise ValueError("multiple sources required without a single source")
        for source in sources:
            _validate_source(source)
        if (
            connection.execute(
                "SELECT 1 FROM features WHERE typeof(source_index)!='integer' "
                "OR source_index<0 OR source_index>=? LIMIT 1",
                (len(sources),),
            ).fetchone()
            is not None
        ):
            raise ValueError("invalid feature source index")
        if (
            connection.execute(
                "SELECT 1 FROM features GROUP BY osm_type,osm_id HAVING count(*)>1 LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise ValueError("duplicate OSM feature identity")
    else:
        if "sources" in metadata:
            raise ValueError("single-source index cannot declare multiple sources")
        _validate_source(metadata["source"])
    return metadata


class OfflineGazetteer:
    """Pinned external map candidates near capture GPS, never asserted photo identities.

    This accepts the documented named-geometry v1 or v2 SQLite index. Its sources, license and
    complete main-file SHA are checked at construction. The input must be a checkpointed,
    immutable snapshot; SQLite WAL/SHM files are never part of the accepted input.
    No memory data is written, no index is downloaded and no coordinate is sent over a network.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        index_sha256: str,
        radius_m: float = DEFAULT_GAZETTEER_RADIUS_M,
        limit: int = DEFAULT_GAZETTEER_LIMIT,
        budget_chars: int = DEFAULT_GAZETTEER_BUDGET_CHARS,
    ) -> None:
        radius = _number(radius_m, "radius_m", 0, DEFAULT_GAZETTEER_RADIUS_M)
        if radius == 0:
            raise ValidationError("gazetteer radius_m must be positive")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= DEFAULT_GAZETTEER_LIMIT
        ):
            raise ValidationError(
                f"gazetteer limit must be an integer between 1 and {DEFAULT_GAZETTEER_LIMIT}"
            )
        if isinstance(budget_chars, bool) or not isinstance(budget_chars, int) or budget_chars <= 0:
            raise ValidationError("gazetteer budget_chars must be a positive integer")
        if not isinstance(index_sha256, str) or re.fullmatch("[0-9a-f]{64}", index_sha256) is None:
            raise ValidationError("gazetteer index_sha256 must be 64 lowercase hexadecimal digits")
        if not isinstance(index_path, str | Path) or not str(index_path).strip():
            raise ValidationError("gazetteer index_path must be a non-empty path")
        try:
            import pyproj  # noqa: F401 - verify this optional capability before accepting it
            import shapely  # noqa: F401 - verify this optional capability before accepting it
        except ImportError:
            raise ValidationError("offline gazetteer requires the 'gazetteer' extra") from None
        self._path = Path(index_path).resolve()
        self._index_sha256, self._radius_m = index_sha256, radius
        self._limit, self._budget_chars = limit, budget_chars
        try:
            self._signature = self._file_signature()
            digest = hashlib.sha256()
            with self._path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != index_sha256:
                raise ValueError("index SHA mismatch")
            with closing(self._connect()) as connection:
                self._metadata = _metadata(connection)
            self._check_file()
        except (
            OSError,
            sqlite3.Error,
            KeyError,
            TypeError,
            ValueError,
            RecursionError,
            StorageError,
        ):
            raise ValidationError(
                "gazetteer index integrity, schema or provenance is invalid"
            ) from None
        self._lookup: _GazetteerLookup = lru_cache(maxsize=512)(self._query)

    @property
    def index_sha256(self) -> str:
        """The explicitly configured complete index SHA256."""
        return self._index_sha256

    @property
    def budget_chars(self) -> int:
        """Total additional serialized map-context characters per answer request."""
        return self._budget_chars

    @property
    def provenance(self) -> dict[str, object]:
        """Return a detached source/license/geometry identity for diagnostics."""
        return copy.deepcopy(
            {
                "index_sha256": self._index_sha256,
                **(
                    {"source": self._metadata["source"]}
                    if "source" in self._metadata
                    else {"sources": self._metadata["sources"]}
                ),
                "attribution": self._metadata["attribution"],
                "license_url": self._metadata["license_url"],
                "geometry_recipe": self._metadata["recipe"],
                "runtime_versions": {name: version(name) for name in ("shapely", "pyproj")},
                "radius_m": self._radius_m,
                "limit": self._limit,
                "budget_chars": self._budget_chars,
            }
        )

    def nearby(self, latitude: float, longitude: float) -> dict[str, object]:
        """Return nearby/covering geometry; unknown coverage never means no place exists."""
        latitude = _number(latitude, "latitude", -90, 90)
        longitude = _number(longitude, "longitude", -180, 180)
        self._check_file()
        result = self._lookup(latitude, longitude)
        self._check_file()
        return copy.deepcopy(result)

    def _file_signature(self) -> tuple[int, int, int, int]:
        value = self._path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    def _check_file(self) -> None:
        try:
            unchanged = self._file_signature() == self._signature
        except OSError:
            unchanged = False
        if not unchanged:
            raise StorageError("offline gazetteer index changed or is unavailable")

    def _connect(self) -> sqlite3.Connection:
        # Only the complete main snapshot is SHA-bound, never adjacent WAL/SHM contents.
        return sqlite3.connect(self._path.as_uri() + "?mode=ro&immutable=1", uri=True)

    def _query(self, latitude: float, longitude: float) -> dict[str, object]:
        result: dict[str, object] = {
            **self.provenance,
            "status": "unknown",
            "candidates": [],
            "global_complete": False,
            "photo_identity_confidence": "unknown",
        }
        raw_sources = self._metadata.get("sources")
        sources: list[object] = (
            [self._metadata["source"]]
            if "source" in self._metadata
            else raw_sources
            if isinstance(raw_sources, list)
            else []
        )
        allowed = []
        for index, value in enumerate(sources):
            bounds = _mapping(value)["extract_bbox_lonlat"]
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            west, south, east, north = (float(bound) for bound in bounds)
            if west <= longitude <= east and south <= latitude <= north:
                allowed.append(index)
        if not allowed:
            return {**result, "reason": "outside_extract_bbox"}
        try:
            with closing(self._connect()) as connection:
                rows = self._prefilter(
                    connection,
                    latitude,
                    longitude,
                    source_indices=tuple(allowed) if "sources" in self._metadata else None,
                )
            if rows is None:
                return {**result, "reason": "candidate_scan_limit", "lookup_incomplete": True}
            candidates = _nearby_geometries(rows, latitude, longitude, self._radius_m)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            raise StorageError("offline gazetteer query failed") from None
        return {
            **result,
            "status": "candidates" if candidates else "unknown",
            "candidates": candidates[: self._limit],
            "reason": None if candidates else "no_indexed_feature_within_radius",
        }

    def _prefilter(
        self,
        connection: sqlite3.Connection,
        latitude: float,
        longitude: float,
        *,
        source_indices: tuple[int, ...] | None = None,
    ) -> list[_GeometryRow] | None:
        delta_lat = self._radius_m / 110_000
        max_lat = min(90, abs(latitude) + delta_lat)
        delta_lon = min(180, self._radius_m / max(1e-12, 110_000 * math.cos(math.radians(max_lat))))
        intervals = [(max(-180, longitude - delta_lon), min(180, longitude + delta_lon))]
        if longitude - delta_lon < -180:
            intervals.append((longitude - delta_lon + 360, 180))
        if longitude + delta_lon > 180:
            intervals.append((-180, longitude + delta_lon - 360))
        found: dict[int, _GeometryRow] = {}
        used = 0
        source_column = ",f.source_index" if source_indices is not None else ""
        source_filter = (
            " AND f.source_index IN (" + ",".join("?" for _ in source_indices) + ")"
            if source_indices is not None
            else ""
        )
        for low, high in intervals:
            for row in connection.execute(
                "SELECT f.id,f.osm_type,f.osm_id,f.kind,f.tags_json,f.geometry_wkb"
                + source_column
                + " "
                "FROM bounds b JOIN features f ON f.id=b.id "
                "WHERE b.min_lon<=? AND b.max_lon>=? AND b.min_lat<=? AND b.max_lat>=?"
                + source_filter,
                (high, low, latitude + delta_lat, latitude - delta_lat, *(source_indices or ())),
            ):
                identifier = int(row[0])
                if identifier not in found:
                    geometry_wkb = bytes(row[5])
                    used += len(geometry_wkb)
                    if len(found) >= _MAX_PREFILTER_ROWS or used > _MAX_GEOMETRY_BYTES:
                        return None
                    if source_indices is not None:
                        found[identifier] = (
                            str(row[1]),
                            int(row[2]),
                            str(row[3]),
                            str(row[4]),
                            geometry_wkb,
                            int(row[6]),
                        )
                    else:
                        found[identifier] = (
                            str(row[1]),
                            int(row[2]),
                            str(row[3]),
                            str(row[4]),
                            geometry_wkb,
                        )
        return list(found.values())


def _nearby_geometries(
    rows: list[_GeometryRow], latitude: float, longitude: float, radius_m: float
) -> list[dict[str, object]]:
    import pyproj
    from shapely import transform, wkb
    from shapely.errors import GEOSException
    from shapely.geometry import Point

    # An explicit grid-free mathematical pipeline cannot fetch a remote datum/grid resource.
    projection = pyproj.Transformer.from_pipeline(
        f"+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f"+step +proj=aeqd +lat_0={latitude} +lon_0={longitude} +ellps=WGS84 +units=m"
    )

    def project(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
        # Lists keep single-coordinate arrays out of pyproj's deprecated scalar coercion.
        return projection.transform(list(x), list(y))

    point, source_point = Point(0, 0), Point(longitude, latitude)
    ranked: list[tuple[float, str, int, dict[str, object]]] = []
    for row in rows:
        osm_type, osm_id, kind, tags_json, raw = row[0], row[1], row[2], row[3], row[4]
        try:
            source_geometry = wkb.loads(raw)
            if source_geometry.is_empty or not source_geometry.is_valid:
                continue
            tags = _mapping(json.loads(tags_json))
            names = {
                key: value
                for key, value in tags.items()
                if key in _NAME_KEYS and isinstance(value, str) and value.strip()
            }
            if not names:
                continue
            covers = bool(source_geometry.covers(source_point))
            geometry = transform(source_geometry, project, interleaved=False)
            if geometry.is_empty or not geometry.is_valid:
                continue
            distance = 0.0 if covers else float(geometry.distance(point))
            if not math.isfinite(distance) or distance > radius_m + 1e-7:
                continue
            polygon = source_geometry.geom_type in {"Polygon", "MultiPolygon"}
            boundary = polygon and bool(source_geometry.touches(source_point))
            relation = (
                "containing_area"
                if polygon and covers and not boundary
                else "area_boundary"
                if boundary
                else "coincident_node"
                if kind == "named_node" and distance < 1e-7
                else "on_road_geometry"
                if kind == "named_road" and distance < 1e-7
                else "on_waterway_geometry"
                if kind == "named_waterway" and distance < 1e-7
                else "nearby"
            )
            candidate: dict[str, object] = {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "kind": kind,
                "geometry_type": str(source_geometry.geom_type),
                **{key: names[key] for key in ("name", "name:en") if key in names},
                "aliases": {
                    key: names[key] for key in ("official_name", "alt_name") if key in names
                },
                "address_tags": {
                    key: value
                    for key, value in tags.items()
                    if key.startswith("addr:") and isinstance(value, str)
                },
                "distance_m": round(distance, 1),
                "covers_coordinate": covers,
                "coordinate_on_boundary": boundary,
                "spatial_relation": relation,
                **({"source_index": row[5]} if len(row) == 6 else {}),
            }
            ranked.append((distance, osm_type, osm_id, candidate))
        except (ValueError, KeyError, TypeError, GEOSException, pyproj.exceptions.ProjError):
            continue
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [entry[3] for entry in ranked]
