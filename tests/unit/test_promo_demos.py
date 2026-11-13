"""Exercise the demonstrator through real isolated public-SDK stores."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from demos.promo.runner import Evidence


def test_promo_demo_evidence_comes_from_sdk_operations(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    photo = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII="
    )
    (assets / "keys.png").write_bytes(photo)
    output = tmp_path / "evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "demos.promo.runner",
            "--output",
            str(output),
            "--assets-dir",
            str(assets),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = cast("Evidence", json.loads(output.read_text()))

    assert evidence["mode"] == "fixture"
    assert "not model quality" in evidence["disclosure"]
    assert json.loads(output.read_text())["schema_version"] == 1
    assert [scenario["id"] for scenario in evidence["scenarios"]] == [
        "sensory",
        "evolving",
        "streaming",
    ]
    failures = [
        (scenario["id"], check["name"], check["detail"])
        for scenario in evidence["scenarios"]
        for check in scenario["checks"]
        if not check["passed"]
    ]
    assert failures == []
    assert (assets / "recalled-keys.png").read_bytes() == photo
    sensory = evidence["scenarios"][0]["result"]
    hits = cast("list[dict[str, object]]", sensory["hits"])
    assert sensory["retrieval_rank"] == next(
        index for index, hit in enumerate(hits, 1) if hit["id"] == sensory["record_id"]
    )
    assert sensory["total_candidates"] == len(hits)
    assert sensory["answer_kind"] == "evidence_only"
    assert "no generated location answer" in str(sensory["quality_observations"])
    evolving = evidence["scenarios"][1]
    assert (
        evolving["result"]["proposal_origin"] == "host-authored; SDK kernel validates and applies"
    )
    streaming = evidence["scenarios"][2]
    assert streaming["result"]["after_final_count"] == 3
    assert streaming["result"]["after_partials_count"] == 2
    assert streaming["result"]["after_cancel_count"] == 3
    assert all(elapsed >= 0 for elapsed in streaming["timings"].values())
