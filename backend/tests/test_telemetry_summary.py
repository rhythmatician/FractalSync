from pathlib import Path
import json
from backend.scripts.telemetry_summary import iter_entries, summarize


def test_summarize(tmp_path, monkeypatch):
    # Create a fake telemetry log with two entries
    p = tmp_path / "telemetry.log"
    entries = [
        {
            "ts": "t1",
            "model_dx": 0.1,
            "model_dy": 0.0,
            "applied_delta": 0.02,
            "sensitivity": 0.1,
            "avg_rms": 0.2,
        },
        {
            "ts": "t2",
            "model_dx": 0.2,
            "model_dy": 0.0,
            "applied_delta": 0.03,
            "sensitivity": 0.2,
            "avg_rms": 0.3,
        },
    ]
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    lines = list(iter_entries(p))
    assert len(lines) == 2

    stats = summarize(lines)
    assert stats["count"] == 2
    assert "model_dx" in stats
    assert stats["model_dx"]["mean"] == 0.15000000000000002
