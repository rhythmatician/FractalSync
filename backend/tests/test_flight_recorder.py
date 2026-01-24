import os
import json
import numpy as np
from src.flight_recorder import FlightRecorder, compute_delta_v


def test_flight_recorder_writes(tmp_path):
    run_id = "test_run"
    base_dir = tmp_path / "logs"
    fr = FlightRecorder(run_id=run_id, base_dir=str(base_dir), save_images=True)
    fr.start_run({"test": True})

    # Create a simple gradient proxy frame
    h, w = 64, 64
    frame1 = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    frame2 = np.roll(frame1, 1, axis=1)

    delta = compute_delta_v(frame2, frame1)
    assert delta >= 0.0

    fr.record_step(
        t=0,
        c=[-0.7, 0.27],
        controller={
            "s": 1.0,
            "alpha": 0.5,
            "omega_scale": 1.0,
            "band_gates": [0.1, 0.2],
        },
        h=0.2,
        band_energies=[0.1, 0.2],
        audio_features=[0.0, 0.1],
        proxy_frame=frame2,
        delta_v=delta,
        notes="unit test",
    )

    fr.close()

    # Check files
    run_dir = base_dir / run_id
    assert (run_dir / "records.ndjson").exists()
    with open(run_dir / "records.ndjson", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # First line should be metadata
    meta = json.loads(lines[0])
    assert meta.get("_meta") is True
    rec = json.loads(lines[1])
    assert rec["t"] == 0
    assert "proxy_frame_path" in rec
    assert (run_dir / rec["proxy_frame_path"]).exists()
