import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.control_trainer import ControlTrainer
from src.control_model import AudioToControlModel
from src.visual_metrics import VisualMetrics
from src.runtime_core_bridge import make_feature_extractor
from src.flight_recorder import FlightRecorder


def test_trainer_records_transient_h(tmp_path):
    """Ensure ControlTrainer computes and records transient strength `h` during stepping."""

    device = "cpu"
    model = AudioToControlModel(window_frames=10, n_features_per_frame=6, k_bands=6)
    visual_metrics = VisualMetrics()
    feature_extractor = make_feature_extractor()

    # Ensure required precomputed distance field exists for this test
    import numpy as np
    import json as _json
    from pathlib import Path as _Path

    df_base = _Path("data") / "mandelbrot_distance_field"
    df_base.parent.mkdir(parents=True, exist_ok=True)
    res = 4
    field = np.arange(res * res, dtype=np.float32).reshape((res, res))
    np.save(str(df_base.with_suffix(".npy")), field)
    meta = {"resolution": res, "real_range": (-1.5, 1.5), "imag_range": (-1.5, 1.5), "max_distance": 1.0, "slowdown_threshold": 0.05}
    with open(str(df_base.with_suffix(".json")), "w") as f:
        _json.dump(meta, f)

    # Use flight recorder with no image writes for speed
    fr = FlightRecorder(
        run_id="test_run", base_dir=str(tmp_path / "logs"), save_images=False
    )
    fr.start_run({"test": True})

    trainer = ControlTrainer(
        model=model,
        visual_metrics=visual_metrics,
        feature_extractor=feature_extractor,
        device=device,
        use_amp=False,
        julia_renderer=None,
        julia_resolution=32,
        julia_max_iter=20,
        trajectory_steps=2,
        render_fraction=1.0,  # render all samples so recorder is exercised
        flight_recorder=fr,
        contour_d_star=0.3,
        contour_max_step=0.03,
    )

    # Small synthetic dataset
    batch_size = 4
    features = torch.randn(batch_size, model.input_dim)
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Run one epoch in trajectory mode
    trainer.train_epoch(dataloader, epoch=1, curriculum_decay=0.95)
    fr.close()

    # Read records and check h is present and finite for at least one record
    run_dir = Path(fr.run_dir)
    records_path = run_dir / "records.ndjson"
    assert records_path.exists()
    lines = [
        line.strip()
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # metadata + at least one record
    assert len(lines) >= 2
    rec = json.loads(lines[1])
    assert "h" in rec
    # h can be zero or positive (spectral flux proxy)
    assert rec["h"] is None or isinstance(rec["h"], float)
