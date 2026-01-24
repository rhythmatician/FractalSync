from src.control_model import AudioToControlModel
from src.visual_metrics import VisualMetrics
from src.control_trainer import ControlTrainer


def test_trainer_instantiates_visual_components():
    model = AudioToControlModel(window_frames=10, n_features_per_frame=6, k_bands=6)
    vm = VisualMetrics()
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

    trainer = ControlTrainer(
        model=model,
        visual_metrics=vm,
        enable_visual_losses=True,
        visual_proxy_resolution=32,
        visual_proxy_max_iter=8,
    )

    # If available, proxy_renderer should be set
    if trainer.proxy_renderer is not None:
        assert hasattr(trainer.proxy_renderer, "render")
    # Loss fns
    assert hasattr(trainer, "ms_deltav") and hasattr(trainer, "speed_loss")
