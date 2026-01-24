from src.control_model import AudioToControlModel
from src.visual_metrics import VisualMetrics
from src.control_trainer import ControlTrainer


def test_trainer_instantiates_visual_components():
    model = AudioToControlModel(window_frames=10, n_features_per_frame=6, k_bands=6)
    vm = VisualMetrics()
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
