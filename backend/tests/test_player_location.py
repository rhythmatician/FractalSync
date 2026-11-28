import json

import numpy as np
import pytest
import torch
import runtime_core

from backend.src.control_model import AudioToControlModel
from backend.src.export_model import export_to_onnx
from backend.src.location_rollout import location_conditioned_rollout


def test_location_rollout_observes_state_before_action_and_resets_segments():
    model = AudioToControlModel(window_frames=2, controls_version="controls/2", observe_location=True)
    features = torch.zeros(3, model.input_dim)
    starts = torch.tensor([-0.7569 + 0.1439j, 0j, -0.73 + 0.16j], dtype=torch.complex128)
    observations = []
    hook = model.register_forward_pre_hook(lambda _, args: observations.append(args[1].detach()))
    try:
        outputs, positions = location_conditioned_rollout(model, features, torch.tensor([0, 0, 1]), starts)
    finally:
        hook.remove()
    assert outputs.requires_grad
    assert observations[0][0, :4].tolist() == pytest.approx([starts[0].real.item(), starts[0].imag.item(), 0, 0])
    assert observations[1][0, :2].tolist() == pytest.approx([positions[0].real, positions[0].imag])
    assert observations[2][0, :4].tolist() == pytest.approx([starts[2].real.item(), starts[2].imag.item(), 0, 0])


def test_exported_policy_consumes_the_same_location_observation(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = AudioToControlModel(window_frames=2, controls_version="controls/2", observe_location=True).eval()
    path = str(tmp_path / "policy.onnx")
    metadata_path = export_to_onnx(
        model, (1, model.input_dim), path, metadata={"model_type": "controls_v2", "controls_version": "controls/2"}
    )
    with open(metadata_path, encoding="utf-8") as source:
        assert json.load(source)["player_location_version"] == runtime_core.player_location_schema()[0]
    x = torch.zeros(1, model.input_dim)
    location = torch.tensor([runtime_core.player_location_observation(-0.7569, 0.1439, 0.02, -0.03)])
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    result = session.run(None, {"audio_features": x.numpy(), "player_location": location.numpy()})[0]
    np.testing.assert_allclose(result, model(x, location).detach().numpy(), atol=1e-6)
