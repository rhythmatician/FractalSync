"""Exercise runtime-core model_io schemas and decoders through public PyO3."""

import json
from pathlib import Path

import pytest
import torch

import runtime_core
from src.control_model import AudioToControlModel
from src.model_schema import apply_named_schema, output_schema


def test_legacy_head_matches_rust_decoder_and_keeps_checkpoint_keys() -> None:
    torch.manual_seed(7)
    model = AudioToControlModel(k_bands=3)
    checkpoint_keys = list(model.state_dict())
    before = set(checkpoint_keys)
    inputs = torch.randn(2, model.input_dim, requires_grad=True)
    outputs = model(inputs)

    decoded = json.loads(runtime_core.decode_orbit_control_json(outputs[0].tolist(), 3))
    assert decoded["alpha"] == pytest.approx(float(outputs[0, 1].detach()))
    assert 0.05 <= decoded["alpha"] <= 0.95
    outputs.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()

    restored = AudioToControlModel(k_bands=3)
    restored.load_state_dict(model.state_dict())
    assert set(restored.state_dict()) == before
    assert [key for key in checkpoint_keys if "head" in key] == [
        f"{head}.{layer}.{field}"
        for head in ("s_head", "alpha_head", "omega_head", "band_gates_head")
        for layer in (0, 2)
        for field in ("weight", "bias")
    ]


def test_actual_zeroed_heads_are_activated_exactly_once() -> None:
    orbit = AudioToControlModel(k_bands=2)
    for parameter in orbit.parameters():
        torch.nn.init.zeros_(parameter)
    actual = orbit(torch.zeros(1, orbit.input_dim))
    expected_omega = 0.1 + 0.5 * torch.nn.functional.softplus(torch.tensor(0.0))
    torch.testing.assert_close(
        actual,
        torch.tensor([[1.6, 0.5, expected_omega, 0.5, 0.5]]),
    )

    controls = AudioToControlModel(controls_version="controls/2")
    for parameter in controls.parameters():
        torch.nn.init.zeros_(parameter)
    decoded = controls.parse_output(controls(torch.zeros(1, controls.input_dim)))
    assert decoded["directionX"] == 0.0
    assert decoded["throttle"] == 0.5
    assert decoded["zoomDelta"] == 0.0


def test_known_logits_apply_each_rust_activation_once() -> None:
    orbit = output_schema("orbit_control", 0)
    actual = apply_named_schema(
        {
            "s_target": torch.tensor([-5.0]),
            "alpha": torch.tensor([0.0]),
            "omega_scale": torch.tensor([5.0]),
        },
        orbit,
    )
    expected = torch.tensor([[
        0.2 + 2.8 * torch.sigmoid(torch.tensor(-5.0)),
        0.05 + 0.9 * torch.sigmoid(torch.tensor(0.0)),
        min(5.0, 0.1 + 0.5 * torch.nn.functional.softplus(torch.tensor(5.0))),
    ]])
    torch.testing.assert_close(actual, expected)

    controls = output_schema("controls_v2", 0)
    zero_heads = {descriptor.name: torch.tensor([0.0]) for descriptor in controls}
    decoded = apply_named_schema(zero_heads, controls)
    by_name = {descriptor.name: decoded[0, index] for index, descriptor in enumerate(controls)}
    assert by_name["directionX"] == 0.0
    assert by_name["throttle"] == 0.5
    assert by_name["zoomDelta"] == 0.0


def test_zero_band_schema_keeps_empty_group() -> None:
    model = AudioToControlModel(k_bands=0)
    output = model(torch.randn(1, model.input_dim))
    assert model.parse_output(output)["band_gates"].shape == (1, 0)


def test_legacy_orbit_onnx_matches_torch_and_rust_metadata(tmp_path: Path) -> None:
    """Exported legacy orbit heads retain their exact nonlinear transforms."""
    import numpy as np
    import onnxruntime as ort

    from src.export_model import export_to_onnx

    model = AudioToControlModel(
        window_frames=2,
        n_features_per_frame=2,
        hidden_dims=[8],
        k_bands=2,
        dropout=0.0,
    ).eval()
    features = torch.tensor([[0.2, -0.4, 0.8, -0.1]], dtype=torch.float32)
    expected = model(features).detach().numpy()
    destination = tmp_path / "legacy_orbit.onnx"
    metadata_path = export_to_onnx(
        model,
        (1, model.input_dim),
        str(destination),
        metadata={"model_type": "orbit_control", "k_bands": 2},
    )

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual = np.asarray(
        session.run(None, {session.get_inputs()[0].name: features.numpy()})[0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    metadata = json.loads(Path(metadata_path).read_text())
    schema = json.loads(runtime_core.orbit_control_schema_json(2))
    assert metadata["parameter_names"] == [field["name"] for field in schema]
    assert metadata["parameter_ranges"]["alpha"] == [0.05, 0.95]
