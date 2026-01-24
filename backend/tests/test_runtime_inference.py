import pytest

from src.runtime_inference import compute_directional_probes, run_policy_step
from src.policy_interface import policy_state_encoder


def dummy_model(inp):
    # Echo a deterministic response compatible with policy_output_decoder shape
    # Output shape: (5 + k_bands,), we assume k_bands=6 -> length 11
    k = 6
    out = [0.0] * (5 + k)
    # Small positive u_x, u_y
    out[0] = 0.01
    out[1] = -0.02
    return out


def test_compute_directional_probes_shape():
    probes = compute_directional_probes(0.1, -0.2, radii=(0.005, 0.02), directions=8)
    assert len(probes) == 16


def test_run_policy_step_with_dummy():
    from src.lobe_state import LobeState

    ls = LobeState(current_lobe=0, n_lobes=2)
    res = run_policy_step(
        dummy_model,
        s=1.02,
        alpha=0.3,
        omega=0.5,
        theta=0.0,
        h_t=0.1,
        loudness=0.0,
        tonalness=0.0,
        noisiness=0.0,
        band_energies=[0.0] * 6,
        lobe_state=ls,
    )
    assert "decoded" in res and "c_new" in res
    assert isinstance(res["c_new"][0], float)
    # dummy_model returns no lobe logits so LobeState should remain unchanged
    assert ls.current_lobe == 0


def test_run_policy_step_with_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    import torch

    # Build a tiny PyTorch model and export to ONNX
    k = 6
    inp_example = policy_state_encoder(
        s=1.02,
        alpha=0.3,
        omega=0.5,
        theta=0.0,
        h_t=0.1,
        loudness=0.0,
        tonalness=0.0,
        noisiness=0.0,
        band_energies=[0.0] * k,
    )
    input_dim = inp_example.shape[0]

    class TinyPolicy(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = torch.nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.fc(x)

    model = TinyPolicy(input_dim, 5 + k)
    onnx_path = tmp_path / "tiny_policy.onnx"
    # Export with batch size 1
    example_in = torch.randn(1, input_dim)
    torch.onnx.export(
        model, example_in, str(onnx_path), input_names=["input"], opset_version=14
    )

    from src.lobe_state import LobeState

    ls = LobeState(current_lobe=0, n_lobes=2)
    res = run_policy_step(
        str(onnx_path),
        s=1.02,
        alpha=0.3,
        omega=0.5,
        theta=0.0,
        h_t=0.1,
        loudness=0.0,
        tonalness=0.0,
        noisiness=0.0,
        band_energies=[0.0] * k,
        lobe_state=ls,
    )
    assert "decoded" in res and "c_new" in res
    assert isinstance(res["c_new"][0], float)
    # ONNX model output included lobe logits: ensure we return them
    assert "lobe_logits" in res["decoded"] or "lobe_logits" in res["decoded"]
