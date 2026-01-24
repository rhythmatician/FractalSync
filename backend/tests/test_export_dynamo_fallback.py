from pathlib import Path

import torch
import torch.nn as nn

from src.export_model import export_to_onnx


class DummyPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


def test_export_raises_when_dynamo_fails(monkeypatch, tmp_path):
    """Dynamo is the single supported path; if dynamo export fails we must surface an error.

    The test simulates a dynamo failure and asserts `export_to_onnx` raises a RuntimeError.
    """
    input_dim = 10
    k = 4
    output_dim = 5 + k
    model = DummyPolicy(input_dim, output_dim)

    onnx_path = str(Path(tmp_path) / "policy_fallback.onnx")

    def fake_export(model_arg, args, output_path_arg, **kwargs):
        # Simulate failure when dynamo is requested
        if kwargs.get("dynamo", False):
            raise RuntimeError("simulated dynamo failure")
        # Should not be called when dynamo fails
        with open(output_path_arg, "wb") as f:
            f.write(b"SIMULATED_ONNX")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    # Expect the exporter to surface the dynamo error (no fallback)
    raised = False
    try:
        export_to_onnx(
            model=model,
            input_shape=(input_dim,),
            output_path=onnx_path,
            metadata={"k_bands": k, "output_dim": output_dim},
        )
    except RuntimeError as e:
        raised = True
        assert "ONNX dynamo export failed" in str(e)

    assert (
        raised
    ), "Expected export_to_onnx to raise RuntimeError when dynamo export fails (no fallback)"
