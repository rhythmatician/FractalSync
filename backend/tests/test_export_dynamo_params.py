import torch
from pathlib import Path

from src.export_model import export_to_onnx


class DummyModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


def test_dynamo_export_uses_dynamic_shapes(monkeypatch, tmp_path):
    input_dim = 8
    output_dim = 5
    model = DummyModel(input_dim, output_dim)

    onnx_path = str(Path(tmp_path) / "dynamo_params.onnx")

    recorded = {}

    def fake_export(model_arg, args, output_path_arg, **kwargs):
        # Record kwargs and simulate write
        recorded.update(kwargs)
        with open(output_path_arg, "wb") as f:
            f.write(b"OK")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    export_to_onnx(model, input_shape=(input_dim,), output_path=onnx_path, metadata={})

    # The first call should have been dynamo=True with dynamic_shapes and opset_version>=18
    assert recorded.get("dynamo", False) is True
    assert "dynamic_shapes" in recorded
    assert recorded.get("opset_version", 0) >= 18
