import json
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


def test_export_falls_back_from_dynamo(monkeypatch, tmp_path, caplog):
    """If torch.onnx.export raises when dynamo=True, exporter should fall back to legacy and still write files."""
    input_dim = 10
    k = 4
    output_dim = 5 + k
    model = DummyPolicy(input_dim, output_dim)

    onnx_path = str(Path(tmp_path) / "policy_fallback.onnx")

    calls = {"attempts": []}

    def fake_export(model_arg, args, output_path_arg, **kwargs):
        # Record whether dynamo was requested
        dynamo = kwargs.get("dynamo", False)
        calls["attempts"].append(bool(dynamo))
        if dynamo:
            raise RuntimeError("simulated dynamo failure")
        # Simulate writing an ONNX file for legacy exporter
        with open(output_path_arg, "wb") as f:
            f.write(b"SIMULATED_ONNX")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    caplog.clear()
    caplog.set_level("WARNING")

    metadata_path = export_to_onnx(
        model=model,
        input_shape=(input_dim,),
        output_path=onnx_path,
        metadata={"k_bands": k, "output_dim": output_dim},
    )

    # Check that dynamo was attempted first, then fallback called
    assert calls["attempts"][0] is True
    assert calls["attempts"][-1] is False

    # Files exist
    assert Path(onnx_path).exists()
    assert Path(metadata_path).exists()

    # Metadata contains expected fields and the hashes
    with open(metadata_path, "r", encoding="utf-8") as f:
        md = json.load(f)

    assert md["output_dim"] == output_dim
    assert "model_hash" in md and len(md["model_hash"]) == 64
    assert "controller_hash" in md and len(md["controller_hash"]) == 64

    # Warning about dynamo fallback should be present
    found = any("ONNX dynamo export failed" in r.message for r in caplog.records)
    assert found, "Expected a warning about ONNX dynamo export falling back to legacy exporter"
