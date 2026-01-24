import json
import tempfile
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


def test_export_policy_metadata(tmp_path):
    # TODO: remove model_type and stale model types
    input_dim = 39
    k = 6
    output_dim = 5 + k
    model = DummyPolicy(input_dim, output_dim)

    onnx_path = str(Path(tmp_path) / "policy.onnx")
    # Provide `k_bands` and `output_dim` and let the exporter derive
    # policy-like parameter names (u_x, u_y, delta_s, delta_omega, alpha_hit, gate_logits_k)
    metadata = {"k_bands": k, "output_dim": output_dim}

    metadata_path = export_to_onnx(
        model=model,
        input_shape=(input_dim,),
        output_path=onnx_path,
        metadata=metadata,
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        md = json.load(f)

    assert md["output_dim"] == output_dim
    assert md["parameter_names"][0] == "u_x"
    assert md["parameter_names"][5].startswith("gate_logits_")
