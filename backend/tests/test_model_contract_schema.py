import json
from pathlib import Path

from src.export_model import export_to_onnx
from src.model_contract import default_contract, build_output_names
import torch.nn as nn


class DummyControl(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


def test_export_metadata_schema_compliance(tmp_path):
    contract = default_contract()
    model = DummyControl(contract.input_dim, contract.output_dim)

    onnx_path = str(Path(tmp_path) / "control.onnx")
    metadata_path = export_to_onnx(
        model=model,
        input_shape=(contract.input_dim,),
        output_path=onnx_path,
        metadata={"k_bands": len(build_output_names()) - 3},
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        md = json.load(f)

    # input_timing checks
    assert "input_timing" in md
    it = md["input_timing"]
    assert isinstance(it.get("sample_rate_hz"), int) and it["sample_rate_hz"] > 0
    assert isinstance(it.get("hop_size_samples"), int) and it["hop_size_samples"] > 0
    assert isinstance(it.get("frame_rate_hz"), (int, float)) and it["frame_rate_hz"] > 0

    # input_normalization checks
    assert "input_normalization" in md
    norm = md["input_normalization"]
    assert norm.get("type") in ("zscore", "none")
    assert norm.get("applied_by") in ("runtime", "model")

    # state inputs
    assert "state_inputs" in md and isinstance(md["state_inputs"], list)
    assert "c_real" in md["state_inputs"]

    # output semantics
    assert "output_semantics" in md and isinstance(md["output_semantics"], dict)
    assert "s_target" in md["output_semantics"]
    st = md["output_semantics"]["s_target"]
    assert isinstance(st.get("range"), list) and len(st.get("range")) == 2
    assert isinstance(st.get("activation"), str)
