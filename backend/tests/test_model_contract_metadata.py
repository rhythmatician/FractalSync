import json
from pathlib import Path

import torch.nn as nn

from src.export_model import export_to_onnx
from src.model_contract import (
    MODEL_INPUT_NAME,
    MODEL_OUTPUT_NAME,
    default_contract,
    build_output_names,
)


class DummyControl(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


def test_export_contract_metadata(tmp_path):
    contract = default_contract()
    model = DummyControl(contract.input_dim, contract.output_dim)

    onnx_path = str(Path(tmp_path) / "control.onnx")

    # Ensure canonical contract file is accessible during export
    repo_root = Path(__file__).resolve().parents[2]
    contract_path = repo_root / "contracts" / "model_io_contract.json"
    assert contract_path.exists()
    cj = json.loads(contract_path.read_text())
    assert "input_timing" in cj and "input_normalization" in cj
    metadata_path = export_to_onnx(
        model=model,
        input_shape=(contract.input_dim,),
        output_path=onnx_path,
        metadata={"k_bands": len(build_output_names()) - 3},
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    assert md["output_dim"] == contract.output_dim
    assert md["input_feature_names"] == contract.input_names
    assert md["output_name"] == MODEL_OUTPUT_NAME
    assert md["input_name"] == MODEL_INPUT_NAME

    # New contract fields should be present (copied from canonical contract)
    if "input_timing" not in md:
        raise AssertionError(
            f"metadata missing input_timing; keys={sorted(list(md.keys()))}; content={md}"
        )
    assert md["input_timing"].get("sample_rate_hz") == 48000
    assert "input_normalization" in md
    assert md["input_normalization"].get("type") == "zscore"
    assert "state_inputs" in md and "c_real" in md["state_inputs"]
    assert "output_semantics" in md and "s_target" in md["output_semantics"]
