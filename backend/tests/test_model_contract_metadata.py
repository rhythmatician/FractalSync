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
    metadata_path = export_to_onnx(
        model=model,
        input_shape=(contract.input_dim,),
        output_path=onnx_path,
        metadata={"k_bands": len(build_output_names()) - 3},
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        md = json.load(f)

    assert md["parameter_names"] == contract.output_names
    assert md["output_dim"] == contract.output_dim
    assert md["input_feature_names"] == contract.input_names
    assert md["output_name"] == MODEL_OUTPUT_NAME
    assert md["input_name"] == MODEL_INPUT_NAME
