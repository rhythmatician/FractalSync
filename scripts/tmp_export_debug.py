from pathlib import Path
import json
from src.export_model import export_to_onnx
from src.model_contract import default_contract, build_output_names
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


contract = default_contract()
model = Dummy(contract.input_dim, contract.output_dim)
onnx_path = "tmp_control.onnx"
metadata_path = export_to_onnx(
    model,
    (contract.input_dim,),
    onnx_path,
    metadata={"k_bands": len(build_output_names()) - 3},
)
print("metadata_path", metadata_path)
print(open(metadata_path).read())
