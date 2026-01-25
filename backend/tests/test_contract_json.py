import json
from pathlib import Path

from src import model_contract as mc


def test_contract_json_matches_python_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    cj_path = repo_root / "contracts" / "model_io_contract.json"
    assert cj_path.exists(), "contracts/model_io_contract.json must exist"
    cj = json.loads(cj_path.read_text())

    inp = cj["input"]
    out = cj["output"]

    # window frames & features
    assert mc.DEFAULT_WINDOW_FRAMES == int(inp["window_frames"])
    assert mc.FEATURE_NAMES == list(inp["feature_names"])

    # output k bands
    assert mc.DEFAULT_K_BANDS == int(out["k_bands"])

    # Build expected inputs and outputs
    expected_inputs = mc.build_input_names(window_frames=mc.DEFAULT_WINDOW_FRAMES)
    expected_outputs = mc.build_output_names(k_bands=mc.DEFAULT_K_BANDS)

    # Ensure lengths match what contract implies
    assert len(expected_inputs) == mc.INPUT_DIM
    assert len(expected_outputs) == mc.OUTPUT_DIM
