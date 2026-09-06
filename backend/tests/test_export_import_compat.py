from pathlib import Path

import json
import onnx
import torch


def _load_exporter():
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Import here so linters don't complain about modifying sys.path at module import time
    from src.export_model import export_to_onnx

    return export_to_onnx


def test_export_writes_model_and_metadata(tmp_path):
    export_to_onnx = _load_exporter()
    out_onnx = tmp_path / "test_model.onnx"

    model = torch.nn.Sequential(torch.nn.Linear(10, 7))

    metadata_path = export_to_onnx(
        model=model,
        input_shape=(10,),
        output_path=str(out_onnx),
    )

    # files exist
    assert out_onnx.exists(), "ONNX model file was not created"
    assert Path(metadata_path).exists(), "Metadata file was not created"

    # Ensure no external sidecar remains after export (we prefer self-contained models)
    assert not list(
        out_onnx.parent.glob(out_onnx.name + ".data")
    ), "External .data sidecar should not exist after export"

    # metadata contains expected keys (keeps the test future-proof by checking presence not exact values)
    with open(metadata_path, "r") as f:
        md = json.load(f)

    for key in ("input_shape", "output_dim", "parameter_names", "parameter_ranges"):
        assert key in md, f"Expected metadata to contain key '{key}'"

    # Legacy visual outputs remain raw identity values at runtime, while their
    # historical browser metadata ranges come from runtime-core's compatibility
    # contract rather than a Python copy.
    import runtime_core

    compatibility = json.loads(runtime_core.legacy_visual_export_ranges_json())
    assert md["parameter_names"] == [field["name"] for field in compatibility]
    assert md["parameter_ranges"] == {
        field["name"]: [field["min"], field["max"]] for field in compatibility
    }

    # Try to load the ONNX model. If it references external data, ensure the data file exists.
    try:
        m = onnx.load(str(out_onnx))
    except Exception as e:
        # Check for missing external-data and validate presence of sidecar file
        msg = str(e)
        if "should be stored in" in msg and "but it doesn't exist" in msg:
            # Find candidate .data files in same directory
            candidates = list(out_onnx.parent.glob(out_onnx.stem + "*.data"))
            assert candidates, "Model requires external data but no .data sidecar found"
            # Now attempt to load again (should succeed if sidecar is present)
            m = onnx.load(str(out_onnx))
        else:
            raise

    # Basic sanity checks on loaded model
    assert m.ir_version >= 3
    assert m.graph is not None


def test_externalize_model_and_loads_ok(tmp_path):
    export_to_onnx = _load_exporter()
    out_onnx = tmp_path / "external_test_model.onnx"

    model = torch.nn.Sequential(torch.nn.Linear(10, 7))

    metadata_path = export_to_onnx(
        model=model,
        input_shape=(10,),
        output_path=str(out_onnx),
    )

    # Load model, then rewrite it to use external data for all initializers
    m = onnx.load(str(out_onnx))

    # Ensure we can externalize all tensors (size_threshold=0 will force all tensors to external data)
    external_data_filename = out_onnx.name + ".data"
    onnx.save_model(
        m,
        str(out_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_filename,
        size_threshold=0,
    )

    # The sidecar must exist now (relative to the model path)
    external_data_path = out_onnx.parent / external_data_filename
    assert Path(external_data_path).exists(), "External data file was not created"

    # Loading the model should succeed now that the sidecar exists
    m2 = onnx.load(str(out_onnx))
    assert m2 is not None

    # metadata still present and valid
    with open(metadata_path, "r") as f:
        md = json.load(f)
    assert "input_shape" in md and "output_dim" in md
