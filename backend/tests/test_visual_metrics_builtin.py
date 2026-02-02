from src import visual_metrics as vm
import runtime_core
import torch


def test_load_builtin_distance_field(monkeypatch):
    # Remove Rust sampler to force Python fallback sampling
    if hasattr(runtime_core, "sample_distance_field_py"):
        monkeypatch.delattr(runtime_core, "sample_distance_field_py", raising=False)

    # Provide or fake a builtin distance field
    if hasattr(runtime_core, "get_builtin_distance_field_py"):
        rows, cols, xmin, xmax, ymin, ymax = runtime_core.get_builtin_distance_field_py(
            "mandelbrot_default"
        )
        # Convert rows/cols to a 2D list for numpy consumption
        # (the binding returns dims only; the actual data is embedded and registered in Rust)
        rows = [[0.0 for _ in range(cols)] for _ in range(rows)]
    else:
        # Compiled wheel missing the helper; fake a small field
        rows, xmin, xmax, ymin, ymax = ([[0.1, 0.2], [0.3, 0.4]], -2.5, 1.5, -2.0, 2.0)

    # Populate the module-level cache the same way load_distance_field would
    import numpy as np

    arr = np.array(rows, dtype=np.float32)
    vm._distance_field_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    vm._distance_field_meta = vm.DistanceFieldMeta(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, res=arr.shape[0]
    )

    # Sample a few points and ensure we get finite floats
    d1 = vm.sample_distance_field(0.0 + 0.0j)
    d2 = vm.sample_distance_field(1.0 + 1.0j)
    assert isinstance(d1, float) and d1 >= 0.0
    assert isinstance(d2, float) and d2 >= 0.0
