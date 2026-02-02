from src import visual_metrics as vm
import runtime_core
import pytest


def test_load_builtin_distance_field(monkeypatch):
    # Remove Rust sampler to force Python fallback sampling
    if not hasattr(runtime_core, "sample_distance_field_py"):
        pytest.fail("runtime_core missing sample_distance_field_py")
    if not hasattr(runtime_core, "get_builtin_distance_field_py"):
        pytest.fail("runtime_core missing get_builtin_distance_field_py")
    monkeypatch.delattr(runtime_core, "sample_distance_field_py", raising=False)

    # Provide or fake a builtin distance field
    rows, cols, xmin, xmax, ymin, ymax = runtime_core.get_builtin_distance_field_py(
        "mandelbrot_default"
    )
    # Convert rows/cols to a 2D list for numpy consumption
    # (the binding returns dims only; the actual data is embedded and registered in Rust)
    rows = [[0.0 for _ in range(cols)] for _ in range(rows)]
    # Sample a few points and ensure we get finite floats
    d1 = vm.sample_distance_field(0.0 + 0.0j)
    d2 = vm.sample_distance_field(1.0 + 1.0j)
    assert isinstance(d1, float) and d1 >= 0.0
    assert isinstance(d2, float) and d2 >= 0.0
