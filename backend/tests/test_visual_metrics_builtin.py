from src import visual_metrics as vm
import runtime_core
import pytest


def test_load_builtin_distance_field(monkeypatch):
    # Ensure the runtime_core bindings expose the expected Rust-backed helpers
    if not hasattr(runtime_core, "sample_distance_field_py"):
        pytest.fail("runtime_core missing sample_distance_field_py")
    if not hasattr(runtime_core, "get_builtin_distance_field_py"):
        pytest.fail("runtime_core missing get_builtin_distance_field_py")

    # Delete the Rust sampler; vm.sample_distance_field should now fail clearly
    monkeypatch.delattr(runtime_core, "sample_distance_field_py", raising=False)

    # Access to the builtin distance field metadata should still work
    rows, cols, xmin, xmax, ymin, ymax = runtime_core.get_builtin_distance_field_py(
        "mandelbrot_default"
    )
    assert rows > 0
    assert cols > 0
    assert xmin < xmax and ymin < ymax

    # Calling vm.sample_distance_field without the Rust sampler should raise,
    # since there is no Python fallback implemented in visual_metrics.
    with pytest.raises(Exception):
        vm.sample_distance_field(0.0 + 0.0j)


def test_runtime_core_auto_loads_builtin():
    import importlib

    importlib.reload(runtime_core)
    # After reload, no field has been explicitly registered in Rust; sampling
    # should auto-load the canonical builtin and return a finite float.
    out = runtime_core.sample_distance_field_py([0.0], [0.0])
    assert isinstance(out, list) and isinstance(out[0], float) and out[0] >= 0.0
