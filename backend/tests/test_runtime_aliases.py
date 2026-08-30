def test_runtime_aliases_exist_and_call():
    import runtime_core

    # Alias functions should exist without the _py suffix
    assert hasattr(runtime_core, "mandelbrot_distance_estimate")
    assert hasattr(runtime_core, "mandelbrot_distance_estimate")

    # Basic behavior checks
    out = runtime_core.mandelbrot_distance_estimate([2.0])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] > 0.0

    out = runtime_core.mandelbrot_distance_estimate([0.0])
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], float)
