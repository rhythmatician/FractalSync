def test_runtime_mandelbrot_estimate_basic():
    import runtime_core

    # simple escaping point (use single coords argument)
    out = runtime_core.mandelbrot_distance_estimate_py([(2.0, 0.0)])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] > 0.0

    # period-2 point
    out = runtime_core.mandelbrot_distance_estimate_py([(-1.0, 0.0)])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] <= 0.0

    # inside origin
    out = runtime_core.mandelbrot_distance_estimate_py([(0.0, 0.0)])
    assert out[0] <= 0.0
