def test_mandelbrot_estimate_accepts_complex_sequence():
    import runtime_core

    # pass a Python sequence of complex numbers
    pts = [0 + 0j, 2 + 0j, -1 + 0j]
    out = runtime_core.mandelbrot_distance_estimate(pts)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[0] <= 0.0  # origin inside
    assert out[1] > 0.0  # 2 escapes
    assert out[2] <= 0.0  # -1 is period-2
