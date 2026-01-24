import runtime_core as rc


def test_runtime_core_smoke():
    # Minimal smoke test to ensure runtime_core bindings are importable and callable
    fe = rc.FeatureExtractor(48000, 1024, 4096, False, False)
    assert fe.test_simple() == [1.0, 2.0, 3.0]

    df = rc.DistanceField([1.0] * 16, 4, (0.0, 4.0), (0.0, 4.0), 1.0, 0.05)
    val = df.lookup(0.5, 0.5)
    assert isinstance(val, float) and 0.0 <= val <= 1.0
