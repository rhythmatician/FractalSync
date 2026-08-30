import sys
import types
import numpy as np

from src import allin1_bridge as bridge


def test_available_false_when_missing(monkeypatch):
    # Ensure that when allin1 is not present, available() is False and analyze raises
    monkeypatch.setitem(sys.modules, "allin1", None)
    # Clear cached import
    bridge._ALLIN1 = None
    assert not bridge.available()
    try:
        bridge.analyze_get_activations("nope.wav")
        assert False, "Expected AllInOneUnavailable"
    except bridge.AllInOneUnavailable:
        pass


class FakeRes:
    def __init__(self, activ):
        self.activations = activ


def test_analyze_get_activations_and_resample(monkeypatch, tmp_path):
    # Create fake allin1 module
    fake = types.SimpleNamespace()

    def fake_analyze(
        path, include_activations=True, model="harmonix-all", device="cpu"
    ):
        # return a segment activation array at 100 FPS with a peak near 0.5s
        seg = np.zeros(200, dtype=float)
        seg[50] = 1.0
        return FakeRes({"segment": seg})

    fake.analyze = fake_analyze

    monkeypatch.setitem(sys.modules, "allin1", fake)
    bridge._ALLIN1 = None

    out = bridge.analyze_get_activations("ignored.wav", device="cpu")
    assert "activations" in out and "frame_rate" in out
    seg = out["activations"]["segment"]
    assert seg.shape[0] == 200
    # resample to times between 0..1 with 10 points; peak near middle
    # sample exactly at 0.5s where the peak is
    dst = np.array([0.5])
    res = bridge.resample_activation_to_times(seg, src_fps=100.0, dst_times=dst)
    assert res.shape == dst.shape
    assert res[0] > 0.0


def test_resample_activation_out_of_range():
    activ = np.array([0.0, 1.0, 0.0])  # 3 frames at 1fps
    dst = np.array([-1.0, -0.1, 3.0])
    res = bridge.resample_activation_to_times(activ, src_fps=1.0, dst_times=dst)
    assert np.all(res == 0.0)
