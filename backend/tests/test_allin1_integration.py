import sys
import types
import numpy as np
import scipy.io.wavfile as wavfile

from src.song_analyzer import SongAnalyzer
from src import allin1_bridge


def _make_wav(path):
    sr = 22050
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    y = np.zeros_like(t)
    # impulse at 0.5s
    y[int(0.5 * sr)] = 1.0
    # scale to int16
    wavfile.write(path, sr, (y * 30000).astype("int16"))


class FakeRes:
    def __init__(self, activ):
        self.activations = activ


def test_analyze_file_fuses_allin1(monkeypatch, tmp_path):
    path = tmp_path / "test.wav"
    _make_wav(str(path))

    # baseline analyzer
    an = SongAnalyzer(sr=22050, hop_length=512, n_fft=2048)
    baseline = an.analyze_song(np.zeros(22050, dtype=np.float32))

    # fake allin1 with a segment activation peaking at 0.5s (100 fps)
    def fake_analyze(p, include_activations=True, model="harmonix-all", device="cpu"):
        seg = np.zeros(100, dtype=float)
        seg[50] = 1.0
        return FakeRes({"segment": seg})

    fake = types.SimpleNamespace()
    fake.analyze = fake_analyze

    monkeypatch.setitem(sys.modules, "allin1", fake)
    allin1_bridge._ALLIN1 = None

    # analyze the file with fusion enabled and large weight to assert effect
    fused = an.analyze_file(str(path), use_allin1=True, allin1_weight_major=0.9)

    maj_before = baseline["section"]["components"]["nov_fused_major"]
    maj_after = fused["section"]["components"]["nov_fused_major"]

    assert maj_after.shape == maj_before.shape
    # The merged activations should have at least one value strictly larger than baseline
    assert np.max(maj_after) >= np.max(maj_before)
    # And because of strong weight, we expect a noticeable increase
    assert np.max(maj_after) - np.max(maj_before) > 1e-6
