"""Smoke test: generate a short WAV and analyze with allin1 fused into SongAnalyzer."""

import os
import sys
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile

try:
    from src.song_analyzer import SongAnalyzer
except Exception as e:
    print("ERROR: failed to import SongAnalyzer:", e)
    raise

SR = 22050
t = np.linspace(0.0, 1.0, SR, endpoint=False)
# 220 Hz sine + small impulse to make some structure
y = 0.3 * np.sin(2 * np.pi * 220.0 * t)
y[int(0.5 * SR)] += 1.0
# write temporary wav
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
wavfile.write(tmp.name, SR, (y * 30000).astype("int16"))

an = SongAnalyzer(sr=SR)
print("Running analyze_file with use_allin1=True")
try:
    res = an.analyze_file(tmp.name, use_allin1=True, allin1_weight_major=0.5)
    print("tempo:", res.get("tempo"))
    sec = res.get("section", {})
    print("section.boundaries_major (frames):", sec.get("boundaries_major")[:10])
    print(
        "nov_fused_major max:", sec.get("components", {}).get("nov_fused_major").max()
    )
    print(
        "activations (sampled):", sec.get("components", {}).get("nov_fused_major")[:10]
    )
except Exception as e:
    print("ERROR during analyze_file:", e)
finally:
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
