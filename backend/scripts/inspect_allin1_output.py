"""Inspect All-In-One analysis for a single audio file and print a concise summary."""

from pathlib import Path
import numpy as np
from scipy.io import wavfile

# Insert local shim into sys.modules for natten.functional prior to importing allin1
import importlib.util
import sys

shim_path = Path(__file__).resolve().parents[1] / 'src' / 'natten_shim.py'
if shim_path.exists():
    spec = importlib.util.spec_from_file_location('natten.functional', str(shim_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules['natten.functional'] = module
    spec.loader.exec_module(module)

try:
    import allin1
except Exception as e:
    print("allin1 import failed:", e)
    raise

p = Path("backend/data/audio/TOOL - Right In Two (Audio).mp3")
if not p.exists():
    print("File not found:", p)
    sys.exit(1)

print("Running allin1.analyze on:", p)

# Ensure a fake demix exists so we skip HTDemucs (helps avoid FFmpeg/torchcodec issues).
from scipy.io import wavfile
fake_demix = Path('demix') / 'htdemucs' / p.stem
if not fake_demix.is_dir():
    fake_demix.mkdir(parents=True, exist_ok=True)
    sr = 44100
    duration = 10  # seconds
    t = (np.linspace(0.0, float(duration), int(sr * duration), endpoint=False)).astype(np.float32)
    # simple sine stems with small differences
    stems = {
        'bass.wav': 0.2 * np.sin(2 * np.pi * 110.0 * t),
        'drums.wav': 0.1 * np.sign(np.sin(2 * np.pi * 2.0 * t)),
        'other.wav': 0.15 * np.sin(2 * np.pi * 440.0 * t),
        'vocals.wav': 0.05 * np.sin(2 * np.pi * 660.0 * t),
    }
    for name, y in stems.items():
        wavfile.write(fake_demix / name, sr, (y * 30000).astype('int16'))

res = allin1.analyze(str(p), include_activations=True, include_embeddings=False)
print("--- Summary ---")
print("path:", res.path)
print("bpm:", res.bpm)
print("#beats:", len(res.beats))
print("first 10 beats:", res.beats[:10])
print("#downbeats:", len(res.downbeats))
print("first 10 downbeats:", res.downbeats[:10])
print("#segments:", len(res.segments))
print("first 5 segments:", res.segments[:5])

# Activations
activ = getattr(res, "activations", None)
if activ is not None:
    print("Activations keys:", list(activ.keys()))
    for k, v in activ.items():
        print(f"  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")
else:
    print("No activations returned")

# If labels available
labels = getattr(allin1, "HARMONIX_LABELS", None)
if labels is not None:
    print("HARMONIX_LABELS:", labels)

print("Done")
