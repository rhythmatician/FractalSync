"""Smoke test: run the training entrypoint with visual losses enabled."""

from pathlib import Path
import sys
import subprocess
import wave
import struct
import math


def _write_sine_wav(path: Path, duration_s: float = 0.6, sr: int = 22050):
    n_frames = int(duration_s * sr)
    amplitude = 0.6
    freq = 440.0
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for i in range(n_frames):
            t = i / sr
            sample = int(amplitude * 32767.0 * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("h", sample))


def test_train_with_visual_losses_smoke(tmp_path):
    backend_dir = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    wav_path = data_dir / "sample.wav"
    _write_sine_wav(wav_path, duration_s=0.6, sr=22050)

    # Write small precomputed distance field required by training (in backend/data)
    import numpy as np
    import json as _json

    df_base = Path(__file__).resolve().parents[1] / "data" / "mandelbrot_distance_field"
    df_base.parent.mkdir(parents=True, exist_ok=True)
    res = 8
    field = np.zeros((res, res), dtype=np.float32)
    np.save(str(df_base.with_suffix(".npy")), field)
    meta = {
        "resolution": res,
        "real_range": (-2.5, 1.0),
        "imag_range": (-1.5, 1.5),
        "max_distance": 0.5,
        "slowdown_threshold": 0.02,
    }
    with open(str(df_base.with_suffix(".json")), "w") as f:
        _json.dump(meta, f)

    save_dir = tmp_path / "checkpoints"

    cmd = [
        sys.executable,
        "train.py",
        "--data-dir",
        str(data_dir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--no-gpu-rendering",
        "--julia-resolution",
        "8",
        "--julia-max-iter",
        "8",
        "--max-files",
        "1",
        "--save-dir",
        str(save_dir),
        "--sequence-training",
        "--sequence-unroll-steps",
        "3",
        "--enable-visual-losses",
        "--visual-proxy-resolution",
        "32",
        "--visual-proxy-iter",
        "8",
    ]

    res = subprocess.run(
        cmd, cwd=str(backend_dir), capture_output=True, text=True, timeout=180
    )
    print(res.stdout)
    print(res.stderr)
    assert res.returncode == 0, f"train.py failed: {res.stderr}"
    assert "Training complete!" in res.stdout
