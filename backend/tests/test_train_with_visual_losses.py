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
