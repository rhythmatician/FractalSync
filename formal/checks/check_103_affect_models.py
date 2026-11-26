"""Offline synthetic controls for MTG's MusiCNN valence/arousal ONNX exports."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import librosa
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

RATE = 16_000
FRAME = 512
HOP = 256
PATCH = 187
PATCH_HOP = 93
ASSETS = {
    "msd-musicnn-1": "49668ffec47e52e94b96f45930bb46a28a1368d4bdfb5c05378fa834aca616e1",
    "deam-msd-musicnn-2": "2cb8f33f188d73fcb651f7c065b049959166c7852f454a59c13e66fe2c092bde",
    "emomusic-msd-musicnn-2": "f9e6ac891f07c30d8b8d53b93a760f9bd0c7c45686241b5a4386ce34cda66407",
}


def session(path: Path, expected: str) -> ort.InferenceSession:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected model digest: {path.name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def synthetic(seconds: int) -> dict[str, NDArray[np.float64]]:
    t = np.arange(RATE * seconds, dtype=np.float64) / RATE
    tone = 0.1 * np.sin(2 * np.pi * 220 * t)
    upper = 0.1 * np.sin(2 * np.pi * 880 * t)
    return {
        "silence": np.zeros_like(t),
        "steady_220hz": tone,
        "gain_x3": 3 * tone,
        "220_to_880hz_equal_amplitude": np.where(t < seconds / 2, tone, upper),
    }


def features(
    signal: NDArray[np.float64], mel: NDArray[np.float32]
) -> NDArray[np.float32]:
    # Analytic windows start at sample zero. The diagnostic discards all partial
    # windows and does not reproduce Essentia FrameCutter's boundary behavior.
    frames = np.lib.stride_tricks.sliding_window_view(signal, FRAME)[::HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(FRAME))) ** 2
    bands = np.log1p(10_000 * (power @ mel.T)) / np.log(10.0)
    return np.stack(
        [bands[k : k + PATCH] for k in range(0, len(bands) - PATCH + 1, PATCH_HOP)]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    models = {
        name: session(args.model_dir / f"{name}.onnx", digest)
        for name, digest in ASSETS.items()
    }
    backbone = models["msd-musicnn-1"]
    heads = {name: model for name, model in models.items() if model is not backbone}
    mel = librosa.filters.mel(sr=RATE, n_fft=FRAME, n_mels=96, htk=False, norm="slaney")

    def evaluate(patches: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        (embeddings,) = backbone.run(
            ["embeddings"], {backbone.get_inputs()[0].name: patches}
        )
        if (
            not isinstance(embeddings, np.ndarray)
            or embeddings.shape != (len(patches), 200)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("Invalid MusiCNN embeddings")
        result = {}
        for name, model in heads.items():
            (values,) = model.run(None, {model.get_inputs()[0].name: embeddings})
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (len(patches), 2)
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Invalid valence/arousal output: {name}")
            result[name] = values
        return result

    evaluate(features(synthetic(5)["silence"], mel))
    rows = []
    for seconds in (5, 10, 20, 30):
        for name, waveform in synthetic(seconds).items():
            start = perf_counter()
            patches = features(waveform, mel)
            frontend_ms = (perf_counter() - start) * 1000
            samples = []
            for _ in range(3):
                start = perf_counter()
                prediction = evaluate(patches)
                samples.append(((perf_counter() - start) * 1000, prediction))
            reference = samples[0][1]
            exact_repeat = all(
                np.array_equal(values[key], reference[key])
                for _, values in samples
                for key in reference
            )
            if not exact_repeat:
                raise AssertionError(
                    f"ONNX output changed on repeat for {name}, {seconds}s"
                )
            support = FRAME + (PATCH - 1) * HOP
            starts = np.arange(len(patches)) * PATCH_HOP * HOP / RATE
            rows.append(
                {
                    "control": name,
                    "seconds": seconds,
                    "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                    "rms": float(np.sqrt(np.mean(waveform**2))),
                    "tensor_shape": patches.shape,
                    "frontend_ms_once": frontend_ms,
                    "onnx_backbone_plus_two_heads_ms_median": median(
                        ms for ms, _ in samples
                    ),
                    "exact_repeat": exact_repeat,
                    "patch_start_seconds": starts.tolist(),
                    "patch_end_exclusive_seconds": (starts + support / RATE).tolist(),
                    "raw_valence_arousal": {
                        key: values.tolist() for key, values in reference.items()
                    },
                }
            )
    print(
        json.dumps(
            {
                "schema": "affect-onnx-synthetic/1",
                "models_sha256": ASSETS,
                "versions": {
                    pkg: version(pkg) for pkg in ("onnxruntime", "numpy", "librosa")
                },
                "execution_provider": "CPUExecutionProvider",
                "threads": 1,
                "frontend": {
                    "sample_rate": RATE,
                    "frame": FRAME,
                    "hop": HOP,
                    "patch": PATCH,
                    "patch_hop": PATCH_HOP,
                },
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
