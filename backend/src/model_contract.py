"""Single source of truth for model I/O contract.

Defines the canonical input/output element ordering and dimensionality for
the ONNX control model used by the runtime and trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


# Defaults (will be overridden if `contracts/model_io_contract.json` exists)
MODEL_INPUT_NAME = "audio_features"
MODEL_OUTPUT_NAME = "control_signals"

# Audio feature names per frame (must align with runtime-core feature extractor).
FEATURE_NAMES: List[str] = [
    "spectral_centroid",
    "spectral_flux",
    "rms_energy",
    "zero_crossing_rate",
    "onset_strength",
    "spectral_rolloff",
]

DEFAULT_WINDOW_FRAMES = 10
DEFAULT_K_BANDS = 6

# Attempt to load canonical JSON contract if present
try:
    import json
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    _contract_path = _root / "contracts" / "model_io_contract.json"
    if _contract_path.exists():
        with open(_contract_path, "r", encoding="utf-8") as _f:
            _cj = json.load(_f)
        _inp = _cj.get("input", {})
        _out = _cj.get("output", {})
        MODEL_INPUT_NAME = _inp.get("name", MODEL_INPUT_NAME)
        FEATURE_NAMES = list(_inp.get("feature_names", FEATURE_NAMES))
        DEFAULT_WINDOW_FRAMES = int(_inp.get("window_frames", DEFAULT_WINDOW_FRAMES))
        DEFAULT_K_BANDS = int(_out.get("k_bands", DEFAULT_K_BANDS))
        MODEL_OUTPUT_NAME = _out.get("name", MODEL_OUTPUT_NAME)
except Exception:
    # Don't fail import; fall back to defaults
    pass


def build_input_names(
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> List[str]:
    """Flattened input feature names in chronological order (oldest -> newest)."""
    names: List[str] = []
    features = list(feature_names)
    for frame_idx in range(window_frames):
        for feat in features:
            names.append(f"frame_{frame_idx}_{feat}")
    return names


def build_output_names(k_bands: int = DEFAULT_K_BANDS) -> List[str]:
    names = ["s_target", "alpha", "omega_scale"]
    names.extend([f"band_gate_{i}" for i in range(int(k_bands))])
    return names


INPUT_NAMES = build_input_names()
OUTPUT_NAMES = build_output_names()
INPUT_DIM = len(INPUT_NAMES)
OUTPUT_DIM = len(OUTPUT_NAMES)


@dataclass(frozen=True)
class ModelContract:
    input_names: List[str]
    output_names: List[str]
    input_dim: int
    output_dim: int


def default_contract() -> ModelContract:
    return ModelContract(
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    )


def contract_for(window_frames: int, k_bands: int) -> ModelContract:
    input_names = build_input_names(window_frames=window_frames)
    output_names = build_output_names(k_bands=k_bands)
    return ModelContract(
        input_names=input_names,
        output_names=output_names,
        input_dim=len(input_names),
        output_dim=len(output_names),
    )
