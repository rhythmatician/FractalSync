"""
ONNX model export utilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import torch


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model
        input_shape: Input shape (batch_size, input_dim) or (input_dim,)
        output_path: Path to save ONNX model
        feature_mean: Feature normalization mean
        feature_std: Feature normalization std
        metadata: Additional metadata to save
    """
    model.eval()

    # Validate shape and create a deterministic dummy input for tracing
    if len(input_shape) == 0:
        raise ValueError("input_shape must not be empty")
    if len(input_shape) == 1:
        dummy_input = torch.zeros(1, *input_shape)
    else:
        dummy_input = torch.zeros(*input_shape)

    # Export using the dynamo-based exporter.
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        input_names=["audio_features"],
        output_names=["visual_parameters"],
        dynamic_axes={
            "audio_features": {0: "batch_size"},
            "visual_parameters": {0: "batch_size"},
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False,
        dynamo=True,
    )

    # Load and verify ONNX model structure
    onnx.load(output_path)
    logging.info(f"Model exported successfully to {output_path}")

    # Save metadata
    metadata_path = str(Path(output_path).with_suffix(".onnx_metadata.json"))

    # Determine parameter names and ranges based on metadata
    if metadata and metadata.get("model_type") == "orbit_control":
        # Orbit-based control model outputs: s_target, alpha, omega_scale, band_gates[k]
        k_bands = metadata.get("k_bands", 6)
        output_dim = metadata.get("output_dim", 3 + k_bands)
        parameter_names = ["s_target", "alpha", "omega_scale"] + [
            f"band_gate_{i}" for i in range(k_bands)
        ]
        parameter_ranges = {
            "s_target": [0.2, 3.0],
            "alpha": [0.0, 1.0],
            "omega_scale": [0.1, 5.0],
        }
        for i in range(k_bands):
            parameter_ranges[f"band_gate_{i}"] = [0.0, 1.0]
    else:
        # Default: physics/visual parameter model (legacy)
        output_dim = 7
        parameter_names = [
            "julia_real",
            "julia_imag",
            "color_hue",
            "color_sat",
            "color_bright",
            "zoom",
            "speed",
        ]
        parameter_ranges = {
            "julia_real": [-2.0, 2.0],
            "julia_imag": [-2.0, 2.0],
            "color_hue": [0.0, 1.0],
            "color_sat": [0.0, 1.0],
            "color_bright": [0.0, 1.0],
            "zoom": [0.1, 10.0],
            "speed": [0.0, 1.0],
        }

    metadata_dict = {
        "input_shape": (
            list(input_shape) if len(input_shape) > 1 else [1, input_shape[0]]
        ),
        "output_dim": output_dim,
        "parameter_names": parameter_names,
        "parameter_ranges": parameter_ranges,
    }

    if feature_mean is not None:
        metadata_dict["feature_mean"] = feature_mean.tolist()
    if feature_std is not None:
        metadata_dict["feature_std"] = feature_std.tolist()

    if metadata:
        metadata_dict.update(metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata_path
