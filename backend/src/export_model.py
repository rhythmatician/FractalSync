"""
ONNX model export utilities.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import onnx
import torch

from .model import AudioToVisualModel


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

    # Create dummy input
    if len(input_shape) == 1:
        dummy_input = torch.randn(1, *input_shape)
    else:
        dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["audio_features"],
        output_names=["visual_parameters"],
        dynamic_axes={
            "audio_features": {0: "batch_size"},
            "visual_parameters": {0: "batch_size"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Model exported to {output_path}")

    # Save metadata
    metadata_path = output_path.replace(".onnx", "_metadata.json")
    metadata_dict = {
        "input_shape": (
            list(input_shape) if len(input_shape) > 1 else [1, input_shape[0]]
        ),
        "output_dim": 7,
        "parameter_names": [
            "julia_real",
            "julia_imag",
            "color_hue",
            "color_sat",
            "color_bright",
            "zoom",
            "speed",
        ],
        "parameter_ranges": {
            "julia_real": [-2.0, 2.0],
            "julia_imag": [-2.0, 2.0],
            "color_hue": [0.0, 1.0],
            "color_sat": [0.0, 1.0],
            "color_bright": [0.0, 1.0],
            "zoom": [0.1, 10.0],
            "speed": [0.0, 1.0],
        },
    }

    if feature_mean is not None:
        metadata_dict["feature_mean"] = feature_mean.tolist()
    if feature_std is not None:
        metadata_dict["feature_std"] = feature_std.tolist()

    if metadata:
        metadata_dict.update(metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"Metadata saved to {metadata_path}")

    return metadata_path


def load_checkpoint_and_export(
    checkpoint_path: str,
    model_class: type = AudioToVisualModel,
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "models",
    input_dim: int = 60,
):
    """
    Load checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        output_dir: Directory to save ONNX model
        input_dim: Input dimension
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model
    model = model_class(input_dim=input_dim, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get normalization stats
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export
    output_path = os.path.join(output_dir, "model.onnx")
    metadata_path = export_to_onnx(
        model,
        input_shape=(1, input_dim),
        output_path=output_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata={
            "epoch": checkpoint.get("epoch", 0),
            "checkpoint_path": checkpoint_path,
        },
    )

    return output_path, metadata_path
