"""
ONNX model export utilities.
"""

import json
import os
import logging
from pathlib import Path
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

    # Validate shape and create a deterministic dummy input for tracing
    if len(input_shape) == 0:
        raise ValueError("input_shape must not be empty")
    if len(input_shape) == 1:
        dummy_input = torch.zeros(1, *input_shape)
    else:
        dummy_input = torch.zeros(*input_shape)

    try:
        # Use legacy export API for better compatibility
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
            dynamo=False,  # Disable dynamo to use legacy export
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    # Load and verify ONNX model structure (skip full validation due to compat issues)
    try:
        onnx.load(output_path)
        logging.info(f"Model exported successfully to {output_path}")
    except Exception as e:
        logging.warning(f"Could not fully validate ONNX model: {e}")

    # Save metadata
    metadata_path = str(Path(output_path).with_suffix(".onnx_metadata.json"))
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

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata_path


def load_checkpoint_and_export(
    checkpoint_path: str,
    model_class: type = AudioToVisualModel,
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "models",
    window_frames: int = 10,
    num_features_per_frame: Optional[int] = None,
):
    """
    Load checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        output_dir: Directory to save ONNX model
        window_frames: Number of audio frames
        num_features_per_frame: Features per frame (6, 12, or 18). If None, inferred from checkpoint or defaults to 6.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Determine num_features_per_frame
    if num_features_per_frame is None:
        # Try to infer from checkpoint metadata or feature stats
        feature_mean = checkpoint.get("feature_mean")
        if feature_mean is not None:
            total_features = len(feature_mean)
            if total_features % window_frames != 0:
                raise ValueError(
                    "feature_mean length does not divide evenly by window_frames; "
                    "cannot infer num_features_per_frame"
                )
            num_features_per_frame = total_features // window_frames
            logging.info(
                f"Inferred num_features_per_frame={num_features_per_frame} from checkpoint"
            )
        else:
            # Default to base features when no stats available
            num_features_per_frame = 6
            logging.info(
                "Using default num_features_per_frame=6 (no feature_mean in checkpoint)"
            )

    # Create model
    model = model_class(
        window_frames=window_frames,
        num_features_per_frame=num_features_per_frame,
        **model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get normalization stats
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")

    # Validate feature stats dimensions if present
    expected_dim = num_features_per_frame * window_frames
    if feature_mean is not None and len(feature_mean) != expected_dim:
        raise ValueError(
            f"feature_mean length {len(feature_mean)} does not match expected input dim {expected_dim}"
        )
    if feature_std is not None and len(feature_std) != expected_dim:
        raise ValueError(
            f"feature_std length {len(feature_std)} does not match expected input dim {expected_dim}"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate input dimension
    input_dim = num_features_per_frame * window_frames

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
            "window_frames": window_frames,
            "num_features_per_frame": num_features_per_frame,
            "input_dim": input_dim,
        },
    )

    return output_path, metadata_path
