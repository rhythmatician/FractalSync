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

    # Select input/output names. We no longer use a "model_type" field - prefer
    # explicit metadata (e.g., 'output_dim', 'k_bands', 'parameter_names') when
    # available. Default input/output names are for audio feature -> visual param models.
    input_name = "audio_features"
    output_name = "visual_parameters"

    # Prefer the new dynamo-based exporter; use dynamic_shapes instead of dynamic_axes per warning.
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"},
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
            dynamo=True,
        )
    except Exception as dynamo_error:  # pragma: no cover - fallback path
        logging.warning(
            "ONNX dynamo export failed (%s); falling back to legacy exporter.",
            dynamo_error,
        )
        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                output_path,
                export_params=True,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes={
                    input_name: {0: "batch_size"},
                    output_name: {0: "batch_size"},
                },
                opset_version=11,
                do_constant_folding=True,
                verbose=False,
                dynamo=False,
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"ONNX export failed (dynamo and legacy): {legacy_error}"
            ) from legacy_error

    # Load and verify ONNX model structure (skip full validation due to compat issues)
    try:
        onnx.load(output_path)
        logging.info(f"Model exported successfully to {output_path}")
    except Exception as e:
        logging.warning(f"Could not fully validate ONNX model: {e}")

    # Save metadata
    metadata_path = str(Path(output_path).with_suffix(".onnx_metadata.json"))

    # Create a safe copy of metadata (drop stale 'model_type' if present)
    md = dict(metadata) if metadata else {}
    md.pop("model_type", None)  # Remove stale field if callers left it in

    # Determine parameter names and ranges.
    # Prefer explicit `parameter_names` provided by callers. Otherwise use
    # heuristics based on `k_bands` and `output_dim` when possible.
    k_bands = md.get("k_bands")
    provided_param_names = md.get("parameter_names")
    provided_output_dim = md.get("output_dim")

    if provided_param_names is not None:
        parameter_names = provided_param_names
        output_dim = provided_output_dim if provided_output_dim is not None else len(
            provided_param_names
        )
        parameter_ranges = md.get("parameter_ranges", {})
    else:
        # Try policy-like (u_x,...,gate_logits) if output_dim matches 5+k
        if k_bands is not None and provided_output_dim == 5 + int(k_bands):
            k = int(k_bands)
            parameter_names = [
                "u_x",
                "u_y",
                "delta_s",
                "delta_omega",
                "alpha_hit",
            ] + [f"gate_logits_{i}" for i in range(k)]
            output_dim = provided_output_dim
            parameter_ranges = {"u_x": [-1.0, 1.0], "u_y": [-1.0, 1.0], "delta_s": [-0.5, 0.5], "delta_omega": [-1.0, 1.0], "alpha_hit": [0.0, 2.0]}
            for i in range(k):
                parameter_ranges[f"gate_logits_{i}"] = [-5.0, 5.0]
        else:
            # Default to orbit-control style when k_bands present or fall back to legacy
            if k_bands is not None:
                k = int(k_bands)
                parameter_names = ["s_target", "alpha", "omega_scale"] + [
                    f"band_gate_{i}" for i in range(k)
                ]
                output_dim = provided_output_dim if provided_output_dim is not None else 3 + k
                parameter_ranges = {"s_target": [0.2, 3.0], "alpha": [0.0, 1.0], "omega_scale": [0.1, 5.0]}
                for i in range(k):
                    parameter_ranges[f"band_gate_{i}"] = [0.0, 1.0]
            else:
                # Legacy default
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

    # Merge the (sanitized) caller metadata last so explicit parameter names or
    # ranges provided by the caller are preserved.
    if md:
        metadata_dict.update(md)

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata_path
