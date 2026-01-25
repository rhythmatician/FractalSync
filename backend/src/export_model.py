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
import hashlib
from pathlib import Path as _Path  # local alias for internal path ops

from .model_contract import (
    MODEL_INPUT_NAME,
    MODEL_OUTPUT_NAME,
    DEFAULT_WINDOW_FRAMES,
    DEFAULT_K_BANDS,
    build_output_names,
    contract_for,
    default_contract,
)


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
    # available.
    input_name = MODEL_INPUT_NAME
    output_name = MODEL_OUTPUT_NAME

    # Prefer the new dynamo-based exporter; use dynamic_shapes instead of dynamic_axes
    # and prefer a newer opset to avoid automatic version conversion.
    try:
        # Dynamo-friendly kwargs
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes={
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"},
            },
            opset_version=18,
            do_constant_folding=True,
            verbose=False,
            dynamo=True,
        )
    except Exception as dynamo_error:
        # Dynamo-based export sometimes fails on older onnxscript/torch setups
        # (TypeError in onnxscript during function signature translation). As a
        # pragmatic compatibility shim, retry without dynamo in those cases so
        # tests and CI on varied environments can still validate the exporter.
        msg = str(dynamo_error)
        if "Expecting a type not f<class 'typing.Union'>" not in msg:
            # Fail fast for other unexpected errors.
            raise RuntimeError(
                f"ONNX dynamo export failed: {dynamo_error}"
            ) from dynamo_error
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
                opset_version=18,
                do_constant_folding=True,
                verbose=False,
                dynamo=False,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"ONNX export failed (dynamo and fallback): {fallback_error}"
            ) from fallback_error

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
        output_dim = (
            provided_output_dim
            if provided_output_dim is not None
            else len(provided_param_names)
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
            parameter_ranges = {
                "u_x": [-1.0, 1.0],
                "u_y": [-1.0, 1.0],
                "delta_s": [-0.5, 0.5],
                "delta_omega": [-1.0, 1.0],
                "alpha_hit": [0.0, 2.0],
            }
            for i in range(k):
                parameter_ranges[f"gate_logits_{i}"] = [-5.0, 5.0]
        else:
            # Default to orbit-control style when k_bands present or fall back to
            # contract defaults (audio_features -> control_signals).
            if k_bands is not None:
                k = int(k_bands)
            else:
                k = default_contract().output_dim - 3
            parameter_names = build_output_names(k)
            output_dim = provided_output_dim if provided_output_dim is not None else 3 + k
            parameter_ranges = {
                "s_target": [0.2, 3.0],
                "alpha": [0.0, 1.0],
                "omega_scale": [0.1, 5.0],
            }
            for i in range(k):
                parameter_ranges[f"band_gate_{i}"] = [0.0, 1.0]

    contract_window_frames = int(md.get("window_frames", DEFAULT_WINDOW_FRAMES))
    contract_k_bands = int(k_bands if k_bands is not None else DEFAULT_K_BANDS)
    contract = contract_for(window_frames=int(contract_window_frames), k_bands=int(contract_k_bands))
    metadata_dict = {
        "input_shape": (
            list(input_shape) if len(input_shape) > 1 else [1, input_shape[0]]
        ),
        "input_name": input_name,
        "output_name": output_name,
        "output_dim": output_dim,
        "parameter_names": parameter_names,
        "parameter_ranges": parameter_ranges,
        "input_feature_names": contract.input_names,
    }

    if feature_mean is not None:
        metadata_dict["feature_mean"] = feature_mean.tolist()
    if feature_std is not None:
        metadata_dict["feature_std"] = feature_std.tolist()

    # Merge the (sanitized) caller metadata last so explicit parameter names or
    # ranges provided by the caller are preserved.
    if md:
        metadata_dict.update(md)

    # Compute model hash (SHA256) from the exported ONNX file
    try:
        with open(output_path, "rb") as onnx_f:
            onnx_bytes = onnx_f.read()
            model_hash = hashlib.sha256(onnx_bytes).hexdigest()
            metadata_dict["model_hash"] = model_hash
    except Exception:
        logging.warning("Could not compute model hash for ONNX file: %s", output_path)

    # Compute controller hash from canonical controller sources (if present)
    # Files considered: backend/src/control_model.py, backend/src/control_trainer.py,
    # and runtime-core/src/controller.rs (if the runtime-core crate exists alongside repo).
    try:
        repo_root = _Path(__file__).resolve().parents[1]
        candidate_paths = [
            _Path(__file__).resolve().parent / "control_model.py",
            _Path(__file__).resolve().parent / "control_trainer.py",
            repo_root / "runtime-core" / "src" / "controller.rs",
        ]
        existing = [p for p in candidate_paths if p.exists()]
        if existing:
            h = hashlib.sha256()
            for p in existing:
                with open(p, "rb") as pf:
                    h.update(pf.read())
            metadata_dict["controller_hash"] = h.hexdigest()
    except Exception:
        logging.warning("Could not compute controller hash (one or more files missing)")

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata_path
