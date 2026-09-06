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

    # Recurrent models (GRU/LSTM/RNN modules) must use the legacy exporter:
    # the dynamo exporter emits malformed Slice nodes on RNN flat weights
    # ("Starts must be a 1-D array" at runtime in onnxruntime-web).
    is_recurrent = any(
        isinstance(m, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN))
        for m in model.modules()
    )

    # Prefer the new dynamo-based exporter; use dynamic_shapes instead of dynamic_axes per warning.
    try:
        if is_recurrent:
            raise RuntimeError(
                "Recurrent model detected; using legacy exporter directly."
            )

        # Dynamo expects dynamic_shapes to be keyed by forward *argument* names
        # (e.g., 'x' in `def forward(self, x):`) rather than ONNX input/output names.
        import inspect

        arg_name = "x"
        sig = inspect.signature(model.forward)
        params = [p.name for p in sig.parameters.values() if p.name != "self"]
        if params:
            arg_name = params[0]

        dynamic_shapes = {arg_name: {0: "batch_size"}}

        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            input_names=["audio_features"],
            output_names=["visual_parameters"],
            # Use dynamic_shapes with the dynamo exporter (preferred) instead of dynamic_axes
            dynamic_shapes=dynamic_shapes,
            # Request a modern opset to avoid post-export version conversion failures
            opset_version=18,
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
                input_names=["audio_features"],
                output_names=["visual_parameters"],
                dynamic_axes={
                    "audio_features": {0: "batch_size"},
                    "visual_parameters": {0: "batch_size"},
                },
                # Request a modern opset to match available implementations
                opset_version=18,
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
        model_proto = onnx.load(output_path)
        opset_version_saved = None
        if model_proto.opset_import:
            opset_version_saved = model_proto.opset_import[0].version
        logging.info(
            f"Model exported successfully to {output_path} (opset={opset_version_saved})"
        )
    except Exception as e:
        logging.warning(f"Could not fully validate ONNX model: {e}")

    # Save metadata
    metadata_path = str(Path(output_path).with_suffix(".onnx_metadata.json"))

    # Ensure model is self-contained (no external data sidecars). If any external data is referenced,
    # load it into the model and re-save so consumers (like the browser) don't need sidecar files.
    try:
        import os
        from onnx import external_data_helper

        model_dir = os.path.dirname(output_path) or "."
        model_proto = onnx.load(output_path)

        # If the model references external data, load it and re-save inline
        try:
            external_data_helper.load_external_data_for_model(model_proto, model_dir)
            # If this succeeded, re-save model with embedded data
            onnx.save_model(model_proto, output_path)

            # Remove any previously generated .data sidecar (we now embed data)
            sidecar_path = f"{output_path}.data"
            if os.path.exists(sidecar_path):
                try:
                    os.remove(sidecar_path)
                except Exception:
                    # Non-fatal: log and continue
                    logging.warning(
                        "Failed to remove external sidecar %s", sidecar_path
                    )
        except Exception:
            # No external data to inline or failed to load; continue silently
            pass
    except Exception:
        # Be conservative: if ONNX runtime or helper isn't available, skip and keep original behavior
        pass

    import runtime_core
    from .model_schema import legacy_visual_export_schema, output_schema

    model_type = None if metadata is None else metadata.get("model_type")
    controls_version = None if metadata is None else metadata.get("controls_version")
    kind = runtime_core.model_output_kind(model_type, controls_version)
    k_bands = 6 if metadata is None else int(metadata.get("k_bands", 6))
    schema = output_schema(kind, k_bands)
    metadata_schema = (
        legacy_visual_export_schema() if kind == "legacy_visual" else schema
    )
    parameter_names = [descriptor.name for descriptor in metadata_schema]
    parameter_ranges = {
        descriptor.name: [descriptor.minimum, descriptor.maximum]
        for descriptor in metadata_schema
    }
    output_dim = len(schema)
    if metadata is not None and "output_dim" in metadata:
        actual = int(metadata["output_dim"])
        if actual != output_dim:
            raise ValueError(
                f"output_dim {actual} does not match Rust {kind} schema length {output_dim}"
            )

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
    # Rust schema fields remain authoritative even when a caller supplies stale
    # copies in otherwise valid auxiliary metadata.
    metadata_dict["output_dim"] = output_dim
    metadata_dict["parameter_names"] = parameter_names
    metadata_dict["parameter_ranges"] = parameter_ranges

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata_path
