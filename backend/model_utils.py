"""Utility functions for managing trained models."""

import json
import shutil
from pathlib import Path
from typing import Optional, Tuple


def find_latest_model(
    checkpoint_dir: str = "checkpoints",
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Find the most recently created ONNX model and its metadata.

    Checks for both timestamped (model_physics_*.onnx) and legacy (model.onnx) models.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Tuple of (model_path, metadata_path) or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find all physics models (prioritize timestamped, fall back to legacy)
    timestamped_models = list(checkpoint_path.glob("model_physics_*.onnx"))
    legacy_models = list(checkpoint_path.glob("model.onnx"))

    # Sort timestamped by modification time (newest first)
    if timestamped_models:
        timestamped_models.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        latest_model = timestamped_models[0]
    elif legacy_models:
        latest_model = legacy_models[0]
    else:
        return None

    # Metadata has same name but .onnx_metadata.json extension
    metadata_name = latest_model.stem + ".onnx_metadata.json"
    metadata_path = checkpoint_path / metadata_name

    if metadata_path.exists():
        return str(latest_model), str(metadata_path)
    else:
        return str(latest_model), None


def copy_latest_to_frontend(
    checkpoint_dir: str = "checkpoints",
    frontend_dir: str = "../frontend/public/models",
) -> bool:
    """
    Copy the latest trained model to frontend.

    Args:
        checkpoint_dir: Backend checkpoint directory
        frontend_dir: Frontend models directory (relative to backend)

    Returns:
        True if successful, False otherwise
    """
    result = find_latest_model(checkpoint_dir)
    if not result:
        print(f"No models found in {checkpoint_dir}")
        return False

    model_path, metadata_path = result

    # Resolve frontend path
    frontend_path = Path(checkpoint_dir).parent / frontend_dir
    frontend_path.mkdir(parents=True, exist_ok=True)

    try:
        # Copy model
        dest_model = frontend_path / "model.onnx"
        shutil.copy2(model_path, dest_model)
        print(f"✓ Copied model to {dest_model}")

        # Copy metadata if available
        if metadata_path:
            dest_metadata = frontend_path / "model.onnx_metadata.json"
            shutil.copy2(metadata_path, dest_metadata)
            print(f"✓ Copied metadata to {dest_metadata}")

        # Also keep timestamped copies for reference
        model_name = Path(model_path).name
        metadata_name = Path(metadata_path).name if metadata_path else None

        ref_model = frontend_path / f"models_archive/{model_name}"
        ref_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, ref_model)

        if metadata_path and metadata_name:
            ref_metadata = frontend_path / f"models_archive/{metadata_name}"
            shutil.copy2(metadata_path, ref_metadata)

        print("✓ Archived timestamped copies in models_archive/")

        return True

    except Exception as e:
        print(f"✗ Failed to copy models: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Allow running as: python model_utils.py [checkpoint_dir] [frontend_dir]
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    frontend_dir = sys.argv[2] if len(sys.argv) > 2 else "../frontend/public/models"

    print("Finding latest model...")
    result = find_latest_model(checkpoint_dir)
    if result:
        model_path, metadata_path = result
        print(f"Latest model: {model_path}")
        print(f"Metadata: {metadata_path}")

        # Display epoch info if available
        if metadata_path:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if "epoch" in metadata:
                        print(f"  → Trained for {metadata['epoch']} epoch(s)")
                    if "window_frames" in metadata:
                        print(f"  → Window frames: {metadata['window_frames']}")
                    if "timestamp" in metadata:
                        print(f"  → Timestamp: {metadata['timestamp']}")
                    if "git_hash" in metadata:
                        git_short = metadata["git_hash"][:8]
                        print(f"  → Git commit: {git_short}")
            except Exception:
                pass

        print(f"\nCopying to frontend ({frontend_dir})...")
        if copy_latest_to_frontend(checkpoint_dir, frontend_dir):
            print("\n✓ Models synced to frontend!")
        else:
            print("\n✗ Failed to sync models")
            sys.exit(1)
    else:
        print("✗ No models found")
        sys.exit(1)
