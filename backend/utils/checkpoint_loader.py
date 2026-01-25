"""
Utility script to load and export model checkpoints.

Usage:
    python -m utils.checkpoint_loader --checkpoint checkpoints/checkpoint_epoch_10.pt --output model_epoch_10.onnx
    python -m utils.checkpoint_loader --checkpoint checkpoints/checkpoint_epoch_10.pt --inspect
"""

import argparse
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control_model import AudioToControlModel  # noqa: E402
from src.export_model import export_to_onnx  # noqa: E402


def inspect_checkpoint(checkpoint_path):
    """Inspect the contents of a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"\n📋 Checkpoint: {checkpoint_path.name}")
        print("=" * 50)

        # Print keys
        print("\n🔑 Keys in checkpoint:")
        for key in checkpoint.keys():
            print(f"  - {key}")

        # Print epoch info
        if "epoch" in checkpoint:
            print(f"\n📊 Epoch: {checkpoint['epoch']}")

        # Print model state_dict info
        if "model_state_dict" in checkpoint:
            print("\n🧠 Model State Dict:")
            state = checkpoint["model_state_dict"]
            # Show first few parameters
            for name, param in list(state.items())[:3]:
                print(f"  - {name}: {param.shape}")
            print(f"  ... ({len(state)} total parameters)")

        # Print metadata if available
        if "metadata" in checkpoint:
            print("\n📝 Metadata:")
            for key, value in checkpoint["metadata"].items():
                print(f"  - {key}: {value}")

        print("\n" + "=" * 50 + "\n")
        return True

    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return False


def load_and_export(
    checkpoint_path,
    output_path=None,
    window_frames=10,
    n_features_per_frame=6,
    hidden_dims=None,
    k_bands=6,
    include_delta=False,
    include_delta_delta=False,
):
    """Load a checkpoint and export it to ONNX format."""
    checkpoint_path = Path(checkpoint_path)

    if hidden_dims is None:
        hidden_dims = [128, 256, 128]

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Create model with parameters from checkpoint metadata if available
        meta = checkpoint.get("metadata", {})
        window_frames = meta.get("window_frames", window_frames)
        n_features_per_frame = meta.get("n_features_per_frame", n_features_per_frame)
        hidden_dims = meta.get("hidden_dims", hidden_dims)
        k_bands = meta.get("k_bands", k_bands)
        include_delta = meta.get("include_delta", include_delta)
        include_delta_delta = meta.get("include_delta_delta", include_delta_delta)

        print(
            f"🧠 Creating model (window_frames={window_frames}, n_features_per_frame={n_features_per_frame}, "
            f"hidden_dims={hidden_dims}, k_bands={k_bands})"
        )
        model = AudioToControlModel(
            window_frames=window_frames,
            n_features_per_frame=n_features_per_frame,
            hidden_dims=hidden_dims,
            k_bands=k_bands,
            include_delta=include_delta,
            include_delta_delta=include_delta_delta,
        )

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("✅ Model loaded successfully")

        # Determine output filename
        if output_path is None:
            epoch = checkpoint.get("epoch", "unknown")
            output_path = f"model_epoch_{epoch}.onnx"

        output_path = Path(output_path)

        # Export to ONNX
        print(f"💾 Exporting to ONNX: {output_path}")
        input_dim = (
            window_frames
            * n_features_per_frame
            * (1 + (1 if include_delta else 0) + (1 if include_delta_delta else 0))
        )

        # Prepare metadata for export function
        from datetime import datetime
        import subprocess

        # Try to get git hash
        git_hash = None
        try:
            git_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(Path(__file__).parent.parent),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()[:8]
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            # Git not available or not a git repo — not fatal for export
            git_hash = None
            print(f"Warning: could not determine git hash: {exc}")

        metadata_dict = {
            "source_checkpoint": str(checkpoint_path),
            "epoch": checkpoint.get("epoch", None),
            "window_frames": window_frames,
            "n_features_per_frame": n_features_per_frame,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "k_bands": k_bands,
            "include_delta": include_delta,
            "include_delta_delta": include_delta_delta,
            "model_class": "AudioToControlModel",
            "timestamp": datetime.now().isoformat(),
            "git_hash": git_hash,
        }

        # Use export_to_onnx for consistent ONNX export logic
        export_to_onnx(
            model,
            (input_dim,),
            str(output_path),
            metadata=metadata_dict,
        )

        print(f"✅ ONNX model exported: {output_path}")
        metadata_path = output_path.with_suffix(".onnx_metadata.json")
        print(f"📝 Metadata saved: {metadata_path}")
        print("\n✅ Export complete!\n")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Load and export model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a checkpoint
  python -m utils.checkpoint_loader --checkpoint checkpoints/checkpoint_epoch_10.pt --inspect

  # Export checkpoint to ONNX
  python -m utils.checkpoint_loader --checkpoint checkpoints/checkpoint_epoch_10.pt --output model_epoch_10.onnx

  # Export with custom model dimensions
  python -m utils.checkpoint_loader --checkpoint checkpoints/checkpoint_epoch_10.pt \\
    --output model_custom.onnx --window-frames 10 --k-bands 8
        """,
    )

    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--output", help="Output ONNX file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect checkpoint without exporting"
    )
    parser.add_argument(
        "--window-frames", type=int, default=10, help="Window frames (default: 10)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=6,
        help="Features per frame (default: 6)",
        dest="n_features_per_frame",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 256, 128],
        help="Hidden layer dimensions (default: 128 256 128)",
    )
    parser.add_argument(
        "--k-bands", type=int, default=6, help="Number of band gates (default: 6)"
    )
    parser.add_argument(
        "--include-delta",
        action="store_true",
        help="Include delta features",
    )
    parser.add_argument(
        "--include-delta-delta",
        action="store_true",
        help="Include delta-delta features",
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.checkpoint)
    else:
        load_and_export(
            args.checkpoint,
            args.output,
            window_frames=args.window_frames,
            n_features_per_frame=args.n_features_per_frame,
            hidden_dims=args.hidden_dims,
            k_bands=args.k_bands,
            include_delta=args.include_delta,
            include_delta_delta=args.include_delta_delta,
        )


if __name__ == "__main__":
    main()
