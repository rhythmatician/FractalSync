"""
Training script for orbit-based control signal model.

Usage:
    python train.py --data-dir data/audio --epochs 100 --use-curriculum
"""

import argparse
import os
import sys
import subprocess
import logging
from datetime import datetime
import traceback

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import AudioDataset  # noqa: E402
from src.control_model import AudioToControlModel  # noqa: E402
from src.control_trainer import ControlTrainer  # noqa: E402
from src.visual_metrics import LossVisualMetrics  # noqa: E402
from src.export_model import export_to_onnx  # noqa: E402
from src.runtime_core_bridge import make_feature_extractor  # noqa: E402

# GPU rendering optimization imports
try:
    from src.julia_gpu import GPUJuliaRenderer

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def main():
    """Main training function."""
    # Configure logging so ControlTrainer messages are visible
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/train.log"),
        ],
    )
    parser = argparse.ArgumentParser(
        description="Train orbit-based control signal model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--window-frames", type=int, default=10, help="Number of frames in input window"
    )
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Use curriculum learning with Mandelbrot orbits",
    )
    parser.add_argument(
        "--curriculum-weight",
        type=float,
        default=1.0,
        help="Initial weight for curriculum loss",
    )
    parser.add_argument(
        "--curriculum-decay",
        type=float,
        default=0.50,
        help="Decay factor for curriculum weight per epoch",
    )
    parser.add_argument(
        "--k-bands",
        type=int,
        default=6,
        help="Number of residual bands (epicycles)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--no-gpu-rendering",
        action="store_true",
        help="Disable GPU-accelerated Julia set rendering",
    )
    parser.add_argument(
        "--julia-resolution",
        type=int,
        default=64,
        help="Julia set image resolution",
    )
    parser.add_argument(
        "--julia-max-iter",
        type=int,
        default=50,
        help="Julia set max iterations",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of audio files to load (for quick runs)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Orbit-Based Control Signal Model Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Window frames: {args.window_frames}")
    print(f"Residual bands (k): {args.k_bands}")
    print(f"Use curriculum: {args.use_curriculum}")
    if args.use_curriculum:
        print(f"  Curriculum weight: {args.curriculum_weight}")
        print(f"  Curriculum decay: {args.curriculum_decay}")
    print(f"Device: {args.device}")
    print("Optimizations:")
    print(f"  GPU rendering: {not args.no_gpu_rendering and GPU_AVAILABLE}")
    print(f"  Julia resolution: {args.julia_resolution}x{args.julia_resolution}")
    print(f"  Julia max iterations: {args.julia_max_iter}")
    print(f"  DataLoader workers: {args.num_workers}")
    print("=" * 60)

    # Ensure a consistent multiprocessing start method on Windows to avoid
    # handle duplication errors when spawning worker processes.
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set; ignore.
        pass

    # Initialize components
    print("\n[1/7] Initializing feature extractor...")
    feature_extractor = make_feature_extractor()

    print("[2/7] Loading audio dataset...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        window_frames=args.window_frames,
        max_files=args.max_files,
        cache_dir="data/cache",
    )

    print(f"Found {len(dataset)} audio files")

    print("[3/7] Initializing visual metrics...")
    visual_metrics = LossVisualMetrics()

    print("[4/7] Initializing GPU renderer (if enabled)...")
    julia_renderer = None
    if not args.no_gpu_rendering and GPU_AVAILABLE:
        try:
            julia_renderer = GPUJuliaRenderer(
                width=args.julia_resolution,
                height=args.julia_resolution,
            )
            print(
                f"  GPU renderer initialized: {args.julia_resolution}x{args.julia_resolution}"
            )
        except Exception as e:
            print(f"  Warning: GPU renderer failed: {e}")
            print("  Falling back to CPU rendering")
            julia_renderer = None
    else:
        print("  GPU rendering disabled, using CPU")

    print("[5/7] Creating orbit-based control model...")
    model = AudioToControlModel(
        window_frames=args.window_frames,
        n_features_per_frame=6,
        hidden_dims=[128, 256, 128],
        k_bands=args.k_bands,
        dropout=0.2,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Output dimension: {model.output_dim}")

    print("[6/7] Initializing control trainer...")
    trainer = ControlTrainer(
        model=model,
        feature_extractor=feature_extractor,
        visual_metrics=visual_metrics,
        device=args.device,
        learning_rate=args.learning_rate,
        use_curriculum=args.use_curriculum,
        curriculum_weight=args.curriculum_weight,
        julia_renderer=julia_renderer,
        julia_resolution=args.julia_resolution,
        julia_max_iter=args.julia_max_iter,
        num_workers=args.num_workers,
        k_residuals=args.k_bands,
    )

    print("[7/7] Starting training...")
    print("=" * 60)
    print(f"\nTraining will save checkpoints every 10 epochs to: {args.save_dir}")
    print("\nArchitecture overview:")
    print("  - Model predicts control signals: s, alpha, omega_scale, band_gates")
    print("  - Orbit synthesizer generates deterministic c(t) from controls")
    print("  - Curriculum learning teaches Mandelbrot orbit geometry")
    print("  - Correlation losses map audio features to visual parameters")
    print("=" * 60)

    final_checkpoint = trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        curriculum_decay=args.curriculum_decay,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Export to ONNX
    print("\nExporting model to ONNX format...")
    os.makedirs(args.save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iso_timestamp = datetime.now().isoformat()

    # Get git commit hash
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        print(f"Warning: Could not get git hash: {e}")

    onnx_model_filename = f"model_orbit_control_{timestamp}.onnx"
    onnx_path = os.path.join(args.save_dir, onnx_model_filename)

    try:
        model.eval()
        export_to_onnx(
            model=model,
            input_shape=(1, model.input_dim),
            output_path=onnx_path,
            feature_mean=(
                np.array(feature_extractor.feature_mean, dtype=np.float32)
                if hasattr(feature_extractor, "feature_mean")
                else None
            ),
            feature_std=(
                np.array(feature_extractor.feature_std, dtype=np.float32)
                if hasattr(feature_extractor, "feature_std")
                else None
            ),
            metadata={
                "model_type": "orbit_control",
                "output_dim": model.output_dim,
                "k_bands": args.k_bands,
                "epoch": args.epochs,
                "window_frames": args.window_frames,
                "num_features_per_frame": 6,
                "input_dim": model.input_dim,
                "timestamp": iso_timestamp,
                "git_hash": git_hash,
            },
        )
        print(f"Model exported to: {onnx_path}")
    except Exception as e:
        print(f"Warning: Could not export to ONNX: {e}")

    print(
        "\nTraining history saved to:",
        os.path.join(args.save_dir, "training_history.json"),
    )
    if final_checkpoint:
        print("Final checkpoint:", final_checkpoint)
    else:
        print("Final checkpoint saved to:", args.save_dir)
    print("\n[OK] Training complete! Orbit-based model ready for deployment.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"\n[ERROR] Training failed: {e}")
