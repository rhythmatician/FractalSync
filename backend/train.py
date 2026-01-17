"""
Training script for physics-based model with curriculum learning.

Usage:
    python train.py --data-dir data/audio --epochs 100 --use-curriculum
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_features import AudioFeatureExtractor  # noqa: E402
from src.data_loader import AudioDataset  # noqa: E402
from src.physics_model import PhysicsAudioToVisualModel  # noqa: E402
from src.physics_trainer import PhysicsTrainer  # noqa: E402
from src.visual_metrics import VisualMetrics  # noqa: E402
from src.export_model import export_to_onnx  # noqa: E402

# GPU rendering optimization imports
try:
    from src.julia_gpu import GPUJuliaRenderer

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train physics-based audio-to-visual model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
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
        "--damping-factor",
        type=float,
        default=0.95,
        help="Velocity damping factor (0-1)",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.1,
        help="Scaling factor for velocity magnitude",
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
    # GPU rendering optimizations (commit 75c1a43)
    parser.add_argument(
        "--no-gpu-rendering",
        action="store_true",
        help="Disable GPU-accelerated Julia set rendering (use CPU instead)",
    )
    parser.add_argument(
        "--julia-resolution",
        type=int,
        default=64,
        help="Julia set image resolution (default: 64x64, original: 128x128)",
    )
    parser.add_argument(
        "--julia-max-iter",
        type=int,
        default=50,
        help="Julia set max iterations (default: 50, original: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel data loading (default: 4, original: 0)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Physics-Based Model Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Window frames: {args.window_frames}")
    print(f"Use curriculum: {args.use_curriculum}")
    if args.use_curriculum:
        print(f"  Curriculum weight: {args.curriculum_weight}")
        print(f"  Curriculum decay: {args.curriculum_decay}")
    print(f"Damping factor: {args.damping_factor}")
    print(f"Speed scale: {args.speed_scale}")
    print(f"Device: {args.device}")
    print("Optimizations:")
    print(f"  GPU rendering: {not args.no_gpu_rendering and GPU_AVAILABLE}")
    print(f"  Julia resolution: {args.julia_resolution}x{args.julia_resolution}")
    print(f"  Julia max iterations: {args.julia_max_iter}")
    print(f"  DataLoader workers: {args.num_workers}")
    print("=" * 60)

    # Initialize components
    print("\n[1/7] Initializing feature extractor...")
    feature_extractor = AudioFeatureExtractor(
        sr=22050,
        hop_length=512,
        n_fft=2048,
    )

    print("[2/7] Loading audio dataset...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        window_frames=args.window_frames,
        cache_dir="data/cache",
    )

    print(f"Found {len(dataset)} audio files")

    print("[3/7] Initializing visual metrics...")
    visual_metrics = VisualMetrics()

    print("[4/7] Initializing GPU renderer (if enabled)...")
    julia_renderer = None
    if not args.no_gpu_rendering and GPU_AVAILABLE:
        try:
            julia_renderer = GPUJuliaRenderer(
                width=args.julia_resolution,
                height=args.julia_resolution,
            )
            print(
                f"  GPU renderer initialized: {args.julia_resolution}x{args.julia_resolution}, {args.julia_max_iter} iterations"
            )
        except Exception as e:
            print(f"  Warning: GPU renderer failed to initialize: {e}")
            print("  Falling back to CPU rendering")
            julia_renderer = None
    else:
        print(
            f"  GPU rendering disabled, using CPU: {args.julia_resolution}x{args.julia_resolution}, {args.julia_max_iter} iterations"
        )

    print("[5/7] Creating physics-based model...")
    model = PhysicsAudioToVisualModel(
        window_frames=args.window_frames,
        hidden_dims=[128, 256, 128],
        output_dim=9,  # 2 velocity + 2 position + 5 other params
        dropout=0.2,
        predict_velocity=True,
        damping_factor=args.damping_factor,
        speed_scale=args.speed_scale,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Output dimension: {model.output_dim}")

    print("[6/7] Initializing physics trainer...")
    trainer = PhysicsTrainer(
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
    )

    print("[7/7] Starting training...")
    print("=" * 60)
    print(f"\nTraining will save checkpoints every 10 epochs to: {args.save_dir}")
    print("Estimated time per epoch: ~30-60 seconds (depends on audio length)")
    print(f"Total estimated time: {args.epochs * 45 / 3600:.1f} hours\n")
    print("Loss breakdown:")
    print("  - Boundary proximity: Rewards c near Mandelbrot boundary")
    print("  - Directional consistency: Penalizes velocity oscillation")
    print("  - Curriculum learning: Teaches Mandelbrot orbits (decays over time)")
    print("  - Correlation losses: Audio-visual feature mapping")
    print("=" * 60)

    trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        curriculum_decay=args.curriculum_decay,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Export to ONNX by default
    print("\nExporting model to ONNX format...")
    os.makedirs(args.save_dir, exist_ok=True)

    # Use dynamic naming based on configuration to avoid overwriting
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

    onnx_model_filename = f"model_physics_{timestamp}.onnx"
    onnx_path = os.path.join(args.save_dir, onnx_model_filename)

    try:
        model.eval()
        export_to_onnx(
            model=model,
            input_shape=(1, model.input_dim),
            output_path=onnx_path,
            feature_mean=feature_extractor.feature_mean,
            feature_std=feature_extractor.feature_std,
            metadata={
                "model_type": "physics",
                "output_dim": model.output_dim,
                "damping_factor": args.damping_factor,
                "speed_scale": args.speed_scale,
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
    print("Final checkpoint saved to:", args.save_dir)
    print("\n[OK] Training complete! Model available via API at /api/model/latest")


if __name__ == "__main__":
    main()
