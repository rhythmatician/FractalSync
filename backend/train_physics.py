"""
Training script for physics-based model with curriculum learning.

Usage:
    python train_physics.py --data-dir data/audio --epochs 100 --use-curriculum
"""

import argparse
import os
import sys

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_features import AudioFeatureExtractor
from src.data_loader import AudioDataset
from src.physics_model import PhysicsAudioToVisualModel
from src.physics_trainer import PhysicsTrainer
from src.visual_metrics import VisualMetrics
from src.export_model import export_to_onnx


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train physics-based audio-to-visual model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
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
        default=0.95,
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
        default="models/physics",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export final model to ONNX format",
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
    print("=" * 60)

    # Initialize components
    print("\n[1/6] Initializing feature extractor...")
    feature_extractor = AudioFeatureExtractor(
        sr=22050,
        hop_length=512,
        n_fft=2048,
    )

    print("[2/6] Loading audio dataset...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        window_frames=args.window_frames,
        cache_dir="data/cache",
    )

    print(f"Found {len(dataset)} audio files")

    print("[3/6] Initializing visual metrics...")
    visual_metrics = VisualMetrics()

    print("[4/6] Creating physics-based model...")
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

    print("[5/6] Initializing physics trainer...")
    trainer = PhysicsTrainer(
        model=model,
        feature_extractor=feature_extractor,
        visual_metrics=visual_metrics,
        device=args.device,
        learning_rate=args.learning_rate,
        use_curriculum=args.use_curriculum,
        curriculum_weight=args.curriculum_weight,
    )

    print("[6/6] Starting training...")
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

    # Export to ONNX if requested
    if args.export_onnx:
        print("\nExporting model to ONNX format...")
        os.makedirs(args.save_dir, exist_ok=True)
        onnx_path = os.path.join(args.save_dir, "physics_model.onnx")

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
                },
            )
            print(f"Model exported to: {onnx_path}")
        except Exception as e:
            print(f"Warning: Could not export to ONNX: {e}")

    print("\nTraining history saved to:", os.path.join(args.save_dir, "training_history.json"))
    print("Final checkpoint saved to:", args.save_dir)


if __name__ == "__main__":
    main()
