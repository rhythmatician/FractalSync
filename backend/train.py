"""
Main training script.
"""

import argparse
import os
from pathlib import Path

import torch

from src.audio_features import AudioFeatureExtractor
from src.data_loader import AudioDataset
from src.export_model import load_checkpoint_and_export
from src.model import AudioToVisualModel
from src.trainer import Trainer
from src.visual_metrics import VisualMetrics


def main():
    parser = argparse.ArgumentParser(description="Train audio-to-visual model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--window-frames", type=int, default=10, help="Number of frames per window"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=60,
        help="Input dimension (n_features * window_frames)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export to ONNX after training"
    )
    parser.add_argument(
        "--onnx-output-dir",
        type=str,
        default="models",
        help="Directory to save ONNX model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.export_onnx:
        os.makedirs(args.onnx_output_dir, exist_ok=True)

    # Initialize components
    print("Initializing components...")
    feature_extractor = AudioFeatureExtractor()
    visual_metrics = VisualMetrics()

    # Create model
    model = AudioToVisualModel(input_dim=args.input_dim)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "feature_mean" in checkpoint:
            feature_extractor.feature_mean = checkpoint["feature_mean"]
        if "feature_std" in checkpoint:
            feature_extractor.feature_std = checkpoint["feature_std"]

    # Create trainer
    trainer = Trainer(
        model=model,
        feature_extractor=feature_extractor,
        visual_metrics=visual_metrics,
        learning_rate=args.learning_rate,
    )

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        window_frames=args.window_frames,
    )

    print(f"Found {len(dataset)} audio files")

    # Train
    trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )

    # Export to ONNX
    if args.export_onnx:
        print("Exporting to ONNX...")
        latest_checkpoint = max(
            Path(args.save_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        load_checkpoint_and_export(
            str(latest_checkpoint),
            output_dir=args.onnx_output_dir,
            input_dim=args.input_dim,
        )
        print("ONNX export complete!")


if __name__ == "__main__":
    main()
