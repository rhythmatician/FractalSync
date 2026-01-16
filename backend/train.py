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

import logging


def main():
    parser = argparse.ArgumentParser(description="Train audio-to-visual model")
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
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--window-frames", type=int, default=10, help="Number of frames per window"
    )
    parser.add_argument(
        "--include-delta",
        action="store_true",
        help="Include velocity (first-order derivative) features",
    )
    parser.add_argument(
        "--include-delta-delta",
        action="store_true",
        help="Include acceleration (second-order derivative) features",
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
    parser.add_argument(
        "--no-gpu-rendering",
        action="store_true",
        help="Disable GPU-accelerated Julia rendering (use original CPU rendering)",
    )
    parser.add_argument(
        "--julia-resolution",
        type=int,
        default=64,
        help="Julia set rendering resolution (default: 64, original: 128)",
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
        help="Number of data loader workers (default: 4, set to 0 to disable)",
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.export_onnx:
        os.makedirs(args.onnx_output_dir, exist_ok=True)

    # Initialize components
    logging.info("Initializing components...")
    feature_extractor = AudioFeatureExtractor(
        include_delta=args.include_delta,
        include_delta_delta=args.include_delta_delta,
    )
    visual_metrics = VisualMetrics()

    # Get number of features per frame
    num_features_per_frame = feature_extractor.get_num_features()
    logging.info(f"Using {num_features_per_frame} features per frame")

    # Create model
    model = AudioToVisualModel(
        window_frames=args.window_frames,
        num_features_per_frame=num_features_per_frame,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "feature_mean" in checkpoint:
            feature_extractor.feature_mean = checkpoint["feature_mean"]
        if "feature_std" in checkpoint:
            feature_extractor.feature_std = checkpoint["feature_std"]

    # Create Julia renderer (GPU or original CPU)
    julia_renderer = None
    if not args.no_gpu_rendering:
        from src.julia_gpu import GPUJuliaRenderer

        julia_renderer = GPUJuliaRenderer(
            width=args.julia_resolution,
            height=args.julia_resolution,
        )
        logging.info(
            f"Using GPU Julia renderer: {args.julia_resolution}x{args.julia_resolution}, "
            f"max_iter={args.julia_max_iter}"
        )
    else:
        logging.info(
            f"Using original CPU rendering: {args.julia_resolution}x{args.julia_resolution}, "
            f"max_iter={args.julia_max_iter}"
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        feature_extractor=feature_extractor,
        visual_metrics=visual_metrics,
        learning_rate=args.learning_rate,
        julia_renderer=julia_renderer,
        julia_resolution=args.julia_resolution,
        julia_max_iter=args.julia_max_iter,
        num_workers=args.num_workers,
    )

    # Load dataset
    logging.info(f"Loading dataset from {args.data_dir}...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        window_frames=args.window_frames,
    )

    logging.info(f"Found {len(dataset)} audio files")

    # Train
    trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )

    # Export to ONNX
    if args.export_onnx:
        logging.info("Exporting to ONNX...")
        latest_checkpoint = max(
            Path(args.save_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        load_checkpoint_and_export(
            str(latest_checkpoint),
            output_dir=args.onnx_output_dir,
            window_frames=args.window_frames,
        )
        logging.info("ONNX export complete!")


if __name__ == "__main__":
    main()
