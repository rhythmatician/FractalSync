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
from src.visual_metrics import VisualMetrics  # noqa: E402
from src.export_model import export_to_onnx  # noqa: E402
from src.runtime_core_bridge import make_feature_extractor  # noqa: E402
from src.song_analyzer import SongAnalyzer  # noqa: E402

# GPU rendering optimization imports
try:
    from src.julia_gpu import GPUJuliaRenderer

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def precompute_sections(
    dataset: AudioDataset,
    section_method: str = "auto",
    cache_dir: str = "data/cache",
) -> dict:
    """
    Precompute section boundaries for all audio files in the dataset.

    Args:
        dataset: AudioDataset instance
        section_method: Section detection method ('auto', 'ruptures', 'librosa')
        cache_dir: Directory to cache section data

    Returns:
        Dictionary mapping filename to section analysis data
    """
    import json
    import hashlib
    import librosa
    from pathlib import Path
    from src.runtime_core_bridge import SAMPLE_RATE

    analyzer = SongAnalyzer(sr=SAMPLE_RATE, section_method=section_method)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    section_data = {}

    for idx, audio_file in enumerate(dataset.audio_files):
        filename = audio_file.name

        # Generate cache key
        file_stat = audio_file.stat()
        cache_payload = {
            "path": str(audio_file.resolve()),
            "mtime_ns": file_stat.st_mtime_ns,
            "section_method": section_method,
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, sort_keys=True).encode()
        ).hexdigest()
        section_cache_file = cache_path / f"sections_{cache_key}.json"

        # Check cache
        if section_cache_file.exists():
            try:
                with open(section_cache_file, "r") as f:
                    cached = json.load(f)

                # Validate cache has local_tempo (invalidate old cache format)
                if "local_tempo" not in cached:
                    print(
                        f"  [{idx+1}/{len(dataset)}] {filename}: cache outdated (missing local_tempo), re-analyzing..."
                    )
                    section_cache_file.unlink(missing_ok=True)
                else:
                    section_data[filename] = cached
                    print(
                        f"  [{idx+1}/{len(dataset)}] {filename}: loaded from cache ({len(cached['section_boundaries'])} sections)"
                    )
                    continue
            except Exception:
                section_cache_file.unlink(missing_ok=True)

        # Load audio and analyze
        print(f"  [{idx+1}/{len(dataset)}] {filename}: analyzing...")
        try:
            audio, _ = librosa.load(
                str(audio_file), sr=SAMPLE_RATE, mono=True, duration=5 * 60
            )
            analysis = analyzer.analyze_song(audio)

            # Convert numpy arrays to lists for JSON serialization
            result = {
                "tempo": analysis["tempo"],
                "local_tempo": analysis["local_tempo"].tolist(),
                "section_boundaries": analysis["section_boundaries"].tolist(),
                "section_times": analyzer.frames_to_time(
                    analysis["section_boundaries"]
                ).tolist(),
                "beat_frames": analysis["beat_frames"].tolist(),
                "n_sections": len(analysis["section_boundaries"]) + 1,
            }

            section_data[filename] = result

            # Compute tempo statistics for display
            local_tempo_array = analysis["local_tempo"]

            # Filter out invalid values (inf, nan)
            valid_tempo = local_tempo_array[np.isfinite(local_tempo_array)]

            if len(valid_tempo) > 0:
                tempo_mean = float(np.mean(valid_tempo))
                tempo_std = float(np.std(valid_tempo))
                tempo_min = float(np.min(valid_tempo))
                tempo_max = float(np.max(valid_tempo))

                print(
                    f"    -> {result['n_sections']} sections, "
                    f"tempo={tempo_mean:.1f}±{tempo_std:.1f} BPM "
                    f"(range: {tempo_min:.1f}-{tempo_max:.1f})"
                )
            else:
                # Fallback to global tempo if local tempo is all invalid
                print(
                    f"    -> {result['n_sections']} sections, "
                    f"tempo={result['tempo']:.1f} BPM (global, local tempo unavailable)"
                )

            # Cache result
            with open(section_cache_file, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"    -> Error: {e}")
            continue

    # Save combined section data
    combined_path = cache_path / "all_sections.json"
    with open(combined_path, "w") as f:
        json.dump(section_data, f, indent=2)
    print(f"\nSection data saved to: {combined_path}")

    return section_data


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
        default=True,
        help="Use curriculum learning with Mandelbrot orbits (default: True)",
    )
    parser.add_argument(
        "--no-curriculum",
        dest="use_curriculum",
        action="store_false",
        help="Disable curriculum learning",
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
    parser.add_argument(
        "--precompute-sections",
        action="store_true",
        default=True,
        help="Precompute section boundaries for all audio files (default: True)",
    )
    parser.add_argument(
        "--no-precompute-sections",
        dest="precompute_sections",
        action="store_false",
        help="Disable section boundary precomputation",
    )
    parser.add_argument(
        "--section-method",
        type=str,
        default="ruptures",
        choices=["auto", "ruptures", "librosa"],
        help="Section detection method (default: ruptures)",
    )
    parser.add_argument(
        "--predict-lobes",
        action="store_true",
        default=True,
        help="Train model to predict lobe switching (default: True, requires --precompute-sections)",
    )
    parser.add_argument(
        "--no-predict-lobes",
        dest="predict_lobes",
        action="store_false",
        help="Disable lobe prediction",
    )
    parser.add_argument(
        "--n-lobes",
        type=int,
        default=9,
        help="Number of lobes for lobe prediction (default: 9)",
    )
    parser.add_argument(
        "--lobe-loss-weight",
        type=float,
        default=0.3,
        help="Weight for lobe prediction loss",
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
    if args.precompute_sections:
        print(f"  Section detection: {args.section_method}")
    print("=" * 60)

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

    # Precompute section boundaries if requested
    section_data = None
    if args.precompute_sections:
        print("\n[2.5/7] Precomputing section boundaries...")
        section_data = precompute_sections(dataset, section_method=args.section_method)

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
        n_lobes=args.n_lobes,
        predict_lobes=args.predict_lobes,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Output dimension: {model.output_dim}")
    if args.predict_lobes:
        print(f"Lobe prediction: ENABLED ({args.n_lobes} lobes)")

    print("[6/7] Initializing control trainer...")
    correlation_weights = {}
    if args.predict_lobes:
        correlation_weights["lobe_prediction"] = args.lobe_loss_weight

    trainer = ControlTrainer(
        model=model,
        feature_extractor=feature_extractor,
        visual_metrics=visual_metrics,
        device=args.device,
        learning_rate=args.learning_rate,
        use_curriculum=args.use_curriculum,
        curriculum_weight=args.curriculum_weight,
        correlation_weights=correlation_weights if correlation_weights else None,
        julia_renderer=julia_renderer,
        julia_resolution=args.julia_resolution,
        julia_max_iter=args.julia_max_iter,
        num_workers=args.num_workers,
        k_residuals=args.k_bands,
    )

    # Generate lobe curriculum if lobe prediction is enabled and sections are precomputed
    if args.predict_lobes and section_data:
        print("\n[6.5/7] Generating lobe curriculum from section boundaries...")
        from src.live_controller import LOBE_CHARACTERISTICS

        # Create lobe to index mapping
        lobe_to_index = {}
        for idx, (lobe_key, chars) in enumerate(sorted(LOBE_CHARACTERISTICS.items())):
            lobe_to_index[lobe_key] = idx

        print(f"  Lobe mapping: {len(lobe_to_index)} lobes")

        # Compute total samples (approximate based on dataset size)
        # This is a simplification - in practice you'd want exact sample counts
        approx_samples = len(dataset) * 100  # Rough estimate

        # For now, use first audio file's section data as template
        # In production, you'd merge all section data properly
        if section_data:
            first_section_data = next(iter(section_data.values()))
            trainer._generate_lobe_curriculum(
                section_data=first_section_data,
                n_samples=approx_samples,
                lobe_to_index=lobe_to_index,
            )
            print(f"  Lobe curriculum generated for {approx_samples} samples")
    elif args.predict_lobes and not section_data:
        print("\n⚠️  WARNING: --predict-lobes requires --precompute-sections")
        print("  Lobe prediction will not be trained without section data.")

    print("[7/7] Starting training...")
    print("=" * 60)
    print(f"\nTraining will save checkpoints every 10 epochs to: {args.save_dir}")
    print("\nArchitecture overview:")
    print("  - Model predicts control signals: s, alpha, omega_scale, band_gates")
    if args.predict_lobes:
        print(f"  - Model predicts lobe switching: {args.n_lobes} lobe classes")
    print("  - Orbit synthesizer generates deterministic c(t) from controls")
    print("  - Curriculum learning teaches Mandelbrot orbit geometry")
    print("  - Correlation losses map audio features to visual parameters")
    if args.predict_lobes:
        print(
            f"  - Lobe prediction loss (weight={args.lobe_loss_weight}) teaches section switching"
        )
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
    except Exception as e:
        traceback.print_exc()
        print(f"\n[ERROR] Training failed: {e}")
