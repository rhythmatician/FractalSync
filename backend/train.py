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
from runtime_core import FeatureExtractor, SAMPLE_RATE, HOP_LENGTH, N_FFT  # noqa: E402

# GPU rendering optimization imports
try:
    from src.julia_gpu import GPUJuliaRenderer

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def _run_preflight_parity() -> None:
    """Run scripts/preflight_parity.py as a subprocess; abort on failure.

    Runs before any long-running work so a diverged training mirror can never
    waste a training session. Subprocess isolation means a broken mirror fails
    cleanly instead of crashing weirdly inside the trainer.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "scripts", "preflight_parity.py")
    if not os.path.exists(script):
        print(
            f"ERROR: preflight parity script not found: {script}\n"
            "Refusing to train without parity verification. "
            "Use --skip-parity-check to bypass (emergency only)."
        )
        sys.exit(1)
    result = subprocess.run([sys.executable, script], cwd=repo_root)
    if result.returncode != 0:
        print(
            "\nAborting training: preflight parity check FAILED "
            f"(exit code {result.returncode}). The training-time mirror and "
            "runtime_core have likely diverged. Fix the mismatch or use "
            "--skip-parity-check (emergency only)."
        )
        sys.exit(result.returncode)


def _runtime_controller_version() -> str:
    """Read CONTROLLER_VERSION from the installed runtime_core.

    The stamp comes from the same Rust source the preflight verifies against,
    so the model records exactly the contract the mirror was checked against.
    """
    try:
        import runtime_core

        version = getattr(runtime_core, "CONTROLLER_VERSION", None)
        if version:
            return str(version)
    except ImportError:
        pass
    return "unknown"


def _runtime_feature_version() -> str:
    """Read FEATURE_VERSION from the installed runtime_core.

    Same contract mechanism as controller_version: the model records the
    feature-extraction semantics the mirror was verified against, and the
    browser refuses mismatched models.
    """
    try:
        import runtime_core

        version = getattr(runtime_core, "FEATURE_VERSION", None)
        if version:
            return str(version)
    except ImportError:
        pass
    return "unknown"


def _runtime_analysis_pipeline_version() -> str:
    """Read ANALYSIS_PIPELINE_VERSION from the installed runtime_core.

    Versions HOW audio reaches the extractor (resampling ownership, hop
    scheduling, epoch semantics) — distinct from FEATURE_VERSION (the
    formulas). The browser refuses models stamped with a different
    pipeline, and refuses pre-timebase models with no stamp at all.
    """
    try:
        import runtime_core

        version = getattr(runtime_core, "ANALYSIS_PIPELINE_VERSION", None)
        if version:
            return str(version)
    except ImportError:
        pass
    return "unknown"


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
    parser.add_argument(
        "--temporal-smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for off-hit control smoothness (speed + jerk)",
    )
    parser.add_argument(
        "--sequence-loss-weight",
        type=float,
        default=0.0,
        help="Weight for sequence-level motion correlation with music hits",
    )
    parser.add_argument(
        "--hit-alignment-weight",
        type=float,
        default=0.0,
        help="Weight for direct hit-to-transition intensity alignment",
    )
    parser.add_argument(
        "--rollout-batch-fraction",
        type=float,
        default=0.0,
        help="Fraction of batches that use runtime-like rollout training",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=64,
        help="Max contiguous frames to include in rollout windows",
    )
    parser.add_argument(
        "--rollout-teacher-forcing",
        type=float,
        default=0.2,
        help="Teacher forcing factor for rollout carryover dynamics",
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.0,
        help="Weight for rollout-mode sequence loss",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume model/optimizer state from",
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="When resuming, load model weights but reset optimizer state",
    )
    parser.add_argument(
        "--no-cspace-proxies",
        action="store_true",
        help="Disable differentiable c-space proxy supervision (falls back to slow rendered-image losses)",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.1,
        help="Weight for c-space coverage/diversity regularizer (0 disables)",
    )
    parser.add_argument(
        "--scheduled-sampling-max",
        type=float,
        default=0.3,
        help="Max scheduled-sampling probability reached after ramping (0 disables)",
    )
    parser.add_argument(
        "--scheduled-sampling-ramp-epochs",
        type=int,
        default=20,
        help="Epochs over which scheduled sampling ramps from 0 to max",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=1,
        help="Contiguous windows per training clip (1 = legacy window batching; 32-128 recommended for sequence training)",
    )
    parser.add_argument(
        "--anti-dwell-weight",
        type=float,
        default=1.0,
        help="Weight for scale-aware anti-dwell penalty keeping c(t) moving (0 disables)",
    )
    parser.add_argument(
        "--anti-dwell-target",
        type=float,
        default=0.15,
        help="Required per-frame c displacement as fraction of local feature scale",
    )
    parser.add_argument(
        "--zone-weight",
        type=float,
        default=2.0,
        help="Weight for visibility-band constraint keeping c near the Mandelbrot boundary (0 disables)",
    )
    parser.add_argument(
        "--zone-min",
        type=float,
        default=0.01,
        help="Minimum cardioid proximity (interior dead-zone edge)",
    )
    parser.add_argument(
        "--zone-max",
        type=float,
        default=0.45,
        help="Maximum cardioid proximity (exterior dust dead-zone edge)",
    )
    parser.add_argument(
        "--julia-stability-weight",
        type=float,
        default=0.0,
        help="Weight for J(c) frame-to-frame stability loss (0 disables). "
        "Penalizes perceptual c displacement in quiet parts; transients exempt.",
    )
    parser.add_argument(
        "--julia-stability-base",
        type=float,
        default=0.02,
        help="Perceptual displacement allowed in silence (local-scale units)",
    )
    parser.add_argument(
        "--julia-stability-loud-gain",
        type=float,
        default=0.08,
        help="Extra allowed displacement per unit audio energy (loud parts drift more)",
    )
    parser.add_argument(
        "--song-identity-weight",
        type=float,
        default=0.0,
        help="Weight for song-identity region loss (0 disables). Pushes each "
        "song's c(t) centroid at least --song-identity-margin from every other "
        "song's, and keeps each song's path coherent around its own centroid.",
    )
    parser.add_argument(
        "--song-identity-margin",
        type=float,
        default=0.35,
        help="Minimum c-space distance between song home regions",
    )
    parser.add_argument(
        "--region-dwell-weight",
        type=float,
        default=0.0,
        help="Weight for region-dwell loss (0 disables). Penalizes c occupying "
        "the same J(c)-region (perceptual neighborhood) for the whole dwell window.",
    )
    parser.add_argument(
        "--region-dwell-window",
        type=int,
        default=240,
        help="Look-back window in frames (240 = 4s at 60fps) for continuous occupation",
    )
    parser.add_argument(
        "--region-dwell-p",
        type=float,
        default=0.08,
        help="Region radius in proximity units (J-space perceptual axis)",
    )
    parser.add_argument(
        "--region-dwell-phi",
        type=float,
        default=0.5,
        help="Region radius in boundary-angle units (radians)",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use GRU-based temporal encoder instead of flat MLP",
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip the preflight parity check (emergency use only: allows "
        "training even if the training mirror and runtime_core have diverged)",
    )

    args = parser.parse_args()

    if not args.skip_parity_check:
        _run_preflight_parity()

    execute_training_workflow(args)


def execute_training_workflow(args):
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
    print(f"  Temporal smoothness weight: {args.temporal_smoothness_weight}")
    print(f"  Sequence loss weight: {args.sequence_loss_weight}")
    print(f"  Hit alignment weight: {args.hit_alignment_weight}")
    print(f"  Rollout batch fraction: {args.rollout_batch_fraction}")
    print(f"  Rollout horizon: {args.rollout_horizon}")
    print(f"  Rollout teacher forcing: {args.rollout_teacher_forcing}")
    print(f"  Rollout loss weight: {args.rollout_loss_weight}")
    print(f"  C-space proxies: {not args.no_cspace_proxies}")
    print(f"  Coverage weight: {args.coverage_weight}")
    print(f"  Scheduled sampling max: {args.scheduled_sampling_max}")
    print(f"  Clip length: {args.clip_length}")
    print(
        f"  Anti-dwell weight: {args.anti_dwell_weight} (target {args.anti_dwell_target})"
    )
    print(
        f"  Zone band: [{args.zone_min}, {args.zone_max}] (weight {args.zone_weight})"
    )
    print(f"  Recurrent encoder: {args.recurrent}")
    print(f"  Resume checkpoint: {args.resume_checkpoint}")
    print(f"  Resume reset optimizer: {args.resume_reset_optimizer}")
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
    feature_extractor = FeatureExtractor(
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )

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
        recurrent=args.recurrent,
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
        temporal_smoothness_weight=args.temporal_smoothness_weight,
        sequence_loss_weight=args.sequence_loss_weight,
        hit_alignment_weight=args.hit_alignment_weight,
        rollout_batch_fraction=args.rollout_batch_fraction,
        rollout_horizon=args.rollout_horizon,
        rollout_teacher_forcing=args.rollout_teacher_forcing,
        rollout_loss_weight=args.rollout_loss_weight,
        use_cspace_proxies=not args.no_cspace_proxies,
        coverage_weight=args.coverage_weight,
        scheduled_sampling_max=args.scheduled_sampling_max,
        scheduled_sampling_ramp_epochs=args.scheduled_sampling_ramp_epochs,
        clip_length=args.clip_length,
        anti_dwell_weight=args.anti_dwell_weight,
        anti_dwell_target=args.anti_dwell_target,
        zone_weight=args.zone_weight,
        zone_min=args.zone_min,
        zone_max=args.zone_max,
        julia_stability_weight=args.julia_stability_weight,
        julia_stability_base=args.julia_stability_base,
        julia_stability_loud_gain=args.julia_stability_loud_gain,
        song_identity_weight=args.song_identity_weight,
        song_identity_margin=args.song_identity_margin,
        region_dwell_weight=args.region_dwell_weight,
        region_dwell_window=args.region_dwell_window,
        region_dwell_p=args.region_dwell_p,
        region_dwell_phi=args.region_dwell_phi,
    )

    if args.resume_checkpoint:
        ckpt_path = os.path.abspath(args.resume_checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        print(f"  Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=args.device)

        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError(
                "Resume checkpoint missing model_state_dict; cannot resume."
            )
        trainer.model.load_state_dict(model_state)

        if not args.resume_reset_optimizer:
            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                trainer.optimizer.load_state_dict(opt_state)

        ckpt_history = checkpoint.get("history")
        if isinstance(ckpt_history, dict):
            for key, values in ckpt_history.items():
                if key in trainer.history and isinstance(values, list):
                    trainer.history[key] = [float(v) for v in values]

        # Feature normalization stats are recomputed in trainer.train() from the
        # current dataset. runtime_core FeatureExtractor binding properties are
        # read-only in this environment, so we intentionally do not assign them
        # here when resuming.

        print("  Resume load complete")

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
                # Controller contract stamp (ADR 0001): the browser refuses
                # to load an orbit_control model whose controller_version
                # differs from its own runtime's version.
                "controller_version": _runtime_controller_version(),
                # Feature-extraction contract stamp (ADR 0001): same
                # mechanism for the audio feature pipeline.
                "feature_version": _runtime_feature_version(),
                # Analysis-pipeline contract stamp (issue #93): versions the
                # ingestion pipeline (Rust resampling, hop scheduling, epoch
                # semantics). The browser refuses mismatches AND pre-timebase
                # models with no stamp.
                "analysis_pipeline_version": _runtime_analysis_pipeline_version(),
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
