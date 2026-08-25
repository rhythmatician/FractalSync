"""
Trainer for control signal model with orbit-based synthesis.

Trains model to predict control signals that drive deterministic orbit synthesis.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .control_model import AudioToControlModel
from .cspace_proxies import cardioid_proximity, synthesize_c
from .data_loader import AudioDataset
from .visual_metrics import LossVisualMetrics
from runtime_core import (
    DEFAULT_BASE_OMEGA,
    DEFAULT_K_RESIDUALS,
    DEFAULT_ORBIT_SEED,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    lobe_point_at_angle,
    ResidualParams,
    OrbitState,
    DEFAULT_RESIDUAL_CAP,
    FeatureExtractor,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
)
from .julia_gpu import GPUJuliaRenderer

logger = logging.getLogger(__name__)


class ControlLoss(nn.Module):
    """Loss for control signal prediction."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self, predicted_controls: torch.Tensor, target_controls: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss between predicted and target control signals."""
        return self.weight * torch.mean((predicted_controls - target_controls) ** 2)


class CorrelationLoss(nn.Module):
    """Negative correlation loss to maximize positive correlation."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        y = y.flatten()
        # Guard against invalid values propagating into correlation math.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        correlation = numerator / (denominator + 1e-8)
        return -correlation


class ControlTrainer:
    """Trainer for control signal model with orbit synthesis."""

    def __init__(
        self,
        model: AudioToControlModel,
        visual_metrics: LossVisualMetrics,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        use_curriculum: bool = True,
        curriculum_weight: float = 1.0,
        correlation_weights: Optional[Dict[str, float]] = None,
        julia_renderer: Optional[GPUJuliaRenderer] = None,
        julia_resolution: int = 128,
        julia_max_iter: int = 100,
        num_workers: int = 0,
        k_residuals: int = DEFAULT_K_RESIDUALS,
        temporal_smoothness_weight: float = 0.0,
        sequence_loss_weight: float = 0.0,
        hit_alignment_weight: float = 0.0,
        rollout_batch_fraction: float = 0.0,
        rollout_horizon: int = 64,
        rollout_teacher_forcing: float = 0.2,
        rollout_loss_weight: float = 0.0,
    ):
        """
        Initialize control trainer.

        Args:
            model: Control signal model
            feature_extractor: Audio feature extractor
            visual_metrics: Visual metrics calculator
            device: Training device
            learning_rate: Learning rate
            use_curriculum: Use curriculum learning
            curriculum_weight: Weight for curriculum loss
            correlation_weights: Weights for correlation losses
            julia_renderer: Optional GPU renderer
            julia_resolution: Julia set resolution
            julia_max_iter: Julia set max iterations
            num_workers: DataLoader workers
            k_residuals: Number of residual circles
            temporal_smoothness_weight: Weight for off-hit control smoothness
            sequence_loss_weight: Weight for sequence-level audio/visual motion correlation
            hit_alignment_weight: Weight for explicit hit-to-transition alignment
            rollout_batch_fraction: Fraction of batches to train with rollout mode
            rollout_horizon: Maximum contiguous window length for rollout mode
            rollout_teacher_forcing: Blend factor for predicted vs carried control state
            rollout_loss_weight: Weight for rollout-mode sequence loss
        """
        self.model: AudioToControlModel = model.to(device)

        # Feature extractor is guaranteed to be present after initialization
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )
        self.visual_metrics = visual_metrics
        self.device = device
        self.use_curriculum = use_curriculum
        self.curriculum_weight = curriculum_weight
        self.julia_renderer = julia_renderer
        self.julia_resolution = julia_resolution
        self.julia_max_iter = julia_max_iter
        self.num_workers = num_workers
        self.k_residuals = k_residuals
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.sequence_loss_weight = sequence_loss_weight
        self.hit_alignment_weight = hit_alignment_weight
        self.rollout_batch_fraction = rollout_batch_fraction
        self.rollout_horizon = rollout_horizon
        self.rollout_teacher_forcing = rollout_teacher_forcing
        self.rollout_loss_weight = rollout_loss_weight
        self.residual_params = ResidualParams(
            k_residuals=k_residuals,
            residual_cap=DEFAULT_RESIDUAL_CAP,
            radius_scale=1.0,
        )

        # Default correlation weights
        default_weights = {
            "timbre_color": 1.0,
            "transient_impact": 1.0,
            "loudness_distance": 1.0,
            "control_loss": 1.0,
        }
        self.correlation_weights = {**default_weights, **(correlation_weights or {})}

        # Loss functions
        self.correlation_loss = CorrelationLoss()
        self.control_loss = ControlLoss(
            weight=self.correlation_weights.get("control_loss", 1.0)
        )

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = 1.0

        # Curriculum data
        self.curriculum_positions: Optional[torch.Tensor] = None
        self.curriculum_velocities: Optional[torch.Tensor] = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "control_loss": [],
            "timbre_color_loss": [],
            "transient_impact_loss": [],
            "loudness_distance_loss": [],
            "temporal_smoothness_loss": [],
            "sequence_perceptual_loss": [],
            "hit_alignment_loss": [],
            "rollout_loss": [],
            "alignment_proxy": [],
        }
        # Track last checkpoint for reporting
        self.last_checkpoint_path: Optional[str] = None
        self.best_checkpoint_path: Optional[str] = None
        self.best_alignment_proxy: Optional[float] = None

    def _generate_curriculum_data(self, n_samples: int):
        """Generate curriculum learning data from preset orbits."""
        logger.info(f"Generating curriculum data: {n_samples} samples")
        thetas = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        positions = []
        velocities = []
        for idx, theta in enumerate(thetas):
            current = lobe_point_at_angle(1, 0, float(theta), 1.02)
            next_theta = thetas[(idx + 1) % len(thetas)]
            nxt = lobe_point_at_angle(1, 0, float(next_theta), 1.02)
            positions.append([current.real, current.imag])
            velocities.append([nxt.real - current.real, nxt.imag - current.imag])

        self.curriculum_positions = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )
        self.curriculum_velocities = torch.tensor(
            velocities, dtype=torch.float32, device=self.device
        )

        logger.info(
            f"Curriculum generated: positions={self.curriculum_positions.shape}, "
            f"velocities={self.curriculum_velocities.shape}"
        )

    def _extract_control_targets_from_curriculum(
        self, positions: torch.Tensor, velocities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract control signal targets from curriculum positions/velocities.

        This is a simplified mapping - in practice, we derive s, alpha, etc.
        from the curriculum orbit parameters.

        Args:
            positions: Tensor of shape (batch_size, 2) with position data
            velocities: Optional tensor of shape (batch_size, 2) with velocity data

        Returns:
            Tensor of shape (batch_size, output_dim) with control targets
        """
        batch_size = positions.shape[0]

        # Compute s from position magnitude (near boundary ~1.0)
        position_mag = torch.norm(positions, dim=1)
        s_target = torch.clamp(position_mag * 1.5, 0.2, 3.0)

        # Compute alpha from velocity magnitude (higher velocity = more residual)
        if velocities is not None:
            velocity_mag = torch.norm(velocities, dim=1)
            alpha = torch.clamp(velocity_mag * 2.0, 0.0, 1.0)
        else:
            alpha = (
                torch.ones(batch_size, device=self.device) * 0.3
            )  # Default amplitude

        # Omega scale from velocity direction changes (placeholder)
        omega_scale = torch.ones(batch_size, device=self.device) * 1.0

        # Band gates (default to open)
        band_gates = torch.ones(batch_size, self.k_residuals, device=self.device) * 0.7

        # Stack control targets
        control_targets = torch.cat(
            [
                s_target.unsqueeze(1),
                alpha.unsqueeze(1),
                omega_scale.unsqueeze(1),
                band_gates,
            ],
            dim=1,
        )

        return control_targets

    @staticmethod
    def _normalize_01(x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] with numerical stability."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = torch.min(x)
        x_max = torch.max(x)
        return (x - x_min) / (x_max - x_min + 1e-8)

    @staticmethod
    def _safe_mean(x: torch.Tensor) -> torch.Tensor:
        """Mean that defaults to zero when tensor has no elements."""
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return torch.mean(x)

    @staticmethod
    def _sanitize_scalar(x: torch.Tensor) -> torch.Tensor:
        """Ensure scalar loss term is finite."""
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _alignment_proxy(
        sequence_perceptual_loss: float,
        hit_alignment_loss: float,
    ) -> float:
        """Higher-is-better proxy for hit-correlated transitions.

        sequence_perceptual_loss is a negative-correlation loss (lower is better),
        while hit_alignment_loss is MSE (lower is better).
        """
        return float(-sequence_perceptual_loss - hit_alignment_loss)

    def _compute_sequence_losses(
        self,
        controls: torch.Tensor,
        spectral_flux: torch.Tensor,
        onset_strength: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hit-aware sequence losses over a contiguous control sequence."""
        n_steps = controls.shape[0]
        if n_steps <= 1:
            zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            return zero, zero, zero

        control_delta = controls[1:] - controls[:-1]
        control_speed = torch.norm(control_delta, dim=1)

        hit_proxy = self._normalize_01(0.5 * spectral_flux + 0.5 * onset_strength)
        hit_levels = hit_proxy[1:]
        off_hit_gate = 1.0 - hit_levels

        speed_sq = control_speed**2
        smooth_speed_loss = self._safe_mean(off_hit_gate * speed_sq)

        if n_steps > 2:
            control_jerk = torch.norm(control_delta[1:] - control_delta[:-1], dim=1)
            jerk_gate = 1.0 - hit_proxy[2:]
            smooth_jerk_loss = self._safe_mean(jerk_gate * (control_jerk**2))
        else:
            smooth_jerk_loss = torch.tensor(
                0.0, device=self.device, dtype=torch.float32
            )

        temporal_smoothness_loss = smooth_speed_loss + 0.5 * smooth_jerk_loss
        sequence_perceptual_loss = self.correlation_loss(hit_levels, control_speed)

        speed_norm = self._normalize_01(control_speed)
        hit_alignment_loss = self._safe_mean((speed_norm - hit_levels) ** 2)
        temporal_smoothness_loss = self._sanitize_scalar(temporal_smoothness_loss)
        sequence_perceptual_loss = self._sanitize_scalar(sequence_perceptual_loss)
        hit_alignment_loss = self._sanitize_scalar(hit_alignment_loss)
        return temporal_smoothness_loss, sequence_perceptual_loss, hit_alignment_loss

    def _compute_rollout_loss(
        self,
        controls: torch.Tensor,
        spectral_flux: torch.Tensor,
        onset_strength: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rollout loss on contiguous same-segment windows.

        This simulates a light runtime-like carryover by blending each step with
        the previously simulated control state.
        """
        if controls.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        seg_cpu = segment_ids.detach().cpu().tolist()
        runs: List[Tuple[int, int]] = []
        run_start = 0
        for idx in range(1, len(seg_cpu)):
            if seg_cpu[idx] != seg_cpu[idx - 1]:
                runs.append((run_start, idx))
                run_start = idx
        runs.append((run_start, len(seg_cpu)))

        rollout_terms: List[torch.Tensor] = []
        carry_weight = max(0.0, min(1.0, 1.0 - self.rollout_teacher_forcing))
        teacher_weight = max(0.0, min(1.0, self.rollout_teacher_forcing))

        for start, end in runs:
            run_len = end - start
            if run_len <= 1:
                continue

            horizon = min(self.rollout_horizon, run_len)
            seq_controls = controls[start : start + horizon]
            seq_flux = spectral_flux[start : start + horizon]
            seq_onset = onset_strength[start : start + horizon]

            sim_controls: List[torch.Tensor] = [seq_controls[0]]
            for t in range(1, horizon):
                sim_controls.append(
                    teacher_weight * seq_controls[t] + carry_weight * sim_controls[-1]
                )

            simulated = torch.stack(sim_controls, dim=0)
            smooth_l, seq_l, hit_l = self._compute_sequence_losses(
                simulated,
                seq_flux,
                seq_onset,
            )
            rollout_terms.append(smooth_l + seq_l + hit_l)

        if not rollout_terms:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return torch.stack(rollout_terms).mean()

    def _synthesize_c_differentiable(
        self,
        s_target: torch.Tensor,
        alpha: torch.Tensor,
        band_gates: torch.Tensor,
        thetas: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable c-space synthesis (delegates to cspace_proxies).

        Thin wrapper over :func:`cspace_proxies.synthesize_c`, which mirrors
        ``runtime_core::controller::synthesize`` and is parity-tested against
        the Rust bindings in backend/tests/test_synthesis_parity.py.
        """
        return synthesize_c(
            s_target=s_target,
            alpha=alpha,
            band_gates=band_gates,
            thetas=thetas,
            k_residuals=self.k_residuals,
            residual_cap=DEFAULT_RESIDUAL_CAP,
        )

    def _cardioid_proximity_differentiable(self, c: torch.Tensor) -> torch.Tensor:
        """Cardioid proximity proxy (delegates to cspace_proxies).

        Mirrors ``runtime_core::proxies::mandelbrot_cardioid_proximity``;
        parity-tested against the Rust bindings.
        """
        return cardioid_proximity(c)
    def train_epoch(
        self, dataloader: DataLoader, epoch: int, curriculum_decay: float = 0.95
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_control_loss = 0.0
        total_timbre_color = 0.0
        total_transient_impact = 0.0
        total_loudness_distance = 0.0
        total_temporal_smoothness = 0.0
        total_sequence_perceptual = 0.0
        total_hit_alignment = 0.0
        total_rollout = 0.0
        n_batches = 0

        # Generate curriculum data if needed
        if self.use_curriculum and self.curriculum_positions is None:
            total_samples = len(dataloader.dataset)  # type: ignore
            self._generate_curriculum_data(total_samples)

        # Curriculum weight decays over epochs
        current_curriculum_weight = self.curriculum_weight * (curriculum_decay**epoch)

        sample_idx = 0

        for batch_idx, batch_item in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Batches",
            leave=True,
            mininterval=0.5,
        ):
            # Extract features
            segment_ids = None
            if isinstance(batch_item, (tuple, list)):
                features = batch_item[0]
                if len(batch_item) > 1:
                    segment_ids = batch_item[1]
            else:
                features = batch_item

            features = features.to(self.device)
            if segment_ids is None:
                segment_ids = torch.arange(features.shape[0], dtype=torch.int64)
            segment_ids = segment_ids.to(self.device)
            batch_size = features.shape[0]

            # Get curriculum targets if available
            control_targets = None
            if self.use_curriculum and self.curriculum_positions is not None:
                end_idx = min(sample_idx + batch_size, len(self.curriculum_positions))
                actual_batch_size = end_idx - sample_idx

                if actual_batch_size > 0:
                    curriculum_pos = self.curriculum_positions[sample_idx:end_idx]
                    curriculum_vel = (
                        self.curriculum_velocities[sample_idx:end_idx]
                        if self.curriculum_velocities is not None
                        else None
                    )

                    control_targets = self._extract_control_targets_from_curriculum(
                        curriculum_pos, curriculum_vel
                    )

                    if actual_batch_size < batch_size:
                        features = features[:actual_batch_size]
                        segment_ids = segment_ids[:actual_batch_size]
                        batch_size = actual_batch_size

            sample_idx += batch_size

            # Forward pass
            predicted_controls = self.model(features)

            # Parse control signals
            parsed = self.model.parse_output(predicted_controls)
            s_target = parsed["s_target"]
            alpha = parsed["alpha"]
            omega_scale = parsed["omega_scale"]
            band_gates = parsed["band_gates"]

            # Synthesize c(t) using runtime_core (cardioid lobe)
            c_values = []
            for i in range(batch_size):
                state = OrbitState.new_with_seed(
                    lobe=1,
                    sub_lobe=0,
                    theta=float(i * 2 * np.pi / batch_size),
                    omega=float(DEFAULT_BASE_OMEGA * omega_scale[i].detach().item()),
                    s=float(s_target[i].detach().item()),
                    alpha=float(alpha[i].detach().item()),
                    k_residuals=self.k_residuals,
                    residual_omega_scale=DEFAULT_RESIDUAL_OMEGA_SCALE,
                    seed=int(DEFAULT_ORBIT_SEED + i),
                )
                rp = self.residual_params or ResidualParams(
                    k_residuals=DEFAULT_K_RESIDUALS,
                    residual_cap=DEFAULT_RESIDUAL_CAP,
                    radius_scale=1.0,
                )
                c = state.synthesize(rp, band_gates[i].detach().cpu().tolist())
                c_values.append(c)

            # Extract audio features for correlation
            n_features_per_frame = self.feature_extractor.num_features_per_frame()
            window_frames = features.shape[1] // n_features_per_frame
            features_reshaped = features.view(
                batch_size, window_frames, n_features_per_frame
            )
            avg_features = features_reshaped.mean(dim=1)

            spectral_centroid = avg_features[:, 0]
            spectral_flux = avg_features[:, 1]
            onset_strength = avg_features[:, 4]

            # Render Julia sets for visual metrics
            images = []
            color_hues = []
            temporal_changes = []

            prev_image = None
            for i in range(batch_size):
                seed = c_values[i]
                if self.julia_renderer is not None:
                    try:
                        image = self.julia_renderer.render(
                            seed=seed,
                            max_iter=self.julia_max_iter,
                        )
                    except Exception as e:
                        logger.warning(f"GPU rendering failed: {e}")
                        image = self.visual_metrics.render_julia_set(
                            seed=seed,
                            width=self.julia_resolution,
                            height=self.julia_resolution,
                            max_iter=self.julia_max_iter,
                        )
                else:
                    image = self.visual_metrics.render_julia_set(
                        seed=seed,
                        width=self.julia_resolution,
                        height=self.julia_resolution,
                        max_iter=self.julia_max_iter,
                    )

                metrics = self.visual_metrics.compute_all_metrics(
                    image, prev_image=prev_image
                )

                images.append(image)
                # Use s_target as proxy for color hue (example correlation)
                color_hues.append(s_target[i])
                temporal_changes.append(
                    torch.tensor(
                        metrics["temporal_change"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )

                prev_image = image

            color_hue_tensor = torch.stack(color_hues)
            temporal_change_tensor = torch.stack(temporal_changes)

            # Compute correlation losses
            timbre_color_loss = self.correlation_loss(
                spectral_centroid, color_hue_tensor
            )
            transient_impact_loss = self.correlation_loss(
                spectral_flux, temporal_change_tensor
            )

            # Loudness-distance (negative correlation) loss
            # Loudness proxy: RMS feature (index 2 of avg_features)
            spectral_rms = avg_features[:, 2]

            # Calculate the distance between `c` and the Mandelbrot set boundary
            distance_tensor = (
                self.visual_metrics.mandelbrot_distance_estimate(c_values)
                .to(self.device)
                .to(torch.float32)
            )
            loudness_distance_loss = self.correlation_loss(
                -spectral_rms, distance_tensor
            )
            timbre_color_loss = self._sanitize_scalar(timbre_color_loss)
            transient_impact_loss = self._sanitize_scalar(transient_impact_loss)
            loudness_distance_loss = self._sanitize_scalar(loudness_distance_loss)

            # Sequence-level terms: smooth off-hit, allow/encourage transitions on hits.
            controls = torch.cat(
                [
                    s_target.unsqueeze(1),
                    alpha.unsqueeze(1),
                    omega_scale.unsqueeze(1),
                    band_gates,
                ],
                dim=1,
            )

            (
                temporal_smoothness_loss,
                sequence_perceptual_loss,
                hit_alignment_loss,
            ) = self._compute_sequence_losses(
                controls,
                spectral_flux,
                onset_strength,
            )

            rollout_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            if (
                self.rollout_loss_weight > 0.0
                and self.rollout_batch_fraction > 0.0
                and torch.rand(1).item() < self.rollout_batch_fraction
            ):
                rollout_loss = self._compute_rollout_loss(
                    controls,
                    spectral_flux,
                    onset_strength,
                    segment_ids,
                )

            # Control loss (curriculum learning)
            if control_targets is not None and current_curriculum_weight > 0.0:
                control_loss_val = self.control_loss(
                    predicted_controls, control_targets
                )
            else:
                control_loss_val = torch.tensor(
                    0.0, device=self.device, dtype=torch.float32
                )
            control_loss_val = self._sanitize_scalar(control_loss_val)

            # Total loss
            total_batch_loss = (
                self.correlation_weights["timbre_color"] * timbre_color_loss
                + self.correlation_weights["transient_impact"] * transient_impact_loss
                + self.correlation_weights["loudness_distance"] * loudness_distance_loss
                + current_curriculum_weight * control_loss_val
                + self.temporal_smoothness_weight * temporal_smoothness_loss
                + self.sequence_loss_weight * sequence_perceptual_loss
                + self.hit_alignment_weight * hit_alignment_loss
                + self.rollout_loss_weight * rollout_loss
            )
            total_batch_loss = self._sanitize_scalar(total_batch_loss)

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_control_loss += control_loss_val.item()
            total_timbre_color += timbre_color_loss.item()
            total_transient_impact += transient_impact_loss.item()
            total_loudness_distance += loudness_distance_loss.item()
            total_temporal_smoothness += temporal_smoothness_loss.item()
            total_sequence_perceptual += sequence_perceptual_loss.item()
            total_hit_alignment += hit_alignment_loss.item()
            total_rollout += rollout_loss.item()
            n_batches += 1

        # Average losses
        avg_losses = {
            "loss": total_loss / n_batches,
            "control_loss": total_control_loss / n_batches,
            "timbre_color_loss": total_timbre_color / n_batches,
            "transient_impact_loss": total_transient_impact / n_batches,
            "loudness_distance_loss": total_loudness_distance / n_batches,
            "temporal_smoothness_loss": total_temporal_smoothness / n_batches,
            "sequence_perceptual_loss": total_sequence_perceptual / n_batches,
            "hit_alignment_loss": total_hit_alignment / n_batches,
            "rollout_loss": total_rollout / n_batches,
        }

        return avg_losses

    def train(
        self,
        dataset: AudioDataset,
        epochs: int = 100,
        batch_size: int = 32,
        save_dir: Optional[str] = None,
        curriculum_decay: float = 0.95,
    ):
        """Train model on dataset.

        Returns:
            Path to the final checkpoint if saved, else None.
        """
        logger.info("Loading audio features...")
        all_features = dataset.load_all_features()
        logger.info(f"Loaded {len(all_features)} feature set(s)")

        if len(all_features) == 0:
            logger.error(
                "No features loaded. Ensure data/audio contains supported files and that feature extraction succeeded."
            )
            raise ValueError("No features loaded from dataset")

        logger.info("Computing normalization statistics...")
        # Flatten all feature windows into a single sequence of 1D feature vectors
        # (Rust binding expects Seq[Seq[float]] where the inner seq is a single window).
        all_windows = [row for f in all_features for row in f.tolist()]
        self.feature_extractor.compute_normalization_stats(all_windows)

        # Normalize each window individually and re-stack per-file arrays. Handle
        # empty feature arrays safely by keeping an empty (0, n_features) array.
        normalized_features: List[NDArray[np.floating]] = []
        for f in all_features:
            if f.shape[0] == 0:
                # Preserve feature dimensionality for empty files
                normalized_features.append(np.empty((0, f.shape[1]), dtype=np.float32))
            else:
                normalized_rows = [
                    self.feature_extractor.normalize_features(row) for row in f.tolist()
                ]
                normalized_features.append(
                    np.vstack(normalized_rows).astype(np.float32)
                )

        try:
            concatenated = np.concatenate(normalized_features, axis=0)
            segment_ids = np.concatenate(
                [
                    np.full(f.shape[0], idx, dtype=np.int64)
                    for idx, f in enumerate(normalized_features)
                ],
                axis=0,
            )
        except Exception as e:
            logger.error(f"Failed to concatenate features: {e}")
            raise
        all_features_tensor = torch.tensor(concatenated, dtype=torch.float32)
        segment_id_tensor = torch.tensor(segment_ids, dtype=torch.int64)

        tensor_dataset = TensorDataset(all_features_tensor, segment_id_tensor)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        logger.info(
            f"Starting control signal training for {epochs} epochs... (total frames: {all_features_tensor.shape[0]})"
        )

        for epoch in tqdm(
            range(epochs), desc="Epochs", total=epochs, leave=True, mininterval=0.5
        ):
            avg_losses = self.train_epoch(dataloader, epoch, curriculum_decay)
            avg_losses["alignment_proxy"] = self._alignment_proxy(
                avg_losses["sequence_perceptual_loss"],
                avg_losses["hit_alignment_loss"],
            )

            for key, value in avg_losses.items():
                self.history[key].append(value)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f'Loss: {avg_losses["loss"]:.4f}, '
                f'Control: {avg_losses["control_loss"]:.4f}, '
                f'AlignProxy: {avg_losses["alignment_proxy"]:.4f}'
            )

            if save_dir and (
                self.best_alignment_proxy is None
                or avg_losses["alignment_proxy"] > self.best_alignment_proxy
            ):
                self.best_alignment_proxy = avg_losses["alignment_proxy"]
                self.save_checkpoint(
                    save_dir,
                    epoch + 1,
                    batch_size,
                    curriculum_decay,
                    epochs,
                    checkpoint_filename="checkpoint_best.pt",
                    update_last_path=False,
                )
                self.best_checkpoint_path = os.path.join(save_dir, "checkpoint_best.pt")
                logger.info(
                    f"Best checkpoint updated at epoch {epoch + 1}: "
                    f"alignment_proxy={self.best_alignment_proxy:.4f}"
                )

            if save_dir and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs):
                self.save_checkpoint(
                    save_dir, epoch + 1, batch_size, curriculum_decay, epochs
                )

        logger.info("Training complete!")
        return self.last_checkpoint_path

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        batch_size: int = 32,
        curriculum_decay: float = 0.95,
        total_epochs: int = 100,
        checkpoint_filename: Optional[str] = None,
        update_last_path: bool = True,
    ):
        """Save model checkpoint with full training configuration."""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "feature_mean": self.feature_extractor.feature_mean,
            "feature_std": self.feature_extractor.feature_std,
            # Training hyperparameters
            "learning_rate": self.learning_rate,
            "batch_size": batch_size,
            "total_epochs": total_epochs,
            "use_curriculum": self.use_curriculum,
            "curriculum_weight": self.curriculum_weight,
            "curriculum_decay": curriculum_decay,
            "correlation_weights": self.correlation_weights,
            "julia_resolution": self.julia_resolution,
            "julia_max_iter": self.julia_max_iter,
            "k_residuals": self.k_residuals,
            "temporal_smoothness_weight": self.temporal_smoothness_weight,
            "sequence_loss_weight": self.sequence_loss_weight,
            "hit_alignment_weight": self.hit_alignment_weight,
            "rollout_batch_fraction": self.rollout_batch_fraction,
            "rollout_horizon": self.rollout_horizon,
            "rollout_teacher_forcing": self.rollout_teacher_forcing,
            "rollout_loss_weight": self.rollout_loss_weight,
            "best_alignment_proxy": self.best_alignment_proxy,
            "best_checkpoint_path": self.best_checkpoint_path,
        }

        if checkpoint_filename is None:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pt"

        checkpoint_path = os.path.join(save_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        # Also emit a console print for immediate visibility
        print(f"[CHECKPOINT] Saved: {checkpoint_path}")
        if update_last_path:
            self.last_checkpoint_path = checkpoint_path

        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")
