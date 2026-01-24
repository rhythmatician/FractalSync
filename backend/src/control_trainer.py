"""
Trainer for control signal model with orbit-based synthesis.

Trains model to predict control signals that drive deterministic orbit synthesis.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from .control_model import AudioToControlModel
from .data_loader import AudioDataset
from .visual_metrics import VisualMetrics
from .runtime_core_bridge import (
    DEFAULT_BASE_OMEGA,
    DEFAULT_K_RESIDUALS,
    DEFAULT_ORBIT_SEED,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    make_feature_extractor,
    make_orbit_state,
    make_residual_params,
    synthesize,
)
import runtime_core as rc

logger = logging.getLogger(__name__)


def compute_mandelbrot_distance(
    c_real: float, c_imag: float, max_iter: int = 256
) -> float:
    """Compute approximate distance to Mandelbrot set boundary.

    Uses escape-time distance estimation. Returns:
    - Negative values: inside the set
    - ~0: near boundary
    - Positive values: outside the set

    Args:
        c_real: Real part of c
        c_imag: Imaginary part of c
        max_iter: Maximum iterations

    Returns:
        Approximate distance to boundary (0 = on boundary)
    """
    z_real, z_imag = 0.0, 0.0
    dz_real, dz_imag = 1.0, 0.0

    for i in range(max_iter):
        z_mag_sq = z_real * z_real + z_imag * z_imag
        if z_mag_sq > 256.0:  # Escaped
            z_mag = np.sqrt(z_mag_sq)
            dz_mag = np.sqrt(dz_real * dz_real + dz_imag * dz_imag)
            # Distance estimate using derivative (clamped to avoid huge values)
            if dz_mag > 0:
                distance = 0.5 * z_mag * np.log(z_mag) / dz_mag
                # Clamp to reasonable range: [0, 2.0]
                distance = min(distance, 2.0)
            else:
                distance = 2.0  # Far outside
            return distance

        # dz = 2 * z * dz + 1
        temp = 2.0 * (z_real * dz_real - z_imag * dz_imag) + 1.0
        dz_imag = 2.0 * (z_real * dz_imag + z_imag * dz_real)
        dz_real = temp

        # z = z^2 + c
        temp = z_real * z_real - z_imag * z_imag + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = temp

    # Inside the set
    return -1.0


def compute_musical_complexity(features: torch.Tensor) -> torch.Tensor:
    """Compute musical complexity from audio features.

    Uses variance of spectral flux as a proxy for musical complexity.

    Args:
        features: Feature tensor of shape (batch_size, window_frames, n_features)

    Returns:
        Complexity score per sample (batch_size,)
    """
    # Spectral flux is at index 1
    spectral_flux = features[:, :, 1]  # (batch_size, window_frames)
    # Compute variance across time window
    complexity = torch.var(spectral_flux, dim=1) + torch.mean(
        torch.abs(spectral_flux), dim=1
    )
    return complexity


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
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        correlation = numerator / (denominator + 1e-8)
        return -correlation


class VisualContinuityLoss(nn.Module):
    """Penalize visual discontinuities that don't align with audio transients."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self, visual_changes: torch.Tensor, transient_strengths: torch.Tensor
    ) -> torch.Tensor:
        """Loss = visual_change * (1 - transient_strength).
        High visual change is OK during transients, penalized otherwise.
        """
        # Normalize transient strengths to [0, 1]
        transient_norm = torch.sigmoid(transient_strengths)
        # Penalize visual changes that occur without transients
        # Scale by 100 to make comparable to boundary loss (temporal_change is typically 0.0-0.1)
        loss = torch.mean(visual_changes * (1.0 - transient_norm)) * 100.0
        return self.weight * loss


class BoundaryExplorationLoss(nn.Module):
    """Encourage c to explore near Mandelbrot boundary (not deep inside)."""

    def __init__(self, weight: float = 1.0, target_distance: float = 0.1):
        super().__init__()
        self.weight = weight
        self.target_distance = target_distance

    def forward(
        self, c_values: torch.Tensor, mandelbrot_distances: torch.Tensor
    ) -> torch.Tensor:
        """Penalize being too far from target distance (slightly outside boundary).

        Args:
            c_values: Complex values as (batch_size, 2) tensor
            mandelbrot_distances: Distance to Mandelbrot set for each c
        """
        # Penalize deviation from target distance
        distance_error = torch.abs(mandelbrot_distances - self.target_distance)
        return self.weight * torch.mean(distance_error)


class ControlTrainer:
    """Trainer for control signal model with orbit synthesis."""

    def __init__(
        self,
        model: AudioToControlModel,
        visual_metrics: VisualMetrics,
        feature_extractor: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        use_curriculum: bool = True,
        curriculum_weight: float = 1.0,
        correlation_weights: Optional[Dict[str, float]] = None,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        julia_renderer: Optional[Any] = None,
        julia_resolution: int = 128,
        julia_max_iter: int = 100,
        num_workers: int = 0,
        k_residuals: int = DEFAULT_K_RESIDUALS,
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
            max_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
            use_amp: Use automatic mixed precision for faster training
            julia_renderer: Optional GPU renderer
            julia_resolution: Julia set resolution
            julia_max_iter: Julia set max iterations
            num_workers: DataLoader workers
            k_residuals: Number of residual circles
        """
        self.model: AudioToControlModel = model.to(device)
        self.feature_extractor = feature_extractor
        self.visual_metrics = visual_metrics
        self.device = device
        self.use_curriculum = use_curriculum
        self.curriculum_weight = curriculum_weight
        self.julia_renderer = julia_renderer
        self.julia_resolution = julia_resolution
        self.julia_max_iter = julia_max_iter
        self.num_workers = num_workers
        self.k_residuals = k_residuals
        self.residual_params = make_residual_params(k_residuals=k_residuals)

        # Default correlation weights
        default_weights = {
            "timbre_color": 0.5,  # Reduced
            "transient_impact": 0.5,  # Reduced
            "control_loss": 1.0,
            "visual_continuity": 1.0,  # Scaled internally by 100x
            "boundary_exploration": 1.5,  # New: encourage boundary exploration
            "complexity_correlation": 10.0,  # Increased to make correlation more visible
        }
        self.correlation_weights = {**default_weights, **(correlation_weights or {})}

        # Runtime-core feature extractor (shared constants)
        self.feature_extractor = feature_extractor or make_feature_extractor()

        # Loss functions
        self.correlation_loss = CorrelationLoss()
        self.control_loss = ControlLoss(
            weight=self.correlation_weights.get("control_loss", 1.0)
        )
        self.visual_continuity_loss = VisualContinuityLoss(
            weight=self.correlation_weights.get("visual_continuity", 2.0)
        )
        self.boundary_loss = BoundaryExplorationLoss(
            weight=self.correlation_weights.get("boundary_exploration", 1.5),
            target_distance=0.05,  # Stay close to boundary
        )

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler (cosine annealing)
        self.scheduler: Optional[CosineAnnealingLR] = (
            None  # Set in train() with total epochs
        )

        # Gradient clipping
        self.max_grad_norm = max_grad_norm

        # Mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training enabled (AMP)")

        # Curriculum data
        self.curriculum_positions: Optional[torch.Tensor] = None
        self.curriculum_velocities: Optional[torch.Tensor] = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "control_loss": [],
            "timbre_color_loss": [],
            "transient_impact_loss": [],
            "visual_continuity_loss": [],
            "boundary_exploration_loss": [],
            "complexity_correlation_loss": [],
        }
        # Track last checkpoint for reporting
        self.last_checkpoint_path: Optional[str] = None

    def _generate_curriculum_data(self, n_samples: int):
        """Generate curriculum learning data from preset orbits."""
        logger.info(f"Generating curriculum data: {n_samples} samples")
        thetas = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        positions = []
        velocities = []
        for idx, theta in enumerate(thetas):
            current = rc.lobe_point_at_angle(1, 0, float(theta), 1.02)
            next_theta = thetas[(idx + 1) % len(thetas)]
            nxt = rc.lobe_point_at_angle(1, 0, float(next_theta), 1.02)
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

    def train_epoch(
        self, dataloader: DataLoader, epoch: int, curriculum_decay: float = 0.95
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_control_loss = 0.0
        total_timbre_color = 0.0
        total_transient_impact = 0.0
        total_visual_continuity = 0.0
        total_boundary_exploration = 0.0
        total_complexity_correlation = 0.0
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
            if isinstance(batch_item, (tuple, list)):
                features = batch_item[0]
            else:
                features = batch_item

            features = features.to(self.device)
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
                        batch_size = actual_batch_size

            sample_idx += batch_size

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
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
                state = make_orbit_state(
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
                c = synthesize(
                    state,
                    residual_params=self.residual_params,
                    band_gates=band_gates[i].detach().cpu().tolist(),
                )

                # Calculate ||c|| and scale down if needed
                c_norm = np.sqrt(c.real**2 + c.imag**2)
                max_magnitude = 2.0
                divisor = max(c_norm, max_magnitude) / max_magnitude

                # SAFETY: Scale c values to reasonable range for Julia sets
                # The orbit synthesis can produce values outside [-2, 2] which are not useful
                # This preserves direction while constraining magnitude
                c_real = c.real / divisor
                c_imag = c.imag / divisor
                c_values.append([c_real, c_imag])

            c_tensor = torch.tensor(c_values, dtype=torch.float32, device=self.device)
            julia_real = c_tensor[:, 0]
            julia_imag = c_tensor[:, 1]

            # Extract audio features for correlation
            n_features_per_frame = self.feature_extractor.num_features_per_frame()
            window_frames = features.shape[1] // n_features_per_frame
            features_reshaped = features.view(
                batch_size, window_frames, n_features_per_frame
            )
            avg_features = features_reshaped.mean(dim=1)

            spectral_centroid = avg_features[:, 0]
            spectral_flux = avg_features[:, 1]

            # Render Julia sets for visual metrics
            images = []
            color_hues = []
            temporal_changes = []
            visual_complexities = []
            mandelbrot_distances = []

            prev_image = None
            for i in range(batch_size):
                if self.julia_renderer is not None:
                    try:
                        image = self.julia_renderer.render(
                            seed_real=julia_real[i].detach().item(),
                            seed_imag=julia_imag[i].detach().item(),
                            max_iter=self.julia_max_iter,
                        )
                    except Exception as e:
                        logger.warning(f"GPU rendering failed: {e}")
                        image = self.visual_metrics.render_julia_set(
                            seed_real=julia_real[i].detach().item(),
                            seed_imag=julia_imag[i].detach().item(),
                            width=self.julia_resolution,
                            height=self.julia_resolution,
                            max_iter=self.julia_max_iter,
                        )
                else:
                    image = self.visual_metrics.render_julia_set(
                        seed_real=julia_real[i].detach().item(),
                        seed_imag=julia_imag[i].detach().item(),
                        width=self.julia_resolution,
                        height=self.julia_resolution,
                        max_iter=self.julia_max_iter,
                    )

                metrics = self.visual_metrics.compute_all_metrics(
                    image, prev_image=prev_image
                )

                # Compute Mandelbrot distance
                c_real = julia_real[i].detach().item()
                c_imag = julia_imag[i].detach().item()
                mb_dist = compute_mandelbrot_distance(c_real, c_imag, max_iter=256)
                mandelbrot_distances.append(mb_dist)

                # Visual complexity from connectedness and edge density
                visual_complexity = metrics["connectedness"] * metrics["edge_density"]

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
                visual_complexities.append(
                    torch.tensor(
                        visual_complexity, device=self.device, dtype=torch.float32
                    )
                )

                prev_image = image

            color_hue_tensor = torch.stack(color_hues)
            temporal_change_tensor = torch.stack(temporal_changes)
            visual_complexity_tensor = torch.stack(visual_complexities)
            mandelbrot_dist_tensor = torch.tensor(
                mandelbrot_distances, device=self.device, dtype=torch.float32
            )

            # Compute correlation losses
            timbre_color_loss = self.correlation_loss(
                spectral_centroid, color_hue_tensor
            )
            transient_impact_loss = self.correlation_loss(
                spectral_flux, temporal_change_tensor
            )

            # Visual continuity loss: penalize visual changes without audio transients
            # Use spectral flux as transient strength indicator
            visual_continuity_loss_val = self.visual_continuity_loss(
                temporal_change_tensor, spectral_flux
            )

            # Boundary exploration loss: encourage c to stay near Mandelbrot boundary
            boundary_loss_val = self.boundary_loss(c_tensor, mandelbrot_dist_tensor)

            # Complexity correlation: match musical and visual complexity
            musical_complexity = compute_musical_complexity(features_reshaped)
            complexity_correlation_loss_val = self.correlation_loss(
                musical_complexity, visual_complexity_tensor
            )

            # Control loss (curriculum learning)
            if control_targets is not None and current_curriculum_weight > 0.0:
                control_loss_val = self.control_loss(
                    predicted_controls, control_targets
                )
            else:
                control_loss_val = torch.zeros(1, device=self.device)

            # Total loss with new components
            total_batch_loss = (
                self.correlation_weights["timbre_color"] * timbre_color_loss
                + self.correlation_weights["transient_impact"] * transient_impact_loss
                + self.correlation_weights["visual_continuity"]
                * visual_continuity_loss_val
                + self.correlation_weights["boundary_exploration"] * boundary_loss_val
                + self.correlation_weights["complexity_correlation"]
                * complexity_correlation_loss_val
                + current_curriculum_weight * control_loss_val
            )

            # Backward pass with mixed precision and gradient clipping
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(total_batch_loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_control_loss += control_loss_val.item()
            total_timbre_color += timbre_color_loss.item()
            total_transient_impact += transient_impact_loss.item()
            total_visual_continuity += visual_continuity_loss_val.item()
            total_boundary_exploration += boundary_loss_val.item()
            total_complexity_correlation += complexity_correlation_loss_val.item()
            n_batches += 1

        # Average losses
        avg_losses = {
            "loss": total_loss / n_batches,
            "control_loss": total_control_loss / n_batches,
            "timbre_color_loss": total_timbre_color / n_batches,
            "transient_impact_loss": total_transient_impact / n_batches,
            "visual_continuity_loss": total_visual_continuity / n_batches,
            "boundary_exploration_loss": total_boundary_exploration / n_batches,
            "complexity_correlation_loss": total_complexity_correlation / n_batches,
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
        self.feature_extractor.compute_normalization_stats(all_features)

        normalized_features = [
            self.feature_extractor.normalize_features(f) for f in all_features
        ]

        try:
            concatenated = np.concatenate(normalized_features, axis=0)
        except Exception as e:
            logger.error(f"Failed to concatenate features: {e}")
            raise
        all_features_tensor = torch.tensor(concatenated, dtype=torch.float32)

        tensor_dataset = TensorDataset(all_features_tensor)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        logger.info(
            f"Starting control signal training for {epochs} epochs... (total frames: {all_features_tensor.shape[0]})"
        )
        logger.info(
            f"Optimizations: AMP={self.use_amp}, GradClip={self.max_grad_norm}, "
            f"LR Schedule=Cosine(eta_min=1e-6)"
        )

        for epoch in tqdm(
            range(epochs), desc="Epochs", total=epochs, leave=True, mininterval=0.5
        ):
            avg_losses = self.train_epoch(dataloader, epoch, curriculum_decay)

            for key, value in avg_losses.items():
                self.history[key].append(value)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f'Loss: {avg_losses["loss"]:.4f}, '
                f'Boundary: {avg_losses["boundary_exploration_loss"]:.4f}, '
                f'Continuity: {avg_losses["visual_continuity_loss"]:.4f}, '
                f'Complexity: {avg_losses["complexity_correlation_loss"]:.4f}'
            )

            if save_dir and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs):
                self.save_checkpoint(save_dir, epoch + 1)

        logger.info("Training complete!")
        return self.last_checkpoint_path

    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "feature_mean": self.feature_extractor.feature_mean,
            "feature_std": self.feature_extractor.feature_std,
        }

        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        # Also emit a console print for immediate visibility
        print(f"[CHECKPOINT] Saved: {checkpoint_path}")
        self.last_checkpoint_path = checkpoint_path

        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")
