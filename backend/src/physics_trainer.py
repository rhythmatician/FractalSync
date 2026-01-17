"""
Physics-aware trainer with curriculum learning support.

Trains models with velocity-based predictions and curriculum learning
using preset Mandelbrot orbits.
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
from torch.utils.data import DataLoader, TensorDataset

from .audio_features import AudioFeatureExtractor
from .data_loader import AudioDataset
from .visual_metrics import VisualMetrics
from .mandelbrot_orbits import generate_curriculum_sequence

logger = logging.getLogger(__name__)


class CorrelationLoss(nn.Module):
    """
    Negative correlation loss encourages positive correlation between two signals.
    Returns -1 × correlation, so minimizing this maximizes positive correlation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute negative correlation loss.

        Args:
            x: First signal (batch_size,)
            y: Second signal (batch_size,)

        Returns:
            Negative correlation coefficient
        """
        # Ensure tensors are 1D
        x = x.flatten()
        y = y.flatten()

        # Center the signals
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)

        # Compute correlation
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))

        # Avoid division by zero
        correlation = numerator / (denominator + 1e-8)

        # Return negative correlation (so minimizing maximizes correlation)
        return -correlation


class SmoothnessLoss(nn.Module):
    """
    Penalize large differences between consecutive outputs.
    Encourages smooth temporal transitions.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss.

        Args:
            current: Current output parameters (batch_size, output_dim)
            previous: Previous output parameters (batch_size, output_dim)

        Returns:
            Mean squared difference between current and previous
        """
        diff = current - previous
        return self.weight * torch.mean(diff**2)


class VelocityLoss(nn.Module):
    """Loss for velocity prediction accuracy."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self, predicted_velocity: torch.Tensor, target_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity prediction loss.

        Args:
            predicted_velocity: Predicted velocity (batch_size, 2)
            target_velocity: Target velocity from curriculum (batch_size, 2)

        Returns:
            MSE loss between predicted and target velocities
        """
        return self.weight * torch.mean((predicted_velocity - target_velocity) ** 2)


class AccelerationSmoothness(nn.Module):
    """Penalize rapid changes in velocity (acceleration)."""

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(
        self, current_velocity: torch.Tensor, previous_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute acceleration smoothness loss.

        Args:
            current_velocity: Current velocity (batch_size, 2)
            previous_velocity: Previous velocity (batch_size, 2)

        Returns:
            Smoothness loss on velocity changes
        """
        acceleration = current_velocity - previous_velocity
        return self.weight * torch.mean(acceleration**2)


class BoundaryProximityLoss(nn.Module):
    """Reward Julia parameter c being near Mandelbrot set boundary."""

    def __init__(
        self, weight: float = 0.1, target_iters: int = 30, max_iters: int = 100
    ):
        super().__init__()
        self.weight = weight
        self.target_iters = (
            target_iters  # Optimal iterations for interesting Julia sets
        )
        self.max_iters = max_iters

    def mandelbrot_escape_time(
        self, c_real: torch.Tensor, c_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Mandelbrot escape time for each c value.

        Args:
            c_real: Real part of c (batch_size,)
            c_imag: Imaginary part of c (batch_size,)

        Returns:
            Number of iterations before escape (batch_size,)
        """
        z_real = torch.zeros_like(c_real)
        z_imag = torch.zeros_like(c_imag)
        iterations = torch.zeros_like(c_real)

        # Track which points are still active (not yet escaped or cycled)
        active = torch.ones_like(c_real, dtype=torch.bool)

        # For cycle detection: track previous z values
        z_real_prev = torch.zeros_like(c_real)
        z_imag_prev = torch.zeros_like(c_imag)

        for i in range(self.max_iters):
            # Only compute for active points
            if not active.any():
                break  # Early exit: all points have escaped or cycled

            # z^2 + c
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag_new = 2 * z_real * z_imag + c_imag

            # Check escape condition |z| > 2
            magnitude_sq = z_real_new * z_real_new + z_imag_new * z_imag_new
            escaped = magnitude_sq > 4.0

            # Check for cycles: if z hasn't changed much, we're in a cycle
            # (This catches period-1, period-2, and most other periodic orbits)
            if i > 0:
                delta_real = torch.abs(z_real_new - z_real_prev)
                delta_imag = torch.abs(z_imag_new - z_imag_prev)
                in_cycle = (delta_real < 1e-6) & (delta_imag < 1e-6) & active

                # Points in cycles are in the Mandelbrot set (never escape)
                iterations = torch.where(
                    in_cycle,
                    torch.tensor(
                        self.max_iters, dtype=torch.float32, device=c_real.device
                    ),
                    iterations,
                )
                active = active & ~in_cycle

            # Update iterations for points that just escaped
            just_escaped = escaped & active
            iterations = torch.where(
                just_escaped,
                torch.tensor(i + 1, dtype=torch.float32, device=c_real.device),
                iterations,
            )

            # Mark escaped points as inactive
            active = active & ~escaped

            # Store current z for cycle detection next iteration
            z_real_prev = z_real_new
            z_imag_prev = z_imag_new

            z_real = z_real_new
            z_imag = z_imag_new

        # Points that never escaped get max_iters
        never_escaped = iterations == 0
        iterations = torch.where(
            never_escaped,
            torch.tensor(self.max_iters, dtype=torch.float32, device=c_real.device),
            iterations,
        )

        return iterations

    def forward(self, c_real: torch.Tensor, c_imag: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary proximity loss.

        Rewards c values that are near the Mandelbrot boundary (escape in 10-50 iterations).
        Penalizes c values deep inside the set (>50 iters) or far outside (<10 iters).

        Args:
            c_real: Real part of Julia parameter (batch_size,)
            c_imag: Imaginary part of Julia parameter (batch_size,)

        Returns:
            Loss value (lower is better, 0 = at target boundary)
        """
        escape_iters = self.mandelbrot_escape_time(c_real, c_imag)

        # Distance from target iteration count
        distance = torch.abs(escape_iters - self.target_iters)

        # Normalize by target_iters so loss is scale-invariant
        normalized_distance = distance / self.target_iters

        return self.weight * torch.mean(normalized_distance)


class DirectionalConsistencyLoss(nn.Module):
    """Penalize velocity direction flipping (oscillation)."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        current_velocity: torch.Tensor,
        previous_velocity: torch.Tensor,
        energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute directional consistency loss.

        Penalizes when velocity direction changes (dot product < 0).
        Encourages smooth exploration rather than back-and-forth oscillation.

        Args:
            current_velocity: Current velocity (batch_size, 2)
            previous_velocity: Previous velocity (batch_size, 2)

        Returns:
            Loss value (0 = same direction, higher = opposite directions)
        """
        # Normalize velocities to unit vectors
        current_norm = current_velocity / (
            torch.norm(current_velocity, dim=1, keepdim=True) + 1e-8
        )
        previous_norm = previous_velocity / (
            torch.norm(previous_velocity, dim=1, keepdim=True) + 1e-8
        )

        # Compute dot product (1 = same direction, -1 = opposite)
        dot_product = torch.sum(current_norm * previous_norm, dim=1)

        # Penalize negative dot products (direction flips)
        # ReLU ensures we only penalize flips, not consistent motion
        direction_flip_penalty = torch.relu(-dot_product)

        # Weight penalty by energy so high-energy audio discourages flips more
        if energy is not None:
            energy_min, energy_max = energy.min(), energy.max()
            energy_norm = (energy - energy_min) / (energy_max - energy_min + 1e-8)
            direction_flip_penalty = direction_flip_penalty * energy_norm

        return self.weight * torch.mean(direction_flip_penalty)


class AudioDrivenMomentumLoss(nn.Module):
    """Encourage velocity magnitude to respond to audio energy (loudness/activity)."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self, velocity_magnitude: torch.Tensor, audio_energy: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute audio-driven momentum loss.

        During silence (low audio energy), velocity can decay to zero.
        During loud/active parts (high energy), velocity should be high.
        Penalizes misalignment between velocity and audio energy.

        Args:
            velocity_magnitude: Magnitude of velocity vectors (batch_size,)
            audio_energy: Audio energy proxy (e.g., RMS energy) (batch_size,)

        Returns:
            Loss value encouraging velocity to follow audio dynamics
        """
        # Normalize each to [0, 1] independently to handle different scales
        vel_min, vel_max = velocity_magnitude.min(), velocity_magnitude.max()
        vel_range = vel_max - vel_min + 1e-8
        vel_normalized = (velocity_magnitude - vel_min) / vel_range

        energy_min, energy_max = audio_energy.min(), audio_energy.max()
        energy_range = energy_max - energy_min + 1e-8
        energy_normalized = (audio_energy - energy_min) / energy_range

        # Penalize difference between normalized velocity and audio energy
        mismatch = torch.abs(vel_normalized - energy_normalized)

        return self.weight * torch.mean(mismatch)


class EnergyScaledVelocityFloorLoss(nn.Module):
    """Enforce a minimum velocity that scales with audio energy."""

    def __init__(self, weight: float = 0.1, v_min: float = 0.05):
        super().__init__()
        self.weight = weight
        self.v_min = v_min

    def forward(
        self, velocity_magnitude: torch.Tensor, energy: torch.Tensor
    ) -> torch.Tensor:
        # Normalize energy to [0, 1]
        energy_min, energy_max = energy.min(), energy.max()
        energy_norm = (energy - energy_min) / (energy_max - energy_min + 1e-8)

        # Target floor scales with energy; silence => near zero floor
        target_floor = self.v_min * energy_norm
        penalty = torch.relu(target_floor - velocity_magnitude)
        return self.weight * penalty.mean()


class ExplorationVarianceLoss(nn.Module):
    """Encourage positional variance to avoid stagnation in one region."""

    def __init__(self, weight: float = 0.05, target_variance: float = 0.02):
        super().__init__()
        self.weight = weight
        self.target_variance = target_variance

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (batch, 2) for c_real, c_imag
        var = torch.var(positions, dim=0).mean()
        shortfall = torch.relu(self.target_variance - var)
        return self.weight * shortfall


class PhysicsTrainer:
    """Trainer for physics-based models with curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: AudioFeatureExtractor,
        visual_metrics: VisualMetrics,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        use_curriculum: bool = True,
        curriculum_weight: float = 1.0,
        correlation_weights: Optional[Dict[str, float]] = None,
        julia_renderer: Optional[Any] = None,
        julia_resolution: int = 128,
        julia_max_iter: int = 100,
        num_workers: int = 0,
    ):
        """
        Initialize physics trainer.

        Args:
            model: PyTorch model (should be PhysicsAudioToVisualModel)
            feature_extractor: Audio feature extractor
            visual_metrics: Visual metrics calculator
            device: Device to train on
            learning_rate: Learning rate
            use_curriculum: Whether to use curriculum learning with preset orbits
            curriculum_weight: Weight for curriculum loss (decreases over training)
            correlation_weights: Weights for different correlation losses
            julia_renderer: Optional GPU renderer for Julia sets (commit 75c1a43)
            julia_resolution: Julia set resolution (commit 75c1a43)
            julia_max_iter: Julia set max iterations (commit 75c1a43)
            num_workers: DataLoader num_workers for parallel loading (commit 75c1a43)
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.visual_metrics = visual_metrics
        self.device = device
        self.use_curriculum = use_curriculum
        self.curriculum_weight = curriculum_weight
        self.julia_renderer = julia_renderer
        self.julia_resolution = julia_resolution
        self.julia_max_iter = julia_max_iter
        self.num_workers = num_workers

        # Default correlation weights
        if correlation_weights is None:
            correlation_weights = {
                "timbre_color": 1.0,
                "transient_impact": 1.0,
                "silence_stillness": 1.0,
                "distortion_roughness": 1.0,
                "smoothness": 0.1,
                "acceleration_smoothness": 0.05,
                "velocity_loss": 1.0,
                "boundary_proximity": 0.2,
                "directional_consistency": 0.15,
                "audio_driven_momentum": 0.1,
                "energy_velocity_floor": 0.1,
                "exploration_variance": 0.15,
            }
        self.correlation_weights = correlation_weights

        # Loss functions
        self.correlation_loss = CorrelationLoss()
        self.smoothness_loss = SmoothnessLoss(
            weight=correlation_weights.get("smoothness", 0.1)
        )
        self.velocity_loss = VelocityLoss(
            weight=correlation_weights.get("velocity_loss", 1.0)
        )
        self.acceleration_smoothness = AccelerationSmoothness(
            weight=correlation_weights.get("acceleration_smoothness", 0.05)
        )
        self.boundary_proximity_loss = BoundaryProximityLoss(
            weight=correlation_weights.get("boundary_proximity", 0.2),
            target_iters=30,
            max_iters=100,
        )
        self.directional_consistency_loss = DirectionalConsistencyLoss(
            weight=correlation_weights.get("directional_consistency", 0.15)
        )
        self.audio_driven_momentum_loss = AudioDrivenMomentumLoss(
            weight=correlation_weights.get("audio_driven_momentum", 0.1)
        )
        self.energy_velocity_floor_loss = EnergyScaledVelocityFloorLoss(
            weight=correlation_weights.get("energy_velocity_floor", 0.1)
        )
        self.exploration_variance_loss = ExplorationVarianceLoss(
            weight=correlation_weights.get("exploration_variance", 0.05)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Curriculum data (generated on first epoch)
        self.curriculum_positions = None
        self.curriculum_velocities = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "velocity_loss": [],
            "acceleration_smoothness": [],
            "boundary_proximity_loss": [],
            "directional_consistency_loss": [],
            "audio_driven_momentum_loss": [],
            "energy_velocity_floor_loss": [],
            "exploration_variance_loss": [],
            "timbre_color_loss": [],
            "transient_impact_loss": [],
            "silence_stillness_loss": [],
            "distortion_roughness_loss": [],
            "smoothness_loss": [],
        }

    def _generate_curriculum_data(self, n_samples: int):
        """
        Generate curriculum learning data from preset orbits.

        Args:
            n_samples: Number of samples to generate
        """
        logger.info(f"Generating curriculum data: {n_samples} samples")
        positions, velocities = generate_curriculum_sequence(n_samples)

        self.curriculum_positions = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )
        self.curriculum_velocities = torch.tensor(
            velocities, dtype=torch.float32, device=self.device
        )

        logger.info(
            f"Curriculum data generated: positions shape={self.curriculum_positions.shape}, "
            f"velocities shape={self.curriculum_velocities.shape}"
        )

    def train_epoch(
        self, dataloader: DataLoader, epoch: int, curriculum_decay: float = 0.95
    ) -> Dict[str, float]:
        """
        Train for one epoch with physics-based learning.

        Args:
            dataloader: Data loader
            epoch: Current epoch number
            curriculum_decay: Decay factor for curriculum weight per epoch

        Returns:
            Dictionary of average losses
        """
        logger.debug(f"train_epoch called: epoch={epoch}")

        self.model.train()

        total_loss = 0.0
        total_velocity_loss = 0.0
        total_acceleration_smoothness = 0.0
        total_boundary_proximity = 0.0
        total_directional_consistency = 0.0
        total_timbre_color = 0.0
        total_transient_impact = 0.0
        total_silence_stillness = 0.0
        total_distortion_roughness = 0.0
        total_smoothness = 0.0
        total_audio_driven_momentum = 0.0
        total_energy_velocity_floor = 0.0
        total_exploration_variance = 0.0

        n_batches = 0

        # State tracking
        previous_velocity = None
        current_positions = None

        # Generate curriculum data if needed
        if self.use_curriculum and self.curriculum_positions is None:
            total_samples = len(dataloader.dataset)
            self._generate_curriculum_data(total_samples)

        # Curriculum weight decays over epochs
        current_curriculum_weight = self.curriculum_weight * (curriculum_decay**epoch)

        logger.debug(
            f"Starting epoch {epoch} with curriculum_weight={current_curriculum_weight:.4f}"
        )

        def _to_tensor(batch: Any) -> torch.Tensor:
            """Convert common batch shapes to torch.Tensor."""
            if isinstance(batch, torch.Tensor):
                return batch
            if isinstance(batch, (list, tuple)):
                return torch.stack(
                    [
                        b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
                        for b in batch
                    ]
                )
            return torch.as_tensor(batch)

        try:
            sample_idx = 0  # Track global sample index for curriculum data

            for batch_idx, batch_item in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                position=1,
                leave=False,
                desc="  Batches",
            ):
                logger.debug(
                    f"Got batch {batch_idx}: type={type(batch_item).__name__}, "
                    f"is_tuple={isinstance(batch_item, tuple)}"
                )

                # Handle batch from TensorDataset
                if isinstance(batch_item, (tuple, list)):
                    features = batch_item[0] if len(batch_item) == 1 else batch_item[0]
                else:
                    features = batch_item

                features = _to_tensor(features).to(self.device)
                batch_size = features.shape[0]

                # Get curriculum targets if available
                curriculum_pos_batch = None
                curriculum_vel_batch = None
                if self.use_curriculum and self.curriculum_positions is not None:
                    end_idx = min(
                        sample_idx + batch_size, len(self.curriculum_positions)
                    )
                    actual_batch_size = end_idx - sample_idx

                    if actual_batch_size > 0:
                        curriculum_pos_batch = self.curriculum_positions[
                            sample_idx:end_idx
                        ]
                        curriculum_vel_batch = self.curriculum_velocities[
                            sample_idx:end_idx
                        ]

                        # Trim features if curriculum batch is smaller
                        if actual_batch_size < batch_size:
                            features = features[:actual_batch_size]
                            batch_size = actual_batch_size

                sample_idx += batch_size

                # Initialize positions if needed
                if current_positions is None:
                    if curriculum_pos_batch is not None:
                        current_positions = curriculum_pos_batch.clone()
                    else:
                        current_positions = torch.zeros(
                            batch_size, 2, device=self.device
                        )

                # Forward pass
                output = self.model(features)

                # Extract velocity and other parameters
                # Output: [v_real, v_imag, c_real, c_imag, hue, sat, bright, zoom, speed]
                predicted_velocity = output[:, 0:2]  # (batch_size, 2)

                # Integrate velocity to get position
                if hasattr(self.model, "integrate_velocity"):
                    current_positions, integrated_velocity = (
                        self.model.integrate_velocity(
                            predicted_velocity, current_positions, previous_velocity
                        )
                    )
                else:
                    # Fallback: simple integration
                    integrated_velocity = predicted_velocity
                    current_positions = current_positions + predicted_velocity

                # Update output with integrated positions
                visual_params = torch.cat(
                    [current_positions, output[:, 4:]], dim=1
                )  # [c_real, c_imag, hue, sat, bright, zoom, speed]

                # Extract parameters for rendering
                julia_real = current_positions[:, 0].detach().cpu().numpy()
                julia_imag = current_positions[:, 1].detach().cpu().numpy()
                zoom = visual_params[:, 5].detach().cpu().numpy()

                # Extract audio features for correlation
                n_features_per_frame = 6
                window_frames = features.shape[1] // n_features_per_frame
                features_reshaped = features.view(
                    batch_size, window_frames, n_features_per_frame
                )
                avg_features = features_reshaped.mean(dim=1)  # (batch, n_features)

                spectral_centroid = avg_features[:, 0]
                spectral_flux = avg_features[:, 1]
                rms_energy = avg_features[:, 2]
                zero_crossing_rate = avg_features[:, 3]

                # Combine energy measures for audio-driven terms
                audio_energy = rms_energy + 0.5 * spectral_flux

                # Render Julia sets for visual metrics
                images = []
                color_hues = []
                temporal_changes = []
                edge_densities = []

                prev_image = None
                for i in range(batch_size):
                    # Use GPU renderer if available, else CPU
                    if self.julia_renderer is not None:
                        try:
                            image = self.julia_renderer.render(
                                seed_real=float(julia_real[i]),
                                seed_imag=float(julia_imag[i]),
                                max_iter=self.julia_max_iter,
                            )
                        except Exception as e:
                            # Fallback to CPU rendering on GPU error
                            logger.warning(f"GPU rendering failed, using CPU: {e}")
                            image = self.visual_metrics.render_julia_set(
                                seed_real=float(julia_real[i]),
                                seed_imag=float(julia_imag[i]),
                                width=self.julia_resolution,
                                height=self.julia_resolution,
                                zoom=float(zoom[i]),
                                max_iter=self.julia_max_iter,
                            )
                    else:
                        # CPU rendering
                        image = self.visual_metrics.render_julia_set(
                            seed_real=float(julia_real[i]),
                            seed_imag=float(julia_imag[i]),
                            width=self.julia_resolution,
                            height=self.julia_resolution,
                            zoom=float(zoom[i]),
                            max_iter=self.julia_max_iter,
                        )

                    metrics = self.visual_metrics.compute_all_metrics(
                        image, prev_image=prev_image
                    )

                    images.append(image)
                    color_hues.append(visual_params[i, 2])
                    temporal_changes.append(
                        torch.tensor(
                            metrics["temporal_change"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                    )
                    edge_densities.append(
                        torch.tensor(
                            metrics["edge_density"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                    )

                    prev_image = image

                # Stack tensors
                color_hue_tensor = torch.stack(color_hues)
                temporal_change_tensor = torch.stack(temporal_changes)
                edge_density_tensor = torch.stack(edge_densities)

                # Compute correlation losses
                timbre_color_loss = self.correlation_loss(
                    spectral_centroid, color_hue_tensor
                )
                transient_impact_loss = self.correlation_loss(
                    spectral_flux, temporal_change_tensor
                )
                silence_stillness_loss = self.correlation_loss(
                    rms_energy, -temporal_change_tensor
                )
                distortion_roughness_loss = self.correlation_loss(
                    zero_crossing_rate, edge_density_tensor
                )

                # Combined audio energy for motion-related losses
                audio_energy = rms_energy + 0.5 * spectral_flux

                # Velocity loss (curriculum learning)
                if curriculum_vel_batch is not None and current_curriculum_weight > 0.0:
                    velocity_loss_val = self.velocity_loss(
                        predicted_velocity, curriculum_vel_batch
                    )
                else:
                    velocity_loss_val = torch.zeros(1, device=self.device)

                # Acceleration smoothness
                if previous_velocity is not None:
                    # Handle size mismatch
                    min_size = min(
                        predicted_velocity.size(0), previous_velocity.size(0)
                    )
                    acceleration_smoothness_val = self.acceleration_smoothness(
                        predicted_velocity[:min_size], previous_velocity[:min_size]
                    )

                    # Directional consistency (penalize oscillation)
                    directional_consistency_val = self.directional_consistency_loss(
                        predicted_velocity[:min_size],
                        previous_velocity[:min_size],
                        energy=audio_energy[:min_size],
                    )
                else:
                    acceleration_smoothness_val = torch.zeros(1, device=self.device)
                    directional_consistency_val = torch.zeros(1, device=self.device)

                # Boundary proximity (reward c near Mandelbrot boundary)
                boundary_proximity_val = self.boundary_proximity_loss(
                    current_positions[:, 0], current_positions[:, 1]
                )

                # Audio-driven momentum (encourage velocity to match audio energy)
                velocity_magnitude = torch.sqrt(
                    predicted_velocity[:, 0] ** 2 + predicted_velocity[:, 1] ** 2
                )
                audio_driven_momentum_val = self.audio_driven_momentum_loss(
                    velocity_magnitude, rms_energy[:batch_size]
                )

                # Energy-scaled velocity floor (enforce motion when audio is active)
                energy_velocity_floor_val = self.energy_velocity_floor_loss(
                    velocity_magnitude, audio_energy[:batch_size]
                )

                # Exploration variance (encourage spread of c over batch)
                exploration_variance_val = self.exploration_variance_loss(
                    current_positions
                )

                # Parameter smoothness (placeholder for future use)
                smoothness = torch.zeros(1, device=self.device)

                # Total loss
                total_batch_loss = (
                    self.correlation_weights["timbre_color"] * timbre_color_loss
                    + self.correlation_weights["transient_impact"]
                    * transient_impact_loss
                    + self.correlation_weights["silence_stillness"]
                    * silence_stillness_loss
                    + self.correlation_weights["distortion_roughness"]
                    * distortion_roughness_loss
                    + current_curriculum_weight * velocity_loss_val
                    + acceleration_smoothness_val
                    + directional_consistency_val
                    + boundary_proximity_val
                    + energy_velocity_floor_val
                    + exploration_variance_val
                    + audio_driven_momentum_val
                    + smoothness
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                total_loss += total_batch_loss.item()
                total_velocity_loss += velocity_loss_val.item()
                total_acceleration_smoothness += acceleration_smoothness_val.item()
                total_boundary_proximity += boundary_proximity_val.item()
                total_directional_consistency += directional_consistency_val.item()
                total_audio_driven_momentum += audio_driven_momentum_val.item()
                total_energy_velocity_floor += energy_velocity_floor_val.item()
                total_exploration_variance += exploration_variance_val.item()
                total_timbre_color += timbre_color_loss.item()
                total_transient_impact += transient_impact_loss.item()
                total_silence_stillness += silence_stillness_loss.item()
                total_distortion_roughness += distortion_roughness_loss.item()
                total_smoothness += smoothness.item()

                n_batches += 1
                # Detach state tensors (velocity and position) so gradients do not backpropagate
                # through the integrated history of previous batches in the curriculum sequence.
                previous_velocity = integrated_velocity.detach()
                current_positions = current_positions.detach()

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {total_batch_loss.item():.4f}, "
                        f"Velocity Loss: {velocity_loss_val.item():.4f}"
                    )

        except Exception as e:
            logger.error(
                f"ERROR in train_epoch loop: error={str(e)}, "
                f"error_type={type(e).__name__}, batch_idx={batch_idx}"
            )
            raise

        # Average losses
        avg_losses = {
            "loss": total_loss / n_batches,
            "velocity_loss": total_velocity_loss / n_batches,
            "acceleration_smoothness": total_acceleration_smoothness / n_batches,
            "boundary_proximity_loss": total_boundary_proximity / n_batches,
            "directional_consistency_loss": total_directional_consistency / n_batches,
            "audio_driven_momentum_loss": total_audio_driven_momentum / n_batches,
            "energy_velocity_floor_loss": total_energy_velocity_floor / n_batches,
            "exploration_variance_loss": total_exploration_variance / n_batches,
            "timbre_color_loss": total_timbre_color / n_batches,
            "transient_impact_loss": total_transient_impact / n_batches,
            "silence_stillness_loss": total_silence_stillness / n_batches,
            "distortion_roughness_loss": total_distortion_roughness / n_batches,
            "smoothness_loss": total_smoothness / n_batches,
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
        """
        Train model on dataset with physics-based learning.

        Args:
            dataset: Audio dataset
            epochs: Number of epochs
            batch_size: Batch size
            save_dir: Directory to save checkpoints
            curriculum_decay: Decay rate for curriculum weight
        """
        # Load all features
        logger.info("Loading audio features...")
        all_features = dataset.load_all_features()

        # Compute normalization stats
        logger.info("Computing normalization statistics...")
        self.feature_extractor.compute_normalization_stats(all_features)

        # Normalize features
        normalized_features = [
            self.feature_extractor.normalize_features(f) for f in all_features
        ]

        # Create dataset
        all_features_tensor = torch.tensor(
            np.concatenate(normalized_features, axis=0), dtype=torch.float32
        )

        # Create data loader
        tensor_dataset = TensorDataset(all_features_tensor)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )  # Don't shuffle for curriculum

        # Training loop
        logger.info(f"Starting physics-based training for {epochs} epochs...")
        logger.info(
            f"Curriculum learning: {self.use_curriculum}, decay: {curriculum_decay}"
        )

        for epoch in tqdm(range(epochs), desc="Training Epochs", position=0):
            avg_losses = self.train_epoch(dataloader, epoch, curriculum_decay)

            # Update history
            for key, value in avg_losses.items():
                self.history[key].append(value)

            # Print progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f'Loss: {avg_losses["loss"]:.4f}, '
                f'Velocity: {avg_losses["velocity_loss"]:.4f}, '
                f'Timbre-Color: {avg_losses["timbre_color_loss"]:.4f}'
            )

            # Save checkpoint
            if save_dir and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs):
                self.save_checkpoint(save_dir, epoch + 1)

        logger.info("Physics-based training complete!")

    def save_checkpoint(self, save_dir: str, epoch: int):
        """
        Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            epoch: Epoch number
        """
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

        # Save history as JSON
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")
