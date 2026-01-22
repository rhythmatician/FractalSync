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


class BoundaryCrossingReward(nn.Module):
    """
    Reward s crossing the Mandelbrot boundary (s=1) when transients are strong.

    This teaches the model to create dramatic visual moments when there are
    big audio hits - crossing s=1 causes significant visual changes in Julia sets.
    """

    def __init__(
        self, boundary: float = 1.0, margin: float = 0.05, weight: float = 1.0
    ):
        """
        Initialize boundary crossing reward.

        Args:
            boundary: The s value boundary to cross (typically 1.0)
            margin: Region around boundary considered "near"
            weight: Loss weight multiplier
        """
        super().__init__()
        self.boundary = boundary
        self.margin = margin
        self.weight = weight

    def forward(
        self, s: torch.Tensor, transient_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary crossing reward/penalty.

        Args:
            s: Predicted s values (batch_size,)
            transient_strength: Audio transient strength (batch_size,), normalized to [0, 1]

        Returns:
            Loss value (negative reward for good crossings, penalty for bad behavior)
        """
        # Detect if s is near the boundary
        near_boundary = (torch.abs(s - self.boundary) < self.margin).float()

        # Detect boundary crossings (s changes sign relative to boundary)
        s_prev = torch.roll(s, 1, dims=0)
        # First element can't have a valid crossing
        crossed = ((s > self.boundary) != (s_prev > self.boundary)).float()
        crossed[0] = 0.0  # Invalid for first element

        # Reward: crossing when transient is strong
        # Penalty: crossing when transient is weak (random/chaotic)
        crossing_reward = crossed * transient_strength
        crossing_penalty = crossed * (1.0 - transient_strength) * 0.5

        # Small penalty for lingering near boundary without crossing
        linger_penalty = (1 - crossed) * near_boundary * 0.1

        # Total: negative reward (we minimize), plus penalties
        loss = -crossing_reward + crossing_penalty + linger_penalty

        return self.weight * torch.mean(loss)


class LobePredictionLoss(nn.Module):
    """
    Cross-entropy loss for lobe prediction at section boundaries.

    Trains the model to switch lobes appropriately based on:
    - Section boundaries (detected via ruptures/novelty)
    - Audio energy levels
    - Musical context

    Uses the LobeScheduler's strategic selection as supervision signal.
    """

    def __init__(self, weight: float = 1.0, smoothing: float = 0.1):
        """
        Initialize lobe prediction loss.

        Args:
            weight: Loss weight multiplier
            smoothing: Label smoothing factor (0 = hard labels, 0.1 = smooth)
        """
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(
        self,
        lobe_logits: torch.Tensor,
        target_lobes: torch.Tensor,
        at_boundary: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute lobe prediction loss.

        Args:
            lobe_logits: Predicted lobe logits (batch_size, n_lobes)
            target_lobes: Target lobe indices (batch_size,) long tensor
            at_boundary: Optional mask for frames at section boundaries (batch_size,)
                        If provided, only compute loss at boundaries

        Returns:
            Loss value
        """
        if at_boundary is not None:
            # Only compute loss at section boundaries
            boundary_mask = at_boundary > 0.5
            if not boundary_mask.any():
                return torch.tensor(0.0, device=lobe_logits.device)

            lobe_logits = lobe_logits[boundary_mask]
            target_lobes = target_lobes[boundary_mask]

        loss = self.cross_entropy(lobe_logits, target_lobes)
        return self.weight * loss


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
            "timbre_color": 1.0,
            "transient_impact": 1.0,
            "control_loss": 1.0,
            "boundary_crossing": 0.5,  # Reward s crossing boundary on transients
            "lobe_prediction": 0.3,  # Teach lobe switching at section boundaries
        }
        self.correlation_weights = {**default_weights, **(correlation_weights or {})}

        # Runtime-core feature extractor (shared constants)
        self.feature_extractor = feature_extractor or make_feature_extractor()

        # Loss functions
        self.correlation_loss = CorrelationLoss()
        self.control_loss = ControlLoss(
            weight=self.correlation_weights.get("control_loss", 1.0)
        )
        self.boundary_crossing_reward = BoundaryCrossingReward(
            boundary=1.0,
            margin=0.05,
            weight=self.correlation_weights.get("boundary_crossing", 0.5),
        )
        self.lobe_prediction_loss = LobePredictionLoss(
            weight=self.correlation_weights.get("lobe_prediction", 0.3),
            smoothing=0.1,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Curriculum data
        self.curriculum_positions: Optional[torch.Tensor] = None
        self.curriculum_velocities: Optional[torch.Tensor] = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "control_loss": [],
            "timbre_color_loss": [],
            "transient_impact_loss": [],
            "boundary_crossing_loss": [],
            "lobe_prediction_loss": [],
        }
        # Track last checkpoint for reporting
        self.last_checkpoint_path: Optional[str] = None

        # Lobe curriculum data (for section-aware training)
        self.curriculum_lobes: Optional[torch.Tensor] = None
        self.section_boundaries: Optional[List[int]] = None

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

    def _generate_lobe_curriculum(
        self,
        section_data: Dict[str, Any],
        n_samples: int,
        lobe_to_index: Dict[Tuple[int, int], int],
    ):
        """
        Generate lobe switching curriculum from section boundary data.

        Uses LobeScheduler to strategically assign lobes to sections based on:
        - Energy levels
        - Section boundaries from audio analysis
        - Musical structure (verse/chorus/etc if available)

        Args:
            section_data: Dict with 'section_boundaries', 'tempo', 'energy_profile'
            n_samples: Total number of samples in training data
            lobe_to_index: Mapping from (lobe, sub_lobe) to index for classification
        """
        from .live_controller import LobeScheduler

        logger.info("Generating lobe curriculum from section boundaries")

        section_boundaries = section_data.get("section_boundaries", [])
        if not section_boundaries or len(section_boundaries) == 0:
            # No section data - assign default lobe (cardioid) to all
            logger.warning(
                "No section boundaries found, using default lobe for all samples"
            )
            default_lobe_idx = lobe_to_index.get((1, 0), 0)
            self.curriculum_lobes = torch.full(
                (n_samples,), default_lobe_idx, dtype=torch.long, device=self.device
            )
            return

        # Initialize scheduler
        scheduler = LobeScheduler()

        # Get local tempo if available
        local_tempo = section_data.get("local_tempo", None)

        # Create lobe assignments for each sample
        lobe_assignments = []
        current_lobe = (1, 0)  # Start with cardioid

        # Convert boundaries to sample indices (assuming uniform sampling)
        boundary_samples = [
            (
                int(b * n_samples / section_boundaries[-1])
                if section_boundaries[-1] > 0
                else 0
            )
            for b in section_boundaries
        ]
        boundary_samples.append(n_samples)  # Add final boundary

        # Compute tempo statistics if local tempo available
        tempo_std = 0.0
        if local_tempo is not None and len(local_tempo) > 0:
            local_tempo_array = np.array(local_tempo)
            tempo_mean = np.mean(local_tempo_array)
            tempo_std = np.std(local_tempo_array)
            logger.info(
                f"  Tempo: {tempo_mean:.1f} ± {tempo_std:.1f} BPM (range: {local_tempo_array.min():.1f}-{local_tempo_array.max():.1f})"
            )
        else:
            local_tempo_array = None
            tempo_mean = 0.0
            logger.critical("  No local tempo data available")

        for i in range(len(boundary_samples)):
            start_sample = boundary_samples[i - 1] if i > 0 else 0
            end_sample = boundary_samples[i]

            # Compute section characteristics
            section_duration = end_sample - start_sample

            # Energy estimation: consider position in song + section length
            # Longer sections often = verses (lower energy), shorter = transitions/chorus (higher)
            relative_pos = i / max(len(boundary_samples) - 1, 1)

            # Base energy from song position (builds toward middle)
            base_energy = 0.3 + 0.7 * np.sin(relative_pos * np.pi)  # 0.3 to 1.0

            # Adjust for section length (shorter sections = more intense)
            avg_section_length = n_samples / len(boundary_samples)
            length_factor = 1.0 - 0.3 * (section_duration / avg_section_length - 1.0)
            length_factor = np.clip(length_factor, 0.5, 1.5)

            energy = np.clip(base_energy * length_factor, 0.2, 1.0)

            # Compute local tempo for this section
            tempo_change = 0.0

            if local_tempo is not None and len(local_tempo) > 0:
                # Map section to tempo frames
                # section_boundaries are in frames, local_tempo is per-frame
                if i < len(section_boundaries):
                    boundary_frame = int(section_boundaries[i])
                    prev_boundary_frame = int(section_boundaries[i - 1]) if i > 0 else 0

                    # Get tempo for this section
                    if boundary_frame < len(
                        local_tempo_array
                    ) and prev_boundary_frame < len(local_tempo_array):
                        section_tempo_values = local_tempo_array[
                            prev_boundary_frame:boundary_frame
                        ]
                        if len(section_tempo_values) > 0:
                            section_tempo = np.mean(section_tempo_values)

                            # Compute tempo change relative to global mean
                            tempo_change = (section_tempo - tempo_mean) / (
                                tempo_std + 1e-8
                            )

            # Map tempo to energy boost
            # Faster tempo → higher energy, slower → lower energy
            tempo_factor = 1.0 + 0.2 * tempo_change  # ±20% based on tempo deviation
            energy = np.clip(energy * tempo_factor, 0.2, 1.0)

            # Novelty: higher at boundaries, especially with tempo changes
            novelty = 0.5 + 0.3 * abs(tempo_change) + 0.2 * np.random.random()
            novelty = np.clip(novelty, 0.0, 1.0)

            # Classify section type based on characteristics
            section_type = None
            if energy < 0.4 and section_duration > avg_section_length * 1.2:
                section_type = "verse"  # Low energy, long section
            elif energy > 0.7 and novelty > 0.6:
                section_type = "chorus"  # High energy, high novelty
            elif abs(tempo_change) > 1.0:  # More than 1 std dev
                section_type = "breakdown"  # Significant tempo change
            elif 0.4 <= energy < 0.7 and novelty > 0.5:
                section_type = "build"  # Medium energy, rising

            # Select lobe for this section
            new_lobe = scheduler.select_next_lobe(
                energy_level=energy,
                novelty=novelty,
                timestamp=float(i * 10),  # Dummy timestamp
                section_type=section_type,
            )

            # Assign this lobe to all samples in the section
            lobe_idx = lobe_to_index.get(new_lobe, lobe_to_index.get((1, 0), 0))
            for _ in range(start_sample, end_sample):
                lobe_assignments.append(lobe_idx)

            # Log transition if lobe changed
            if new_lobe != current_lobe:
                old_chars = scheduler.get_characteristics(current_lobe)
                new_chars = scheduler.get_characteristics(new_lobe)
                logger.debug(
                    f"  Section {i}: {old_chars.name} → {new_chars.name} "
                    f"(energy={energy:.2f}, tempo_change={tempo_change:.2f}, {section_type or 'neutral'})"
                )

            current_lobe = new_lobe

        # Convert to tensor
        self.curriculum_lobes = torch.tensor(
            lobe_assignments[:n_samples],  # Ensure exact length
            dtype=torch.long,
            device=self.device,
        )

        # Mark section boundaries for targeted training
        self.section_boundaries = boundary_samples[:-1]  # Exclude final boundary

        logger.info(
            f"Lobe curriculum generated: {len(set(lobe_assignments))} unique lobes assigned, "
            f"{len(self.section_boundaries)} section boundaries"
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
        total_boundary_crossing = 0.0
        total_lobe_prediction = 0.0
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
                c_values.append([c.real, c.imag])

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

            # Boundary crossing reward: reward crossing s=1 when spectral_flux is high
            # Normalize spectral flux to [0, 1] range for transient strength
            flux_min = spectral_flux.min()
            flux_max = spectral_flux.max()
            transient_strength = (spectral_flux - flux_min) / (
                flux_max - flux_min + 1e-8
            )
            boundary_crossing_loss = self.boundary_crossing_reward(
                s_target, transient_strength
            )

            # Lobe prediction loss (if model predicts lobes and curriculum is available)
            lobe_prediction_loss_val = torch.zeros(1, device=self.device)
            if self.model.predict_lobes and self.curriculum_lobes is not None:
                lobe_logits = parsed.get("lobe_logits")
                if lobe_logits is not None:
                    # Get target lobes for this batch
                    end_idx = min(sample_idx + batch_size, len(self.curriculum_lobes))
                    actual_batch_size = end_idx - sample_idx
                    if actual_batch_size > 0:
                        target_lobes = self.curriculum_lobes[sample_idx:end_idx]

                        # Create boundary mask if section boundaries are available
                        at_boundary = None
                        if self.section_boundaries:
                            at_boundary = torch.zeros(
                                actual_batch_size, device=self.device
                            )
                            for boundary_idx in self.section_boundaries:
                                if sample_idx <= boundary_idx < end_idx:
                                    local_idx = boundary_idx - sample_idx
                                    at_boundary[local_idx] = 1.0

                        lobe_prediction_loss_val = self.lobe_prediction_loss(
                            lobe_logits[:actual_batch_size], target_lobes, at_boundary
                        )

            # Control loss (curriculum learning)
            if control_targets is not None and current_curriculum_weight > 0.0:
                control_loss_val = self.control_loss(
                    predicted_controls, control_targets
                )
            else:
                control_loss_val = torch.zeros(1, device=self.device)

            # Total loss
            total_batch_loss = (
                self.correlation_weights["timbre_color"] * timbre_color_loss
                + self.correlation_weights["transient_impact"] * transient_impact_loss
                + boundary_crossing_loss
                + lobe_prediction_loss_val
                + current_curriculum_weight * control_loss_val
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_control_loss += control_loss_val.item()
            total_timbre_color += timbre_color_loss.item()
            total_transient_impact += transient_impact_loss.item()
            total_boundary_crossing += boundary_crossing_loss.item()
            total_lobe_prediction += lobe_prediction_loss_val.item()
            n_batches += 1

        # Average losses
        avg_losses = {
            "loss": total_loss / n_batches,
            "control_loss": total_control_loss / n_batches,
            "timbre_color_loss": total_timbre_color / n_batches,
            "transient_impact_loss": total_transient_impact / n_batches,
            "boundary_crossing_loss": total_boundary_crossing / n_batches,
            "lobe_prediction_loss": total_lobe_prediction / n_batches,
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

        logger.info(
            f"Starting control signal training for {epochs} epochs... (total frames: {all_features_tensor.shape[0]})"
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
                f'Control: {avg_losses["control_loss"]:.4f}'
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
