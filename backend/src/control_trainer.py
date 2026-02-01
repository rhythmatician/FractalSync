"""
Trainer for control signal model with step-based synthesis.

Trains model to predict delta steps that are applied by the runtime controller.
"""

import json
import os
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .control_model import AudioToControlModel
from .data_loader import AudioDataset
from .visual_metrics import LossVisualMetrics
from runtime_core import (
    FeatureExtractor,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    StepController,
    StepState,
)
from .julia_gpu import GPUJuliaRenderer

logger = logging.getLogger(__name__)


class UphillDeltaLoss(nn.Module):
    """Penalize Δc movement in the uphill direction when gradient is steep.

    Given delta (batch,2) and grad (batch,2):
      proj = (g · d) / (||g|| + eps)
      penalty_per_sample = weight * (ReLU(proj))^2 * ReLU(||g|| - grad_thresh)

    Returns average penalty across batch.
    """

    def __init__(
        self,
        weight: float = 1e-2,
        grad_thresh: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.weight = float(weight)
        self.grad_thresh = float(grad_thresh)
        self.eps = float(eps)

    def forward(self, delta: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Compute uphill penalty.

        Args:
            delta: Tensor (batch, 2) - Δc
            grad: Tensor (batch, 2) - ∇f(c)

        Returns:
            scalar tensor (mean penalty)
        """
        if delta.numel() == 0:
            return torch.tensor(0.0, device=delta.device)

        # compute dot products and gradient norms
        dot = torch.sum(grad * delta, dim=1)
        grad_norm = torch.sqrt(torch.sum(grad * grad, dim=1) + self.eps)

        proj = dot / (grad_norm + self.eps)

        # Only penalize positive projection (moving uphill)
        uphill = torch.nn.functional.relu(proj)

        # Only apply penalty when gradient magnitude exceeds threshold
        grad_mask = torch.nn.functional.relu(grad_norm - self.grad_thresh)

        penalty_per_sample = self.weight * (uphill**2) * grad_mask

        return torch.mean(penalty_per_sample)


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
        sx = torch.sum(x_centered**2)
        sy = torch.sum(y_centered**2)
        eps = 1e-6
        # If either input has (near-)zero variance, return zero loss. Preserve
        # gradient capability only if inputs require gradients (so backward() is
        # well-behaved in training & tests).
        if sx.item() < eps or sy.item() < eps:
            # Return a zero scalar that preserves a gradient path when one of the
            # inputs requires gradients so that backward() produces finite (zero)
            # gradients rather than being disconnected.
            if x.requires_grad:
                return (x * 0.0).sum()
            if y.requires_grad:
                return (y * 0.0).sum()
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        denominator = torch.sqrt(sx * sy)
        denom_clamped = torch.clamp(denominator, min=eps)
        correlation = numerator / denom_clamped
        return -correlation


class ControlTrainer:
    """Trainer for control signal model with step-based synthesis."""

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
        self.context_dim = getattr(model, "context_dim", 0)
        # Debug flag: enable verbose diagnostics when True
        self.debug = False

        # Default correlation weights
        default_weights = {
            "timbre_color": 1.0,
            "transient_impact": 1.0,
            "control_loss": 1.0,
            # Encourage negative correlation between loudness (rms) and
            # Mandelbrot distance proxy (larger = further from the set)
            "loudness_distance": 1.0,
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

        # Curriculum data
        self.curriculum_positions: Optional[torch.Tensor] = None
        self.curriculum_velocities: Optional[torch.Tensor] = None

        # Minimap context controller for on-the-fly context sampling during training
        # (uses the runtime_core minimap via StepController/context_for_state)
        self.step_controller: Optional[StepController] = None
        try:
            self.step_controller = StepController()
        except Exception:
            self.step_controller = None
            logger.warning(
                "Could not initialize StepController for context sampling; training will use zero context"
            )

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "control_loss": [],
            "timbre_color_loss": [],
            "transient_impact_loss": [],
            "loudness_distance_loss": [],
        }
        # Track last checkpoint for reporting
        self.last_checkpoint_path: Optional[str] = None

    def _generate_curriculum_data(self, n_samples: int):
        """Generate curriculum learning data from a simple circular path."""
        logger.info(f"Generating curriculum data: {n_samples} samples")
        thetas = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        positions = []
        velocities = []
        for idx, theta in enumerate(thetas):
            radius = 0.8
            current = np.array([radius * np.cos(theta), radius * np.sin(theta)])
            next_theta = thetas[(idx + 1) % len(thetas)]
            nxt = np.array([radius * np.cos(next_theta), radius * np.sin(next_theta)])
            positions.append(current.tolist())
            velocities.append((nxt - current).tolist())

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
        Extract delta-step targets from curriculum positions/velocities.

        Args:
            positions: Tensor of shape (batch_size, 2) with position data
            velocities: Optional tensor of shape (batch_size, 2) with velocity data

        Returns:
            Tensor of shape (batch_size, output_dim) with delta targets
        """
        if velocities is None:
            return torch.zeros(positions.shape[0], 2, device=self.device)

        return velocities[:, :2]

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

            # Sanity check: ensure model parameters are healthy before forward
            for pname, pparam in self.model.named_parameters():
                try:
                    if torch.isnan(pparam).any() or torch.isinf(pparam).any():
                        n_nan = int(torch.isnan(pparam).sum().item())
                        n_inf = int(torch.isinf(pparam).sum().item())
                        logger.error(
                            f"Model parameter {pname} contains NaNs/InFs before forward: NaNs={n_nan}, Infs={n_inf}"
                        )
                        # Instead of attempting a noisy reinitialization, abort clearly
                        raise RuntimeError(
                            "Model parameters contain NaNs/InFs before forward; aborting"
                        )
                except Exception:
                    logger.exception("Failed checking model parameters for NaN/Inf")
                    raise

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

            features_with_context = features
            if self.context_dim > 0:
                # Attempt to populate minimap-based context when curriculum positions
                # are available. Otherwise, fail loud.
                if (
                    self.step_controller is not None
                    and self.use_curriculum
                    and self.curriculum_positions is not None
                ):
                    # Build context per-sample from curriculum positions/velocities
                    ctx_list = []
                    for i in range(batch_size):
                        # Determine corresponding curriculum index (we generated positions earlier)
                        idx_in_curr = sample_idx - batch_size + i
                        # Clamp
                        idx_in_curr = max(
                            0, min(len(self.curriculum_positions) - 1, idx_in_curr)
                        )
                        pos = (
                            self.curriculum_positions[idx_in_curr]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        vel = (
                            self.curriculum_velocities[idx_in_curr]
                            .detach()
                            .cpu()
                            .numpy()
                            if self.curriculum_velocities is not None
                            else np.array([0.0, 0.0])
                        )
                        try:
                            # pos may be a numpy array or tensor; extract floats
                            c_real = float(pos[0])
                            c_imag = float(pos[1])
                            vd0 = float(vel[0])
                            vd1 = float(vel[1])
                            state = StepState(c_real, c_imag, vd0, vd1)
                            ctx = self.step_controller.context_for_state(state)
                            # ctx may be a StepContext object; prefer as_feature_vec if present
                            if hasattr(ctx, "as_feature_vec"):
                                fv = ctx.as_feature_vec()
                            elif isinstance(ctx, dict) and "feature_vec" in ctx:
                                fv = ctx["feature_vec"]
                            else:
                                fv = (
                                    getattr(ctx, "feature_vec", None)
                                    or [0.0] * self.context_dim
                                )
                        except Exception as e:
                            logger.warning(
                                f"Minimap context sampling failed for curriculum idx {idx_in_curr}: {e}"
                            )
                            fv = [0.0] * self.context_dim
                        ctx_list.append(fv)
                    # Convert to tensor
                    context = torch.tensor(
                        ctx_list, dtype=features.dtype, device=self.device
                    )
                    features_with_context = torch.cat([features, context], dim=1)
                else:
                    # Zero padding fallback (don't require minimap unless curriculum is enabled)
                    context = torch.zeros(
                        (batch_size, self.context_dim),
                        dtype=features.dtype,
                        device=self.device,
                    )
                    features_with_context = torch.cat([features, context], dim=1)

            # Forward pass
            predicted_controls = self.model(features_with_context)

            # Quick check: detect NaNs in model outputs
            try:
                pc_nans = int(torch.isnan(predicted_controls).sum().item())
                pc_infs = int(torch.isinf(predicted_controls).sum().item())
                if pc_nans > 0 or pc_infs > 0:
                    logger.error(
                        f"predicted_controls contains NaNs/Infs: NaNs={pc_nans}, Infs={pc_infs}"
                    )
                    logger.error(
                        f"predicted_controls sample: {predicted_controls.view(-1)[:10].detach().cpu().numpy().tolist()}"
                    )
            except Exception:
                logger.exception("Failed to inspect predicted_controls for NaN/Inf")

            # Parse delta outputs
            parsed = self.model.parse_output(predicted_controls)
            delta_real = parsed["delta_real"]
            delta_imag = parsed["delta_imag"]

            c_tensor = torch.stack([delta_real, delta_imag], dim=1)
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
                # Use delta_real as proxy for color hue (example correlation)
                color_hues.append(delta_real[i])
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

            # Sanitize tensors to avoid NaN/Inf propagation into the loss
            def _sanitize(t: torch.Tensor) -> torch.Tensor:
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, device=self.device, dtype=torch.float32)
                t = t.to(self.device)
                t = torch.where(torch.isnan(t), torch.zeros_like(t), t)
                t = torch.where(torch.isinf(t), torch.sign(t) * 1e6, t)
                return t

            spectral_centroid = _sanitize(spectral_centroid)
            color_hue_tensor = _sanitize(color_hue_tensor)
            spectral_flux = _sanitize(spectral_flux)
            temporal_change_tensor = _sanitize(temporal_change_tensor)

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

            # Compute a simple distance proxy per-sample. Use nu_norm from
            # the minimap when available (distance = 1 - nu_norm), or fail loud.
            distance_list = []
            for i in range(batch_size):
                c_real_val = julia_real[i].detach().item()
                c_imag_val = julia_imag[i].detach().item()
                if self.step_controller is None:
                    raise RuntimeError(
                        "Minimap StepController not available for loudness-distance loss calculation"
                    )
                try:
                    state = StepState(c_real_val, c_imag_val, 0.0, 0.0)
                    ctx = self.step_controller.context_for_state(state)
                    nu_norm = getattr(ctx, "nu_norm", None)
                    if nu_norm is not None:
                        distance_list.append(1.0 - float(nu_norm))
                        continue
                except Exception as e:
                    raise RuntimeError(
                        f"Minimap sampling failed for loudness-distance at batch idx {i}: {e}"
                    )

            distance_tensor = torch.tensor(
                distance_list, dtype=torch.float32, device=self.device
            )
            loudness_distance_loss = self.correlation_loss(
                -spectral_rms, distance_tensor
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
                + self.correlation_weights["loudness_distance"] * loudness_distance_loss
                + current_curriculum_weight * control_loss_val
            )

            # Diagnostic: detect NaNs and log key tensors before backward
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                # Minimal reporting in normal runs; verbose output only when debug=True
                try:
                    # Always log top-level scalar losses so failures are visible
                    logger.error(
                        f"NaN/Inf detected in total_batch_loss at batch {batch_idx}: "
                        f"timbre_color={getattr(timbre_color_loss, 'item', lambda: 'NA')()}, "
                        f"transient_impact={getattr(transient_impact_loss, 'item', lambda: 'NA')()}, "
                        f"loudness_distance={getattr(loudness_distance_loss, 'item', lambda: 'NA')()}, "
                        f"control_loss={getattr(control_loss_val, 'item', lambda: 'NA')()}"
                    )
                except Exception:
                    logger.exception("Failed to read scalar loss items")

                if self.debug:
                    # Scalar stats
                    try:
                        logger.debug(
                            f"predicted_controls: mean={predicted_controls.mean().item()}, std={predicted_controls.std().item()}"
                        )
                    except Exception:
                        logger.exception("Failed to compute predicted_controls stats")

                    try:
                        logger.debug(
                            f"features: mean={features.mean().item()}, std={features.std().item()}"
                        )
                    except Exception:
                        logger.exception("Failed to compute features stats")

                    try:
                        logger.debug(
                            f"distance: min={distance_tensor.min().item()}, max={distance_tensor.max().item()}, mean={distance_tensor.mean().item()}"
                        )
                    except Exception:
                        logger.exception("Failed to compute distance tensor stats")

                    # Check NaNs in key tensors
                    for name, t in (
                        ("predicted_controls", predicted_controls),
                        ("color_hue_tensor", color_hue_tensor),
                        ("spectral_rms", spectral_rms),
                        ("distance_tensor", distance_tensor),
                    ):
                        try:
                            n_nan = int(torch.isnan(t).sum().item())
                            n_inf = int(torch.isinf(t).sum().item())
                            logger.debug(f"{name}: NaNs={n_nan}, Infs={n_inf}")
                        except Exception:
                            logger.exception(f"Failed to check NaN/Inf for {name}")

                    # If predicted_controls has NaNs, run a layer-by-layer forward to find where NaNs originate
                    try:
                        x_debug = features_with_context.clone().detach()
                        for idx, layer in enumerate(self.model.encoder):
                            # If the layer has parameters, check them for NaNs/Infs first
                            try:
                                if hasattr(layer, "weight"):
                                    w = layer.weight.detach()
                                    b = (
                                        layer.bias.detach()
                                        if hasattr(layer, "bias")
                                        and layer.bias is not None
                                        else None
                                    )
                                    w_nan = int(torch.isnan(w).sum().item())
                                    w_inf = int(torch.isinf(w).sum().item())
                                    logger.debug(
                                        f"Encoder layer {idx} ({type(layer).__name__}) params: weight NaNs={w_nan}, Infs={w_inf}"
                                    )
                                    try:
                                        # sample few weight elements for inspection
                                        w_sample = (
                                            w.view(-1)[:10].cpu().numpy().tolist()
                                        )
                                        logger.debug(
                                            f"Encoder layer {idx} weight sample: {w_sample}"
                                        )
                                    except Exception:
                                        logger.exception("Failed to dump weight sample")
                                    if b is not None:
                                        b_nan = int(torch.isnan(b).sum().item())
                                        b_inf = int(torch.isinf(b).sum().item())
                                        logger.debug(
                                            f"Encoder layer {idx} ({type(layer).__name__}) params: bias NaNs={b_nan}, Infs={b_inf}"
                                        )
                                        try:
                                            b_sample = (
                                                b.view(-1)[:10].cpu().numpy().tolist()
                                            )
                                            logger.debug(
                                                f"Encoder layer {idx} bias sample: {b_sample}"
                                            )
                                        except Exception:
                                            logger.exception(
                                                "Failed to dump bias sample"
                                            )
                            except Exception:
                                logger.exception("Failed to inspect layer parameters")

                            mean = float(x_debug.mean().detach().cpu().item())
                            std = float(x_debug.std().detach().cpu().item())
                            logger.debug(
                                f"Encoder layer {idx} ({type(layer).__name__}): mean={mean:.6f}, std={std:.6f}"
                            )

                        # Check delta head layers
                        x_head = x_debug
                        for idx, layer in enumerate(self.model.delta_head):
                            x_head = layer(x_head)
                            n_nan = int(torch.isnan(x_head).sum().item())
                            n_inf = int(torch.isinf(x_head).sum().item())
                            mean = float(x_head.mean().detach().cpu().item())
                            std = float(x_head.std().detach().cpu().item())
                            logger.debug(
                                f"Delta head layer {idx} ({type(layer).__name__}): mean={mean:.6f}, std={std:.6f}, NaNs={n_nan}, Infs={n_inf}"
                            )
                            if n_nan > 0 or n_inf > 0:
                                break
                    except Exception:
                        logger.exception("Layerwise forward failed")

                # Always abort this batch so upstream code can handle the failure
                raise RuntimeError(
                    "Aborting training due to NaN/Inf in total_batch_loss - see logs for details"
                )

            # Backward pass
            self.optimizer.zero_grad()

            # If loss does not depend on model parameters (no grad_fn), skip backward
            if not (
                hasattr(total_batch_loss, "requires_grad")
                and total_batch_loss.requires_grad
            ):
                logger.info(
                    "total_batch_loss has no gradient graph; skipping backward and optimizer step for this batch"
                )
            else:
                total_batch_loss.backward()

                # Detect NaN/Inf in gradients to avoid corrupting parameters
                grad_bad = False
                for g_name, g_param in self.model.named_parameters():
                    if g_param.grad is None:
                        continue
                    try:
                        if (
                            torch.isnan(g_param.grad).any()
                            or torch.isinf(g_param.grad).any()
                        ):
                            n_nan = int(torch.isnan(g_param.grad).sum().item())
                            n_inf = int(torch.isinf(g_param.grad).sum().item())
                            logger.error(
                                f"Gradient for {g_name} contains NaNs/Infs: NaNs={n_nan}, Infs={n_inf}"
                            )
                            grad_bad = True
                    except Exception:
                        logger.exception(f"Failed to inspect gradient for {g_name}")
                        grad_bad = True

                if grad_bad:
                    logger.warning(
                        "Invalid gradients detected; aborting training. Inspect runtime-core and its pybindings if this persists"
                    )
                    # Raise an exception instead of exiting the process to allow
                    # test harnesses and higher-level callers to handle/fail gracefully.
                    raise RuntimeError("Invalid gradients detected; aborting training")
                else:
                    self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_control_loss += control_loss_val.item()
            total_timbre_color += timbre_color_loss.item()
            total_transient_impact += transient_impact_loss.item()
            total_loudness_distance += loudness_distance_loss.item()
            n_batches += 1

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_control_loss += control_loss_val.item()
            total_timbre_color += timbre_color_loss.item()
            total_transient_impact += transient_impact_loss.item()
            total_loudness_distance += loudness_distance_loss.item()
            n_batches += 1

        # Average losses
        avg_losses = {
            "loss": total_loss / n_batches,
            "control_loss": total_control_loss / n_batches,
            "timbre_color_loss": total_timbre_color / n_batches,
            "transient_impact_loss": total_transient_impact / n_batches,
            "loudness_distance_loss": total_loudness_distance / n_batches,
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

        # Quick sanity: Ensure no param corruption happened during feature computations
        for pname, pparam in self.model.named_parameters():
            if torch.isnan(pparam).any() or torch.isinf(pparam).any():
                logger.error(
                    f"Model parameter {pname} contains NaNs/InFs after compute_normalization_stats: NaNs={int(torch.isnan(pparam).sum().item())}, Infs={int(torch.isinf(pparam).sum().item())}"
                )
                raise RuntimeError(
                    "Model parameters corrupted after compute_normalization_stats; aborting"
                )

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
                f'Control: {avg_losses["control_loss"]:.4f}, '
                f'LoudnessDist: {avg_losses["loudness_distance_loss"]:.4f}'
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
            "context_dim": self.context_dim,
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
