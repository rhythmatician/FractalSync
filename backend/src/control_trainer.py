"""
Trainer for control signal model with orbit-based synthesis.

Trains model to predict control signals that drive deterministic orbit synthesis.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .control_model import AudioToControlModel
from .cspace_proxies import (
    cardioid_mu,
    cardioid_proximity,
    canonical_hop_dt,
    orbit_controller_momentum_sequence,
    orbit_controller_oracle_sequence,
    shore_proximity,
    synthesize_c,
)
from .data_loader import AudioDataset
from .c_trace_plot import collect_c_traces, plot_c_traces
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

# Candidate locations of the baked mip pyramid artifacts (issue #88). The
# bake script writes them to its CWD; the repo root and backend/ are both
# plausible depending on how training was launched.
_MIP_PYRAMID_CANDIDATES = [
    ("mandel_F_mips_f32.bin", "mandel_S_mips_f32.bin", "mandel_mips_meta.json"),
    (
        "../mandel_F_mips_f32.bin",
        "../mandel_S_mips_f32.bin",
        "../mandel_mips_meta.json",
    ),
]

_pyramid_load_attempted = False
_pyramid_loaded = False


def _ensure_mip_pyramid_loaded() -> bool:
    """Load the baked mip pyramid into runtime-core (idempotent).

    Returns True when the pyramid is available so shore-proximity losses read
    the minimaps; False when the artifacts are missing and callers should fall
    back to the deprecated cardioid approximation.
    """
    global _pyramid_load_attempted, _pyramid_loaded
    if _pyramid_load_attempted:
        return _pyramid_loaded
    _pyramid_load_attempted = True

    import runtime_core

    for f_bin, s_bin, meta in _MIP_PYRAMID_CANDIDATES:
        if os.path.exists(f_bin) and os.path.exists(s_bin) and os.path.exists(meta):
            try:
                runtime_core.load_mip_pyramid_py(f_bin, s_bin, meta)
                _pyramid_loaded = True
                logger.info("Mip pyramid loaded from %s", os.path.abspath(meta))
                return True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load mip pyramid from %s: %s", meta, exc)
                break

    logger.warning(
        "Baked mip pyramid not found; falling back to deprecated cardioid "
        "proximity. Run scripts/bake_mandel_maps_gl.py to generate it."
    )
    return False


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
        use_cspace_proxies: bool = True,
        coverage_weight: float = 0.1,
        scheduled_sampling_start: float = 0.0,
        scheduled_sampling_max: float = 0.3,
        scheduled_sampling_ramp_epochs: int = 20,
        clip_length: int = 1,
        anti_dwell_weight: float = 1.0,
        anti_dwell_target: float = 0.15,
        zone_weight: float = 1.0,
        zone_min: float = 0.01,
        zone_max: float = 0.45,
        julia_stability_weight: float = 0.0,
        julia_stability_base: float = 0.02,
        julia_stability_loud_gain: float = 0.08,
        song_identity_weight: float = 0.0,
        song_identity_margin: float = 0.35,
        region_dwell_weight: float = 0.0,
        region_dwell_window: int = 240,
        region_dwell_p: float = 0.08,
        region_dwell_phi: float = 0.5,
        region_dwell_depth_gate: float = 0.6,
        region_dwell_onset_gate: float = 0.5,
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
            use_cspace_proxies: Supervise via differentiable c-space proxies
                (cardioid proximity, orbit speed) instead of rendered images.
                Removes the slow non-differentiable render loop from training.
            coverage_weight: Weight for the c-space coverage (anti-revisit)
                diversity regularizer.
            scheduled_sampling_start: Initial probability of feeding the model's
                previous prediction back as input context.
            scheduled_sampling_max: Final scheduled-sampling probability after ramp.
            scheduled_sampling_ramp_epochs: Epochs over which to ramp from start to max.
            clip_length: Number of contiguous windows per training clip.
                1 disables clip mode (legacy per-window batches). Values of
                32–128 enable truncated-BPTT-style sequence training.
            anti_dwell_weight: Weight for the scale-aware anti-dwell penalty
                that keeps c(t) moving through c-space over time.
            anti_dwell_target: Minimum required per-frame displacement of c,
                normalized by local feature scale (cardioid proximity).
                Scale-free: near the Mandelbrot boundary tiny moves count;
                far from it, larger travel is demanded for equal visual change.
            zone_weight: Weight for the visibility-band constraint that keeps
                c within the region where Julia sets are visually interesting.
            zone_min: Minimum cardioid proximity — below this c is deep in the
                interior where Julia sets are a solid blob.
            zone_max: Maximum cardioid proximity — above this c is far outside
                the set where Julia sets are a sparse dust (mostly black).
        """
        self.model: AudioToControlModel = model.to(device)

        # Load the Map's mip pyramid (issue #88) so shore-proximity losses can
        # read the minimaps. Best-effort: if the baked artifacts are missing,
        # fall back to the deprecated cardioid approximation with a warning.
        self._use_minimap_proximity = _ensure_mip_pyramid_loaded()

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
        self.use_cspace_proxies = use_cspace_proxies
        self.coverage_weight = coverage_weight
        self.scheduled_sampling_start = scheduled_sampling_start
        self.scheduled_sampling_max = scheduled_sampling_max
        self.scheduled_sampling_ramp_epochs = max(1, scheduled_sampling_ramp_epochs)
        self.clip_length = max(1, clip_length)
        self.anti_dwell_weight = anti_dwell_weight
        self.anti_dwell_target = anti_dwell_target
        self.zone_weight = zone_weight
        self.zone_min = zone_min
        self.zone_max = zone_max
        # J(c) frame-to-frame stability (see _julia_stability_loss).
        self.julia_stability_weight = julia_stability_weight
        # Perceptual displacement allowed in silence (in local-scale units).
        # Must exceed anti_dwell_target (0.15) so the two losses never
        # conflict: allowed(e) = base + gain*e >= required(e) = 0.15*e
        # for all e in [0,1] requires base >= 0.15 and gain >= 0.15.
        self.julia_stability_base = max(julia_stability_base, 0.16)
        # Extra allowed displacement per unit of audio energy (loud parts
        # may drift a little more).
        self.julia_stability_loud_gain = max(julia_stability_loud_gain, 0.16)
        # Song-identity region loss (see _song_identity_loss): different
        # songs explore different areas, consistently within each song.
        self.song_identity_weight = song_identity_weight
        # Minimum distance between song home regions (c-space units).
        self.song_identity_margin = song_identity_margin
        # Region-dwell loss (see _region_dwell_loss): penalize c occupying
        # the same J(c)-region for too long. "Region" is defined in J-space
        # via the perceptual coordinates (proximity, boundary angle).
        self.region_dwell_weight = region_dwell_weight
        # Look-back window (frames) for continuous occupation. 240 = 4 s.
        self.region_dwell_window = region_dwell_window
        # Region radius in proximity units (J-space perceptual axis).
        self.region_dwell_p = region_dwell_p
        # Region radius in boundary-angle units (radians).
        self.region_dwell_phi = region_dwell_phi
        # Occupation fraction above which a frame counts as dwelling.
        self.region_dwell_depth_gate = region_dwell_depth_gate
        # Onset level that resets the dwell window (a hit frees c to jump).
        self.region_dwell_onset_gate = region_dwell_onset_gate
        # c-trace plot metadata (set in train()): dataset file order and
        # window size, so per-epoch plots replay the same songs.
        self._trace_dataset_files: List[Path] = []
        self._trace_window_frames: int = 10
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
            "coverage_loss": [],
            "anti_dwell_loss": [],
            "julia_stability_loss": [],
            "song_identity_loss": [],
            "region_dwell_loss": [],
            "zone_loss": [],
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
        teacher_forcing_override: Optional[float] = None,
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
        effective_tf = (
            self.rollout_teacher_forcing
            if teacher_forcing_override is None
            else teacher_forcing_override
        )
        carry_weight = max(0.0, min(1.0, 1.0 - effective_tf))
        teacher_weight = max(0.0, min(1.0, effective_tf))

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

    def _scheduled_sampling_prob(self, epoch: int) -> float:
        """Ramp scheduled-sampling probability from start to max over epochs.

        The probability is interpreted as how much of the rollout carryover
        comes from the model's own previously simulated controls instead of
        ground-truth predictions (i.e. ``1 - effective teacher forcing``).
        Early epochs train close to ground truth; later epochs progressively
        expose the model to its own drift, matching inference conditions.
        """
        if self.scheduled_sampling_max <= 0.0:
            return 0.0
        frac = min(1.0, max(0.0, epoch / float(self.scheduled_sampling_ramp_epochs)))
        return self.scheduled_sampling_start + frac * (
            self.scheduled_sampling_max - self.scheduled_sampling_start
        )

    def _synthesize_c_differentiable(
        self,
        s_target: torch.Tensor,
        alpha: torch.Tensor,
        band_gates: torch.Tensor,
        thetas: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable c-space synthesis (delegates to cspace_proxies).

        Thin wrapper over :func:`cspace_proxies.synthesize_c`, which mirrors
        ``runtime_core::controller::synthesize``. The residual phases come from
        ``runtime_core.residual_phases_for_seed_py`` (the same single source
        of truth the runtime controller uses), so training and runtime share
        identical phase statistics — no golden-angle approximation.
        """
        import runtime_core

        phases = runtime_core.residual_phases_for_seed_py(
            DEFAULT_ORBIT_SEED, self.k_residuals
        )
        return synthesize_c(
            s_target=s_target,
            alpha=alpha,
            band_gates=band_gates,
            thetas=thetas,
            k_residuals=self.k_residuals,
            residual_cap=DEFAULT_RESIDUAL_CAP,
            phases=phases,
        )

    def _cardioid_proximity_differentiable(self, c: torch.Tensor) -> torch.Tensor:
        """Shore proximity for supervision (delegates to cspace_proxies).

        Sunset note (issue #88): the cardioid approximation is retired as the
        shore-distance oracle; this now samples the Map's mip pyramid S field
        via the Rust minimap reader when the pyramid is loaded, falling back
        to the deprecated cardioid approximation otherwise. Kept under the
        old name so existing loss call sites remain stable.
        """
        if self._use_minimap_proximity:
            return shore_proximity(c)
        return cardioid_proximity(c)

    def _region_dwell_loss(
        self,
        c_sequence: torch.Tensor,
        segment_ids: torch.Tensor,
        onset_strength: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize c dwelling in the same J(c)-region for too long.

        The user's framing, made precise. "Region" is defined in J(c)-space,
        not c-space: two points a, b are in the same region when J(a) ≈ J(b).
        We use the cardioid-proximity coordinate p(c) = ||mu|-1| as the
        perceptual axis — it is continuous in c outside the set, and J(c)'s
        character varies primarily along it (deep interior = blob, boundary =
        filigree, far exterior = dust). Two c points with similar p AND
        similar angle around the cardioid render similar Julia sets.

        Region signature per frame: (p(c), phi(c)) where phi is the angle of
        mu = 1 - sqrt(1-4c) — the position ON the cardioid boundary nearest
        c. Frames whose signatures stay within (region_p, region_phi) of a
        recently-visited signature are "dwelling".

        Loss: for each frame, look back over a dwell window; if ALL recent
        signatures are within the region radius, that frame is dwelling and
        incurs loss proportional to how deep into the window it sits (early
        occupancy is fine — a section may open in one area — but lingering
        the whole window is not). Transients reset the window: a hit lets c
        jump regions freely (that's the desired section-change behavior).

        This directly encodes "okay to jitter in one area for a section,
        but then it's gotta move" without forbidding returns: coming BACK
        to an area after visiting others is not penalized — only continuous
        occupation is.
        """
        if c_sequence.shape[0] < 2 or self.region_dwell_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        n = c_sequence.shape[0]
        # Analytic proximity (differentiable). The minimap S field is a
        # ridge detector (0 interior / 1 boundary) with zero gradient
        # inside the set — unusable as a distance axis for losses.
        proximity = cardioid_proximity(c_sequence)

        # mu = 1 - sqrt(1-4c): the cardioid parameterization. Its angle is
        # the position along the boundary — the second perceptual axis.
        # Shared helper (cspace_proxies) — do not re-derive inline.
        mu = cardioid_mu(c_sequence)
        phi = torch.atan2(mu.imag.float(), mu.real.float())

        # Region signature: (p, phi). Two frames are "same region" when both
        # coordinates are close. phi wraps at ±pi, so compare via sin/cos.
        p = proximity.float()

        onset = torch.nan_to_num(onset_strength.reshape(-1).float(), nan=0.0)
        onset = torch.clamp(onset, 0.0, 1.0)

        window = self.region_dwell_window
        losses: List[torch.Tensor] = []
        for t in range(window // 2, n):
            lo = max(0, t - window // 2)
            # SPREAD-based dwell detection (not fixed-region occupation):
            # measure how much the recent perceptual signatures have SPREAD.
            # A dwelling c has near-zero spread in (p, phi); a traveling c
            # accumulates spread even if it moves slowly or revisits. This
            # is robust to region size choices and directly encodes "has c
            # stopped exploring?".
            dp = torch.abs(p[lo:t] - p[lo:t].mean())
            dphi = torch.abs(phi[lo:t] - phi[lo:t].mean())
            dphi = torch.minimum(dphi, 2.0 * torch.pi - dphi)  # wrap
            spread_p = dp.max()
            spread_phi = dphi.max()

            # Normalized dwell: how far below the region radius the recent
            # spread sits. spread >= region radius -> not dwelling (0 loss).
            dwell_p = torch.relu(self.region_dwell_p - spread_p) / self.region_dwell_p
            dwell_phi = (
                torch.relu(self.region_dwell_phi - spread_phi) / self.region_dwell_phi
            )
            # Both axes must be confined for a true dwell (moving along one
            # axis alone still changes J(c) meaningfully).
            dwell = dwell_p * dwell_phi

            # Transient escape: hits in the recent window PARTIALLY forgive
            # dwelling — c was allowed to jump regions, but if it stayed in
            # the same J-neighborhood anyway, the dwell still counts (with
            # reduced weight per hit). Full forgiveness only when hits are
            # frequent relative to the window.
            n_hits = (onset[lo:t] > self.region_dwell_onset_gate).float().sum()
            hit_fraction = torch.clamp(n_hits / max(1, (t - lo) / 60.0), 0.0, 1.0)
            dwell = dwell * (1.0 - 0.5 * hit_fraction)

            losses.append(dwell)

        if not losses:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def _coverage_loss(
        self, c_sequence: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Anti-revisit diversity regularizer over contiguous same-segment runs.

        Penalizes points that fall inside the convex hull of earlier points in
        the same run: low hull coverage means the orbit keeps revisiting the
        same region of c-space (the "gets old after a few seconds" failure).
        Implemented as mean pairwise-distance deficit: for each run we compute
        the mean pairwise distance between sampled points; runs whose points
        cluster tightly produce a small value, which we invert into a loss via
        negative-mean normalized by the expected spread of controls.
        """
        if c_sequence.shape[0] < 2 or self.coverage_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        seg_cpu = segment_ids.detach().cpu().tolist()
        runs: List[Tuple[int, int]] = []
        run_start = 0
        for idx in range(1, len(seg_cpu)):
            if seg_cpu[idx] != seg_cpu[idx - 1]:
                runs.append((run_start, idx))
                run_start = idx
        runs.append((run_start, len(seg_cpu)))

        terms: List[torch.Tensor] = []
        max_points_per_run = 64
        for start, end in runs:
            pts = c_sequence[start:end]
            n = pts.shape[0]
            if n < 2:
                continue
            if n > max_points_per_run:
                idx_sel = torch.linspace(0, n - 1, max_points_per_run).long()
                pts = pts[idx_sel]
                n = pts.shape[0]
            re = pts.real.float()
            im = pts.imag.float()
            diff_re = re.unsqueeze(1) - re.unsqueeze(0)
            diff_im = im.unsqueeze(1) - im.unsqueeze(0)
            dists = torch.sqrt(diff_re**2 + diff_im**2 + 1e-12)
            # Mean off-diagonal pairwise distance = spatial spread of the run.
            mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
            spread = dists[mask].mean()
            # Loss = inverse spread: tight clustering → large loss.
            terms.append(1.0 / (spread + 1e-3))

        if not terms:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return torch.stack(terms).mean()

    def _temporal_thetas(
        self,
        omega_scale: torch.Tensor,
        segment_ids: torch.Tensor,
        audio_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Physics-based carrier phase: velocity integrates acceleration with drag.

        The model's omega_scale output is an *acceleration* signal, not a
        velocity. Angular velocity evolves as:

            v(t) = drag * v(t-1) + accel_scale * omega_scale(t)
            theta(t) = theta(t-1) + v(t)

        where ``drag`` (slightly < 1) is a constant weak friction. This gives
        the exact behavior the user specified: c's motion is driven by audio —
        sustained loud audio builds up angular velocity, and during silence the
        drag gradually decays velocity to zero (c coasts to a stop, no hard
        threshold). Fully differentiable w.r.t. omega_scale via the scan.

        Phase resets at segment boundaries (new file/clip).
        """
        drag = 0.92  # per-frame velocity retention; ~e-folding over ~12 frames
        accel_gain = 0.02  # rad/frame^2 at omega_scale = 1

        accel = accel_gain * omega_scale.reshape(-1).float()
        if audio_energy is not None:
            # Audio gates the acceleration: silence produces no thrust.
            accel = accel * audio_energy.reshape(-1).float()

        seg = segment_ids.reshape(-1)
        n = accel.shape[0]

        # Sequential scan (differentiable through time).
        velocity = torch.zeros(n, device=accel.device, dtype=accel.dtype)
        theta = torch.zeros(n, device=accel.device, dtype=accel.dtype)
        v = torch.zeros((), device=accel.device, dtype=accel.dtype)
        th = torch.zeros((), device=accel.device, dtype=accel.dtype)
        for i in range(n):
            if i > 0 and seg[i] != seg[i - 1]:
                v = torch.zeros_like(v)
                th = torch.zeros_like(th)
            v = drag * v + accel[i]
            th = th + v
            velocity[i] = v
            theta[i] = th
        return theta % (2.0 * np.pi)

    def _anti_dwell_loss(
        self,
        c_sequence: torch.Tensor,
        segment_ids: torch.Tensor,
        audio_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scale-aware penalty on c(t) dwelling, gated by audio energy.

        The user's insight: 'region' is not definable in absolute c-space units
        because visual feature size varies enormously across the Mandelbrot set
        (a sliver outside the boundary shows as much variety as an entire lobe).
        We therefore normalize displacement by the LOCAL feature scale, proxied
        by cardioid proximity ||mu|-1| (distance from the boundary in multiplier
        space): near-boundary points have tiny features, far points have large
        ones.

        Physics model: c is a body with inertia; audio provides the only
        accelerating force, and a weak constant friction always opposes
        motion. There is deliberately NO silence threshold — instead the
        required movement scales smoothly with audio energy, so during
        silence the requirement decays toward zero and friction naturally
        brings c to a stop. Loud audio demands movement; silence permits rest.
        """
        if c_sequence.shape[0] < 2 or self.anti_dwell_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Analytic proximity (differentiable): S field is flat 0 inside the
        # set, which would zero the anti-dwell scale everywhere c actually
        # parks.
        proximity = cardioid_proximity(c_sequence)
        # Local feature scale: floor at a small epsilon so deep-interior points
        # (proximity ~ 0) still demand some minimal motion.
        local_scale = torch.clamp(proximity, min=0.02)

        seg = segment_ids.reshape(-1)
        step_ok = torch.ones_like(proximity, dtype=torch.bool)
        if seg.shape[0] > 1:
            step_ok[1:] = seg[1:] == seg[:-1]
        step_ok[0] = False  # no predecessor for first point

        dc = c_sequence[1:] - c_sequence[:-1]
        displacement = torch.abs(dc)

        # Audio-gated requirement: energy in [0, 1] scales the target.
        # energy = RMS (index 2) when provided; silence -> ~0 requirement.
        if audio_energy is not None:
            energy = torch.nan_to_num(audio_energy.reshape(-1).float(), nan=0.0)
            energy = torch.clamp(energy, 0.0, 1.0)
        else:
            energy = torch.ones_like(displacement)
        required = self.anti_dwell_target * local_scale[1:] * energy[1:]

        # Smooth displacement over a short horizon so single-frame jitter
        # cannot satisfy the requirement while c overall stays put: running
        # max of displacement over `dwell_window` frames.
        window = min(8, displacement.shape[0])
        if window > 1:
            disp_used = (
                torch.nn.functional.max_pool1d(
                    displacement.unsqueeze(0).unsqueeze(0),
                    kernel_size=window,
                    stride=1,
                    padding=window // 2,
                )
                .squeeze(0)
                .squeeze(0)[: displacement.shape[0]]
            )
        else:
            disp_used = displacement

        # Hinge: penalize frames where even the best recent movement falls
        # short of the energy-scaled target.
        shortfall = torch.relu(required - disp_used)
        # Mask out segment-boundary transitions (no meaningful predecessor).
        valid = step_ok[1:]
        if valid.any():
            return shortfall[valid].mean()
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def _zone_loss(self, c_sequence: torch.Tensor) -> torch.Tensor:
        """Keep c inside the visibility band where Julia sets are interesting.

        Two dead zones exist in c-space:
        - Deep interior (proximity ~ 0): Julia set is a connected blob that
          barely changes shape — the "still image" failure.
        - Far exterior (large proximity): Julia set is disconnected Cantor
          dust — mostly black, sparse, boring.

        The interesting band is a thin shell hugging the set boundary. We
        penalize proximity outside [zone_min, zone_max] with a smooth hinge.

        IMPORTANT: this uses the ANALYTIC cardioid proximity (differentiable
        closed form), NOT the minimap S field. S is a ridge detector — 0
        across the whole interior, ~1 at the boundary — so a zone band built
        on S has no gradient inside the set (flat 0) and actively PENALIZES
        the boundary (S≈1 > zone_max). That bug trained the model to park in
        the gravity valley and never approach the Shore.
        """
        if self.zone_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        proximity = cardioid_proximity(c_sequence)
        below = torch.relu(self.zone_min - proximity)
        above = torch.relu(proximity - self.zone_max)
        return (below**2 + above**2).mean()

    def _julia_stability_loss(
        self,
        c_sequence: torch.Tensor,
        segment_ids: torch.Tensor,
        audio_energy: torch.Tensor,
        onset_strength: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-to-frame stability of the rendered Julia set J(c).

        The user's framing: what matters visually is not |dc| but how much
        J(c) changes frame to frame. Two regimes:

        - Quiet parts: J(c) should be nearly STILL. Penalize any c
          displacement, scaled by the local feature size (proximity) so the
          penalty is perceptual, not absolute — the same |dc| barely changes
          J far from the boundary but completely transforms it near it.
        - Transients: c crossing in/out of the Mandelbrot set flips J(c)
          between connected and dust — a full-frame change. That is DESIRED
          on transients/section transitions, so displacement there is
          exempt (gated by onset strength).

        Loss = mean over frames of (quiet-gate * perceptual-displacement),
        where perceptual-displacement = |dc| / local_scale and the quiet
        gate fades out on transients. Fully differentiable w.r.t. the
        controls through c_sequence.
        """
        if c_sequence.shape[0] < 2 or self.julia_stability_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Analytic proximity (differentiable): same flat-S reasoning as
        # anti-dwell — the perceptual scale must be nonzero in the interior.
        proximity = cardioid_proximity(c_sequence)
        # Local feature scale: near the boundary J(c) is infinitely detailed,
        # so tiny dc = huge visual change; far away the same dc is invisible.
        local_scale = torch.clamp(proximity, min=0.02)

        seg = segment_ids.reshape(-1)
        step_ok = torch.ones_like(proximity, dtype=torch.bool)
        if seg.shape[0] > 1:
            step_ok[1:] = seg[1:] == seg[:-1]
            step_ok[0] = False

        dc = torch.abs(c_sequence[1:] - c_sequence[:-1])
        perceptual_dc = dc / local_scale[1:]

        # Quiet gate: full penalty in silence/quiet, fading to zero on
        # transients (where big J(c) changes are wanted).
        onset = torch.nan_to_num(onset_strength.reshape(-1).float(), nan=0.0)
        onset = torch.clamp(onset, 0.0, 1.0)[1:]
        quiet_gate = 1.0 - onset

        # Energy scaling: loud-but-steady parts may drift a little more
        # than silence, per the user's "less movement for quiet parts,
        # a little more for loud parts".
        #
        # IMPORTANT: this allowance must stay ABOVE the anti-dwell loss's
        # requirement (anti_dwell_target * energy = 0.15*energy local
        # units). With the old base=0.02/gain=0.08 the two losses were in
        # direct conflict at Tool-level energy (anti-dwell demanded 0.09
        # while stability allowed 0.068 at e=0.6) — the model resolved the
        # fight by parking in one region. The band below guarantees
        # allowed >= required + margin for all energy levels.
        energy = torch.nan_to_num(audio_energy.reshape(-1).float(), nan=0.0)
        energy = torch.clamp(energy, 0.0, 1.0)[1:]
        allowed = self.julia_stability_base + self.julia_stability_loud_gain * energy

        excess = torch.relu(perceptual_dc - allowed) * quiet_gate
        valid = step_ok[1:]
        if valid.any():
            return excess[valid].mean()
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def _song_identity_loss(
        self,
        c_sequence: torch.Tensor,
        segment_ids: torch.Tensor,
        song_fingerprints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Song-identity region loss: different songs explore different areas.

        The user's requirement, stated mathematically. Let mu_s be the
        centroid of c(t) for song s and sigma_s its spread. Two terms:

        1. SEPARATION (between songs): penalize small pairwise distances
           between song centroids. For songs s != s':
               L_sep = sum_{s<s'} max(0, margin - |mu_s - mu_s'|)
           pushing each song's home region at least `margin` away from
           every other song's. This is the direct answer to "why does it
           always converge to the same region" — same region now costs.

        2. CONSISTENCY (within a song): penalize large per-frame deviation
           from the song's own centroid:
               L_cons = mean_s mean_t |c_s(t) - mu_s| / margin
           normalized by margin so its scale matches term 1. This is the
           "prefer certain regions for certain songs" part: a song keeps
           coming home to its own area instead of wandering uniformly.

        The fingerprint tensor (optional) carries a learned embedding per
        song; when provided, songs with similar audio get similar target
        regions via a contrastive pull. Without it, the loss is purely
        repulsive — every song claims distinct territory.

        Fully differentiable w.r.t. the controls through c_sequence.
        """
        if c_sequence.shape[0] < 2 or self.song_identity_weight <= 0.0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        seg = segment_ids.reshape(-1)
        unique_songs = torch.unique(seg)
        n_songs = unique_songs.numel()
        if n_songs < 1:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Per-song centroids (differentiable means over each song's frames).
        centroids = []
        for s in unique_songs:
            mask = seg == s
            if mask.sum() < 2:
                continue
            centroids.append(c_sequence[mask].real.float().mean())
            centroids.append(c_sequence[mask].imag.float().mean())
        if len(centroids) < 4:
            # Fewer than 2 usable songs: only the consistency term applies.
            pass

        n_usable = len(centroids) // 2
        margin = self.song_identity_margin

        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # ---- Term 1: separation between song centroids ----
        if n_usable >= 2:
            mu = torch.stack(centroids).view(n_usable, 2)
            # Pairwise centroid distances.
            diff = mu.unsqueeze(1) - mu.unsqueeze(0)
            dists = torch.sqrt(diff**2 + 1e-12).sum(dim=2)
            iu = torch.triu_indices(n_usable, n_usable, offset=1)
            pair_d = dists[iu[0], iu[1]]
            separation = torch.relu(margin - pair_d).pow(2).mean()
            loss = loss + separation

        # ---- Term 2: within-song consistency ----
        cons_terms: List[torch.Tensor] = []
        for s in unique_songs:
            mask = seg == s
            if mask.sum() < 2:
                continue
            pts = c_sequence[mask]
            mu_re = pts.real.float().mean()
            mu_im = pts.imag.float().mean()
            dev = torch.sqrt(
                (pts.real.float() - mu_re) ** 2
                + (pts.imag.float() - mu_im) ** 2
                + 1e-12
            )
            # Cap: deviations beyond 2*margin are already "wandering" —
            # the coverage loss handles diversity; we only penalize the
            # middle ground so songs stay coherent without being pinned.
            excess = torch.relu(dev - 2.0 * margin)
            cons_terms.append(excess.mean())
        if cons_terms:
            loss = loss + torch.stack(cons_terms).mean() / margin

        return loss

    def _make_clip_dataloader(
        self,
        features: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_size: int,
        shuffle_clips: bool = True,
    ) -> DataLoader:
        """Build a DataLoader whose samples are contiguous same-file clips.

        Each dataset item is a (clip_length, feature_dim) tensor plus its
        segment id; the DataLoader stacks them into batches of shape
        (batch, clip_length, feature_dim). The trainer's train_epoch flattens
        these back to (batch * clip_length, feature_dim) so all existing loss
        machinery operates on temporally contiguous runs with correct
        segment_ids — enabling true sequence supervision and TBPTT-style
        gradient flow across the clip.
        """
        clip_len = self.clip_length
        clips: List[torch.Tensor] = []
        clip_segments: List[torch.Tensor] = []

        seg_np = segment_ids.numpy()
        feat_np = features.numpy()
        n = len(seg_np)
        start = 0
        while start + clip_len <= n:
            if seg_np[start] == seg_np[start + clip_len - 1]:
                clips.append(torch.tensor(feat_np[start : start + clip_len]))
                clip_segments.append(
                    torch.full((clip_len,), seg_np[start], dtype=torch.int64)
                )
                start += clip_len
            else:
                # Skip over the file boundary.
                start += 1

        if not clips:
            logger.warning(
                "No complete clips of length %d could be formed; falling back to window batching.",
                clip_len,
            )
            self.clip_length = 1
            return DataLoader(
                TensorDataset(features, segment_ids),
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        clip_dataset = TensorDataset(torch.stack(clips), torch.stack(clip_segments))
        logger.info(
            "Clip training enabled: %d clips of length %d (batch covers %d contiguous frames)",
            len(clip_dataset),
            clip_len,
            batch_size * clip_len,
        )
        return DataLoader(
            clip_dataset,
            batch_size=batch_size,
            shuffle=shuffle_clips,
            num_workers=self.num_workers,
        )

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
        total_coverage = 0.0
        total_anti_dwell = 0.0
        total_julia_stability = 0.0
        total_song_identity = 0.0
        total_region_dwell = 0.0
        total_zone = 0.0
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

            # Clip mode: (batch, clip_length, feature_dim) →
            # (batch * clip_length, feature_dim) so downstream losses see one
            # contiguous temporal run per clip with correct segment ids.
            if features.dim() == 3:
                batch_size_clip, clip_len, feat_dim = features.shape
                features = features.reshape(batch_size_clip * clip_len, feat_dim)
                if segment_ids is not None:
                    segment_ids = segment_ids.reshape(batch_size_clip * clip_len)

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

            if self.use_cspace_proxies:
                # ---- Differentiable c-space supervision (fast path) ----
                # Physics-based theta: audio-driven acceleration vs constant
                # drag. Silence decays velocity to zero gradually (no hard
                # threshold), matching the desired friction model.
                try:
                    n_features_per_frame = (
                        self.feature_extractor.num_features_per_frame()
                    )
                except Exception:
                    n_features_per_frame = 6
                window_frames = features.shape[1] // n_features_per_frame
                features_reshaped = features.view(
                    batch_size, window_frames, n_features_per_frame
                )
                avg_features = features_reshaped.mean(dim=1)

                _rms_for_thrust = avg_features[:, 2]
                thrust_for_c = (
                    torch.sigmoid(_rms_for_thrust) * 0.06
                )  # sigmoid so centered norm RMS still orbits
                # Music push energy (orbit-controller/3): sigmoid of RMS in
                # [0,1] — drives the uphill push toward the Shore in the
                # mirror integrator (gravity provides the counterforce).
                energy_for_c = torch.sigmoid(_rms_for_thrust)
                # Transient (hit) signal for the shore wall. Same
                # half-flux-half-onset proxy used by sequence losses, then
                # normalized to [0,1] so the Rust wall gating sees a
                # well-scaled h. Loud attacks open the wall (boundary
                # crossing becomes easy); quiet frames hold c inside.
                _flux = avg_features[:, 1]
                _onset = avg_features[:, 4]
                _hit_raw = 0.5 * _flux + 0.5 * _onset
                _hit_min = float(_hit_raw.min().item())
                _hit_max = float(_hit_raw.max().item())
                if _hit_max - _hit_min > 1e-8:
                    h_for_c = (_hit_raw - _hit_min) / (_hit_max - _hit_min)
                else:
                    h_for_c = torch.zeros_like(_hit_raw)
                # Supervise through the SAME controller the browser executes:
                # OrbitController with momentum + shore_bias ON (drag 0.90) —
                # the runtime enables these refinements for smooth audio-
                # driven motion that hugs the Shore. The forward simulation
                # uses the Rust-oracle path so the model is trained on the
                # exact physics the browser runs; parity is pinned by
                # preflight checks (e) and (e4).
                # Domain-randomized initial c per segment (orbit |c|~boundary or |c|~2).
                _n = s_target.shape[0]
                _dev = s_target.device
                _seg = segment_ids.reshape(-1)
                _uniq = torch.unique(_seg)
                _starts_re = {}
                _starts_im = {}
                for _sid in _uniq.tolist():
                    if torch.rand((), device=_dev).item() < 0.10:
                        _ang = (
                            torch.rand((), device=_dev).item() * 2 * 3.141592653589793
                        )
                        _r = 1.8 + torch.rand((), device=_dev).item() * 0.4
                        import math as _math

                        _starts_re[_sid] = _math.cos(_ang) * _r
                        _starts_im[_sid] = _math.sin(_ang) * _r
                    else:
                        _t = torch.rand((), device=_dev).item() * 2 * 3.141592653589793
                        import math as _math2

                        _mu = complex(_math2.cos(_t), _math2.sin(_t))
                        _cb = _mu * 0.5 - _mu * _mu * 0.25
                        _j = (torch.rand((), device=_dev).item() - 0.5) * 0.30
                        _starts_re[_sid] = _cb.real + _j * _mu.real * 0.5
                        _starts_im[_sid] = _cb.imag + _j * _mu.imag * 0.5
                _ic_re = torch.tensor(
                    [_starts_re[int(s)] for s in _seg.tolist()],
                    device=_dev,
                    dtype=torch.float32,
                )
                _ic_im = torch.tensor(
                    [_starts_im[int(s)] for s in _seg.tolist()],
                    device=_dev,
                    dtype=torch.float32,
                )
                _initial_c = torch.complex(_ic_re, _ic_im)
                # Forward simulation routes through the Rust contour step
                # (orbit_controller_oracle_sequence) so the trainer
                # supervises EXACTLY the physics the browser executes.
                # The integrator remains differentiable; only the
                # contour step is non-grad (PyO3 boundary).
                c_complex = orbit_controller_oracle_sequence(
                    s_target=s_target,
                    alpha=alpha,
                    omega=1.0,
                    band_gates=band_gates,
                    segment_ids=segment_ids,
                    dt=canonical_hop_dt(),
                    drag=0.90,
                    thrust=thrust_for_c,
                    initial_c=_initial_c,
                    energy=energy_for_c,
                    h=h_for_c,
                    level=0,
                    d_star=0.5,
                    max_step=0.05,
                )

                spectral_centroid = avg_features[:, 0]
                spectral_flux = avg_features[:, 1]
                onset_strength = avg_features[:, 4]
                spectral_rms = avg_features[:, 2]

                # Timbre ↔ color: brighter timbre → larger s (carrier radius).
                timbre_color_loss = self.correlation_loss(spectral_centroid, s_target)

                # Transient impact ↔ motion: flux → c-space step speed proxy.
                # alpha * omega_scale drives how fast/wild the orbit moves.
                motion_proxy = alpha.reshape(-1) * omega_scale.reshape(-1)
                transient_impact_loss = self.correlation_loss(
                    spectral_flux, motion_proxy
                )

                # Loudness ↔ boundary proximity: louder audio should push c
                # closer to the Mandelbrot boundary (more intricate visuals).
                # Analytic proximity: S field is 0 across the interior, so a
                # correlation against S would be degenerate exactly where c
                # parks.
                proximity = cardioid_proximity(c_complex)
                loudness_distance_loss = self.correlation_loss(-spectral_rms, proximity)

                timbre_color_loss = self._sanitize_scalar(timbre_color_loss)
                transient_impact_loss = self._sanitize_scalar(transient_impact_loss)
                loudness_distance_loss = self._sanitize_scalar(loudness_distance_loss)

                temporal_change_tensor = torch.zeros_like(spectral_flux)
            else:
                # ---- Legacy rendered-image supervision (slow path) ----
                c_values = []
                for i in range(batch_size):
                    state = OrbitState.new_with_seed(
                        lobe=1,
                        sub_lobe=0,
                        theta=float(i * 2 * np.pi / batch_size),
                        omega=float(
                            DEFAULT_BASE_OMEGA * omega_scale[i].detach().item()
                        ),
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

                try:
                    n_features_per_frame = (
                        self.feature_extractor.num_features_per_frame()
                    )
                except Exception:
                    n_features_per_frame = 6
                window_frames = features.shape[1] // n_features_per_frame
                features_reshaped = features.view(
                    batch_size, window_frames, n_features_per_frame
                )
                avg_features = features_reshaped.mean(dim=1)

                spectral_centroid = avg_features[:, 0]
                spectral_flux = avg_features[:, 1]
                onset_strength = avg_features[:, 4]
                spectral_rms = avg_features[:, 2]

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

                timbre_color_loss = self.correlation_loss(
                    spectral_centroid, color_hue_tensor
                )
                transient_impact_loss = self.correlation_loss(
                    spectral_flux, temporal_change_tensor
                )

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

                # Legacy path also supervises through OrbitController (momentum
                # ON) so both supervision paths match runtime physics.
                # Note: thrust/initial_c domain randomization only in the c-space path;
                # legacy path uses (0,0) starts and no thrust (rendered supervision is dominant).
                # The oracle forward keeps the legacy supervision consistent
                # with the browser physics even though this path's primary
                # signal is the rendered image loss.
                c_complex = orbit_controller_oracle_sequence(
                    s_target=s_target,
                    alpha=alpha,
                    omega=1.0,
                    band_gates=band_gates,
                    segment_ids=segment_ids,
                    dt=canonical_hop_dt(),
                    drag=0.90,
                )

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
                # Scheduled sampling: ramp reliance on the model's own carried
                # state up over epochs (teacher forcing decays accordingly).
                ss_prob = self._scheduled_sampling_prob(epoch)
                effective_tf = max(0.0, 1.0 - ss_prob)
                rollout_loss = self._compute_rollout_loss(
                    controls,
                    spectral_flux,
                    onset_strength,
                    segment_ids,
                    teacher_forcing_override=effective_tf,
                )

            # Coverage/diversity: penalize c(t) clustering in c-space.
            coverage_loss = self._coverage_loss(c_complex, segment_ids)
            coverage_loss = self._sanitize_scalar(coverage_loss)

            # Anti-dwell: scale-aware penalty on c(t) staying put, gated by
            # audio energy — silence decays the requirement so friction can
            # bring c to rest; loud audio demands movement.
            anti_dwell_loss = self._anti_dwell_loss(
                c_complex, segment_ids, audio_energy=spectral_rms
            )
            anti_dwell_loss = self._sanitize_scalar(anti_dwell_loss)

            # Zone: keep c in the visually interesting band near the boundary.
            zone_loss = self._zone_loss(c_complex)
            zone_loss = self._sanitize_scalar(zone_loss)

            # J(c) frame-to-frame stability: quiet parts nearly still,
            # loud parts drift a little, transients exempt (full-frame
            # changes land there by design).
            julia_stability_loss = self._julia_stability_loss(
                c_complex, segment_ids, spectral_rms, onset_strength
            )
            julia_stability_loss = self._sanitize_scalar(julia_stability_loss)

            # Song identity: different songs claim different home regions,
            # consistently within each song. Directly answers "why does it
            # always converge to the same region".
            song_identity_loss = self._song_identity_loss(c_complex, segment_ids)
            song_identity_loss = self._sanitize_scalar(song_identity_loss)

            # Region dwell: c must not occupy the same J(c)-region for the
            # whole dwell window. Jitter in one area for a section is fine;
            # lingering all window is not. Hits reset the window.
            region_dwell_loss = self._region_dwell_loss(
                c_complex, segment_ids, onset_strength
            )
            region_dwell_loss = self._sanitize_scalar(region_dwell_loss)

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
                + self.coverage_weight * coverage_loss
                + self.anti_dwell_weight * anti_dwell_loss
                + self.zone_weight * zone_loss
                + self.julia_stability_weight * julia_stability_loss
                + self.song_identity_weight * song_identity_loss
                + self.region_dwell_weight * region_dwell_loss
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
            total_coverage += coverage_loss.item()
            total_anti_dwell += anti_dwell_loss.item()
            total_julia_stability += julia_stability_loss.item()
            total_song_identity += song_identity_loss.item()
            total_region_dwell += region_dwell_loss.item()
            total_zone += zone_loss.item()
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
            "coverage_loss": total_coverage / n_batches,
            "anti_dwell_loss": total_anti_dwell / n_batches,
            "julia_stability_loss": total_julia_stability / n_batches,
            "song_identity_loss": total_song_identity / n_batches,
            "region_dwell_loss": total_region_dwell / n_batches,
            "zone_loss": total_zone / n_batches,
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

        # Remember the dataset layout for the per-epoch c-trace plots.
        self._trace_dataset_files = list(dataset.audio_files)
        self._trace_window_frames = dataset.window_frames

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

        if self.clip_length > 1:
            # Clip-based training: batches are contiguous clips of clip_length
            # windows drawn from the same file, shuffled per epoch. This gives
            # the sequence losses real temporal structure instead of relying
            # on incidental batch adjacency.
            dataloader = self._make_clip_dataloader(
                all_features_tensor, segment_id_tensor, batch_size, shuffle_clips=True
            )
        else:
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
                f"Loss: {avg_losses['loss']:.4f}, "
                f"Control: {avg_losses['control_loss']:.4f}, "
                f"AlignProxy: {avg_losses['alignment_proxy']:.4f}"
            )

            # c-trace diagnostic: plot the model's c(t) path per song over
            # the Mandelbrot set. Shows which regions are explored and —
            # critically — whether/where c gets stuck.
            if save_dir:
                self._plot_c_traces(save_dir, epoch + 1)

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

    def _plot_c_traces(self, save_dir: str, epoch: int) -> Optional[str]:
        """Generate the per-song c-trace plot for the current model state."""
        try:
            dataset_files = self._trace_dataset_files
            if not dataset_files:
                return None
            traces = collect_c_traces(
                self.model,
                self.feature_extractor,
                dataset_files,
                window_frames=self._trace_window_frames,
            )
            if not traces:
                return None
            out = plot_c_traces(
                traces,
                Path(save_dir) / "c_traces" / f"c_trace_epoch_{epoch:04d}.png",
                title=f"c(t) trajectories — epoch {epoch}",
            )
            return str(out) if out else None
        except Exception as exc:  # noqa: BLE001 - diagnostics must never kill training
            logger.warning("c-trace plot failed (non-fatal): %s", exc)
            return None

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
