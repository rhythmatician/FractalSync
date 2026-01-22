"""
Advanced loss components for emotional coherence and variety in training.

These losses help the model:
1. Stay visually interesting during high-intensity audio (membership proximity)
2. Correlate visual detail with audio timbre (edge density correlation)
3. Explore different lobes across the song (lobe variety)
4. Avoid staying in small neighborhoods (neighborhood penalty)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MembershipProximityLoss(nn.Module):
    """
    Penalizes sparse Julia sets during high-intensity audio.
    
    Uses escape-time of orbit of 0 under f_c(z) = z^2 + c to approximate
    Mandelbrot set membership. Points closer to the set produce more
    visually interesting, connected Julia sets.
    
    Loss is weighted by audio intensity - during loud/intense moments,
    we strongly penalize c values that are far from the Mandelbrot set.
    """

    def __init__(
        self,
        target_membership: float = 0.75,
        max_iter: int = 50,
        escape_radius: float = 2.0,
        weight: float = 1.0,
    ):
        """
        Initialize membership proximity loss.

        Args:
            target_membership: Target membership ratio (0-1). Higher means closer to M.
            max_iter: Maximum iterations for escape-time computation
            escape_radius: Escape radius threshold
            weight: Loss weight multiplier
        """
        super().__init__()
        self.target_membership = target_membership
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        self.weight = weight

    def compute_membership_proxy(
        self, c_real: torch.Tensor, c_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute membership proxy for complex values c.
        
        Returns ratio of iterations before escape (0-1).
        1.0 means didn't escape (inside M), 0.0 means escaped immediately.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)

        Returns:
            Membership proxy values (batch_size,) in [0, 1]
        """
        batch_size = c_real.shape[0]
        device = c_real.device

        # Initialize z = 0 for all samples
        z_real = torch.zeros_like(c_real)
        z_imag = torch.zeros_like(c_imag)

        # Track iteration count for each sample
        iter_count = torch.zeros(batch_size, device=device)
        not_escaped = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Iterate z_{n+1} = z_n^2 + c
        for i in range(self.max_iter):
            # z^2 = (z_real + i*z_imag)^2 = z_real^2 - z_imag^2 + 2*i*z_real*z_imag
            z_real_new = z_real**2 - z_imag**2 + c_real
            z_imag_new = 2 * z_real * z_imag + c_imag

            z_real = z_real_new
            z_imag = z_imag_new

            # Check which samples escaped
            magnitude_sq = z_real**2 + z_imag**2
            escaped = magnitude_sq > self.escape_radius**2

            # Update iteration count for samples that haven't escaped yet
            iter_count = torch.where(not_escaped & ~escaped, iter_count + 1, iter_count)

            # Update not_escaped mask
            not_escaped = not_escaped & ~escaped

            # Early exit if all escaped
            if not not_escaped.any():
                break

        # Normalize to [0, 1]
        membership = iter_count / self.max_iter

        return membership

    def forward(
        self,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        audio_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute membership proximity loss.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)
            audio_intensity: Audio intensity values (batch_size,) in [0, 1]

        Returns:
            Scalar loss value
        """
        # Ensure audio_intensity is in valid range [0, 1]
        audio_intensity = torch.clamp(audio_intensity, 0.0, 1.0)
        
        # Compute membership proxy
        membership = self.compute_membership_proxy(c_real, c_imag)

        # Penalize when membership < target, weighted by intensity
        shortfall = torch.clamp(self.target_membership - membership, min=0.0)
        
        # Weight by audio intensity - higher intensity = stronger penalty
        weighted_shortfall = audio_intensity * shortfall

        # MSE-style loss
        loss = torch.mean(weighted_shortfall**2)

        return self.weight * loss


class EdgeDensityCorrelationLoss(nn.Module):
    """
    Correlates visual edge density with audio spectral brightness.
    
    Encourages model to produce:
    - Jagged, detailed fractals during bright/high-frequency audio
    - Smooth, blobby fractals during dark/low-frequency audio
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize edge density correlation loss.

        Args:
            weight: Loss weight multiplier
        """
        super().__init__()
        self.weight = weight

    def forward(
        self, edge_density: torch.Tensor, spectral_centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative correlation between edge density and spectral centroid.

        Args:
            edge_density: Edge density values (batch_size,)
            spectral_centroid: Spectral centroid values (batch_size,)

        Returns:
            Scalar loss value (negative correlation)
        """
        # Flatten tensors
        x = edge_density.flatten()
        y = spectral_centroid.flatten()

        # Center the values
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)

        # Compute correlation
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))

        correlation = numerator / (denominator + 1e-8)

        # Return negative correlation (we want to maximize positive correlation)
        return -self.weight * correlation


class LobeVarietyLoss(nn.Module):
    """
    Encourages exploration of different lobes across a song.
    
    Tracks the distribution of c values in parameter space and penalizes
    concentrating in a small set of lobes. Uses a history buffer to
    measure variety over recent time windows.
    """

    def __init__(
        self,
        history_size: int = 100,
        n_clusters: int = 5,
        weight: float = 1.0,
        target_variety_scale: float = 0.3,
    ):
        """
        Initialize lobe variety loss.

        Args:
            history_size: Number of recent c values to track
            n_clusters: Expected number of different regions/lobes
            weight: Loss weight multiplier
            target_variety_scale: Scaling factor for target variety calculation.
                This value (default 0.3) represents the expected std per cluster
                in c-space, where the Mandelbrot set has diameter ~3. The target
                variety is computed as: target_variety_scale * sqrt(n_clusters).
        """
        super().__init__()
        self.history_size = history_size
        self.n_clusters = n_clusters
        self.weight = weight
        self.target_variety_scale = target_variety_scale
        self.c_history: List[Tuple[float, float]] = []

    def update_history(self, c_real: torch.Tensor, c_imag: torch.Tensor):
        """
        Update history buffer with new c values.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)
        """
        # Detach and convert to CPU for history tracking
        c_real_cpu = c_real.detach().cpu()
        c_imag_cpu = c_imag.detach().cpu()

        for i in range(len(c_real_cpu)):
            self.c_history.append((float(c_real_cpu[i]), float(c_imag_cpu[i])))

        # Trim to history size
        if len(self.c_history) > self.history_size:
            self.c_history = self.c_history[-self.history_size :]

    def compute_variety_score(self) -> float:
        """
        Compute variety score from history.
        
        Uses standard deviation in c-space as a proxy for exploration.
        Higher std = more exploration.

        Returns:
            Variety score (higher is more diverse)
        """
        if len(self.c_history) < 2:
            return 0.0

        c_real_vals = [c[0] for c in self.c_history]
        c_imag_vals = [c[1] for c in self.c_history]

        # Compute standard deviation in both dimensions
        std_real = float(np.std(c_real_vals))
        std_imag = float(np.std(c_imag_vals))

        # Combined variety score (Euclidean norm of standard deviations)
        variety = float(np.linalg.norm([std_real, std_imag]))

        return variety

    def forward(
        self, c_real: torch.Tensor, c_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute lobe variety loss.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)

        Returns:
            Scalar loss value (penalizes low variety)
        """
        # Update history
        self.update_history(c_real, c_imag)

        # Compute variety score
        variety = self.compute_variety_score()

        # Target variety based on expected n_clusters and typical Mandelbrot extent
        target_variety = self.target_variety_scale * np.sqrt(self.n_clusters)

        # Penalize when variety is below target
        shortfall = max(0.0, target_variety - variety)

        # Convert to tensor loss
        loss = torch.tensor(shortfall**2, device=c_real.device, dtype=torch.float32)

        return self.weight * loss


class NeighborhoodPenaltyLoss(nn.Module):
    """
    Penalizes staying in a small neighborhood of c-space for extended periods.
    
    Tracks recent c values and penalizes when they cluster too tightly,
    encouraging the model to move around and explore different regions.
    """

    def __init__(
        self,
        window_size: int = 32,
        min_radius: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Initialize neighborhood penalty loss.

        Args:
            window_size: Number of recent frames to consider
            min_radius: Minimum acceptable radius of recent c values
            weight: Loss weight multiplier
        """
        super().__init__()
        self.window_size = window_size
        self.min_radius = min_radius
        self.weight = weight
        self.recent_c: List[Tuple[float, float]] = []

    def update_recent(self, c_real: torch.Tensor, c_imag: torch.Tensor):
        """
        Update recent c values buffer.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)
        """
        # Detach and convert to CPU
        c_real_cpu = c_real.detach().cpu()
        c_imag_cpu = c_imag.detach().cpu()

        for i in range(len(c_real_cpu)):
            self.recent_c.append((float(c_real_cpu[i]), float(c_imag_cpu[i])))

        # Trim to window size
        if len(self.recent_c) > self.window_size:
            self.recent_c = self.recent_c[-self.window_size :]

    def compute_neighborhood_radius(self) -> float:
        """
        Compute radius of recent c values from their centroid.

        Returns:
            Average radius from centroid
        """
        if len(self.recent_c) < 2:
            return float("inf")  # Not enough samples to penalize

        c_real_vals = np.array([c[0] for c in self.recent_c])
        c_imag_vals = np.array([c[1] for c in self.recent_c])

        # Compute centroid
        centroid_real = np.mean(c_real_vals)
        centroid_imag = np.mean(c_imag_vals)

        # Compute distances from centroid
        distances = np.sqrt(
            (c_real_vals - centroid_real) ** 2 + (c_imag_vals - centroid_imag) ** 2
        )

        # Average distance (radius)
        radius = float(np.mean(distances))

        return radius

    def forward(
        self, c_real: torch.Tensor, c_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute neighborhood penalty loss.

        Args:
            c_real: Real parts of c values (batch_size,)
            c_imag: Imaginary parts of c values (batch_size,)

        Returns:
            Scalar loss value (penalizes small neighborhoods)
        """
        # Update recent c values
        self.update_recent(c_real, c_imag)

        # Compute neighborhood radius
        radius = self.compute_neighborhood_radius()

        # Penalize when radius is below minimum
        if radius < self.min_radius:
            shortfall = self.min_radius - radius
            loss = torch.tensor(shortfall**2, device=c_real.device, dtype=torch.float32)
        else:
            loss = torch.zeros(1, device=c_real.device, dtype=torch.float32)

        return self.weight * loss
