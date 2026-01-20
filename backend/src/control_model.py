"""
Control signal model for orbit-based Julia parameter synthesis.

Predicts control signals (s, alpha, ω, band_gates) instead of raw c(t).
The orbit synthesizer uses these signals to generate deterministic c(t).
"""

import torch
import torch.nn as nn


class AudioToControlModel(nn.Module):
    """
    Neural network that predicts orbit control signals from audio features.

    Outputs:
        - s_target: Radius scaling factor [0.2, 3.0]
        - alpha: Residual amplitude [0, 1]
        - omega_scale: Angular velocity scale [0.1, 5.0]
        - band_gates: Per-band residual gates [0, 1]^k (k=6 default)

    The lobe/sub_lobe are controlled by section detection (not predicted per-frame).
    """

    def __init__(
        self,
        window_frames: int = 10,
        n_features_per_frame: int = 6,
        hidden_dims: list[int] = [128, 256, 128],
        k_bands: int = 6,
        dropout: float = 0.2,
        include_delta: bool = False,
        include_delta_delta: bool = False,
    ):
        """
        Initialize control signal model.

        Args:
            window_frames: Number of time frames
            n_features_per_frame: Number of features per frame (6 base, +6 delta, +6 delta-delta)
            hidden_dims: List of hidden layer dimensions
            k_bands: Number of band gates (residual epicycles)
            dropout: Dropout rate
            include_delta: Include delta (velocity) features
            include_delta_delta: Include delta-delta (acceleration) features
        """
        super().__init__()

        self.window_frames = window_frames
        self.n_features_per_frame = n_features_per_frame
        self.k_bands = k_bands
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta

        # Calculate input dimension based on feature configuration
        features_multiplier = 1
        if include_delta:
            features_multiplier += 1
        if include_delta_delta:
            features_multiplier += 1

        self.input_dim = n_features_per_frame * features_multiplier * window_frames

        # Output dimension: s_target(1) + alpha(1) + omega_scale(1) + band_gates(k_bands)
        self.output_dim = 3 + k_bands

        # Build encoder layers
        encoder_layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Control signal heads
        self.s_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Alpha in [0, 1]
        )

        self.omega_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.band_gates_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, k_bands),
            nn.Sigmoid(),  # Gates in [0, 1]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting control signals.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Control signals of shape (batch_size, output_dim)
            Format: [s_target, alpha, omega_scale, band_gate_0, ..., band_gate_k-1]
        """
        # Validate input shape
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {x.shape[1]}. "
                f"Config: window_frames={self.window_frames}, "
                f"n_features_per_frame={self.n_features_per_frame}, "
                f"include_delta={self.include_delta}, "
                f"include_delta_delta={self.include_delta_delta}"
            )

        # Encode features
        encoded = self.encoder(x)

        # Predict control signals
        s_raw = self.s_head(encoded)  # (batch_size, 1)
        alpha = self.alpha_head(encoded)  # (batch_size, 1)
        omega_raw = self.omega_head(encoded)  # (batch_size, 1)
        band_gates = self.band_gates_head(encoded)  # (batch_size, k_bands)

        # Apply activation functions to constrain outputs
        # s_target: map to [0.2, 3.0] using sigmoid + scaling
        s_target = 0.2 + 2.8 * torch.sigmoid(s_raw)  # [0.2, 3.0]

        # omega_scale: map to [0.1, 5.0] using softplus
        omega_scale = 0.1 + torch.nn.functional.softplus(omega_raw) * 0.5  # ~[0.1, 5.0]
        omega_scale = torch.clamp(omega_scale, 0.1, 5.0)

        # Concatenate all control signals
        output = torch.cat([s_target, alpha, omega_scale, band_gates], dim=1)

        return output

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        ranges = {
            "s_target": (0.2, 3.0),
            "alpha": (0.0, 1.0),
            "omega_scale": (0.1, 5.0),
        }
        for k in range(self.k_bands):
            ranges[f"band_gate_{k}"] = (0.0, 1.0)
        return ranges

    def parse_output(self, output: torch.Tensor) -> dict:
        """
        Parse model output into named control signals.

        Args:
            output: Model output tensor (batch_size, output_dim)

        Returns:
            Dictionary with keys: s_target, alpha, omega_scale, band_gates
        """
        return {
            "s_target": output[:, 0],
            "alpha": output[:, 1],
            "omega_scale": output[:, 2],
            "band_gates": output[:, 3:],
        }
