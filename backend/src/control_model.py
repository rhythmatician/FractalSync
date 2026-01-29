"""
Control signal model for step-based Julia parameter synthesis.

Predicts a complex delta step Δc = (dx, dy) instead of orbit parameters.
The runtime step controller applies contextual throttling and safety
redirection before updating c(t).
"""

import torch
import torch.nn as nn


class AudioToControlModel(nn.Module):
    """
    Neural network that predicts Δc steps from audio + minimap context features.

    Outputs:
        - delta_real
        - delta_imag
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
        context_dim: int = 265,
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
        self.context_dim = context_dim
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta

        # Calculate input dimension based on feature configuration
        features_multiplier = 1
        if include_delta:
            features_multiplier += 1
        if include_delta_delta:
            features_multiplier += 1

        self.base_input_dim = n_features_per_frame * features_multiplier * window_frames
        self.input_dim = self.base_input_dim + context_dim

        # Output dimension: delta_real, delta_imag
        self.output_dim = 2

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

        self.delta_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
            Format: [delta_real, delta_imag]
        """
        # Compatibility shim: pad or truncate to expected input dimension.
        if x.shape[1] != self.input_dim:
            if x.shape[1] < self.input_dim:
                pad = torch.zeros(
                    (x.shape[0], self.input_dim - x.shape[1]), device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, : self.input_dim]

        # Encode features
        encoded = self.encoder(x)

        return self.delta_head(encoded)

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {
            "delta_real": (-0.02, 0.02),
            "delta_imag": (-0.02, 0.02),
        }

    def parse_output(self, output: torch.Tensor) -> dict:
        """
        Parse model output into named control signals.

        Args:
            output: Model output tensor (batch_size, output_dim)

        Returns:
            Dictionary with keys: delta_real, delta_imag
        """
        return {
            "delta_real": output[:, 0],
            "delta_imag": output[:, 1],
        }
