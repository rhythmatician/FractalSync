"""
Control signal model for height-field Julia parameter synthesis.

Predicts a model step Δc plus height-control parameters used by the
height-field controller to stay on level sets of f(c) = log|z_N(c)|.
"""

import torch
import torch.nn as nn


class AudioToControlModel(nn.Module):
    """
    Neural network that predicts height-field control signals from audio features.

    Outputs:
        - delta_c_real: Proposed Δc real component
        - delta_c_imag: Proposed Δc imag component
        - target_height: Desired height f(c)
        - normal_risk: How much normal motion to allow [0, 1]
    """

    def __init__(
        self,
        window_frames: int = 10,
        n_features_per_frame: int = 6,
        hidden_dims: list[int] = [128, 256, 128],
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
            dropout: Dropout rate
            include_delta: Include delta (velocity) features
            include_delta_delta: Include delta-delta (acceleration) features
        """
        super().__init__()

        self.window_frames = window_frames
        self.n_features_per_frame = n_features_per_frame
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta

        # Calculate input dimension based on feature configuration
        features_multiplier = 1
        if include_delta:
            features_multiplier += 1
        if include_delta_delta:
            features_multiplier += 1

        self.input_dim = n_features_per_frame * features_multiplier * window_frames

        # Output dimension: delta_c(2) + target_height(1) + normal_risk(1)
        self.output_dim = 4

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
        self.delta_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.height_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.normal_risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
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
            Format: [delta_c_real, delta_c_imag, target_height, normal_risk]
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
        delta_raw = self.delta_head(encoded)  # (batch_size, 2)
        height_raw = self.height_head(encoded)  # (batch_size, 1)
        normal_risk = self.normal_risk_head(encoded)  # (batch_size, 1)

        # Apply activation functions to constrain outputs
        delta_scale = 0.02
        delta_c = torch.tanh(delta_raw) * delta_scale  # [-0.02, 0.02]

        # target_height: keep within a compact range
        target_height = torch.tanh(height_raw) * 2.0  # ~[-2, 2]

        # Concatenate all control signals
        output = torch.cat([delta_c, target_height, normal_risk], dim=1)

        return output

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        ranges = {
            "delta_c_real": (-0.02, 0.02),
            "delta_c_imag": (-0.02, 0.02),
            "target_height": (-2.0, 2.0),
            "normal_risk": (0.0, 1.0),
        }
        return ranges

    def parse_output(self, output: torch.Tensor) -> dict:
        """
        Parse model output into named control signals.

        Args:
            output: Model output tensor (batch_size, output_dim)

        Returns:
            Dictionary with keys: delta_c_real, delta_c_imag, target_height, normal_risk
        """
        return {
            "delta_c_real": output[:, 0],
            "delta_c_imag": output[:, 1],
            "target_height": output[:, 2],
            "normal_risk": output[:, 3],
        }
