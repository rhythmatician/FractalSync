"""
PyTorch model architecture for mapping audio features to visual parameters.
"""

import torch
import torch.nn as nn


class AudioToVisualModel(nn.Module):
    """
    Neural network mapping audio features to visual parameters.

    Architecture: CNN encoder → MLP decoder
    Output: 7 parameters (Julia seed real/imag, color hue/sat/bright, zoom, speed)
    """

    def __init__(
        self,
        input_dim: int = 60,  # 6 features * 10 frames
        hidden_dims: list[int] = [128, 256, 128],
        output_dim: int = 7,
        dropout: float = 0.2,
    ):
        """
        Initialize model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output parameter dimension (default 7)
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim

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

        # Decoder to output parameters
        self.decoder = nn.Sequential(
            nn.Linear(prev_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
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
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Visual parameters of shape (batch_size, output_dim)
        """
        # Validate input shape
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[1]}")

        # Encode features
        encoded = self.encoder(x)

        # Decode to visual parameters
        params = self.decoder(encoded)

        # Apply activation functions to constrain outputs
        # Julia seed: real and imag components (unbounded, but we'll clip in training)
        # Color: hue [0, 1], saturation [0, 1], brightness [0, 1]
        # Zoom: positive (use exp or sigmoid)
        # Speed: positive (use exp or sigmoid)

        # Split parameters
        julia_real = params[:, 0]
        julia_imag = params[:, 1]
        color_hue = torch.sigmoid(params[:, 2])
        color_sat = torch.sigmoid(params[:, 3])
        color_bright = torch.sigmoid(params[:, 4])
        zoom = torch.exp(params[:, 5])  # Ensure positive
        speed = torch.sigmoid(params[:, 6])  # [0, 1]

        # Concatenate back
        output = torch.stack(
            [julia_real, julia_imag, color_hue, color_sat, color_bright, zoom, speed],
            dim=1,
        )

        return output

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {
            "julia_real": (-2.0, 2.0),
            "julia_imag": (-2.0, 2.0),
            "color_hue": (0.0, 1.0),
            "color_sat": (0.0, 1.0),
            "color_bright": (0.0, 1.0),
            "zoom": (0.1, 10.0),
            "speed": (0.0, 1.0),
        }


class TransformerAudioToVisualModel(nn.Module):
    """
    Alternative architecture using Transformer encoder.
    Better for capturing temporal dependencies in audio features.
    """

    def __init__(
        self,
        input_dim: int = 6,  # Per-frame features
        num_frames: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        output_dim: int = 7,
        dropout: float = 0.2,
    ):
        """
        Initialize Transformer-based model.

        Args:
            input_dim: Feature dimension per frame
            num_frames: Number of frames in sequence
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output parameter dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frames = num_frames
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_frames, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling and decoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, num_frames, input_dim)

        Returns:
            Visual parameters of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (batch, frames, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoding
        encoded = self.transformer(x)  # (batch, frames, d_model)

        # Global pooling (average over frames)
        pooled = encoded.mean(dim=1)  # (batch, d_model)

        # Decode to parameters
        params = self.decoder(pooled)

        # Apply same activations as CNN model
        julia_real = params[:, 0]
        julia_imag = params[:, 1]
        color_hue = torch.sigmoid(params[:, 2])
        color_sat = torch.sigmoid(params[:, 3])
        color_bright = torch.sigmoid(params[:, 4])
        zoom = torch.exp(params[:, 5])
        speed = torch.sigmoid(params[:, 6])

        output = torch.stack(
            [julia_real, julia_imag, color_hue, color_sat, color_bright, zoom, speed],
            dim=1,
        )

        return output
