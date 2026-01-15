"""
Physics-based model for Julia parameter prediction.

Predicts velocity of the Julia parameter c, treating it as a physical object
whose speed is influenced by audio loudness (RMS energy).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PhysicsAudioToVisualModel(nn.Module):
    """
    Neural network that predicts velocity of Julia parameter c and other visual parameters.

    The Julia parameter c is treated as a physical object in the complex plane:
    - Model predicts velocity (dc/dt) instead of position
    - Speed magnitude is influenced by audio loudness (RMS energy)
    - Direction is learned from audio features
    - Position is integrated from velocity with optional damping
    """

    def __init__(
        self,
        window_frames: int = 10,
        hidden_dims: list[int] = [128, 256, 128],
        output_dim: int = 9,  # 2 (c velocity) + 7 (other params)
        dropout: float = 0.2,
        predict_velocity: bool = True,
        damping_factor: float = 0.95,
        speed_scale: float = 0.1,
    ):
        """
        Initialize physics-based model.

        Args:
            window_frames: Number of time frames (input_dim = 6 * window_frames)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (9 for velocity mode: 2 velocity + 7 other params)
            dropout: Dropout rate
            predict_velocity: If True, predict velocity; if False, predict position directly
            damping_factor: Velocity damping factor (0-1, higher = less damping)
            speed_scale: Scaling factor for velocity magnitude
        """
        super().__init__()

        self.window_frames = window_frames
        self.input_dim = 6 * window_frames  # 6 features per frame
        self.output_dim = output_dim
        self.predict_velocity = predict_velocity
        self.damping_factor = damping_factor
        self.speed_scale = speed_scale

        # State tracking (position and velocity)
        self.register_buffer("current_position", torch.zeros(2))  # [c_real, c_imag]
        self.register_buffer("current_velocity", torch.zeros(2))  # [v_real, v_imag]

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

        # Velocity predictor (for Julia parameter c)
        self.velocity_predictor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [v_real, v_imag]
        )

        # Other parameters decoder (color, zoom, speed)
        self.params_decoder = nn.Sequential(
            nn.Linear(prev_dim, 64), nn.ReLU(), nn.Linear(64, 5)  # color (3) + zoom + speed
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

    def forward(
        self, x: torch.Tensor, audio_rms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with physics-based velocity prediction.

        Args:
            x: Input features of shape (batch_size, input_dim)
            audio_rms: Optional RMS energy values (batch_size,) to modulate speed

        Returns:
            Visual parameters of shape (batch_size, output_dim)
            If predict_velocity=True: [v_real, v_imag, c_real, c_imag, hue, sat, bright, zoom, speed]
            If predict_velocity=False: [c_real, c_imag, hue, sat, bright, zoom, speed]
        """
        # Validate input shape
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[1]}")

        batch_size = x.shape[0]

        # Extract RMS energy if not provided (from input features)
        if audio_rms is None:
            # RMS is the 3rd feature (index 2) in each frame
            n_features_per_frame = 6
            features_reshaped = x.view(batch_size, self.window_frames, n_features_per_frame)
            audio_rms = features_reshaped[:, :, 2].mean(dim=1)  # Average RMS over window

        # Encode features
        encoded = self.encoder(x)

        # Predict velocity for Julia parameter
        velocity_raw = self.velocity_predictor(encoded)  # (batch_size, 2)

        # Modulate velocity magnitude by audio loudness
        # Higher RMS → faster movement
        velocity_magnitude = torch.norm(velocity_raw, dim=1, keepdim=True)  # (batch_size, 1)
        velocity_direction = velocity_raw / (velocity_magnitude + 1e-8)  # Normalized direction

        # Scale magnitude by RMS and speed_scale
        scaled_magnitude = velocity_magnitude * audio_rms.unsqueeze(1) * self.speed_scale
        velocity = velocity_direction * scaled_magnitude  # (batch_size, 2)

        # Predict other parameters
        other_params = self.params_decoder(encoded)  # (batch_size, 5)

        # Apply activation functions to constrain outputs
        color_hue = torch.sigmoid(other_params[:, 0])
        color_sat = torch.sigmoid(other_params[:, 1])
        color_bright = torch.sigmoid(other_params[:, 2])
        zoom = torch.exp(other_params[:, 3])  # Ensure positive
        speed = torch.sigmoid(other_params[:, 4])  # [0, 1]

        if self.predict_velocity:
            # Return velocity + integrated position
            # Update internal state (for inference mode)
            if not self.training and batch_size == 1:
                # Single-sample inference: update state
                self.current_velocity = (
                    self.current_velocity * self.damping_factor + velocity[0].detach()
                )
                self.current_position = self.current_position + self.current_velocity

                # Constrain position to |c| < 2 (visually interesting Julia sets)
                position_magnitude = torch.norm(self.current_position)
                if position_magnitude > 2.0:
                    self.current_position = self.current_position / position_magnitude * 2.0

                c_real = self.current_position[0]
                c_imag = self.current_position[1]
            else:
                # Training mode: use batch positions (can be provided externally or initialized)
                # For now, just constrain velocity and return zeros for position
                # (position integration happens in trainer)
                c_real = torch.zeros(batch_size, device=x.device)
                c_imag = torch.zeros(batch_size, device=x.device)

            # Return velocity and position
            output = torch.stack(
                [
                    velocity[:, 0],  # v_real
                    velocity[:, 1],  # v_imag
                    c_real,  # c_real (integrated position)
                    c_imag,  # c_imag (integrated position)
                    color_hue,
                    color_sat,
                    color_bright,
                    zoom,
                    speed,
                ],
                dim=1,
            )
        else:
            # Direct position prediction mode (fallback)
            julia_real = 2.0 * torch.tanh(velocity_raw[:, 0])
            julia_imag = 2.0 * torch.tanh(velocity_raw[:, 1])

            output = torch.stack(
                [julia_real, julia_imag, color_hue, color_sat, color_bright, zoom, speed],
                dim=1,
            )

        return output

    def reset_state(self, position: Optional[Tuple[float, float]] = None):
        """
        Reset internal position and velocity state.

        Args:
            position: Optional initial position (c_real, c_imag). Defaults to (0, 0).
        """
        if position is None:
            position = (0.0, 0.0)

        self.current_position = torch.tensor(position, dtype=torch.float32)
        self.current_velocity = torch.zeros(2, dtype=torch.float32)

    def integrate_velocity(
        self,
        velocity: torch.Tensor,
        position: torch.Tensor,
        prev_velocity: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate velocity to get new position (for training).

        Args:
            velocity: Current velocity (batch_size, 2)
            position: Current position (batch_size, 2)
            prev_velocity: Previous velocity for damping (batch_size, 2)
            dt: Time step

        Returns:
            Tuple of (new_position, new_velocity) both (batch_size, 2)
        """
        # Apply damping if previous velocity provided
        if prev_velocity is not None:
            # Handle batch size mismatch (last batch may be smaller)
            batch_size = velocity.size(0)
            if prev_velocity.size(0) != batch_size:
                # Trim or pad prev_velocity to match current batch size
                prev_velocity = prev_velocity[:batch_size]
            
            damped_velocity = prev_velocity * self.damping_factor + velocity * (
                1.0 - self.damping_factor
            )
        else:
            damped_velocity = velocity

        # Integrate position
        new_position = position + damped_velocity * dt

        # Constrain to |c| < 2 (avoid in-place operations for autograd)
        position_magnitude = torch.norm(new_position, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Scale down any positions that exceed magnitude 2
        scale_factor = torch.where(
            position_magnitude > 2.0,
            2.0 / position_magnitude,
            torch.ones_like(position_magnitude)
        )
        new_position = new_position * scale_factor

        return new_position, damped_velocity

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        if self.predict_velocity:
            return {
                "v_real": (-0.5, 0.5),
                "v_imag": (-0.5, 0.5),
                "julia_real": (-2.0, 2.0),
                "julia_imag": (-2.0, 2.0),
                "color_hue": (0.0, 1.0),
                "color_sat": (0.0, 1.0),
                "color_bright": (0.0, 1.0),
                "zoom": (0.1, 10.0),
                "speed": (0.0, 1.0),
            }
        else:
            return {
                "julia_real": (-2.0, 2.0),
                "julia_imag": (-2.0, 2.0),
                "color_hue": (0.0, 1.0),
                "color_sat": (0.0, 1.0),
                "color_bright": (0.0, 1.0),
                "zoom": (0.1, 10.0),
                "speed": (0.0, 1.0),
            }
