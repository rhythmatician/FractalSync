"""
Control signal model for orbit-based Julia parameter synthesis.

Predicts control signals (s, alpha, ω, band_gates) instead of raw c(t).
The orbit synthesizer uses these signals to generate deterministic c(t).

Supports two encoder modes:
- ``feedforward`` (default, legacy): MLP over a flattened feature window.
  Fully backward compatible with existing checkpoints and ONNX export.
- ``recurrent``: per-frame features pass through a shared frame encoder,
  then a GRU integrates temporal context across the window. This gives the
  model real state so it can react to feature *dynamics* (rising vs falling
  flux), not just window averages.
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
        recurrent: bool = False,
        gru_hidden_dim: int = 128,
        controls_version: str = "orbit_control",
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
            recurrent: Use a GRU over per-frame features instead of a flat MLP
            gru_hidden_dim: Hidden size of the GRU when ``recurrent`` is True
        """
        super().__init__()

        self.window_frames = window_frames
        self.n_features_per_frame = n_features_per_frame
        self.k_bands = k_bands
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta
        self.recurrent = recurrent

        # Calculate input dimension based on feature configuration
        features_multiplier = 1
        if include_delta:
            features_multiplier += 1
        if include_delta_delta:
            features_multiplier += 1

        self.features_per_frame = n_features_per_frame * features_multiplier
        self.input_dim = self.features_per_frame * window_frames

        self.controls_version = controls_version
        # Output dimension: Controls v2 (13) vs legacy orbit_control (3 + k_bands)
        if controls_version == "controls/2":
            # Canonical 13-channel: directionX,Y,throttle,brake,grip,impulse + 7 Julia deltas; Rust owns order/ranges
            try:
                import runtime_core
                self.output_dim = len(runtime_core.ControlsV2.model_output_order())
            except Exception:
                self.output_dim = 13
        else:
            self.output_dim = 3 + k_bands

        if recurrent:
            # Shared per-frame encoder → GRU → heads.
            frame_encoder_layers = []
            prev_frame_dim = self.features_per_frame
            for hidden_dim in hidden_dims[:2]:
                frame_encoder_layers.extend(
                    [
                        nn.Linear(prev_frame_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_frame_dim = hidden_dim
            self.frame_encoder = nn.Sequential(*frame_encoder_layers)
            self.gru = nn.GRU(
                input_size=prev_frame_dim,
                hidden_size=gru_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            head_input_dim = gru_hidden_dim
        else:
            # Build encoder layers (legacy flat MLP)
            encoder_layers = []
            prev_dim = self.input_dim

            for hidden_dim in hidden_dims:
                encoder_layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*encoder_layers)
            head_input_dim = prev_dim

        # Control signal heads
        if self.controls_version == "controls/2":
            # 13-channel ControlsV2: motion 6 + view 7. Activations match Rust parameter_ranges:
            # directionX,Y, zoom/rotation/hue/chroma/lightness/accent/harmonyShift are tanh [-1,1];
            # throttle, brake, grip, impulse are sigmoid [0,1].
            self.direction_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
            self.throttle_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.brake_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.grip_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.impulse_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.view_head = nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 7),
            )
        else:
            self.s_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Alpha in [0, 1], rescaled below

        )

        self.omega_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.band_gates_head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder appropriate for this model's mode.

        Returns an encoding of shape (batch_size, head_input_dim).
        """
        if not self.recurrent:
            return self.encoder(x)

        batch_size = x.shape[0]
        frames = x.view(batch_size, self.window_frames, self.features_per_frame)
        encoded_frames = self.frame_encoder(frames)  # (B, T, F_enc)
        _, hidden = self.gru(encoded_frames)  # hidden: (1, B, H)
        return hidden.squeeze(0)

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
                f"include_delta_delta={self.include_delta_delta}, "
                f"recurrent={self.recurrent}"
            )

        # Encode features
        encoded = self._encode(x)

        if self.controls_version == "controls/2":
            # 13-channel ControlsV2: directionX,Y (tanh), throttle/brake/grip/impulse (sigmoid), 7 view deltas (tanh)
            direction = torch.tanh(self.direction_head(encoded))  # (B,2) in [-1,1]
            throttle = self.throttle_head(encoded)  # (B,1) sigmoid already
            brake = self.brake_head(encoded)
            grip = self.grip_head(encoded)
            impulse = self.impulse_head(encoded)
            view_deltas = torch.tanh(self.view_head(encoded))  # (B,7) in [-1,1]
            output = torch.cat([direction, throttle, brake, grip, impulse, view_deltas], dim=1)
            return output

        # Legacy orbit_control: s_target, alpha, omega_scale, band_gates
        s_raw = self.s_head(encoded)  # (batch_size, 1)
        alpha = self.alpha_head(encoded)  # (batch_size, 1)
        omega_raw = self.omega_head(encoded)  # (batch_size, 1)
        band_gates = self.band_gates_head(encoded)  # (batch_size, k_bands)

        # Apply activation functions to constrain outputs
        s_target = 0.2 + 2.8 * torch.sigmoid(s_raw)  # [0.2, 3.0]
        alpha = 0.05 + 0.90 * alpha  # [0.05, 0.95]
        omega_scale = 0.1 + torch.nn.functional.softplus(omega_raw) * 0.5  # ~[0.1, 5.0]
        omega_scale = torch.clamp(omega_scale, 0.1, 5.0)
        output = torch.cat([s_target, alpha, omega_scale, band_gates], dim=1)
        return output

    def get_parameter_ranges(self) -> dict:
        """
        Get expected ranges for each output parameter.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        if self.controls_version == "controls/2":
            # Canonical ControlsV2 ranges from Rust ControlsV2::parameter_ranges()
            try:
                import runtime_core
                order = runtime_core.ControlsV2.model_output_order()
                # Use Rust ranges if available via python helper; fallback to hardcoded
                return {
                    "directionX": (-1.0, 1.0),
                    "directionY": (-1.0, 1.0),
                    "throttle": (0.0, 1.0),
                    "brake": (0.0, 1.0),
                    "grip": (0.0, 1.0),
                    "impulse": (0.0, 1.0),
                    "zoomDelta": (-1.0, 1.0),
                    "rotationDelta": (-1.0, 1.0),
                    "hueDelta": (-1.0, 1.0),
                    "chromaDelta": (-1.0, 1.0),
                    "lightnessDelta": (-1.0, 1.0),
                    "accentDelta": (-1.0, 1.0),
                    "harmonyShift": (-1.0, 1.0),
                }
            except Exception:
                return {
                    "directionX": (-1.0, 1.0),
                    "directionY": (-1.0, 1.0),
                    "throttle": (0.0, 1.0),
                    "brake": (0.0, 1.0),
                    "grip": (0.0, 1.0),
                    "impulse": (0.0, 1.0),
                    "zoomDelta": (-1.0, 1.0),
                    "rotationDelta": (-1.0, 1.0),
                    "hueDelta": (-1.0, 1.0),
                    "chromaDelta": (-1.0, 1.0),
                    "lightnessDelta": (-1.0, 1.0),
                    "accentDelta": (-1.0, 1.0),
                    "harmonyShift": (-1.0, 1.0),
                }
        ranges = {
            "s_target": (0.2, 3.0),
            "alpha": (0.05, 0.95),
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
            Dictionary with keys depending on controls_version:
              controls/2 -> directionX,Y,throttle,brake,grip,impulse,view deltas
              orbit_control -> s_target, alpha, omega_scale, band_gates
        """
        if self.controls_version == "controls/2":
            return {
                "directionX": output[:, 0],
                "directionY": output[:, 1],
                "throttle": output[:, 2],
                "brake": output[:, 3],
                "grip": output[:, 4],
                "impulse": output[:, 5],
                "zoomDelta": output[:, 6],
                "rotationDelta": output[:, 7],
                "hueDelta": output[:, 8],
                "chromaDelta": output[:, 9],
                "lightnessDelta": output[:, 10],
                "accentDelta": output[:, 11],
                "harmonyShift": output[:, 12],
            }
        return {
            "s_target": output[:, 0],
            "alpha": output[:, 1],
            "omega_scale": output[:, 2],
            "band_gates": output[:, 3:],
        }
