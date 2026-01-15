"""
Velocity-based parameter prediction for smooth visual transitions.

This module implements physics-inspired momentum for visual parameters,
enabling smoother transitions and more natural-looking animations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class VelocityPredictor:
    """
    Tracks parameter velocities and predicts smooth transitions.
    
    Uses exponential moving average for velocity estimation and
    momentum-based prediction for smooth parameter evolution.
    """
    
    def __init__(
        self,
        num_params: int = 7,
        momentum: float = 0.8,
        velocity_decay: float = 0.95,
        max_velocity: float = 0.5,
    ):
        """
        Initialize velocity predictor.
        
        Args:
            num_params: Number of visual parameters to track
            momentum: Momentum factor for velocity updates (0-1, higher = smoother)
            velocity_decay: Decay factor for velocity over time (0-1)
            max_velocity: Maximum allowed velocity magnitude (clipping threshold)
        """
        self.num_params = num_params
        self.momentum = momentum
        self.velocity_decay = velocity_decay
        self.max_velocity = max_velocity
        
        # State tracking
        self.previous_params: Optional[np.ndarray] = None
        self.current_velocity: Optional[np.ndarray] = None
        
    def reset(self):
        """Reset internal state (call when starting new audio sequence)."""
        self.previous_params = None
        self.current_velocity = None
        
    def update(
        self, 
        current_params: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Update velocity estimate and predict smooth parameters.
        
        Args:
            current_params: Current parameter values (shape: [batch_size, num_params] or [num_params])
            dt: Time step (default 1.0 for frame-to-frame)
            
        Returns:
            Smoothed parameters with velocity-based prediction
        """
        # Handle batch dimension
        is_batched = len(current_params.shape) > 1
        if not is_batched:
            current_params = current_params[np.newaxis, :]
            
        batch_size = current_params.shape[0]
        
        # Initialize on first call
        if self.previous_params is None:
            self.previous_params = current_params.copy()
            self.current_velocity = np.zeros_like(current_params)
            return current_params.squeeze() if not is_batched else current_params
            
        # Compute instantaneous velocity
        instantaneous_velocity = (current_params - self.previous_params) / dt
        
        # Clip extreme velocities
        instantaneous_velocity = np.clip(
            instantaneous_velocity,
            -self.max_velocity,
            self.max_velocity,
        )
        
        # Update velocity with momentum (exponential moving average)
        self.current_velocity = (
            self.momentum * self.current_velocity +
            (1 - self.momentum) * instantaneous_velocity
        )
        
        # Apply velocity decay (reduces oscillation)
        self.current_velocity *= self.velocity_decay
        
        # Predict next parameters using velocity
        predicted_params = self.previous_params + self.current_velocity * dt
        
        # Blend prediction with actual parameters (smoothing)
        smoothed_params = (
            self.momentum * predicted_params +
            (1 - self.momentum) * current_params
        )
        
        # Update state
        self.previous_params = smoothed_params.copy()
        
        return smoothed_params.squeeze() if not is_batched else smoothed_params
        
    def get_velocity(self) -> Optional[np.ndarray]:
        """Get current velocity estimate."""
        return self.current_velocity
        
    def predict_next(self, steps: int = 1, dt: float = 1.0) -> Optional[np.ndarray]:
        """
        Predict future parameters based on current velocity.
        
        Args:
            steps: Number of time steps to predict ahead
            dt: Time step size
            
        Returns:
            Predicted parameters or None if not initialized
        """
        if self.previous_params is None or self.current_velocity is None:
            return None
            
        # Simple linear prediction (could be extended to higher-order)
        predicted = self.previous_params + self.current_velocity * dt * steps
        
        return predicted


class VelocityLoss(nn.Module):
    """
    Loss function that penalizes large velocity changes (jerk).
    
    This encourages smooth acceleration/deceleration rather than
    abrupt velocity changes, creating more natural-looking animations.
    """
    
    def __init__(self, weight: float = 0.05):
        """
        Initialize velocity loss.
        
        Args:
            weight: Weight for velocity penalty
        """
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        current_params: torch.Tensor,
        previous_params: torch.Tensor,
        previous_velocity: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute velocity-based smoothness loss.
        
        Args:
            current_params: Current parameters (batch_size, num_params)
            previous_params: Previous parameters (batch_size, num_params)
            previous_velocity: Previous velocity if available (batch_size, num_params)
            dt: Time step
            
        Returns:
            Velocity smoothness loss
        """
        # Compute current velocity
        current_velocity = (current_params - previous_params) / dt
        
        if previous_velocity is None:
            # If no previous velocity, just penalize high velocities
            velocity_loss = torch.mean(current_velocity ** 2)
        else:
            # Penalize velocity changes (jerk = dv/dt)
            jerk = (current_velocity - previous_velocity) / dt
            velocity_loss = torch.mean(jerk ** 2)
            
        return self.weight * velocity_loss


class VelocityAwarePredictor(nn.Module):
    """
    Neural network module that predicts both parameters and their velocities.
    
    This can be used to extend the AudioToVisualModel to output velocity
    predictions alongside parameter predictions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 7):
        """
        Initialize velocity-aware predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of parameters (outputs 2x this: params + velocities)
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Separate heads for parameters and velocities
        self.param_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Constrain velocity to reasonable range
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (parameters, velocities)
        """
        params = self.param_head(x)
        velocities = self.velocity_head(x)
        
        return params, velocities
        
    def predict_with_velocity(
        self,
        x: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict parameters incorporating velocity.
        
        Args:
            x: Input features
            dt: Time step for velocity integration
            
        Returns:
            Parameters adjusted by predicted velocity
        """
        params, velocities = self.forward(x)
        
        # Apply velocity to get smoother predictions
        adjusted_params = params + velocities * dt
        
        return adjusted_params
