"""
Section detector for audio-reactive lobe switching.

Detects section changes in audio based on feature statistics
to trigger lobe transitions during training and inference.
"""

import numpy as np
import torch
from typing import Optional, Tuple


class SectionDetector:
    """
    Detects section changes using moving averages of audio features.
    
    Uses exponential moving average with change detection and hysteresis
    to identify transitions between song sections (verse, chorus, etc.).
    """
    
    def __init__(
        self,
        smoothing: float = 0.95,
        change_threshold: float = 0.15,
        cooldown_frames: int = 10,
        num_lobes: int = 4,
    ):
        """
        Initialize section detector.
        
        Args:
            smoothing: EMA smoothing factor (higher = more smoothing)
            change_threshold: Threshold for section change detection
            cooldown_frames: Minimum frames between section changes
            num_lobes: Number of Mandelbrot lobes to cycle through
        """
        self.smoothing = smoothing
        self.change_threshold = change_threshold
        self.cooldown_frames = cooldown_frames
        self.num_lobes = num_lobes
        
        # State
        self.ema_features: Optional[np.ndarray] = None
        self.current_lobe = 1
        self.cooldown_counter = 0
        
    def reset(self):
        """Reset detector state."""
        self.ema_features = None
        self.current_lobe = 1
        self.cooldown_counter = 0
        
    def detect_section_change(
        self, features: np.ndarray
    ) -> Tuple[bool, int]:
        """
        Detect if current features indicate a section change.
        
        Args:
            features: Feature vector (shape: [n_features])
            
        Returns:
            Tuple of (section_changed, current_lobe)
        """
        # Initialize EMA on first call
        if self.ema_features is None:
            self.ema_features = features.copy()
            return False, self.current_lobe
            
        # Update EMA
        prev_ema = self.ema_features.copy()
        self.ema_features = (
            self.smoothing * self.ema_features + 
            (1.0 - self.smoothing) * features
        )
        
        # Compute feature change magnitude
        change = np.linalg.norm(self.ema_features - prev_ema)
        normalized_change = change / (np.linalg.norm(prev_ema) + 1e-8)
        
        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, self.current_lobe
            
        # Detect section change
        section_changed = normalized_change > self.change_threshold
        
        if section_changed:
            # Switch to next lobe
            self.current_lobe = (self.current_lobe % self.num_lobes) + 1
            self.cooldown_counter = self.cooldown_frames
            return True, self.current_lobe
            
        return False, self.current_lobe
    
    def detect_batch(
        self, features_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect sections for a batch of feature vectors.
        
        Args:
            features_batch: Batch of features (shape: [batch_size, n_features])
            
        Returns:
            Tensor of lobe assignments (shape: [batch_size])
        """
        lobes = []
        
        for i in range(features_batch.shape[0]):
            features_np = features_batch[i].detach().cpu().numpy()
            _, lobe = self.detect_section_change(features_np)
            lobes.append(lobe)
            
        return torch.tensor(lobes, dtype=torch.long, device=features_batch.device)
