"""
Training pipeline with correlation-based loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path

from .model import AudioToVisualModel
from .visual_metrics import VisualMetrics
from .audio_features import AudioFeatureExtractor
from .data_loader import AudioDataset


class CorrelationLoss(nn.Module):
    """Loss based on correlation between audio features and visual metrics."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        audio_feature: torch.Tensor,
        visual_metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative correlation loss.
        
        Args:
            audio_feature: Audio feature values (batch_size,)
            visual_metric: Visual metric values (batch_size,)
            
        Returns:
            Negative correlation (to maximize correlation, minimize this)
        """
        # Normalize
        audio_norm = (audio_feature - audio_feature.mean()) / (
            audio_feature.std() + 1e-8
        )
        visual_norm = (visual_metric - visual_metric.mean()) / (
            visual_metric.std() + 1e-8
        )
        
        # Compute correlation
        correlation = (audio_norm * visual_norm).mean()
        
        # Return negative correlation (we want to maximize correlation)
        return -correlation


class SmoothnessLoss(nn.Module):
    """Penalize rapid changes in visual parameters."""
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize smoothness loss.
        
        Args:
            weight: Weight for smoothness penalty
        """
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        current_params: torch.Tensor,
        previous_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness loss.
        
        Args:
            current_params: Current parameter values (batch_size, n_params)
            previous_params: Previous parameter values (batch_size, n_params)
            
        Returns:
            Smoothness loss
        """
        # Compute parameter change
        param_diff = torch.abs(current_params - previous_params)
        
        # Sum over parameters and average over batch
        smoothness_loss = param_diff.mean()
        
        return self.weight * smoothness_loss


class Trainer:
    """Main training class."""
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: AudioFeatureExtractor,
        visual_metrics: VisualMetrics,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        correlation_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            feature_extractor: Audio feature extractor
            visual_metrics: Visual metrics calculator
            device: Device to train on
            learning_rate: Learning rate
            correlation_weights: Weights for different correlation losses
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.visual_metrics = visual_metrics
        self.device = device
        
        # Default correlation weights
        if correlation_weights is None:
            correlation_weights = {
                'timbre_color': 1.0,
                'transient_impact': 1.0,
                'silence_stillness': 1.0,
                'distortion_roughness': 1.0,
                'smoothness': 0.1
            }
        self.correlation_weights = correlation_weights
        
        # Loss functions
        self.correlation_loss = CorrelationLoss()
        self.smoothness_loss = SmoothnessLoss(
            weight=correlation_weights.get('smoothness', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training history
        self.history = {
            'loss': [],
            'timbre_color_loss': [],
            'transient_impact_loss': [],
            'silence_stillness_loss': [],
            'distortion_roughness_loss': [],
            'smoothness_loss': []
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        # #region agent log
        import json
        import time
        try:
            with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:170", "message": "train_epoch called", "data": {"epoch": epoch, "dataloader_type": str(type(dataloader))}, "timestamp": int(time.time() * 1000)}) + '\n')
        except Exception as e:
            pass
        # #endregion
        
        self.model.train()
        
        total_loss = 0.0
        total_timbre_color = 0.0
        total_transient_impact = 0.0
        total_silence_stillness = 0.0
        total_distortion_roughness = 0.0
        total_smoothness = 0.0
        
        n_batches = 0
        previous_params = None
        
        # #region agent log
        try:
            with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:195", "message": "About to iterate dataloader", "data": {"epoch": epoch}, "timestamp": int(time.time() * 1000)}) + '\n')
        except Exception as e:
            pass
        # #endregion
        
        try:
            for batch_idx, batch_item in enumerate(dataloader):
                # #region agent log
                try:
                    with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:200", "message": "Got batch from dataloader", "data": {"batch_idx": batch_idx, "type": str(type(batch_item)), "is_tuple": isinstance(batch_item, tuple), "len": len(batch_item) if isinstance(batch_item, (tuple, list)) else "N/A", "has_shape": hasattr(batch_item, 'shape')}, "timestamp": int(time.time() * 1000)}) + '\n')
                except Exception as e:
                    pass
                # #endregion
                
                # Handle both tuple and single tensor cases
                if isinstance(batch_item, tuple):
                    # #region agent log
                    try:
                        with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:210", "message": "Batch is tuple, unpacking", "data": {"tuple_len": len(batch_item)}, "timestamp": int(time.time() * 1000)}) + '\n')
                    except Exception as e:
                        pass
                    # #endregion
                    features, _ = batch_item
                else:
                    # #region agent log
                    try:
                        with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:218", "message": "Batch is not tuple, using directly", "data": {"type": str(type(batch_item))}, "timestamp": int(time.time() * 1000)}) + '\n')
                    except Exception as e:
                        pass
                    # #endregion
                    features = batch_item
                
                features = features.to(self.device)
                batch_size = features.shape[0]
                
                # #region agent log
                try:
                    with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:230", "message": "Features extracted successfully", "data": {"batch_size": batch_size, "features_shape": list(features.shape)}, "timestamp": int(time.time() * 1000)}) + '\n')
                except Exception as e:
                    pass
                # #endregion
                
                # Forward pass
                visual_params = self.model(features)
                
                # Render Julia sets and compute visual metrics
                # Extract parameters
                julia_real = visual_params[:, 0].detach().cpu().numpy()
                julia_imag = visual_params[:, 1].detach().cpu().numpy()
                zoom = visual_params[:, 5].detach().cpu().numpy()
                
                # Extract audio features for correlation
                # Features shape: (batch, n_features * window_frames)
                # We need to extract individual features
                # For simplicity, assume features are flattened: [centroid, flux, rms, zcr, onset, rolloff] * window_frames
                n_features_per_frame = 6
                window_frames = features.shape[1] // n_features_per_frame
                
                # Average features over window to get per-sample values
                features_reshaped = features.view(
                    batch_size, window_frames, n_features_per_frame
                )
                avg_features = features_reshaped.mean(dim=1)  # (batch, n_features)
                
                spectral_centroid = avg_features[:, 0]
                spectral_flux = avg_features[:, 1]
                rms_energy = avg_features[:, 2]
                zero_crossing_rate = avg_features[:, 3]
                
                # Render Julia sets (simplified - use small resolution for speed)
                images = []
                color_hues = []
                temporal_changes = []
                edge_densities = []
                
                prev_image = None
                for i in range(batch_size):
                    # Render Julia set
                    image = self.visual_metrics.render_julia_set(
                        seed_real=float(julia_real[i]),
                        seed_imag=float(julia_imag[i]),
                        width=128,  # Small for training speed
                        height=128,
                        zoom=float(zoom[i])
                    )
                    
                    # Compute metrics
                    metrics = self.visual_metrics.compute_all_metrics(
                        image,
                        prev_image=prev_image
                    )
                    
                    images.append(image)
                    color_hues.append(visual_params[i, 2].item())
                    temporal_changes.append(metrics['temporal_change'])
                    edge_densities.append(metrics['edge_density'])
                    
                    prev_image = image
                
                # Convert to tensors
                color_hue_tensor = torch.tensor(
                    color_hues, device=self.device, dtype=torch.float32
                )
                temporal_change_tensor = torch.tensor(
                    temporal_changes, device=self.device, dtype=torch.float32
                )
                edge_density_tensor = torch.tensor(
                    edge_densities, device=self.device, dtype=torch.float32
                )
                
                # Compute correlation losses
                timbre_color_loss = self.correlation_loss(
                    spectral_centroid, color_hue_tensor
                )
                
                transient_impact_loss = self.correlation_loss(
                    spectral_flux, temporal_change_tensor
                )
                
                silence_stillness_loss = self.correlation_loss(
                    rms_energy, -temporal_change_tensor  # Negative: silence -> low change
                )
                
                distortion_roughness_loss = self.correlation_loss(
                    zero_crossing_rate, edge_density_tensor
                )
                
                # Smoothness loss
                smoothness = torch.tensor(0.0, device=self.device)
                if previous_params is not None:
                    smoothness = self.smoothness_loss(
                        visual_params, previous_params
                    )
                
                # Total loss
                total_batch_loss = (
                    self.correlation_weights['timbre_color'] * timbre_color_loss +
                    self.correlation_weights['transient_impact'] * transient_impact_loss +
                    self.correlation_weights['silence_stillness'] * silence_stillness_loss +
                    self.correlation_weights['distortion_roughness'] * distortion_roughness_loss +
                    smoothness
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += total_batch_loss.item()
                total_timbre_color += timbre_color_loss.item()
                total_transient_impact += transient_impact_loss.item()
                total_silence_stillness += silence_stillness_loss.item()
                total_distortion_roughness += distortion_roughness_loss.item()
                total_smoothness += smoothness.item()
                
                n_batches += 1
                previous_params = visual_params.detach()
                
                if batch_idx % 10 == 0:
                    print(
                        f'Epoch {epoch}, Batch {batch_idx}, '
                        f'Loss: {total_batch_loss.item():.4f}'
                    )
        except Exception as e:
            # #region agent log
            try:
                with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run2", "hypothesisId": "C", "location": "trainer.py:330", "message": "ERROR in train_epoch loop", "data": {"error": str(e), "error_type": str(type(e).__name__)}, "timestamp": int(time.time() * 1000)}) + '\n')
            except Exception as log_err:
                pass
            # #endregion
            raise
        
        # Average losses
        avg_losses = {
            'loss': total_loss / n_batches,
            'timbre_color_loss': total_timbre_color / n_batches,
            'transient_impact_loss': total_transient_impact / n_batches,
            'silence_stillness_loss': total_silence_stillness / n_batches,
            'distortion_roughness_loss': total_distortion_roughness / n_batches,
            'smoothness_loss': total_smoothness / n_batches
        }
        
        return avg_losses
    
    def train(
        self,
        dataset: AudioDataset,
        epochs: int = 100,
        batch_size: int = 32,
        save_dir: Optional[str] = None
    ):
        """
        Train model on dataset.
        
        Args:
            dataset: Audio dataset
            epochs: Number of epochs
            batch_size: Batch size
            save_dir: Directory to save checkpoints
        """
        # Load all features
        print("Loading audio features...")
        all_features = dataset.load_all_features()
        
        # Compute normalization stats
        print("Computing normalization statistics...")
        self.feature_extractor.compute_normalization_stats(all_features)
        
        # Normalize features
        normalized_features = [
            self.feature_extractor.normalize_features(f) for f in all_features
        ]
        
        # Create dataset
        all_features_tensor = torch.tensor(
            np.concatenate(normalized_features, axis=0),
            dtype=torch.float32
        )
        
        # Create data loader
        tensor_dataset = TensorDataset(all_features_tensor)
        
        # #region agent log
        import json
        with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
            sample_item = tensor_dataset[0]
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "trainer.py:357", "message": "TensorDataset sample item", "data": {"type": str(type(sample_item)), "is_tuple": isinstance(sample_item, tuple), "len": len(sample_item) if isinstance(sample_item, (tuple, list)) else "N/A", "shape": list(sample_item.shape) if hasattr(sample_item, 'shape') else "N/A"}, "timestamp": int(__import__('time').time() * 1000)}) + '\n')
        # #endregion
        
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # #region agent log
        with open('c:\\Users\\JeffHall\\git\\FractalSync\\.cursor\\debug.log', 'a') as f:
            # Test first batch
            first_batch = next(iter(dataloader))
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "trainer.py:365", "message": "DataLoader first batch", "data": {"type": str(type(first_batch)), "is_tuple": isinstance(first_batch, tuple), "len": len(first_batch) if isinstance(first_batch, (tuple, list)) else "N/A"}, "timestamp": int(__import__('time').time() * 1000)}) + '\n')
        # #endregion
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            avg_losses = self.train_epoch(dataloader, epoch)
            
            # Update history
            for key, value in avg_losses.items():
                self.history[key].append(value)
            
            # Print progress
            print(
                f'Epoch {epoch + 1}/{epochs}: '
                f'Loss: {avg_losses["loss"]:.4f}, '
                f'Timbre-Color: {avg_losses["timbre_color_loss"]:.4f}, '
                f'Transient-Impact: {avg_losses["transient_impact_loss"]:.4f}'
            )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch + 1)
        
        print("Training complete!")
    
    def save_checkpoint(self, save_dir: str, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint
            epoch: Epoch number
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'feature_mean': self.feature_extractor.feature_mean,
            'feature_std': self.feature_extractor.feature_std
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save history as JSON
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
