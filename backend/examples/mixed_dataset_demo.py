"""
Example: Integrating synthetic orbit data with real audio training.

This script demonstrates how to combine synthetic data from orbit_engine
with real audio data for improved model training.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orbit_engine import create_synthetic_dataset
from src.audio_features import AudioFeatureExtractor
from src.data_loader import AudioDataset


class MixedDataset(Dataset):
    """
    Dataset that combines real audio data with synthetic orbital trajectories.
    
    This allows the model to learn from both:
    - Real audio-visual mappings (ground truth)
    - Synthetic trajectories that cover parameter space systematically
    """
    
    def __init__(
        self,
        audio_dataset: AudioDataset,
        synthetic_ratio: float = 0.3,
        window_frames: int = 10,
        n_audio_features: int = 6,
    ):
        """
        Initialize mixed dataset.
        
        Args:
            audio_dataset: Real audio dataset
            synthetic_ratio: Fraction of synthetic data (0-1)
            window_frames: Window size for features
            n_audio_features: Number of audio features
        """
        self.audio_dataset = audio_dataset
        self.window_frames = window_frames
        self.n_audio_features = n_audio_features
        
        # Calculate number of synthetic samples to generate
        n_real = len(audio_dataset)
        n_synthetic = int(n_real * synthetic_ratio / (1 - synthetic_ratio))
        
        print(f"Creating mixed dataset:")
        print(f"  - Real samples: {n_real}")
        print(f"  - Synthetic samples: {n_synthetic}")
        print(f"  - Total: {n_real + n_synthetic}")
        print(f"  - Synthetic ratio: {synthetic_ratio:.1%}")
        
        # Generate synthetic data
        print("Generating synthetic trajectories...")
        self.synthetic_audio, self.synthetic_visual, self.synthetic_metadata = (
            create_synthetic_dataset(
                n_samples=n_synthetic,
                window_frames=window_frames,
                n_audio_features=n_audio_features,
            )
        )
        
        self.n_real = n_real
        self.n_synthetic = len(self.synthetic_audio)
        self.total_samples = self.n_real + self.n_synthetic
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a sample (audio features, visual parameters).
        
        Returns samples from real data first, then synthetic data.
        """
        if idx < self.n_real:
            # Return real audio sample
            return self.audio_dataset[idx]
        else:
            # Return synthetic sample
            synthetic_idx = idx - self.n_real
            audio_feat = torch.from_numpy(self.synthetic_audio[synthetic_idx]).float()
            visual_param = torch.from_numpy(self.synthetic_visual[synthetic_idx]).float()
            return (audio_feat, visual_param)
    
    def get_metadata(self, idx):
        """Get metadata about a sample (useful for debugging)."""
        if idx < self.n_real:
            return {"source": "real", "real_idx": idx}
        else:
            synthetic_idx = idx - self.n_real
            metadata = self.synthetic_metadata[synthetic_idx].copy()
            metadata["source"] = "synthetic"
            metadata["synthetic_idx"] = synthetic_idx
            return metadata


def demo_mixed_dataset():
    """Demonstrate creating and using a mixed dataset."""
    print("=" * 60)
    print("MIXED DATASET DEMO")
    print("=" * 60)
    
    # Check if real audio data exists
    data_dir = Path("data/audio")
    if not data_dir.exists() or len(list(data_dir.glob("*.mp3")) + list(data_dir.glob("*.wav"))) == 0:
        print(f"\n⚠ No audio files found in {data_dir}")
        print("This demo will create a synthetic-only dataset for illustration.")
        
        # Create synthetic dataset
        print("\nGenerating synthetic dataset...")
        windowed_audio, windowed_visual, metadata = create_synthetic_dataset(
            n_samples=1000,
            window_frames=10,
            n_audio_features=6
        )
        
        print(f"\nSynthetic Dataset Created:")
        print(f"  - Samples: {len(windowed_audio)}")
        print(f"  - Input shape: {windowed_audio.shape}")
        print(f"  - Output shape: {windowed_visual.shape}")
        
        # Create DataLoader
        class SimpleDataset(Dataset):
            def __init__(self, audio, visual):
                self.audio = torch.from_numpy(audio).float()
                self.visual = torch.from_numpy(visual).float()
            
            def __len__(self):
                return len(self.audio)
            
            def __getitem__(self, idx):
                return (self.audio[idx], self.visual[idx])
        
        dataset = SimpleDataset(windowed_audio, windowed_visual)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"\nDataLoader Created:")
        print(f"  - Batches: {len(dataloader)}")
        print(f"  - Batch size: 32")
        
        # Show first batch
        audio_batch, visual_batch = next(iter(dataloader))
        print(f"\nFirst Batch:")
        print(f"  - Audio batch shape: {audio_batch.shape}")
        print(f"  - Visual batch shape: {visual_batch.shape}")
        
        return
    
    # Create real audio dataset
    print(f"\n1. Loading real audio data from {data_dir}...")
    extractor = AudioFeatureExtractor(
        sr=22050,
        hop_length=512,
        n_fft=2048,
        include_delta=False,
        include_delta_delta=False
    )
    
    audio_dataset = AudioDataset(
        data_dir=str(data_dir),
        feature_extractor=extractor,
        window_frames=10
    )
    
    print(f"   Loaded {len(audio_dataset)} real samples")
    
    # Create mixed dataset
    print(f"\n2. Creating mixed dataset (30% synthetic)...")
    mixed_dataset = MixedDataset(
        audio_dataset=audio_dataset,
        synthetic_ratio=0.3,
        window_frames=10,
        n_audio_features=6
    )
    
    # Create DataLoader
    print(f"\n3. Creating DataLoader...")
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # Show first batch
    print(f"\n4. Inspecting first batch...")
    audio_batch, visual_batch = next(iter(dataloader))
    print(f"   Audio batch shape: {audio_batch.shape}")
    print(f"   Visual batch shape: {visual_batch.shape}")
    
    # Show metadata for some samples
    print(f"\n5. Sample metadata:")
    for i in [0, len(mixed_dataset) // 2, len(mixed_dataset) - 1]:
        metadata = mixed_dataset.get_metadata(i)
        print(f"   Sample {i}: {metadata['source']}")
    
    print("\n" + "=" * 60)
    print("✓ Mixed dataset demo complete!")
    print("=" * 60)


def main():
    """Run the demo."""
    try:
        demo_mixed_dataset()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
