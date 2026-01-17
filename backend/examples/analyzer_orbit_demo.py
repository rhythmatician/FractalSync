"""
Example demonstrating the song analyzer and orbit engine modules.

This script shows how to:
1. Analyze an audio file for tempo and section boundaries
2. Generate synthetic training data using orbit engine
3. Combine real and synthetic data for training
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.song_analyzer import SongAnalyzer, analyze_audio_file
from src.orbit_engine import OrbitEngine, create_synthetic_dataset


def demo_song_analyzer():
    """Demonstrate song analyzer functionality."""
    print("=" * 60)
    print("SONG ANALYZER DEMO")
    print("=" * 60)
    
    # Create synthetic audio (5 seconds)
    print("\n1. Creating synthetic audio signal (5 seconds)...")
    sr = 22050
    duration = 5.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    
    # Analyze the audio
    print("\n2. Analyzing audio...")
    analyzer = SongAnalyzer(sr=sr)
    analysis = analyzer.analyze_song(audio)
    
    # Print results
    print(f"\n3. Analysis Results:")
    print(f"   - Global Tempo: {analysis['tempo']:.1f} BPM")
    print(f"   - Local Tempo Shape: {analysis['local_tempo'].shape}")
    print(f"   - Section Boundaries: {len(analysis['section_boundaries'])} sections detected")
    print(f"   - Onset Frames: {len(analysis['onset_frames'])} onsets detected")
    print(f"   - Beat Frames: {len(analysis['beat_frames'])} beats detected")
    
    # Extract hit events
    hit_events = analyzer.get_hit_events(
        analysis['onset_frames'],
        analysis['onset_strength'],
        threshold=0.5
    )
    print(f"   - Hit Events: {len(hit_events)} strong hits detected")
    
    # Time conversion examples
    frame_10 = 10
    time_at_10 = analyzer.frames_to_time(frame_10)
    print(f"\n4. Frame/Time Conversion:")
    print(f"   - Frame {frame_10} = {time_at_10:.3f} seconds")
    print(f"   - 1.0 second = frame {analyzer.time_to_frames(1.0)}")
    

def demo_orbit_engine():
    """Demonstrate orbit engine functionality."""
    print("\n" + "=" * 60)
    print("ORBIT ENGINE DEMO")
    print("=" * 60)
    
    # Create orbit engine
    print("\n1. Initializing orbit engine...")
    engine = OrbitEngine(n_audio_features=6, sr=22050, hop_length=512)
    
    # Generate a single trajectory
    print("\n2. Generating synthetic trajectory (velocity correlation)...")
    n_samples = 100
    audio_features, visual_params = engine.generate_synthetic_trajectory(
        orbit_name="cardioid_boundary",
        n_samples=n_samples,
        audio_correlation="velocity"
    )
    
    print(f"\n3. Trajectory Results:")
    print(f"   - Audio Features Shape: {audio_features.shape}")
    print(f"   - Visual Params Shape: {visual_params.shape}")
    print(f"   - Audio Features Range: [{audio_features.min():.3f}, {audio_features.max():.3f}]")
    print(f"   - Visual Params (c) Range:")
    print(f"     - Real: [{visual_params[:, 0].min():.3f}, {visual_params[:, 0].max():.3f}]")
    print(f"     - Imag: [{visual_params[:, 1].min():.3f}, {visual_params[:, 1].max():.3f}]")
    
    # Generate mixed curriculum
    print("\n4. Generating mixed curriculum dataset...")
    audio_feat, visual_params, metadata = engine.generate_mixed_curriculum(
        n_samples=500,
        orbit_names=["cardioid_boundary", "period2_boundary", "period3_boundary"]
    )
    
    print(f"\n5. Curriculum Dataset:")
    print(f"   - Total Samples: {len(audio_feat)}")
    print(f"   - Unique Orbits Used: {len(set(m['orbit'] for m in metadata))}")
    print(f"   - Correlation Types: {set(m['correlation'] for m in metadata)}")
    
    # Generate windowed features
    print("\n6. Converting to windowed format for model input...")
    windowed_audio, windowed_visual = engine.generate_windowed_features(
        audio_feat, visual_params, window_frames=10
    )
    
    print(f"\n7. Windowed Dataset:")
    print(f"   - Windowed Audio Shape: {windowed_audio.shape}")
    print(f"   - Windowed Visual Shape: {windowed_visual.shape}")
    print(f"   - Input Dimensionality: {windowed_audio.shape[1]} (6 features × 10 frames)")


def demo_synthetic_dataset_creation():
    """Demonstrate convenience function for dataset creation."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATASET CREATION DEMO")
    print("=" * 60)
    
    print("\n1. Creating complete synthetic dataset...")
    windowed_audio, windowed_visual, metadata = create_synthetic_dataset(
        n_samples=1000,
        window_frames=10,
        n_audio_features=6
    )
    
    print(f"\n2. Dataset Statistics:")
    print(f"   - Training Samples: {len(windowed_audio)}")
    print(f"   - Input Shape: {windowed_audio.shape}")
    print(f"   - Output Shape: {windowed_visual.shape}")
    print(f"   - Unique Orbits: {len(set(m['orbit'] for m in metadata))}")
    
    # Show some orbit distribution
    orbit_counts = {}
    for m in metadata:
        orbit = m['orbit']
        orbit_counts[orbit] = orbit_counts.get(orbit, 0) + 1
    
    print(f"\n3. Orbit Distribution (top 5):")
    for orbit, count in sorted(orbit_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   - {orbit}: {count} samples")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("SONG ANALYZER & ORBIT ENGINE DEMO")
    print("=" * 60)
    
    try:
        demo_song_analyzer()
        demo_orbit_engine()
        demo_synthetic_dataset_creation()
        
        print("\n" + "=" * 60)
        print("✓ All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
