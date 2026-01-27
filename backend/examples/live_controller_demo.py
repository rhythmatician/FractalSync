"""
Demo script for the new LiveController architecture.

Shows how to use the impact detection and section boundary detection
to drive real-time Julia set visualization.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.live_controller import (
    LiveController,
    AudioFeatureStream,
    ImpactDetector,
    NoveltyBoundaryDetector,
)


def demo_live_controller():
    """Demonstrate the live controller with synthetic audio."""
    print("=" * 70)
    print("LIVE CONTROLLER DEMO")
    print("=" * 70)
    
    # Create controller
    controller = LiveController(sr=22050, render_rate=60.0)
    
    print("\n1. Creating synthetic audio (10 seconds with impacts)...")
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base tone
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Add some "impacts" at regular intervals
    impact_times = [1.0, 2.5, 4.0, 5.5, 7.0, 8.5]
    for impact_time in impact_times:
        idx = int(impact_time * sr)
        # Add transient
        transient_len = int(0.1 * sr)
        transient = np.exp(-10 * np.linspace(0, 1, transient_len))
        noise = np.random.randn(transient_len)
        if idx + transient_len < len(audio):
            audio[idx:idx + transient_len] += 0.8 * transient * noise
    
    # Add some sustained frequency changes (section boundaries)
    # Change frequency at 5 seconds
    audio[int(5*sr):] += 0.2 * np.sin(2 * np.pi * 550 * t[int(5*sr):])
    
    print(f"   Audio length: {len(audio)} samples ({duration} seconds)")
    print(f"   Sample rate: {sr} Hz")
    
    # Process in chunks
    print("\n2. Processing audio through live controller...")
    chunk_size = int(sr / 60)  # 60 Hz render rate
    n_chunks = len(audio) // chunk_size
    
    c_trajectory = []
    timestamps = []
    
    for i in range(min(n_chunks, 600)):  # Process up to 10 seconds
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio[start_idx:end_idx]
        timestamp = i / 60.0
        
        # Process chunk
        c = controller.process_audio_frame(chunk, timestamp)
        c_trajectory.append(c)
        timestamps.append(timestamp)
        
        # Print debug info every second
        if i % 60 == 0:
            print(f"\n   t={timestamp:.1f}s:")
            print(controller.get_debug_overlay())
    
    print("\n3. Results:")
    print(f"   Generated {len(c_trajectory)} c(t) values")
    print(f"   Total impacts detected: {len(controller.impact_events)}")
    print(f"   Total boundaries detected: {len(controller.boundary_events)}")
    
    # Print impact events
    print("\n4. Impact Events:")
    for event in controller.impact_events:
        print(f"   t={event.timestamp:.3f}s: score={event.score:.3f}, s_overshoot={event.s_overshoot:.3f}")
    
    # Print boundary events
    print("\n5. Boundary Events:")
    for event in controller.boundary_events:
        print(f"   t={event.timestamp:.3f}s: novelty={event.novelty_score:.3f}, "
              f"target_height={event.target_height:.3f}, "
              f"dwell={event.previous_dwell_time:.1f}s")
    
    # Visualize trajectory
    print("\n6. Trajectory Statistics:")
    c_array = np.array(c_trajectory)
    print(f"   Mean |c|: {np.mean(np.abs(c_array)):.4f}")
    print(f"   Std |c|: {np.std(np.abs(c_array)):.4f}")
    print(f"   Min |c|: {np.min(np.abs(c_array)):.4f}")
    print(f"   Max |c|: {np.max(np.abs(c_array)):.4f}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def demo_impact_detector():
    """Demo just the impact detector."""
    print("\n" + "=" * 70)
    print("IMPACT DETECTOR DEMO")
    print("=" * 70)
    
    # Create components
    feature_stream = AudioFeatureStream(sr=22050)
    detector = ImpactDetector(threshold_percentile=60.0, refractory_ms=100.0)
    
    # Create audio with clear impacts
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    # Add 5 distinct impacts
    impact_times = [0.5, 1.5, 2.5, 3.5, 4.5]
    for impact_time in impact_times:
        idx = int(impact_time * sr)
        transient_len = int(0.05 * sr)
        transient = np.exp(-20 * np.linspace(0, 1, transient_len))
        noise = np.random.randn(transient_len)
        if idx + transient_len < len(audio):
            audio[idx:idx + transient_len] += 1.5 * transient * noise
    
    print(f"Created audio with impacts at: {impact_times}")
    
    # Process in chunks
    chunk_size = int(sr / 100)  # 100 Hz
    detected_impacts = []
    
    for i in range(len(audio) // chunk_size):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio[start_idx:end_idx]
        timestamp = i / 100.0
        
        features = feature_stream.compute_fast_features(chunk, timestamp)
        is_impact, score = detector.detect(features)
        
        if is_impact:
            detected_impacts.append(timestamp)
            print(f"Impact detected at t={timestamp:.3f}s, score={score:.3f}")
    
    print(f"\nTotal impacts detected: {len(detected_impacts)}")
    print(f"Expected: {len(impact_times)}")


def demo_novelty_detector():
    """Demo the novelty boundary detector."""
    print("\n" + "=" * 70)
    print("NOVELTY BOUNDARY DETECTOR DEMO")
    print("=" * 70)
    
    # Create components
    feature_stream = AudioFeatureStream(sr=22050)
    detector = NoveltyBoundaryDetector(
        baseline_window_sec=5.0,
        min_dwell_sec=5.0,
        cooldown_sec=3.0
    )
    
    # Create audio with clear section change at 5 seconds
    sr = 22050
    duration = 12.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # First section: 440 Hz
    audio = np.zeros_like(t)
    audio[:int(5*sr)] = 0.3 * np.sin(2 * np.pi * 440 * t[:int(5*sr)])
    
    # Second section: 550 Hz with different timbre
    audio[int(5*sr):int(10*sr)] = (
        0.3 * np.sin(2 * np.pi * 550 * t[int(5*sr):int(10*sr)]) +
        0.1 * np.sin(2 * np.pi * 1100 * t[int(5*sr):int(10*sr)])
    )
    
    # Third section: back to 440 Hz
    audio[int(10*sr):] = 0.3 * np.sin(2 * np.pi * 440 * t[int(10*sr):])
    
    print("Created audio with section changes at t=5s and t=10s")
    
    # Process in chunks
    chunk_size = int(sr / 10)  # 10 Hz
    detected_boundaries = []
    
    for i in range(len(audio) // chunk_size):
        start_idx = i * chunk_size
        end_idx = start_idx + int(sr * 0.5)  # 0.5s window
        if end_idx > len(audio):
            break
        chunk = audio[start_idx:end_idx]
        timestamp = i / 10.0
        
        features = feature_stream.compute_slow_features(chunk, timestamp)
        is_boundary, novelty = detector.update(features)
        
        if is_boundary:
            detected_boundaries.append((timestamp, novelty))
            print(f"Boundary detected at t={timestamp:.3f}s, novelty={novelty:.3f}")
    
    print(f"\nTotal boundaries detected: {len(detected_boundaries)}")
    print("Expected: 2 (at ~5s and ~10s)")


if __name__ == "__main__":
    try:
        demo_impact_detector()
        demo_novelty_detector()
        demo_live_controller()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
