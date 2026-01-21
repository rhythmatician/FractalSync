"""Debug test for Rust feature extractor hanging bug."""

import runtime_core as rc
import numpy as np
import sys
import threading
import time

print("Step 1: Creating FeatureExtractor...", flush=True)
extractor = rc.FeatureExtractor(
    sr=48000,
    hop_length=1024,
    n_fft=4096,
    include_delta=False,
    include_delta_delta=False,
)
print("  SUCCESS", flush=True)

print("Step 2: Calling num_features_per_frame()...", flush=True)
n = extractor.num_features_per_frame()
print(f"  SUCCESS: {n} features", flush=True)

print("Step 3: Calling with empty numpy array...", flush=True)
try:
    audio = np.array([], dtype=np.float32)
    result = extractor.extract_windowed_features(audio, window_frames=2)
    print(f"  SUCCESS: {result}", flush=True)
except Exception as e:
    print(f"  EXCEPTION (expected): {e}", flush=True)

print("Step 4: Calling with single-element array [0.5]...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

result_holder = {"result": None, "error": None, "done": False}


def call_function():
    try:
        audio = np.array([0.5], dtype=np.float32)
        print("    [THREAD] About to call extract_windowed_features...", flush=True)
        result_holder["result"] = extractor.extract_windowed_features(
            audio, window_frames=2
        )
        print("    [THREAD] Call completed!", flush=True)
    except Exception as e:
        result_holder["error"] = str(e)
        print(f"    [THREAD] Exception: {e}", flush=True)
    finally:
        result_holder["done"] = True


thread = threading.Thread(target=call_function)
thread.daemon = True
print("  Starting thread...", flush=True)
thread.start()

for i in range(20):
    time.sleep(0.5)
    if not thread.is_alive():
        print(f"    Thread finished after {i*0.5}s", flush=True)
        break
    if i % 2 == 0:
        print(f"    Still waiting... {i*0.5}s", flush=True)

if thread.is_alive():
    print("  HUNG DETECTED - thread still alive after 10s!", flush=True)
    sys.exit(1)
else:
    if result_holder["error"]:
        print(f'  EXCEPTION: {result_holder["error"]}', flush=True)
    elif result_holder["result"] is not None:
        print(f'  SUCCESS: {len(result_holder["result"])} windows', flush=True)
    else:
        print("  COMPLETED but no result?", flush=True)

print("\nStep 5: Testing with 5000 samples...", flush=True)
audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 5000)) * 0.3).astype(np.float32)
print(f"  Audio shape: {audio.shape}", flush=True)
result = extractor.extract_windowed_features(audio, window_frames=3)
print(
    f"  SUCCESS: {len(result)} windows, first window has {len(result[0])} features",
    flush=True,
)
