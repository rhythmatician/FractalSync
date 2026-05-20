import time
import time
import numpy as np
from src.runtime_core_bridge import make_feature_extractor


def bench(n_iters=2000, n_samples=1024):
    extractor = make_feature_extractor()
    audio = (np.random.RandomState(42).randn(n_samples) * 0.1).astype(np.float32)
    arr = np.ascontiguousarray(audio)
    lst = list(audio)

    # Warmup
    for _ in range(10):
        extractor.extract_windowed_features(arr, window_frames=4)
        extractor.extract_windowed_features(lst, window_frames=4)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        extractor.extract_windowed_features(arr, window_frames=4)
    t1 = time.perf_counter()
    arr_time = (t1 - t0) / n_iters * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        extractor.extract_windowed_features(lst, window_frames=4)
    t1 = time.perf_counter()
    list_time = (t1 - t0) / n_iters * 1000.0

    print(f"ndarray avg {arr_time:.3f} ms; list avg {list_time:.3f} ms; delta {list_time - arr_time:.3f} ms")


if __name__ == '__main__':
    bench()
