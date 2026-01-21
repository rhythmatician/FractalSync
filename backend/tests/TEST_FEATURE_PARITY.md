# Audio Feature Parity Test

This test ensures that the Python fallback feature extractor (`python_feature_extractor.py`) produces output compatible with the Rust implementation (`features.rs`).

## Current Status

✅ **Python Consistency**: PASS - Python extractor is deterministic  
⚠️ **Python-Rust Parity**: SHAPE MISMATCH

### Known Differences

1. **Window Count Difference**: 
   - Python (librosa): 38 windows from 48000 samples
   - Rust: 34 windows from 48000 samples
   
   **Cause**: Librosa's `stft()` uses `center=True` by default, which pads the audio signal symmetrically. The Rust implementation doesn't pad.
   
2. **Acceptable Tolerance**: Features should match within `1e-4` absolute difference once window count is aligned.

## How to Run

```bash
python test_feature_parity.py
```

## Fixing the Parity

To achieve exact parity, we need to either:
1. Disable centering in Python: `librosa.stft(..., center=False)`
2. Add padding in Rust to match librosa's behavior

**Recommendation**: Disable centering in Python since:
- Simpler change
- Frontend WASM will use the Rust implementation without padding
- Backend should match frontend behavior

## Test Architecture

```
Python Test
    ↓
Generates test audio (numpy)
    ↓
    ├──→ Python Extractor (direct call)
    │       ↓
    │    features_python.npy
    │
    └──→ Rust Extractor (via cargo test)
            ↓
         Rust test reads .npy
            ↓
         Rust extract_windowed_features()
            ↓
         features_rust.json
            ↓
    Compare numpy vs JSON
```

## Future Work

- [ ] Align window counts by adjusting STFT centering
- [ ] Test with delta and delta-delta features
- [ ] Test with different window_frames values  
- [ ] Add CI integration to catch regressions
