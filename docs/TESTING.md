# Running Tests

This document describes how to run tests for each component of FractalSync.

## Backend Tests (Python/pytest)

Run all backend tests:
```bash
pytest backend/tests/
```

Run specific test file:
```bash
pytest backend/tests/test_feature_parity.py
```

Run with verbose output:
```bash
pytest backend/tests/ -v
```

**Test Coverage:**
- ✅ End-to-end workflow tests

> E2E note: The Playwright E2E harness will ensure a model is available for tests — if no ONNX model exists the test setup will train a tiny 1-epoch model on a small generated WAV and export it to ONNX so the frontend can fetch it. This makes the E2E suite self-contained and reliable in CI.
- ✅ Feature extraction parity (Python vs Rust)
- ✅ Integration tests (runtime-core via PyO3)
- ✅ Song analyzer tests
- ✅ Visual metrics tests

**Total: 36 tests**

---

## Runtime-Core Tests (Rust/cargo)

Run all library tests:
```bash
cd runtime-core
cargo test --lib
```

Run specific test:
```bash
cargo test --lib test_feature_extraction
```

Run with output (useful for debugging):
```bash
cargo test --lib -- --nocapture
```

Run in release mode (faster):
```bash
cargo test --lib --release
```

**Test Coverage:**
- ✅ Feature extraction (empty, small, normal audio)
- ✅ Orbit synthesis and geometry
- ✅ Parity extraction (for Python comparison)

**Total: 4 tests** (unit tests only, integration via backend)

---

## Frontend Tests (TypeScript/vitest)

Run all frontend tests:
```bash
cd frontend
npm test
```

Run in watch mode (interactive):
```bash
npm run test:watch
```

Run with UI:
```bash
npm run test:ui
```

**Test Coverage:**
- ⚠️ Audio feature extraction (needs AudioContext mock)
- ✅ Model inference parity (6 tests, skipped without model file)

**Total: 14 tests** (6 pass, 1 needs AudioContext mock, 6 skip without trained model)

### Frontend Test Notes

**AudioContext Mock Issue:**
The audio feature test fails because `AudioContext` is not available in the jsdom test environment. To fix:

```typescript
// Mock AudioContext in test setup
global.AudioContext = class MockAudioContext {
  // ... mock implementation
};
```

**Model Inference Tests:**
These tests skip when no trained model is present. To run with a real model:
1. Train a model: `python backend/train.py --epochs 1` (or PowerShell-safe: `Push-Location backend; try { python train.py --epochs 1 } finally { Pop-Location }`).
2. Copy model to frontend test fixtures: `cp backend/checkpoints/model_*.onnx frontend/src/lib/__tests__/fixtures/`

---

## Running All Tests

To run the complete test suite:

```bash
# Backend tests
pytest backend/tests/

# Runtime-core tests  
Push-Location runtime-core; try { cargo test --lib --release } finally { Pop-Location }

# Frontend tests
npm --prefix frontend test
```

---

## Continuous Integration

For CI pipelines, use:

```yaml
# Backend
- run: pip install -r backend/requirements.txt
- run: pytest backend/tests/

# Runtime-core (requires Rust toolchain)
- run: cargo test --lib --release --manifest-path runtime-core/Cargo.toml

# Frontend (requires Node.js)
- run: npm --prefix frontend ci && npm --prefix frontend test
```

---

## Test Results Summary

| Component     | Tests | Status | Notes                           |
|---------------|-------|--------|---------------------------------|
| Backend       | 36    | ✅ PASS | All passing                     |
| Runtime-Core  | 4     | ✅ PASS | Pure Rust tests                 |
| Frontend      | 14    | ⚠️ SKIP | Needs AudioContext mock + model |

**Overall: 40/54 tests passing, 14 skipped/fixable**
