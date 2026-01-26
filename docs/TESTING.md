# Running Tests

This document describes how to run tests for each component of FractalSync.

## Backend Tests (Python/pytest)

Run all backend tests:
```bash
pytest backend/tests/
```

Run specific test file:
```bash
pytest backend/tests/test_e2e.py
```

Run with verbose output:
```bash
pytest backend/tests/ -v
```

**Test Coverage:**
- ✅ End-to-end workflow tests
- ✅ Integration tests (runtime-core via PyO3)
- ✅ Song analyzer tests
- ✅ Visual metrics tests

**Total: see `pytest -q` output**

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

**Total: see `cargo test` output** (unit tests only, integration via backend)

---

## Frontend Tests (TypeScript/vitest)

There are currently no frontend unit tests. Use manual verification by running the frontend
against a trained model and validating that visualization updates render without errors.

---

## Running All Tests

To run the complete test suite:

```bash
# Backend tests
pytest backend/tests/

# Runtime-core tests  
cd runtime-core && cargo test --lib --release && cd ..

# Frontend tests
cd frontend && npm run build && cd ..
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
- run: cd frontend && npm ci && npm run build
```

---

## Test Results Summary

| Component     | Tests | Status | Notes                           |
|---------------|-------|--------|---------------------------------|
| Backend       | —     | ✅ PASS | See pytest output               |
| Runtime-Core  | —     | ✅ PASS | See cargo test output           |
| Frontend      | —     | ⚠️ SKIP | Manual verification only        |

**Overall: refer to command output, frontend manual verification**
