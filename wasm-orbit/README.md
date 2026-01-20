# WebAssembly Orbit Synthesizer

## Overview

The orbit synthesizer logic is implemented **once** in Rust and compiled to WebAssembly for use in the browser. This ensures DRY (Don't Repeat Yourself) principles are maintained.

## Single Source of Truth

**Rust Implementation**: `wasm-orbit/src/lib.rs`

This contains:
- Orbit synthesis formula: `amplitude_k = α × (s × radius) / (k + 1)²`
- Mandelbrot geometry calculations
- State management and time evolution

## Building the WASM Module

After modifying `wasm-orbit/src/lib.rs`:

```bash
cd wasm-orbit
wasm-pack build --target web
```

This generates compiled files in `wasm-orbit/pkg/`:
- `orbit_synth_wasm.wasm` - WebAssembly binary
- `orbit_synth_wasm.js` - JavaScript bindings
- Type definitions

## Deploying to Frontend

Copy the built WASM files to the frontend:

```bash
# PowerShell
Copy-Item -Path "wasm-orbit\pkg\*" -Destination "frontend\public\wasm\" -Recurse -Force

# Bash
cp -r wasm-orbit/pkg/* frontend/public/wasm/
```

Or use the provided script:
```bash
cd wasm-orbit
npm run deploy  # (if configured in package.json)
```

## Development Workflow

1. **Modify Rust code**: Edit `wasm-orbit/src/lib.rs`
2. **Build WASM**: `cd wasm-orbit && wasm-pack build --target web`
3. **Copy to frontend**: Copy `pkg/*` to `frontend/public/wasm/`
4. **Test**: Refresh browser (hard refresh: Ctrl+Shift+R)

## Frontend Integration

The frontend loads the WASM module dynamically:

```typescript
// In modelInference.ts
const wasmModule = await import('/wasm/orbit_synth_wasm.js');
await wasmModule.default();  // Initialize WASM

const synthesizer = new wasmModule.OrbitSynthesizer(6, 0.5);
const state = new wasmModule.OrbitState(1, 0, 0.0, 1.0, 1.02, 0.3, 6, 2.0);

// Use it
const result = synthesizer.step(state, dt, bandGates);
console.log(result.c);  // { real: ..., imag: ... }
```

## Dependencies

- **Rust**: Install from https://rustup.rs/
- **wasm-pack**: `cargo install wasm-pack`

On Windows:
```powershell
winget install --id Rustlang.Rustup
cargo install wasm-pack
```

## Advantages

✅ **DRY**: Formula exists in exactly one place  
✅ **Performance**: WASM is faster than JavaScript  
✅ **Type Safety**: Rust's type system catches errors at compile time  
✅ **Single codebase**: No manual TypeScript translation needed  

## Troubleshooting

### WASM not loading in browser
- Check that files exist in `frontend/public/wasm/`
- Hard refresh browser (Ctrl+Shift+R)
- Check browser console for errors

### Build failures
- Ensure Rust and wasm-pack are installed
- Update Rust: `rustup update`
- Clean and rebuild: `cargo clean && wasm-pack build --target web`

### Type errors in frontend
- Regenerate TypeScript declarations
- Check that `frontend/src/wasm.d.ts` matches WASM API

## Python Backend

The Python backend (`backend/src/orbit_synth.py`) is still used for:
- Model training
- Offline batch processing
- Testing and validation

The Rust implementation should match the Python logic exactly to ensure consistency between training and inference.

## Testing Equivalence

To verify Rust and Python produce identical results:

```bash
# TODO: Add equivalence tests
cd wasm-orbit
cargo test
```

This compares outputs from Rust and Python for the same inputs.
