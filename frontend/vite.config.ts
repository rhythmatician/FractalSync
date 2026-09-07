import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { copyFileSync, existsSync, mkdirSync } from 'fs'

// Unit tests import src/wasm; the app loads /wasm as an untransformed asset.
// Always serve the same compiled binding in both places.
const runtimeWasmDir = resolve(__dirname, 'src/wasm');
const publicWasmDir = resolve(__dirname, 'public/wasm');
mkdirSync(publicWasmDir, { recursive: true });
for (const file of ['orbit_synth_wasm.js', 'orbit_synth_wasm_bg.wasm']) {
  const source = resolve(runtimeWasmDir, file);
  if (!existsSync(source)) throw new Error(`Missing ${source}. Build wasm-orbit into frontend/src/wasm first.`);
  copyFileSync(source, resolve(publicWasmDir, file));
}

// Auto-copy ONNX Runtime WASM file (canonical: single-thread, non-SIMD) on startup
const preferredWasm = 'node_modules/onnxruntime-web/dist/ort-wasm.wasm';
const fallbackWasm = 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm';
const wasmTarget = 'public/ort-wasm.wasm';
if (existsSync(preferredWasm)) {
  copyFileSync(preferredWasm, wasmTarget);
} else {
  console.warn('[vite] preferred WASM artifact not found; please install onnxruntime-web or provide "ort-wasm.wasm" in node_modules/dist/');
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // Standalone #111 debug cockpit (no model, no audio; wasm physics +
        // Three.js third-person manifold view).
        debugCockpit: resolve(__dirname, 'debugCockpit.html'),
      },
    },
  },
  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  preview: {
    port: 4173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
})
