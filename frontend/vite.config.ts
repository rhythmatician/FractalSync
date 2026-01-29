import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'fs'
import { join } from 'path'

// Auto-copy ONNX Runtime WASM file (canonical: single-thread, non-SIMD) on startup
const preferredWasm = 'node_modules/onnxruntime-web/dist/ort-wasm.wasm';
const fallbackWasm = 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm';
const wasmTarget = 'public/ort-wasm.wasm';
if (existsSync(preferredWasm)) {
  copyFileSync(preferredWasm, wasmTarget);
} else {
  console.warn('[vite] preferred WASM artifact not found; please install onnxruntime-web or provide "ort-wasm.wasm" in node_modules/dist/');
}

const wasmOrbitSource = '../wasm-orbit/pkg';
const wasmOrbitTarget = 'public/wasm';
if (existsSync(wasmOrbitSource)) {
  mkdirSync(wasmOrbitTarget, { recursive: true });
  for (const entry of readdirSync(wasmOrbitSource)) {
    const sourcePath = join(wasmOrbitSource, entry);
    if (statSync(sourcePath).isFile()) {
      copyFileSync(sourcePath, join(wasmOrbitTarget, entry));
    }
  }
} else {
  console.warn('[vite] wasm-orbit pkg not found; run `wasm-pack build --target web` in wasm-orbit before frontend dev/build.');
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      external: ['/wasm/orbit_synth_wasm.js']
    }
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
