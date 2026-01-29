import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, existsSync } from 'fs'

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
