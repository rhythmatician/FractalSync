import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, existsSync } from 'fs'
import { join } from 'path'

// Auto-copy ONNX Runtime WASM files on startup
const wasmSource = 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm';
const wasmTarget = 'public/ort-wasm-simd-threaded.wasm';
if (existsSync(wasmSource)) {
  copyFileSync(wasmSource, wasmTarget);
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
