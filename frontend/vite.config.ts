import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, existsSync, mkdirSync } from 'fs'
import { join } from 'path'

// Copy ONNX Runtime WASM files to public directory
const copyWasmFiles = () => {
  const files = [
    'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd.wasm',
  ];

  const sourceDir = 'node_modules/onnxruntime-web/dist';
  const targetDir = 'public';

  if (!existsSync(targetDir)) {
    mkdirSync(targetDir, { recursive: true });
  }

  files.forEach(file => {
    const sourcePath = join(sourceDir, file);
    const targetPath = join(targetDir, file);
    
    if (existsSync(sourcePath)) {
      try {
        copyFileSync(sourcePath, targetPath);
        console.log(`Copied ${file} to public directory`);
      } catch (err) {
        console.warn(`Failed to copy ${file}:`, err);
      }
    }
  });
};

// Run on startup
copyWasmFiles();

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
