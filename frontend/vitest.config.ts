import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    testTimeout: 60_000,   // per test
    hookTimeout: 60_000,   // before/after hooks
    environment: 'jsdom',
    setupFiles: ['./src/lib/__tests__/setup.ts'],
    globals: true,
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'src/lib/__tests__/']
    }
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    }
  }
});
import { defineConfig } from "vitest/config";

