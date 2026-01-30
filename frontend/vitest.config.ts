import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/lib/__tests__/setup.ts'],
    globals: true,
    exclude: ['node_modules/**'],
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
