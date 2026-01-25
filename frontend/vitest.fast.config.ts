import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['frontend/src/lib/__tests__/**/*.test.ts'],
    exclude: ['frontend/src/lib/__tests__/featureParity*.test.ts', 'frontend/src/lib/__tests__/featureParityRuntime*.test.ts']
  }
});
