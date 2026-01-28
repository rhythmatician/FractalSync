import { defineConfig } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// ESM-friendly __dirname shim
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : [['list']],
  use: {
    headless: true,
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry'
  },
  webServer: {
    command: 'npm run dev',
    cwd: path.resolve(__dirname),
    url: 'http://localhost:3000',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI
  },
  globalSetup: './playwright-global-setup.cjs',
  globalTeardown: './playwright-global-teardown.cjs'
});
