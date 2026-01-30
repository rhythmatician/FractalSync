import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import os from 'os';

// Use a committed sample WAV fixture for E2E to avoid flaky generation and ensure a stable IO contract
import { fileURLToPath } from 'url';
function getSampleWavPath() {
  // Try several candidate locations since test runner CWD may be repo root or frontend/
  const candidates: string[] = [];
  candidates.push(path.resolve(process.cwd(), 'backend', 'data', 'testing', 'sample.wav'));
  candidates.push(path.resolve(process.cwd(), '..', 'backend', 'data', 'testing', 'sample.wav'));
  try {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    candidates.push(path.resolve(__dirname, '..', '..', '..', 'backend', 'data', 'testing', 'sample.wav'));
  } catch (e) {
    // ignore if import.meta isn't available
  }

  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }

  // default to repo-relative path
  return candidates[0];
}

test('visualizer loads model and visualizes audio without console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  await page.goto('/');

  // Wait for model to be loaded and UI to indicate readiness
  await page.waitForSelector('text=✓ Ready', { timeout: 45000 });

  // Use committed sample WAV fixture for a stable test input
  const samplePath = getSampleWavPath();
  if (!fs.existsSync(samplePath)) {
    throw new Error(`Sample WAV not found at ${samplePath}`);
  }

  try {
    // Upload the file
    const fileInput = await page.$('input[type=file]');
    if (!fileInput) {
      throw new Error('File input not found');
    }
    await fileInput.setInputFiles(samplePath);

    // Click Start Visualization
    await page.click('button:has-text("Start Visualization")');

    // Wait for the renderer to draw something. Poll for a non-black pixel for up to 8s.
    const pixelHandle = await page.waitForFunction(() => {
      const c = document.getElementById('visualizerCanvas') as HTMLCanvasElement | null;
      if (!c) return null;

      // Try WebGL first
      const gl = (c.getContext('webgl') || c.getContext('webgl2')) as WebGLRenderingContext | WebGL2RenderingContext | null;
      try {
        if (gl) {
          const pixels = new Uint8Array(4);
          // read a single pixel at (0,0)
          gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          const arr = Array.from(pixels);
          if (!arr.every(v => v === 0)) return arr; // only return if non-black
          return null;
        }
      } catch (e) {
        // ignore
      }

      // Fallback to 2D
      const ctx = c.getContext('2d');
      if (ctx) {
        const img = ctx.getImageData(0, 0, 1, 1).data;
        const arr = Array.from(img);
        if (!arr.every(v => v === 0)) return arr; // only return if non-black
        return null;
      }

      return null;
    }, { timeout: 8000 });

    const pixel = await pixelHandle.jsonValue() as number[] | null;

    // The function returns an array or null
    expect(pixel).not.toBeNull();
    if (pixel) {
      const arr = pixel as number[];
      const allZero = arr.every(v => v === 0);
      expect(allZero).toBe(false);
    }

    // Fail if any console.error was emitted
    expect(errors).toEqual([]);
  } finally {
    // No cleanup needed when using committed fixture
  }
});
