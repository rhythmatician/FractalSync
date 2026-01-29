import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import os from 'os';

// Helper: create a short silent WAV file so we can upload it in the test
function createSilentWav(filePath: string, durationSec = 0.5, sampleRate = 8000) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const numSamples = Math.floor(durationSec * sampleRate);
  const byteRate = sampleRate * numChannels * bytesPerSample;
  const blockAlign = numChannels * bytesPerSample;

  const dataSize = numSamples * numChannels * bytesPerSample;
  const buffer = Buffer.alloc(44 + dataSize);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);

  // fmt subchunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16); // Subchunk1Size
  buffer.writeUInt16LE(1, 20); // AudioFormat PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bytesPerSample * 8, 34);

  // data subchunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  // Fill samples with silence (zeros) - already zeroed by Buffer.alloc
  fs.writeFileSync(filePath, buffer);
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

  // Create a temporary silent WAV
  const tmpFile = path.join(os.tmpdir(), `sample-${Date.now()}.wav`);
  createSilentWav(tmpFile);

  try {
    // Upload the file
    const fileInput = await page.$('input[type=file]');
    if (!fileInput) {
      throw new Error('File input not found');
    }
    await fileInput.setInputFiles(tmpFile);

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
    // Clean up temporary file
    try {
      fs.unlinkSync(tmpFile);
    } catch (e) {
      // Ignore cleanup errors
    }
  }
});
