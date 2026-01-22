/**
 * Vitest setup file for browser API mocks
 * Provides mocks for Web Audio API and other browser APIs needed in tests
 */

import { vi } from 'vitest';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Mock Web Audio API
class MockAudioContext {
  private analyser: MockAnalyserNode;

  constructor() {
    this.analyser = new MockAnalyserNode();
  }

  createAnalyser(): MockAnalyserNode {
    return this.analyser;
  }

  get state(): string {
    return 'running';
  }

  get sampleRate(): number {
    return 44100;
  }

  get currentTime(): number {
    return 0;
  }

  get destination(): any {
    return {};
  }

  createGain(): any {
    return { connect: () => {}, disconnect: () => {}, gain: { value: 1 } };
  }

  createMediaElementAudioSource(): any {
    return { connect: () => {}, disconnect: () => {} };
  }

  createMediaStreamAudioSource(): any {
    return { connect: () => {}, disconnect: () => {} };
  }

  createBiquadFilter(): any {
    return { connect: () => {}, disconnect: () => {}, type: 'lowpass', frequency: { value: 350 } };
  }
}

class MockAnalyserNode {
  private dataArray: Uint8Array = new Uint8Array(2048);
  private frequencyData: Uint8Array = new Uint8Array(2048);

  constructor() {
    // Fill with some test data
    for (let i = 0; i < this.dataArray.length; i++) {
      this.dataArray[i] = Math.floor(Math.random() * 255);
      this.frequencyData[i] = Math.floor(Math.random() * 255);
    }
  }

  get fftSize(): number {
    return 2048;
  }

  set fftSize(_value: number) {
    // no-op for mock
  }

  get frequencyBinCount(): number {
    return 1024;
  }

  getByteFrequencyData(array: Uint8Array): void {
    array.set(this.frequencyData);
  }

  getByteTimeDomainData(array: Uint8Array): void {
    array.set(this.dataArray);
  }

  getFloatFrequencyData(array: Float32Array): void {
    for (let i = 0; i < Math.min(array.length, this.frequencyData.length); i++) {
      array[i] = this.frequencyData[i] / 255;
    }
  }

  getFloatTimeDomainData(array: Float32Array): void {
    for (let i = 0; i < Math.min(array.length, this.dataArray.length); i++) {
      array[i] = (this.dataArray[i] - 128) / 128;
    }
  }

  connect(): void {
    // no-op for mock
  }

  disconnect(): void {
    // no-op for mock
  }
}

// Set up globalThis AudioContext for jsdom
if (!(globalThis as any).AudioContext) {
  (globalThis as any).AudioContext = MockAudioContext;
}

if (!(globalThis as any).webkitAudioContext) {
  (globalThis as any).webkitAudioContext = MockAudioContext;
}

// Mock fetch for fixture paths in tests
const originalFetch = globalThis.fetch;
globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();

  // Handle relative fixture paths
  if (url.startsWith('./src/lib/__tests__/fixtures/')) {
    const fixturePath = resolve(__dirname, url.replace('./src/lib/__tests__/', ''));
    try {
      const content = readFileSync(fixturePath, 'utf-8');
      return {
        ok: true,
        json: async () => JSON.parse(content),
      } as any;
    } catch (_error) {
      return {
        ok: false,
      } as any;
    }
  }

  return originalFetch(input, init);
}) as any;

// Mock Blob and URL.createObjectURL for ONNX Runtime
if (typeof (globalThis as any).Blob === 'undefined') {
  (globalThis as any).Blob = class Blob {
    constructor(public parts: any[], public options: any = {}) {}
  };
}

if (!URL.createObjectURL) {
  (URL as any).createObjectURL = (blob: any) => `blob:${Math.random()}`;
  (URL as any).revokeObjectURL = () => {};
}

// Mock WASM worker loading for ONNX Runtime
vi.stubGlobal(
  'Worker',
  class MockWorker {
    url: string;
    onmessage: ((event: any) => void) | null = null;

    constructor(stringUrl: string) {
      this.url = stringUrl;
    }

    postMessage(_msg: any) {
      // no-op for mock
    }

    terminate() {
      // no-op for mock
    }
  }
);
