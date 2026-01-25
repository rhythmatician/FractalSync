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
// Always override to ensure tests use the Mock implementation regardless of environment
const _MockAudioContextCtor = function (..._args: any[]) {
  // allow invoked with or without `new`
  return new (MockAudioContext as any)();
} as any;
_MockAudioContextCtor.prototype = MockAudioContext.prototype;
(globalThis as any).AudioContext = _MockAudioContextCtor;
(globalThis as any).webkitAudioContext = _MockAudioContextCtor;

// Mock fetch for fixture paths in tests
const originalFetch = globalThis.fetch;
const TEST_NO_NETWORK = (process.env.TEST_NO_NETWORK === '1' || process.env.TEST_NO_NETWORK === 'true');
if (TEST_NO_NETWORK) {
  // Disable any network calls during tests
  (globalThis as any).fetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
    throw new Error('Network disabled during tests (TEST_NO_NETWORK=1)');
  }) as any;
} else {
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
}

// Mock Blob and URL.createObjectURL for ONNX Runtime
if (typeof (globalThis as any).Blob === 'undefined') {
  (globalThis as any).Blob = class Blob {
    constructor(public parts: any[], public options: any = {}) {}
  };
}

if (!URL.createObjectURL) {
  // Return a relative path so onnxruntime-web's worker filename validation passes
  // (it rejects blob: URLs). The test harness uses a mocked Worker so no actual
  // file is required.
  (URL as any).createObjectURL = (_blob: any) => './__onnx_worker_stub.js';
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

// Mock onnxruntime-web's InferenceSession.create to avoid heavy WASM/worker initialization during unit tests.
// Behavior is dynamic: if TEST_NO_NETWORK=1 when tests run, `create` will reject so tests avoid network/WASM work.
vi.mock('onnxruntime-web', async (importOriginal) => {
  // Import the real module by default so we preserve all exports needed by normal tests
  const actual = await importOriginal();
  // If test-fast / no-network mode is active, override the heavy path
  if (process.env.TEST_NO_NETWORK === '1' || process.env.TEST_NO_NETWORK === 'true') {
    return {
      ...(actual as any),
      InferenceSession: {
        create: async (_src: any) => {
          throw new Error('onnxruntime InferenceSession.create blocked by TEST_NO_NETWORK in tests');
        },
      },
    } as any;
  }
  // otherwise return the actual module unmodified
  return actual; 
});

// Mark that the ort mock is installed so tests can assert it (avoids importing the real package in fast mode)
(globalThis as any).__ORT_MOCK_INSTALLED = true;
// Also set on common Node globals to be sure it's visible in all runner contexts
try { (global as any).__ORT_MOCK_INSTALLED = true; } catch (_) { /* ignore */ }
