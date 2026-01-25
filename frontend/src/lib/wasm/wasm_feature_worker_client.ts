// Main-thread client wrapper for FEATURE_EXTRACTOR worker
// Usage:
// const client = new WasmFeatureWorkerClient();
// await client.init({ sr, hop, nfft });
// const features = await client.extract(audioFloat32Array, window_frames);

export class WasmFeatureWorkerClient {
  private worker: Worker;
  private pending = new Map<number, (v: any) => void>();
  private counter = 1;

  constructor() {
    // Vite-style worker bundling (module worker)
    // The bundler will rewrite `new URL()` and provide a worker entry
    this.worker = new Worker(new URL('./feature_extractor_worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg.type === 'inited') return; // ignore
      if (msg.type === 'result') {
        const cb = this.pending.get(msg.id);
        if (cb) {
          // features comes as transferred ArrayBuffer, wrap as Float64Array
          const arr = new Float64Array(msg.features);
          cb(arr);
          this.pending.delete(msg.id);
        }
      }
      if (msg.type === 'error') {
        const cb = this.pending.get(msg.id);
        if (cb) {
          cb(Promise.reject(new Error(msg.error)));
          this.pending.delete(msg.id);
        }
      }
    };
  }

  async init(opts: { sr: number; hop: number; nfft: number; include_delta?: boolean; include_delta_delta?: boolean; }) {
    return new Promise<void>((resolve) => {
      const onInit = (ev: MessageEvent) => {
        const msg = ev.data;
        if (msg && msg.type === 'inited') {
          this.worker.removeEventListener('message', onInit as any);
          resolve();
        }
      };
      this.worker.addEventListener('message', onInit as any);
      this.worker.postMessage({ type: 'init', ...opts });
    });
  }

  async extract(audio: Float32Array, window_frames: number): Promise<Float64Array> {
    const id = this.counter++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve as any);
      // Transfer audio buffer to worker as ArrayBuffer copy (Float32 -> underlying buffer)
      this.worker.postMessage({ type: 'extract', id, audio: audio.buffer, window_frames }, [audio.buffer]);
      // Note: transferring audio.buffer invalidates it on caller side; clone if needed
    });
  }

  terminate() { this.worker.terminate(); }
}
