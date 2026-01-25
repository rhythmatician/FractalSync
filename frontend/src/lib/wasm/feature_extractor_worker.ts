/* Worker: loads wasm FeatureExtractor and handles extract requests

Protocol:
- message { type: 'init', sr, hop, nfft, include_delta?, include_delta_delta? }
  -> respond { type: 'inited', ok: true }
- message { type: 'extract', id, audio: ArrayBuffer, window_frames }
  -> respond { type: 'result', id, features: Float64Array (flattened, transferred) }
*/

import wasmInit, { FeatureExtractor } from "../../wasm/orbit_synth_wasm.js";

let fe: any = null;

self.addEventListener('message', async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      // initialize wasm; the bundler will resolve wasmUrl in module
      await (wasmInit as any)();
      // create the extractor instance
      fe = new FeatureExtractor(msg.sr, msg.hop, msg.nfft, !!msg.include_delta, !!msg.include_delta_delta);
      (self as any).postMessage({ type: 'inited', ok: true });
      return;
    }

    if (msg.type === 'extract') {
      if (!fe) throw new Error('not-initialized');
      // Received audio ArrayBuffer; convert to Float32Array
      const audio = new Float32Array(msg.audio);
      // Call wasm extractor (returns nested array of numbers)
      const nested = fe.extract_windowed_features(audio, msg.window_frames) as any[];
      // Flatten nested into Float64Array for stable transfer
      const flat: number[] = [];
      for (let i = 0; i < nested.length; i++) {
        const row = nested[i];
        for (let v of row as any) flat.push(Number(v));
      }

      const buf = new Float64Array(flat);
      (self as any).postMessage({ type: 'result', id: msg.id, features: buf.buffer }, [buf.buffer]);
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg && msg.id, error: String(err) });
  }
});
