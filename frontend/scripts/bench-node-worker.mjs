import { parentPort } from 'worker_threads';
import wasmInit, { FeatureExtractor } from '../src/wasm/orbit_synth_wasm.js';
import fs from 'fs/promises';

// stub fetch in worker
global.fetch = async (_url) => ({ arrayBuffer: async () => fs.readFile(new URL('../src/wasm/orbit_synth_wasm_bg.wasm', import.meta.url)) });

let fe = null;

parentPort.on('message', async (msg) => {
    if (msg.type === 'init') {
        await wasmInit();
        fe = new FeatureExtractor(msg.sr, msg.hop, msg.nfft, false, false);
        parentPort.postMessage({ type: 'inited' });
        return;
    }

    if (msg.type === 'extract') {
        const audio = new Float32Array(msg.audio);
        const nested = fe.extract_windowed_features(audio, msg.window_frames);
        // flatten
        const flat = [];
        for (let i = 0; i < nested.length; i++) {
            const row = nested[i];
            for (let v of row) flat.push(Number(v));
        }
        parentPort.postMessage({ type: 'result', id: msg.id, features: flat });
    }
});
