#!/usr/bin/env node
import fs from 'fs/promises';
import wasmInit, { FeatureExtractor } from '../src/wasm/orbit_synth_wasm.js';
import { Worker } from 'worker_threads';

// stub fetch for Node importing of wasm initializer
global.fetch = async (_url) => ({ arrayBuffer: async () => fs.readFile(new URL('../src/wasm/orbit_synth_wasm_bg.wasm', import.meta.url)) });

async function benchDirect(sr = 22050, hop = 256, nfft = 1024, windowFrames = 1, runs = 50) {
    await wasmInit();
    const fe = new FeatureExtractor(sr, hop, nfft, false, false);
    const len = (windowFrames - 1) * hop + nfft;
    const audio = new Float32Array(len);
    for (let i = 0; i < len; i++) audio[i] = Math.sin(2 * Math.PI * 440 * (i / sr)) * 0.6;

    const times = [];
    for (let r = 0; r < runs; r++) {
        const t0 = process.hrtime.bigint();
        const nested = fe.extract_windowed_features(audio, windowFrames);
        const t1 = process.hrtime.bigint();
        times.push(Number(t1 - t0) / 1e6);
    }
    const avg = times.reduce((s, v) => s + v, 0) / times.length;
    console.log(`direct: avg ${avg.toFixed(3)} ms/frame over ${runs} runs`);
    return { avg, times };
}

async function benchWorker(sr = 22050, hop = 256, nfft = 1024, windowFrames = 1, runs = 50) {
    const worker = new Worker(new URL('./bench-node-worker.mjs', import.meta.url));
    await new Promise((resolve) => {
        worker.once('message', (m) => { if (m.type === 'inited') resolve(); });
        worker.postMessage({ type: 'init', sr, hop, nfft });
    });

    const len = (windowFrames - 1) * hop + nfft;
    const audio = new Float32Array(len);
    for (let i = 0; i < len; i++) audio[i] = Math.sin(2 * Math.PI * 440 * (i / sr)) * 0.6;

    function extractOnce(id) {
        return new Promise((resolve) => {
            const onMsg = (m) => {
                if (m.type === 'result' && m.id === id) {
                    worker.off('message', onMsg);
                    resolve(m.features);
                }
            };
            worker.on('message', onMsg);
            worker.postMessage({ type: 'extract', id, audio: audio.buffer, window_frames: windowFrames });
        });
    }

    const times = [];
    for (let r = 0; r < runs; r++) {
        const t0 = process.hrtime.bigint();
        await extractOnce(r + 1);
        const t1 = process.hrtime.bigint();
        times.push(Number(t1 - t0) / 1e6);
    }
    const avg = times.reduce((s, v) => s + v, 0) / times.length;
    console.log(`worker roundtrip: avg ${avg.toFixed(3)} ms over ${runs} runs`);
    worker.terminate();
    return { avg, times };
}

(async () => {
    console.log('Running direct benchmark...');
    await benchDirect(22050, 256, 1024, 1, 50);
    console.log('Running worker roundtrip benchmark...');
    await benchWorker(22050, 256, 1024, 1, 50);
})();