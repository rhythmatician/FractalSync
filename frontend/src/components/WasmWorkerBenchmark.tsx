import { useState } from 'react';
import { WasmFeatureWorkerClient } from '../lib/wasm/wasm_feature_worker_client';

export default function WasmWorkerBenchmark() {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState<string | null>(null);

  async function runBench() {
    setStatus('initializing');
    const client = new WasmFeatureWorkerClient();
    const sr = 22050, hop = 256, nfft = 1024;
    await client.init({ sr, hop, nfft });

    setStatus('generating audio');
    // create synthetic audio long enough for multiple frames
    const windowFrames = 6;
    const len = (windowFrames - 1) * hop + nfft + 0; // exact
    const audio = new Float32Array(len);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * (i / sr)) * 0.6;

    setStatus('running benchmark');
    const runs = 20;
    const times: number[] = [];
    for (let r = 0; r < runs; r++) {
      const aCopy = audio.slice(); // worker transfer invalidates buffer
      const t0 = performance.now();
      const _features = await client.extract(aCopy, windowFrames);
      void _features; // used to satisfy TS (bench doesn't assert features)
      const t1 = performance.now();
      times.push(t1 - t0);
    }
    client.terminate();
    const avg = times.reduce((s, v) => s + v, 0) / times.length;
    setResult(`avg=${avg.toFixed(3)}ms over ${runs} runs; samples=${times.map(v=>v.toFixed(2)).join(',')}`);
    setStatus('done');
  }

  return (
    <div style={{ padding: 16 }}>
      <h3>WASM Feature Extractor Worker Benchmark</h3>
      <p>Status: {status}</p>
      <button onClick={runBench}>Run benchmark</button>
      {result && <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre>}
    </div>
  );
}
