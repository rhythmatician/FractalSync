import ort from 'onnxruntime-web';
import fs from 'fs/promises';

(async () => {
  const path = 'backend/checkpoints/model_orbit_control_20260127_234646.onnx';
  const b = await fs.readFile(path);
  console.log('loaded file size', b.length);
  try {
    // pass buffer to web runtime
    const session = await ort.InferenceSession.create(new Uint8Array(b), { executionProviders: ['wasm'] });
    console.log('session created (web build)');
    console.log('inputs', session.inputNames);
  } catch (e) {
    console.error('onnxruntime-web failed:', e);
  }
})();