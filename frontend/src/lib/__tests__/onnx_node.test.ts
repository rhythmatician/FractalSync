import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

// Use onnxruntime-node for Node integration smoke test
import * as ort from 'onnxruntime-node';

describe('onnxruntime-node smoke test', () => {
  it('loads model and runs an inference', async () => {
    // Prefer a committed model fixture in frontend/tests/fixtures
    let modelPath = path.resolve(process.cwd(), 'frontend', 'tests', 'fixtures', 'model.onnx');

    // If no committed fixture, try the backend checkpoint export path
    if (!fs.existsSync(modelPath)) {
      const fallback = path.resolve(process.cwd(), 'backend', 'checkpoints', 'model.onnx');
      if (fs.existsSync(fallback)) {
        modelPath = fallback;
        console.log('Using fallback model from backend checkpoints:', modelPath);
      } else {
        console.warn('No ONNX model available for Node integration test; skipping test');
        return; // skip the test gracefully
      }
    }

    const metadataPath = modelPath.replace('.onnx', '.onnx_metadata.json');
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

    const session = await ort.InferenceSession.create(modelPath);
    expect(session).toBeDefined();

    // Use detected IO names and create dummy input according to metadata
    const inputName = (session.inputNames && session.inputNames[0]) || 'input';
    const outputName = (session.outputNames && session.outputNames[0]) || 'output';

    const inputDim = metadata.input_dim || (metadata.input_shape && metadata.input_shape[1]) || 60;
    const outDim = metadata.output_dim || 7;

    const tensor = new ort.Tensor('float32', new Float32Array(inputDim), [1, inputDim]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = tensor;

    const results = await session.run(feeds as any);
    expect(results).toBeDefined();
    expect(results[outputName]).toBeDefined();

    const arr = Array.from((results[outputName].data as Float32Array));
    expect(arr.length).toBe(outDim);
  }, 30000);
});