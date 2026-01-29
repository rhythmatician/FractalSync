import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

// Use onnxruntime-node for Node integration smoke test
import * as ort from 'onnxruntime-node';

describe('onnxruntime-node smoke test', () => {
  it('loads model and runs an inference', async () => {
    const modelPath = path.resolve(process.cwd(), 'models_i_like', 'model_orbit_control_20260128_002625.onnx');
    expect(fs.existsSync(modelPath)).toBe(true);

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