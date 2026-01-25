import { describe, it, expect, vi } from 'vitest';

// Mock orbit runtime to avoid pulling in wasm during unit tests
vi.mock('../orbitRuntime', () => ({
  OrbitRuntime: class {
    async initialize(_opts?: any) { return; }
    updateState(_s: any) { return; }
    step(_dt: number, _gates: number[], _h: number) { return { real: 0.0, imag: 0.0 }; }
    reset(_seed?: number) { return; }
    getLastC() { return { real: 0.0, imag: 0.0 }; }
    getDebug() { return {}; }
    getLobe() { return 1; }
    setLobe(_l: number) { return; }
  }
}));

import { ModelInference } from '../modelInference';
import { MODEL_INPUT_NAME } from '../modelContract';

// Lightweight fake session that captures the last input tensor and returns a deterministic output tensor
class FakeSession {
  public inputNames = [MODEL_INPUT_NAME];
  public outputNames = ['output'];
  public lastInput: Float32Array | null = null;

  async run(feeds: Record<string, any>) {
    const t = feeds[MODEL_INPUT_NAME];
    this.lastInput = t.data as Float32Array;
    // Return a deterministic legacy-style output of length 7
    const out = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    return { output: { data: out, dims: [1, out.length] } } as any;
  }
}

describe('ModelInference unit tests (fake session)', () => {
  it('applies runtime normalization when contract requests it', async () => {
    const model = new ModelInference();

    // Inject metadata requesting runtime normalization
    (model as any).metadata = {
      input_dim: 6,
      input_normalization: { applied_by: 'runtime', mean_field: 'feature_mean', std_field: 'feature_std' },
      feature_mean: [0, 0, 0, 0, 0, 0],
      feature_std: [2, 2, 2, 2, 2, 2],
    } as any;

    const fake = new FakeSession();
    (model as any).session = fake;

    // Provide six features so windowFrames >= 1 and audio-reactive post-processing won't divide by zero
    const input = [1, 1, 1, 1, 1, 1];
    const vp = await model.infer(input);

    // The fake session should have received normalized input (x - mean) / std = 0.5 for each entry
    expect(fake.lastInput).not.toBeNull();
    expect(Array.from(fake.lastInput!)).toEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

    // The returned visual params map from the deterministic output; since legacy model mapping uses
    // the first 7 outputs directly, check a few expected fields derived from our stub output
    // After post-processing: juliaReal transformed by (x*0.6)%1.4 - 0.7
    const expectedJuliaReal = (1 * 0.6) % 1.4 - 0.7; // = -0.1
    expect(vp.juliaReal).toBeCloseTo(expectedJuliaReal);
    // colorHue wraps modulo 1.0
    expect(vp.colorHue).toBeCloseTo(3 % 1.0);
    // zoom is clamped and scaled: zoom = clamp(params[5]*2+1.5, 1.5, 4.0)
    expect(vp.zoom).toBeCloseTo(4.0);
  });

  it("doesn't normalize when contract indicates model applies normalization", async () => {
    const model = new ModelInference();

    (model as any).metadata = {
      input_dim: 6,
      input_normalization: { applied_by: 'model', mean_field: 'feature_mean', std_field: 'feature_std' },
      feature_mean: [0, 0, 0, 0, 0, 0],
      feature_std: [1, 1, 1, 1, 1, 1],
    } as any;

    const fake = new FakeSession();
    (model as any).session = fake;

    // use six features to match input_dim
    const input = [1.23, 4.56, -2.0, 0.1, -0.2, 0.5];
    await model.infer(input);

    // When the contract says the model applies normalization, ModelInference should NOT normalize
    // Float32Array conversion may introduce small rounding errors; compare with tolerance
    const got = Array.from(fake.lastInput!);
    for (let i = 0; i < input.length; i++) {
      expect(got[i]).toBeCloseTo(input[i], 6);
    }
  });
});