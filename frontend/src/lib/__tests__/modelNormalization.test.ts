import { describe, it, expect, vi } from 'vitest';
import { ModelInference } from '../modelInference';
import { MODEL_OUTPUT_NAME, INPUT_DIM, DEFAULT_K_BANDS } from '../modelContract';

function makeCapturingMockSession(kBands: number) {
  let lastInput: Float32Array | null = null;
  return {
    run: async (feeds: any) => {
      const inData = feeds["audio_features"].data as Float32Array;
      lastInput = inData;
      const outDim = 3 + kBands;
      const data = new Float32Array(outDim).fill(0);
      return { [MODEL_OUTPUT_NAME]: { data, dims: [1, outDim] } };
    },
    getLastInput: () => lastInput,
  } as any;
}

vi.mock('../orbitRuntime', () => ({ OrbitRuntime: vi.fn().mockImplementation(() => ({ initialize: async () => {}, updateState: () => {}, step: () => ({ real: 0, imag: 0 }), reset: () => {}, getDebug: () => null })), }));

describe('ModelInference normalization wiring', () => {
  it('applies runtime normalization when contract requests it and uses configured mean/std fields', async () => {
    const model = new ModelInference();
    const kBands = DEFAULT_K_BANDS;
    const session = makeCapturingMockSession(kBands);
    (model as any).session = session;

    // Provide metadata with explicit normalization policy + alternate field names
    (model as any).metadata = {
      input_dim: INPUT_DIM,
      output_dim: 3 + kBands,
      parameter_names: new Array(3 + kBands).fill('x'),
      k_bands: kBands,
      input_normalization: { type: 'zscore', applied_by: 'runtime', mean_field: 'feature_mean', std_field: 'feature_std' },
      feature_mean: new Array(INPUT_DIM).fill(0).map((_, i) => i * 1.0),
      feature_std: new Array(INPUT_DIM).fill(1).map((_, i) => (i % 2 === 0 ? 2.0 : 1.0)),
    } as any;
    (model as any).isControlModel = true;

    const feats = new Array(INPUT_DIM).fill(0).map((_, i) => i + 1); // [1,2,3,...]

    const out = await model.infer(feats);
    expect(out).toBeDefined();

    // Validate that the session received normalized inputs
    const lastInput = (session as any).getLastInput() as Float32Array;
    expect(lastInput).toBeInstanceOf(Float32Array);
    // Check a couple of entries are normalized as (x - mean)/std
    // element 0: (1 - 0)/2 = 0.5
    expect(Math.abs(lastInput[0] - 0.5)).toBeLessThan(1e-6);
    // element 1: (2 - 1)/1 = 1
    expect(Math.abs(lastInput[1] - 1.0)).toBeLessThan(1e-6);
  });

  it('does not normalize when contract indicates model applies normalization', async () => {
    const model = new ModelInference();
    const kBands = DEFAULT_K_BANDS;
    const session = makeCapturingMockSession(kBands);
    (model as any).session = session;

    (model as any).metadata = {
      input_dim: INPUT_DIM,
      output_dim: 3 + kBands,
      parameter_names: new Array(3 + kBands).fill('x'),
      k_bands: kBands,
      input_normalization: { type: 'zscore', applied_by: 'model' },
      feature_mean: new Array(INPUT_DIM).fill(0),
      feature_std: new Array(INPUT_DIM).fill(1),
    } as any;
    (model as any).isControlModel = true;

    const feats = new Array(INPUT_DIM).fill(1.23);
    await model.infer(feats);
    const lastInput = (session as any).getLastInput() as Float32Array;
    // Should be raw input (not normalized)
    expect(Math.abs(lastInput[0] - 1.23)).toBeLessThan(1e-6);
  });
});
