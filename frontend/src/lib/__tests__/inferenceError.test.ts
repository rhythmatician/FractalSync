import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock OrbitRuntime module to avoid pulling in WASM bindings during unit tests
vi.mock('../orbitRuntime', () => {
  return {
    OrbitRuntime: vi.fn().mockImplementation(() => ({
      initialize: async () => {},
      updateState: () => {},
      step: (_dt: number, _bandGates: number[], _h: number) => ({ real: 0, imag: 0 }),
      reset: (_seed: number) => {},
      getDebug: () => null,
    })),
  };
});

import { ModelInference } from '../modelInference';
import { MODEL_OUTPUT_NAME, INPUT_DIM, DEFAULT_K_BANDS } from '../modelContract';

// Minimal mock session to satisfy ModelInference inference path
function makeMockSession(kBands: number) {
  return {
    run: async (_feeds: any) => {
      const outDim = 3 + kBands;
      const data = new Float32Array(outDim).fill(0);
      return { [MODEL_OUTPUT_NAME]: { data, dims: [1, outDim] } };
    },
  } as any;
}

class ThrowingOrbitRuntime {
  callCount = 0;
  throwOnCalls = new Set<number>();

  updateState(_params: any) {}
  getDebug() {
    return null;
  }

  // step will throw on configured call numbers to simulate wasm borrow/ownership errors
  step(_dt: number, _bandGates: number[], _h: number) {
    this.callCount += 1;
    if (this.throwOnCalls.has(this.callCount)) {
      throw new Error('recursive use of an object detected which would lead to unsafe aliasing in rust');
    }
    return { real: 0.1 * this.callCount, imag: 0.2 * this.callCount };
  }
}

describe('Inference error handling', () => {
  let model: ModelInference;
  const kBands = DEFAULT_K_BANDS;

  beforeEach(() => {
    model = new ModelInference();
    // Inject mock session
    (model as any).session = makeMockSession(kBands);
    // Make it appear as a control model
    (model as any).metadata = {
      input_dim: INPUT_DIM,
      output_dim: 3 + kBands,
      parameter_names: new Array(3 + kBands).fill('x'),
      k_bands: kBands,
    };
    (model as any).isControlModel = true;
  });

  it('handles orbit runtime step throwing gracefully and records diagnostics', async () => {
    const runtime = new ThrowingOrbitRuntime();
    runtime.throwOnCalls.add(2); // throw on second call
    (model as any).orbitRuntime = runtime;

    const features = new Array(INPUT_DIM).fill(0.1);

    // first call should succeed
    await expect(model.infer(features)).resolves.toBeDefined();

    // second call (runtime.step would throw) should be handled by runtime and still succeed
    await expect(model.infer(features)).resolves.toBeDefined();

    // diagnostics should reflect the step error was recorded (inference-level)
    expect((model as any).lastInferenceFailures).toBeGreaterThanOrEqual(1);

    // third call should succeed again
    await expect(model.infer(features)).resolves.toBeDefined();
  });

  it('records intermittent runtime step failures but continues to return results', async () => {
    const runtime = new ThrowingOrbitRuntime();
    // throw on 2,3,5 to simulate intermittent failures
    [2, 3, 5].forEach(n => runtime.throwOnCalls.add(n));
    (model as any).orbitRuntime = runtime;

    const features = new Array(INPUT_DIM).fill(0.2);

    let success = 0;

    for (let i = 0; i < 6; i++) {
      const out = await model.infer(features);
      expect(out).toBeDefined();
      success += 1;
    }

    // All calls should succeed (failures are recorded in diagnostics)
    expect(success).toBe(6);

    // inference-level failure count reflects 3 intermittent runtime throws
    expect((model as any).lastInferenceFailures).toBe(3);
  });
});
