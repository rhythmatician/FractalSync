import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import type { AnalysisTick } from '../analysisTimebase';
import { ModelInference } from '../modelInference';
import { setWasmModuleForTesting } from '../orbitSynthesizer';

let compiledWasm: unknown;

beforeAll(async () => {
  const generated = await import('../../../../wasm-orbit/pkg/orbit_synth_wasm.js');
  const wasmPath = resolve(process.cwd(), '../wasm-orbit/pkg/orbit_synth_wasm_bg.wasm');
  generated.initSync({ module: readFileSync(wasmPath) });
  compiledWasm = generated;
});

afterEach(() => setWasmModuleForTesting(null));

describe('ModelInference with compiled model_io WASM', () => {
  it('returns the browser Complex shape from a legacy ONNX output', async () => {
    setWasmModuleForTesting(compiledWasm as never);
    const inference = new ModelInference();
    inference.setAudioReactivePostProcessing(false);
    const fakeSession = {
      run: async () => ({
        visual_parameters: {
          data: Float32Array.from([0.5, -0.5, 0.2, 0.4, 0.8, 0.6, 0.9]),
        },
      }),
    };
    (inference as unknown as { session: unknown }).session = fakeSession;

    const tick: AnalysisTick = {
      features: new Array(60).fill(0),
      sampleIndex: 1024,
      timeSeconds: 1024 / 48000,
      dtSeconds: 1024 / 48000,
      streamEpoch: 0,
    };
    const visual = await inference.infer(tick);

    expect(visual.juliaSeed.real).toBeCloseTo(-0.4);
    expect(visual.juliaSeed.imag).toBeCloseTo(-1);
    expect(visual.colorHue).toBeCloseTo(0.2);
    expect(visual.colorSat).toBeCloseTo(0.82);
    expect(visual.colorBright).toBeCloseTo(0.9);
    expect(visual.zoom).toBeCloseTo(2.7);
    expect(visual.speed).toBeCloseTo(0.7);
  });
});
