import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { beforeAll, describe, expect, it } from 'vitest';

type ModelIoWasm = {
  decodeLegacyVisualJson(values: Float64Array, audioReactive: boolean, rms: number, onset: number): string;
  decodeOrbitControlJson(values: Float64Array, kBands: number): string;
  legacyOrbitDriveInputsJson(rms: number, onset: number): string;
  legacyVisualExportRangesJson(): string;
  orbitControlSchemaJson(kBands: number): string;
};

let modelIo: ModelIoWasm;

beforeAll(async () => {
  const generated = await import('../../../../wasm-orbit/pkg/orbit_synth_wasm.js');
  const wasmPath = resolve(process.cwd(), '../wasm-orbit/pkg/orbit_synth_wasm_bg.wasm');
  generated.initSync({ module: readFileSync(wasmPath) });
  modelIo = generated as unknown as ModelIoWasm;
});

describe('compiled Rust model_io WASM authority', () => {
  it('exports and decodes the legacy orbit contract', () => {
    const schema = JSON.parse(modelIo.orbitControlSchemaJson(2));
    expect(schema.map((field: { name: string }) => field.name)).toEqual([
      's_target', 'alpha', 'omega_scale', 'band_gate_0', 'band_gate_1',
    ]);
    expect(JSON.parse(modelIo.decodeOrbitControlJson(
      new Float64Array([1.2, 0.4, 0.8, 0.1, 0.9]), 2,
    ))).toEqual({ sTarget: 1.2, alpha: 0.4, omegaScale: 0.8, bandGates: [0.1, 0.9] });

    const zeroBandSchema = JSON.parse(modelIo.orbitControlSchemaJson(0));
    expect(zeroBandSchema.map((field: { name: string }) => field.name)).toEqual([
      's_target', 'alpha', 'omega_scale',
    ]);
    expect(JSON.parse(modelIo.decodeOrbitControlJson(
      new Float64Array([1.2, 0.4, 0.8]), 0,
    ))).toEqual({ sTarget: 1.2, alpha: 0.4, omegaScale: 0.8, bandGates: [] });
  });

  it('runs product projections and preserves historical export metadata', () => {
    expect(JSON.parse(modelIo.legacyOrbitDriveInputsJson(0, 1.4))).toEqual({
      transient: 1, energy: 0.5, thrust: 0.03,
    });
    const ranges = JSON.parse(modelIo.legacyVisualExportRangesJson());
    expect(ranges.map((field: { name: string; min: number; max: number }) =>
      [field.name, field.min, field.max])).toEqual([
      ['julia_real', -2, 2], ['julia_imag', -2, 2],
      ['color_hue', 0, 1], ['color_sat', 0, 1], ['color_bright', 0, 1],
      ['zoom', 0.1, 10], ['speed', 0, 1],
    ]);
    const visual = JSON.parse(modelIo.decodeLegacyVisualJson(
      new Float64Array([0.5, -0.5, 0.2, 0.4, 0.8, 0.6, 0.9]), true, 0.1, 0.5,
    ));
    expect(Object.keys(visual).sort()).toEqual([
      'colorBright', 'colorHue', 'colorSat', 'juliaSeed', 'speed', 'zoom',
    ]);
    expect(visual.juliaSeed.real).toBeCloseTo(-0.4);
    expect(visual.juliaSeed.imag).toBeCloseTo(-1);
    expect(visual.colorHue).toBeCloseTo(0.4);
    expect(visual.colorSat).toBeCloseTo(0.85);
    expect(visual.colorBright).toBeCloseTo(0.63);
    expect(visual.zoom).toBeCloseTo(2.7);
    expect(visual.speed).toBeCloseTo(0.7);
  });
});
