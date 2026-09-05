import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  decodeOrbitControl,
  legacyOrbitDriveInputs,
  setWasmModuleForTesting,
} from '../orbitSynthesizer';

afterEach(() => setWasmModuleForTesting(null));

describe('Rust model output adapters', () => {
  it('forwards exact model channels and returns the native decoded payload', () => {
    const decode = vi.fn((_values: Float64Array, _kBands: number) => JSON.stringify({
      sTarget: 1.2, alpha: 0.4, omegaScale: 0.8, bandGates: [0.1, 0.9],
    }));
    setWasmModuleForTesting({ decodeOrbitControlJson: decode } as never);

    expect(decodeOrbitControl([1.2, 0.4, 0.8, 0.1, 0.9], 2)).toEqual({
      sTarget: 1.2, alpha: 0.4, omegaScale: 0.8, bandGates: [0.1, 0.9],
    });
    expect(Array.from(decode.mock.calls[0][0])).toEqual([1.2, 0.4, 0.8, 0.1, 0.9]);
    expect(decode.mock.calls[0][1]).toBe(2);
  });

  it('does not hide a native malformed-output failure', () => {
    setWasmModuleForTesting({
      legacyOrbitDriveInputsJson: () => { throw new Error('non-finite RMS'); },
    } as never);
    expect(() => legacyOrbitDriveInputs(Number.NaN, 0)).toThrow('non-finite RMS');
  });
});
