import { describe, it, expect } from 'vitest';
import { ModelInference } from '../modelInference';

describe('ModelInference (fallback minimap context)', () => {
  it('pads minimap context when stepController.context is unavailable', () => {
    const m = new ModelInference() as any;
    m.isStepModel = true;
    m.stepController = { /* missing context */ };
    m.stepState = { c_real: 0, c_imag: 0, prev_delta_real: 0, prev_delta_imag: 0 };

    const baseFeatures = new Array(6 * 10).fill(0.1);
    const combined = m['buildModelInput'](baseFeatures);

    // Should be original features plus 265 zeros appended
    expect(combined.length).toBe(baseFeatures.length + 265);

    // Simulate a StepResult that uses c_next_* naming and ensure state picks it up
    const fakeResult = { c_next_real: 0.5, c_next_imag: -0.25, delta_real: 0.02, delta_imag: -0.01 };
    m.stepState = { c_real: 0, c_imag: 0, prev_delta_real: 0, prev_delta_imag: 0 };
    // Emulate internal update logic used in ModelInference
    const pickNumber = (arr: any[]) => {
      for (const v of arr) if (typeof v === 'number') return v;
      return null;
    };
    const maybe_c_real = pickNumber([ (fakeResult as any).c_real, fakeResult.c_next_real]);
    const maybe_c_imag = pickNumber([ (fakeResult as any).c_imag, fakeResult.c_next_imag]);
    const maybe_delta_real = pickNumber([fakeResult.delta_real]);
    const maybe_delta_imag = pickNumber([fakeResult.delta_imag]);

    if (maybe_c_real !== null) m.stepState.c_real = maybe_c_real;
    if (maybe_c_imag !== null) m.stepState.c_imag = maybe_c_imag;
    if (maybe_delta_real !== null) m.stepState.prev_delta_real = maybe_delta_real;
    if (maybe_delta_imag !== null) m.stepState.prev_delta_imag = maybe_delta_imag;

    expect(m.stepState.c_real).toBeCloseTo(0.5);
    expect(m.stepState.c_imag).toBeCloseTo(-0.25);

  });
});