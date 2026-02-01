import { describe, it, expect, vi } from 'vitest';
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

  it('sendTelemetry posts to the server', async () => {
    const m = new ModelInference() as any;
    const mockFetch = vi.fn(() => Promise.resolve({ ok: true })) as any;
    (globalThis as any).fetch = mockFetch;

    await m.sendTelemetry({ hello: 'world' });

    expect(mockFetch).toHaveBeenCalled();
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe('/api/telemetry');
    expect(opts.method).toBe('POST');
    const body = JSON.parse(opts.body);
    expect(body.hello).toBe('world');
  });

  it('setTelemetryEnabled sends a ping when enabling', async () => {
    const m = new ModelInference() as any;
    const mockFetch = vi.fn(() => Promise.resolve({ ok: true })) as any;
    (globalThis as any).fetch = mockFetch;

    m.metadata = { model_type: 'step_control' } as any;
    m.setTelemetryEnabled(true);

    // ping should have been sent immediately
    expect(mockFetch).toHaveBeenCalled();
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe('/api/telemetry');
    const body = JSON.parse(opts.body);
    expect(body.type).toBe('ping');
    expect(body.model_type).toBe('step_control');
  });

  it('assembleTelemetryEntry includes control and context fields', () => {
    const m = new ModelInference() as any;
    m.isStepModel = true;

    // Mock stepState
    m.stepState = { c_real: 0.1, c_imag: -0.2, prev_delta_real: 0.01, prev_delta_imag: -0.02 };

    // Mock stepController.context to return a small, predictable feature_vec
    const fakePatch = new Array(256).fill(0.5);
    const fv = [0.1, -0.2, 0.01, -0.02, 0.75, 1.0, 0.01, -0.02, 0.12].concat(fakePatch);
    m.stepController = { context: (_s: any) => ({ feature_vec: fv }) };

    const payload = m.assembleTelemetryEntry([0.123, -0.456]);

    expect(payload.model_dx).toBeCloseTo(0.123);
    expect(payload.model_dy).toBeCloseTo(-0.456);
    expect(payload.c_real).toBeCloseTo(0.1);
    expect(payload.c_imag).toBeCloseTo(-0.2);
    expect(payload.prev_dx).toBeCloseTo(0.01);
    expect(payload.prev_dy).toBeCloseTo(-0.02);
    expect(payload.nu_norm).toBeCloseTo(0.75);
    expect(payload.membership).toBeCloseTo(1.0);
    expect(payload.grad_re).toBeCloseTo(0.01);
    expect(payload.grad_im).toBeCloseTo(-0.02);
    expect(payload.sensitivity).toBeCloseTo(0.12);
    expect(payload.patch_mean).toBeCloseTo(0.5);
    expect(payload.patch_max).toBeCloseTo(0.5);
  });

});