/**
 * Tick-ordering test for the inference boundary (issue #91, invariant 7).
 *
 * Analysis ticks can reach the async ONNX path faster than inference
 * completes. Because Physics is stateful, ticks must be applied in arrival
 * order — asynchronous completion must not reorder them. This test injects a
 * fake ONNX session whose `run()` resolves after a controllable, varying
 * delay and asserts that the stateful consumer observes ticks strictly in
 * order.
 */

import { describe, it, expect } from 'vitest';
import { ModelInference } from '../modelInference';
import { initOrbitSynth } from '../orbitSynthesizer';

(globalThis as { __vitest?: boolean }).__vitest = true;
import type { AnalysisTick } from '../analysisTimebase';

/** Build a minimal legacy-model tick (7 outputs, no orbit synthesizer). */
function makeTick(sampleIndex: number, marker: number): AnalysisTick {
  // 6 features/frame × 10 frames; we only need a stable marker in the data.
  const features = new Array(60).fill(marker);
  return {
    features,
    sampleIndex,
    timeSeconds: sampleIndex / 48000,
    dtSeconds: 1024 / 48000,
    streamEpoch: 0,
  };
}

/** Install a fake session on a ModelInference, returning `marker` outputs. */
function installFakeSession(
  mi: ModelInference,
  delays: number[],
  outputFor: (callIndex: number) => number[]
): void {
  let call = 0;
  const fakeSession = {
    run: async () => {
      const i = call++;
      const delay = delays[i % delays.length];
      await new Promise((r) => setTimeout(r, delay));
      return {
        visual_parameters: {
          data: Float32Array.from(outputFor(i)),
        },
      };
    },
  };
  // Reach into the private field for the test seam (no public setter).
  (mi as unknown as { session: unknown }).session = fakeSession;
  // Force the legacy (non-orbit) path so no orbit synthesizer is needed.
  (mi as unknown as { isOrbitModel: boolean }).isOrbitModel = false;
  (mi as unknown as { useAudioReactivePostProcessing: boolean }).useAudioReactivePostProcessing =
    false;
}

describe('ModelInference tick ordering (invariant 7)', () => {
  it('applies ticks in arrival order despite varying inference latency', async () => {
    await initOrbitSynth();
    const mi = new ModelInference();

    // First inference is SLOW, later ones FAST — without ordering, tick 1
    // would finish after tick 2 and reorder the stateful consumer.
    const delays = [40, 1, 1, 1, 1];
    // Each call returns its own call index in output[0] so we can detect
    // the order the consumer observes completed Physics steps.
    installFakeSession(mi, delays, (i) => [i, 0, 0, 0, 0, 0, 0]);

    const observedOrder: number[] = [];
    const ticks = [0, 1, 2, 3, 4].map((i) => makeTick((i + 1) * 1024, i));

    // Enqueue all ticks; each resolves with that tick's visual params.
    const promises = ticks.map((t) =>
      mi.inferTick(t).then((params) => {
        // The legacy path maps juliaSeed.real = (x*0.6)%1.4-0.7 where x is
        // output[0] = the call index. Record the transformed value; the
        // expected ordered sequence is the transform of [0,1,2,3,4].
        observedOrder.push(params.juliaSeed.real);
      })
    );

    await Promise.all(promises);

    // The consumer must observe completions in tick arrival order even
    // though tick 0's inference was the slowest. Expected = transform of
    // [0,1,2,3,4] through the legacy post-processing.
    const transform = (x: number) => ((x * 0.6) % 1.4) - 0.7;
    expect(observedOrder).toEqual([0, 1, 2, 3, 4].map(transform));
  });

  it('serializes concurrent inferTick calls (no interleaving)', async () => {
    await initOrbitSynth();
    const mi = new ModelInference();
    let active = 0;
    let maxActive = 0;
    const fakeSession = {
      run: async () => {
        active++;
        maxActive = Math.max(maxActive, active);
        await new Promise((r) => setTimeout(r, 5));
        active--;
        return { visual_parameters: { data: Float32Array.from([0, 0, 0, 0, 0, 0, 0]) } };
      },
    };
    (mi as unknown as { session: unknown }).session = fakeSession;
    (mi as unknown as { isOrbitModel: boolean }).isOrbitModel = false;

    await Promise.all(
      [0, 1, 2, 3].map((i) => mi.inferTick(makeTick((i + 1) * 1024, i)))
    );

    // At most one inference in flight at a time → strictly serialized.
    expect(maxActive).toBe(1);
  });
});
