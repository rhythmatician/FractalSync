/**
 * Unit tests for manual Controls v2 driving (issue #121).
 *
 * Checks:
 * 1. Keyboard mapping maps standard PC keys (WASD, arrows, Shift, Space) to bounded Controls v2.
 * 2. Direction vector is properly normalized or zero when neutral.
 * 3. Spacebar edge-triggered impulse behavior.
 * 4. ManualDriver executes on canonical ticks via accumulator, not rAF delta.
 * 5. ManualDriver records every step into a CockpitTrajectory without mutating c/v directly.
 * 6. Reset reproduces initial state cleanly.
 * 7. Replay of recorded manual trajectory reproduces identical snapshots.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  initOrbitSynth,
  setWasmModuleForTesting,
  OrbitSynthesizer,
} from '../orbitSynthesizer';
import mockModule from './orbitSynthesizer.mock';
import {
  CANONICAL_DT,
  currentSnapshot,
} from '../debugCockpit';
import {
  mapKeyboardToMotionControls,
  ManualDriver,
  type KeyboardState,
} from '../manualDriving';

describe('manual Controls v2 driving (#121)', () => {
  beforeAll(async () => {
    setWasmModuleForTesting(mockModule as never);
    await initOrbitSynth();
  });

  it('maps neutral keyboard input to zero throttle, neutral steering, default grip', () => {
    const keys: KeyboardState = {
      up: false,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    };
    const ctrl = mapKeyboardToMotionControls(keys);
    expect(ctrl.throttle).toBe(0);
    expect(ctrl.brake).toBe(0);
    expect(ctrl.direction).toEqual([1, 0]); // Default forward facing or neutral
    expect(ctrl.grip).toBe(1.0); // Full grip when not drifting
    expect(ctrl.impulse).toBe(0);
  });

  it('maps forward key (W / Up) to full throttle and directional keys to steering vector', () => {
    const keys: KeyboardState = {
      up: true,
      down: false,
      left: false,
      right: true,
      drift: false,
      impulse: false,
    };
    const ctrl = mapKeyboardToMotionControls(keys, 0, 2.0, CANONICAL_DT);
    expect(ctrl.throttle).toBe(1.0);
    expect(ctrl.brake).toBe(0);
    expect(ctrl.grip).toBe(1.0);
    // Steering right from 0 radians turns clockwise: direction has negative Y component
    expect(ctrl.direction[1]).toBeLessThan(0);
    expect(Math.hypot(ctrl.direction[0], ctrl.direction[1])).toBeCloseTo(1.0, 5);
  });

  it('maps down key (S / Down) to brake', () => {
    const keys: KeyboardState = {
      up: false,
      down: true,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    };
    const ctrl = mapKeyboardToMotionControls(keys);
    expect(ctrl.brake).toBe(1.0);
    expect(ctrl.throttle).toBe(0);
  });

  it('maps shift key to reduced grip (drift mode)', () => {
    const keys: KeyboardState = {
      up: true,
      down: false,
      left: false,
      right: false,
      drift: true,
      impulse: false,
    };
    const ctrl = mapKeyboardToMotionControls(keys);
    expect(ctrl.grip).toBe(0.2); // Dissipative reduced grip
    expect(ctrl.throttle).toBe(1.0);
  });

  it('maps spacebar to bounded impulse', () => {
    const keys: KeyboardState = {
      up: false,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: true,
    };
    const ctrl = mapKeyboardToMotionControls(keys);
    expect(ctrl.impulse).toBe(1.0);
  });

  it('ManualDriver advances strictly on canonical ticks via accumulator', () => {
    const driver = new ManualDriver();
    driver.reset([0, 0], [0, 0]);

    // Initial state: 1 snapshot at t=0
    expect(driver.trajectory.snapshots.length).toBe(1);

    // Feed a small frame delta (< CANONICAL_DT): should NOT step physics
    const steppedSmall = driver.update(CANONICAL_DT * 0.4, {
      up: true,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    });
    expect(steppedSmall).toBe(0);
    expect(driver.trajectory.snapshots.length).toBe(1);

    // Feed enough to cross CANONICAL_DT: should step exactly once
    const steppedOne = driver.update(CANONICAL_DT * 0.7, {
      up: true,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    });
    expect(steppedOne).toBe(1);
    expect(driver.trajectory.snapshots.length).toBe(2);
    expect(driver.trajectory.snapshots[1].timeSeconds).toBeCloseTo(CANONICAL_DT, 6);
  });

  it('ManualDriver records raw and effective Controls v2 in each step snapshot', () => {
    const driver = new ManualDriver();
    driver.reset([0, 0], [0, 0]);

    driver.update(CANONICAL_DT, {
      up: true,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    });

    const snap = driver.trajectory.snapshots[1];
    expect(snap.action).not.toBeNull();
    expect(snap.action!.raw.throttle).toBe(1.0);
    expect(snap.action!.effective.throttle).toBe(1.0);
  });

  it('ManualDriver impulse is edge-triggered (resets after step)', () => {
    const driver = new ManualDriver();
    driver.reset([0, 0], [0, 0]);

    // Hold impulse over two canonical updates
    driver.update(CANONICAL_DT, {
      up: false,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: true,
    });
    expect(driver.trajectory.snapshots[1].action?.raw.impulse).toBe(1.0);

    driver.update(CANONICAL_DT, {
      up: false,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: true,
    });
    // Second step must not re-trigger impulse while held (edge-triggered per #121 test 6)
    expect(driver.trajectory.snapshots[2].action?.raw.impulse).toBe(0.0);
  });

  it('ManualDriver reset reproduces initial state', () => {
    const driver = new ManualDriver();
    driver.reset([0.2, -0.1], [0.01, 0]);
    expect(driver.trajectory.snapshots.length).toBe(1);
    expect(driver.trajectory.snapshots[0].physics.c[0]).toBeCloseTo(0.2, 5);

    driver.update(CANONICAL_DT * 3, {
      up: true,
      down: false,
      left: false,
      right: false,
      drift: false,
      impulse: false,
    });
    expect(driver.trajectory.snapshots.length).toBe(4);

    driver.reset();
    expect(driver.trajectory.snapshots.length).toBe(1);
    expect(driver.trajectory.snapshots[0].physics.c[0]).toBeCloseTo(0.2, 5);
  });

  it('replay of recorded manual trajectory produces identical snapshots', () => {
    const driver = new ManualDriver([0, 0], [0, 0]);
    // Simulate a few manual driving steps with varying throttle and steering
    driver.update(CANONICAL_DT, {
      up: true,
      down: false,
      left: false,
      right: true,
      drift: false,
      impulse: false,
    });
    driver.update(CANONICAL_DT, {
      up: true,
      down: false,
      left: true,
      right: false,
      drift: true,
      impulse: false,
    });

    const recordedSnaps = driver.trajectory.snapshots;
    expect(recordedSnaps.length).toBe(3); // 1 initial + 2 steps

    // Replay manually using a new OrbitSynthesizer directly with recorded actions
    const replaySynth = new OrbitSynthesizer(6);
    replaySynth.seed({ re: 0, im: 0 }, { re: 0, im: 0 });

    for (let i = 1; i < recordedSnaps.length; i++) {
      const raw = recordedSnaps[i].action!.raw;
      replaySynth.stepWithControls(CANONICAL_DT, raw);
      const snap = currentSnapshot(replaySynth);
      expect(snap.physics.c[0]).toBeCloseTo(recordedSnaps[i].physics.c[0], 10);
      expect(snap.physics.c[1]).toBeCloseTo(recordedSnaps[i].physics.c[1], 10);
      expect(snap.physics.kinetic).toBeCloseTo(recordedSnaps[i].physics.kinetic, 10);
    }
  });
});
