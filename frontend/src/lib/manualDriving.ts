/**
 * Manual driving adapter for Controls v2 (issue #121).
 *
 * Implements PC keyboard driving through the authoritative Rust destination
 * physics seam (ADR 0001).
 *
 * Invariants per #121:
 * 1. Human input enters ONLY through the low-dimensional Controls v2 surface:
 *    MotionControls { direction: [dx, dy], throttle, brake, grip, impulse }.
 *    Never sets c, velocity, rho, sigma, or forces directly.
 * 2. Physics steps ONLY on canonical ticks (CANONICAL_DT = 1024 / 48000), using
 *    a fixed-timestep accumulator in the render loop. requestAnimationFrame delta
 *    never becomes physics delta.
 * 3. Spacebar / face button impulse is edge-triggered and bounded in dual metric.
 * 4. Every step records a DebugSnapshot, producing a CockpitTrajectory that can
 *    be paused, scrubbed, or replayed.
 */

import {
  OrbitSynthesizer,
} from './orbitSynthesizer';
import {
  CANONICAL_DT,
  currentSnapshot,
  type CockpitTrajectory,
  type DebugSnapshot,
} from './debugCockpit';
import type { CrossingVariantSpec } from './shoreCrossingVariants';

/** Raw state of driver-relevant keyboard inputs. */
export interface KeyboardState {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  drift: boolean;
  impulse: boolean;
}

/** Controls v2 motion payload. */
export interface MotionControlsPayload {
  direction: [number, number];
  throttle: number;
  brake: number;
  grip: number;
  impulse: number;
}

/**
 * Maps standard PC keys (WASD / arrows / Shift / Space) to bounded Controls v2.
 *
 * Mapping per #121:
 * - W / Up -> throttle (0 or 1)
 * - S / Down -> brake (0 or 1)
 * - A / D / Left / Right -> normalized planar direction [dx, dy]
 * - Shift -> grip (0.2 drift / 1.0 full grip)
 * - Space -> impulse (edge-triggered, 1.0 or 0.0)
 *
 * In c-space (Mandelbrot plane):
 * - +X is real axis (right)
 * - -X is left
 * - +Y is imaginary axis (up)
 * - -Y is down
 * If no steering keys are pressed, direction defaults to [1, 0] (forward real axis).
 */
/**
 * Maps standard PC controls to bounded Controls v2.
 *
 * Controls semantics:
 * - W / Up -> Forward Throttle (accelerate along current heading)
 * - S / Down -> Dissipative Brake
 * - A / Left -> Steer Left (counter-clockwise yaw in c-space)
 * - D / Right -> Steer Right (clockwise yaw in c-space)
 * - Shift -> Drift (reduces grip to 0.2)
 * - Space -> Impulse (edge-triggered)
 *
 * @param keys Active keyboard state
 * @param currentHeading Current heading angle theta in c-space radians (atan2(vy, vx))
 * @param steerAngularRate Radians to turn heading per second when steering key is held (default ~1.5 rad/s)
 * @param dt Timestep in seconds
 */
export function mapKeyboardToMotionControls(
  keys: KeyboardState,
  currentHeading: number = 0,
  steerAngularRate: number = 2.0,
  dt: number = CANONICAL_DT
): MotionControlsPayload {
  const throttle = keys.up ? 1.0 : 0.0;
  const brake = keys.down ? 1.0 : 0.0;

  // Steering adjusts heading angle relative to current heading:
  // A (steer left in c-space) -> +dTheta
  // D (steer right in c-space) -> -dTheta
  let heading = currentHeading;
  if (keys.left && !keys.right) {
    heading += steerAngularRate * dt;
  } else if (keys.right && !keys.left) {
    heading -= steerAngularRate * dt;
  }

  const direction: [number, number] = [Math.cos(heading), Math.sin(heading)];

  // Shift modulates grip: 1.0 (firm) -> 0.2 (drift)
  const grip = keys.drift ? 0.2 : 1.0;

  // Spacebar gives bounded impulse
  const impulse = keys.impulse ? 1.0 : 0.0;

  return {
    direction,
    throttle,
    brake,
    grip,
    impulse,
  };
}

/**
 * Manages an active live OrbitSynthesizer instance driven by manual controls.
 * Uses an accumulator to ensure steps occur strictly at CANONICAL_DT.
 */
export class ManualDriver {
  private synth: OrbitSynthesizer;
  private accumulator = 0;
  private lastImpulseState = false;
  private pendingImpulse = false;
  private headingAngle = 0;
  private initialC: [number, number] = [0, 0];
  private initialV: [number, number] = [0, 0];

  public trajectory: CockpitTrajectory;

  constructor(initialC: [number, number] = [0, 0], initialV: [number, number] = [0, 0]) {
    this.initialC = [...initialC];
    this.initialV = [...initialV];
    this.synth = new OrbitSynthesizer(6);
    this.trajectory = this.createEmptyTrajectory();
    this.reset(initialC, initialV);
  }

  private createEmptyTrajectory(): CockpitTrajectory {
    const spec: CrossingVariantSpec = {
      name: 'manual_flight',
      description: 'Interactive manual flight via Controls v2 (#121)',
      actions: [],
      initialC: this.initialC,
      initialV: this.initialV,
    };
    return {
      spec,
      snapshots: [],
      crossingStep: null,
      crossed: false,
      maxPotential: -Infinity,
      crestedRidge: false,
    };
  }

  /**
   * Resets the synthesizer and trajectory to the given or current seed coordinates.
   */
  public reset(c?: [number, number], v?: [number, number]): void {
    if (c) this.initialC = [...c];
    if (v) this.initialV = [...v];

    this.synth = new OrbitSynthesizer(6);
    this.synth.seed(
      { re: this.initialC[0], im: this.initialC[1] },
      { re: this.initialV[0], im: this.initialV[1] }
    );

    this.accumulator = 0;
    this.lastImpulseState = false;
    this.pendingImpulse = false;
    const initialSpeed = Math.hypot(this.initialV[0], this.initialV[1]);
    this.headingAngle = initialSpeed > 1e-7 ? Math.atan2(this.initialV[1], this.initialV[0]) : 0;
    this.trajectory = this.createEmptyTrajectory();

    // Capture baseline initial snapshot (t=0, step 0)
    const initSnap = currentSnapshot(this.synth);
    this.trajectory.snapshots.push(initSnap);
    this.trajectory.maxPotential = initSnap.physics.potential;
  }

  /**
   * Advances the manual flight by dtSeconds.
   * Steps the authoritative physics synthesizer once for every CANONICAL_DT slice.
   * Returns the number of canonical steps executed.
   */
  public update(dtSeconds: number, keys: KeyboardState): number {
    // Edge-triggered impulse: register on rising edge, latch until a canonical step consumes it
    if (keys.impulse && !this.lastImpulseState) {
      this.pendingImpulse = true;
    }
    this.lastImpulseState = keys.impulse;

    this.accumulator += dtSeconds;
    let stepsRun = 0;

    // Follow velocity heading if actively moving, otherwise retain steerable heading angle
    const lastSnap = this.trajectory.snapshots[this.trajectory.snapshots.length - 1];
    if (lastSnap) {
      const vx = lastSnap.physics.velocity[0];
      const vy = lastSnap.physics.velocity[1];
      if (Math.hypot(vx, vy) > 1e-4) {
        this.headingAngle = Math.atan2(vy, vx);
      }
    }

    while (this.accumulator >= CANONICAL_DT) {
      // Consume edge-triggered impulse on first step of this tick batch
      const stepImpulse = this.pendingImpulse;
      this.pendingImpulse = false;

      const ctrl = mapKeyboardToMotionControls(
        {
          ...keys,
          impulse: stepImpulse,
        },
        this.headingAngle,
        2.5,
        CANONICAL_DT
      );

      // Update internal heading angle to match emitted direction
      this.headingAngle = Math.atan2(ctrl.direction[1], ctrl.direction[0]);

      this.synth.stepWithControls(CANONICAL_DT, ctrl);
      const snap = currentSnapshot(this.synth);

      this.trajectory.snapshots.push(snap);
      this.trajectory.maxPotential = Math.max(
        this.trajectory.maxPotential,
        snap.physics.potential
      );
      if (snap.diagnostics.crestPotential) {
        this.trajectory.crestedRidge =
          this.trajectory.maxPotential > snap.diagnostics.crestPotential - 1.0;
      }
      if (this.trajectory.crossingStep === null && snap.physics.signedDistance > 0) {
        this.trajectory.crossingStep = this.trajectory.snapshots.length - 1;
        this.trajectory.crossed = true;
      }

      this.accumulator -= CANONICAL_DT;
      stepsRun++;
    }

    return stepsRun;
  }

  /** Gets the latest snapshot. */
  public get latestSnapshot(): DebugSnapshot {
    return this.trajectory.snapshots[this.trajectory.snapshots.length - 1];
  }
}
