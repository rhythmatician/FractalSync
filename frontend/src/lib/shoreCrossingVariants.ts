/**
 * Hand-authored Shore-ridge crossing variants for the debug cockpit
 * (issue #111 Phase A) and the #82 prototype.
 *
 * These are ordinary Controls v2 action sequences replayed through the Rust
 * destination physics (OrbitSynthesizer.stepWithControls -> wasm
 * integrate_motion_controls). Every replay starts from the controller's
 * default state (c=0, v=0) — a fresh synthesizer per variant — so variants
 * differ only by their action sequences.
 *
 * No scripted c(t), no transient-gated wall, no musical feature: every frame
 * is an ordinary MotionControls consumed by the destination physics seam.
 */

export interface HandAuthoredAction {
  /** World-aligned drive direction (unit disk). */
  direction: [number, number];
  /** Drive magnitude in [0,1]. */
  throttle: number;
  /** Explicit extra dissipation in [0,1]. */
  brake: number;
  /** Traction in [0,1]. */
  grip: number;
  /** Bounded tap impulse in [0,1]. */
  impulse: number;
  /** How many consecutive frames this action is held. */
  frames: number;
}

export interface CrossingVariantSpec {
  name: string;
  description: string;
  actions: HandAuthoredAction[];
}

function buildAction(frames: number): HandAuthoredAction {
  return { direction: [1.0, 0.0], throttle: 0.25, brake: 0.0, grip: 0.0, impulse: 0.0, frames };
}

function commitAction(throttle: number): HandAuthoredAction {
  return { direction: [1.0, 0.0], throttle, brake: 0.0, grip: 0.0, impulse: 0.0, frames: 1 };
}

function settleAction(frames: number): HandAuthoredAction {
  return { direction: [1.0, 0.0], throttle: 0.4, brake: 0.6, grip: 1.0, impulse: 0.0, frames };
}

/**
 * The measured variant family. The replay runs the wasm
 * `OrbitController::step_with_controls` destination seam with the
 * controller-default manifold config (kappa=1.0, drag=0.1) starting at c=0
 * (the Rust PlayerState default), so the tuned action families differ from
 * the kappa=0.5 Python-harness family they were originally tuned against.
 * Measured structure:
 *
 * - commit=0.05: never crosses; barely climbs (max U ~ -0.61);
 * - commit=0.4:  never crosses; climbs to U ~ 0.36 and stalls;
 * - commit=0.8:  never crosses; deep penetration to U ~ 6.3 of the 9.97
 *   crest but the drive runs out below it;
 * - commit=1.0:  crosses at step ~781 and reaches the full crest
 *   U = log2(d_ref/epsilon) ~ 9.97.
 *
 * The commit phase alternates one commit frame with one settle frame
 * ("drip-feed"): the settle frames bleed off the crest-instability kinetic
 * spikes that the FD-derived curvature injects near D~0, so the crossing
 * commits instead of ricocheting. The hold-settle tail keeps the far side.
 */
export function baselineVariants(): CrossingVariantSpec[] {
  const make = (commitThrottle: number, pairs: number, hold: number): CrossingVariantSpec => {
    // Interleave one commit frame with one settle frame for `pairs` frames
    // (the drip cadence), then hold settle for `hold` frames.
    const actions: HandAuthoredAction[] = [buildAction(120)];
    for (let i = 0; i < pairs; i++) {
      actions.push(commitAction(commitThrottle));
      actions.push(settleAction(1));
    }
    actions.push(settleAction(hold));
    return {
      name: `commit_${commitThrottle}`,
      description: `build (t=0.25) x120 -> drip commit(${commitThrottle})/settle x${pairs} -> hold settle x${hold}`,
      actions,
    };
  };

  return [
    make(0.05, 400, 300),
    make(0.4, 400, 300),
    make(0.8, 400, 300),
    make(1.0, 400, 300),
  ];
}

/**
 * Expand a spec's action list into a flat per-frame action array.
 */
export function expandActions(variant: CrossingVariantSpec): HandAuthoredAction[] {
  const out: HandAuthoredAction[] = [];
  for (const a of variant.actions) {
    for (let i = 0; i < a.frames; i++) {
      out.push({
        direction: a.direction,
        throttle: a.throttle,
        brake: a.brake,
        grip: a.grip,
        impulse: a.impulse,
        frames: 1,
      });
    }
  }
  return out;
}
