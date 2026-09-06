/**
 * Adaptive vertical gain for Mandelbrot shore visualization.
 *
 * Dynamically adjusts the presentation-only vertical scale (Y display) based
 * on local terrain slope, keeping the apparent slope near a useful target
 * rather than becoming arbitrarily steep.
 *
 * This is PRESENTATION ONLY — it affects how we display the geometry, not
 * the authoritative Rust physics (rho, sigma, lambda, metric, forces).
 *
 * Issue #82: The Mandelbrot Shore is often too steep/spiky to be visually
 * useful. Fixed nonlinear mappings (asinh on top of logarithm) stack without
 * addressing the root cause: local slope varies radically across Mandelbrot
 * scales. This module makes the rendered vertical scale adapt to the local
 * terrain around the rider.
 */

import type { DebugSnapshot } from './debugCockpit';

/**
 * Configuration for adaptive vertical gain behavior.
 */
export interface AdaptiveGainConfig {
  /** Target apparent slope in display space (dimensionless). */
  targetSlope: number;
  /** Minimum allowed vertical gain (prevents over-compression). */
  minGain: number;
  /** Maximum allowed vertical gain (prevents over-expansion). */
  maxGain: number;
  /** Temporal smoothing time constant in seconds (slew rate limiter). */
  smoothingTau: number;
  /** Epsilon for gradient magnitude to avoid division by zero. */
  epsilon: number;
}

/**
 * Default configuration: target slope ~0.4 (gentle readable terrain),
 * gain bounds 0.1x to 10x, ~0.3s smoothing for stable transitions.
 */
export const DEFAULT_GAIN_CONFIG: AdaptiveGainConfig = {
  targetSlope: 0.4,
  minGain: 0.1,
  maxGain: 10.0,
  smoothingTau: 0.3,
  epsilon: 1e-6,
};

/**
 * Adaptive vertical gain state — maintains current smoothed gain and tracks
 * temporal evolution.
 */
export class AdaptiveVerticalGain {
  private currentGain: number;
  private readonly config: AdaptiveGainConfig;

  constructor(config: AdaptiveGainConfig = DEFAULT_GAIN_CONFIG) {
    this.config = config;
    this.currentGain = 1.0; // Start neutral
  }

  /**
   * Compute the raw target vertical gain from the local terrain slope.
   *
   * In scale-normalized treadmill coordinates:
   *   m_local = lambda * rho0 * ||∇σ(c0)||
   *
   * Target gain:
   *   g_z = m_target / (m_local + ε)
   *
   * Larger raw slope → smaller gain (compress steep terrain).
   * Smaller raw slope → larger gain (expand gentle terrain).
   *
   * The snapshot provides:
   *   - physics.rho: current rho0
   *   - physics.scaleGradient: [∂σ/∂x, ∂σ/∂y] = ∇σ(c0)
   *
   * Lambda is already baked into the Rust embedding height (z = lambda*σ),
   * but we need lambda explicitly for the slope calculation. Under the
   * controller-default config, lambda^2 = 1, so lambda = 1. If lambda
   * becomes configurable, retrieve it from the wasm module's ManifoldConfig.
   */
  private computeRawGain(snap: DebugSnapshot): number {
    const rho0 = Math.max(snap.physics.rho, this.config.epsilon);
    const [gx, gy] = snap.physics.scaleGradient;
    const gradMag = Math.hypot(gx, gy);

    // Lambda = 1 for now (controller-default). If lambda becomes configurable,
    // retrieve from wasm: getWasmModule().getManifoldConfig().lambda_sq ** 0.5
    const lambda = 1.0;

    const localSlope = lambda * rho0 * gradMag;
    const rawGain = this.config.targetSlope / (localSlope + this.config.epsilon);

    // Clamp to configured bounds
    return Math.max(
      this.config.minGain,
      Math.min(this.config.maxGain, rawGain)
    );
  }

  /**
   * Update the adaptive gain with temporal smoothing.
   *
   * Uses exponential smoothing with slew rate limiting:
   *   g(t+dt) = g(t) + α * (g_target - g(t))
   * where α = 1 - exp(-dt/τ)
   *
   * This prevents visible pumping/jitter/snapping as the player moves.
   *
   * @param snap - Current authoritative snapshot
   * @param dt - Time delta in seconds
   * @returns Current smoothed vertical gain
   */
  update(snap: DebugSnapshot, dt: number): number {
    const rawGain = this.computeRawGain(snap);

    // Exponential smoothing coefficient
    const alpha = 1.0 - Math.exp(-dt / this.config.smoothingTau);

    // Smooth transition toward raw target
    this.currentGain += alpha * (rawGain - this.currentGain);

    return this.currentGain;
  }

  /**
   * Get the current smoothed vertical gain without updating.
   */
  get gain(): number {
    return this.currentGain;
  }

  /**
   * Get diagnostic information for debug display.
   */
  getDiagnostics(snap: DebugSnapshot): {
    rho0: number;
    gradMag: number;
    localSlope: number;
    rawGain: number;
    smoothedGain: number;
  } {
    const rho0 = Math.max(snap.physics.rho, this.config.epsilon);
    const [gx, gy] = snap.physics.scaleGradient;
    const gradMag = Math.hypot(gx, gy);
    const lambda = 1.0;
    const localSlope = lambda * rho0 * gradMag;
    const rawGain = this.computeRawGain(snap);

    return {
      rho0,
      gradMag,
      localSlope,
      rawGain,
      smoothedGain: this.currentGain,
    };
  }

  /**
   * Reset the gain to neutral (for mode switches or scene resets).
   */
  reset(): void {
    this.currentGain = 1.0;
  }
}
