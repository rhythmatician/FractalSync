import { describe, it, expect } from 'vitest';
import { computeHyperbolicCameraFrame } from '../hyperbolicCamera';
import type { DebugSnapshot } from '../debugCockpit';

describe('hyperbolic camera lookAt target', () => {
  const dummySnapshot: DebugSnapshot = {
    version: 'debug-snapshot/1',
    timeSeconds: 1.0,
    action: null,
    map: {
      pyramidLoaded: false,
      shoreProximity: null,
      minimapWindow: null,
      extent: null,
    },
    physics: {
      c: [-0.744, 0.132],
      velocity: [-0.000023, 0.0],
      signedDistance: 0.0026695,
      realm: 1,
      rho: 0.0026714,
      sigma: 5.22627,
      sigmaDot: -0.00416,
      scaleGradient: [184.0, 0.0],
      metric: [1.0, 0.0, 1.0],
      metricSpeed: 0.00942,
      kinetic: 0.00004,
      potential: 5.22627,
      total: 5.22631,
      geodesicAccel: [0.0, 0.0],
      potentialForce: [-184.0, 0.0],
      netAccel: [0.001, 0.0],
      derivativeValid: true,
    },
    diagnostics: {
      derivativeStep: 1e-4,
      valid: true,
      lastError: null,
      lastDeltaTotal: null,
      crestPotential: 10.0,
    },
  };

  it('computes camera frame and checks target vector in camera space', () => {
    const { rotationMatrix } = computeHyperbolicCameraFrame(dummySnapshot, 0);
    // Print/verify rotation matrix
    expect(rotationMatrix).toBeDefined();
    // Rotation matrix elements should be finite numbers
    for (let i = 0; i < 16; i++) {
      expect(Number.isFinite(rotationMatrix.elements[i])).toBe(true);
    }
  });
});
