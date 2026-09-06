import { describe, it, expect } from 'vitest';
import {
  computeHyperbolicCameraFrame,
  computeUpperHalfScaleA,
  projectPointToCameraPoincare,
  poincareToSceneVector,
} from '../hyperbolicCamera';
import type { DebugSnapshot } from '../debugCockpit';

describe('hyperbolic camera lookAt target direction', () => {
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

  it('checks rider vector in camera space after applying inverse rotation', () => {
    const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(dummySnapshot, 0);
    const invRot = rotationMatrix.clone().invert();

    const [cx, cy] = dummySnapshot.physics.c;
    const a = computeUpperHalfScaleA(1.0);
    const zRider = a * dummySnapshot.physics.rho;

    const b = projectPointToCameraPoincare({ x: cx, y: cy, z: zRider }, cameraUpperHalf);
    const vScene = poincareToSceneVector(b);
    const vCam = vScene.clone().applyMatrix4(invRot);

    // In camera space:
    // Forward should be along -Z!
    expect(vCam.z).toBeLessThan(0);
    // And nearly centered on X:
    expect(Math.abs(vCam.x)).toBeLessThan(0.01);
  });
});
