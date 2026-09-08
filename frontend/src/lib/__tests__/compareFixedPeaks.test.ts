import { describe, it } from 'vitest';
import * as THREE from 'three';
import {
  transformMeshToHyperbolic,
} from './hyperbolicDiagnosticFixture';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import { buildTerrainMesh } from '../cockpitScene';

describe('compare real terrain patch with sampleTerrainPatch', () => {
  function makeSnapshot(sigma: number, rho: number, cx = -0.75, cy = 0.1): DebugSnapshot {
    return {
      version: 'debug-snapshot/1',
      timeSeconds: 1.0,
      action: null,
      map: { pyramidLoaded: false, shoreProximity: null, minimapWindow: null, extent: null },
      physics: {
        c: [cx, cy],
        velocity: [0.0, 0.0],
        signedDistance: rho,
        realm: 1,
        rho: rho,
        sigma: sigma,
        sigmaDot: 0.0,
        scaleGradient: [0.0, 0.0],
        metric: [1.0, 0.0, 1.0],
        metricSpeed: 0.0,
        kinetic: 0.0,
        potential: sigma,
        total: sigma,
        geodesicAccel: [0.0, 0.0],
        potentialForce: [0.0, 0.0],
        netAccel: [0.0, 0.0],
        derivativeValid: true,
      },
      diagnostics: { derivativeStep: 1e-4, valid: true, lastError: null, lastDeltaTotal: null, crestPotential: 10.0 },
    };
  }

  it('checks what happens when rider is at rho=0.25 vs rho=0.001', () => {
    // Suppose we have two mountain peaks in c-space at fixed world coordinates c1 and c2:
    // c1 = (-0.745, 0.1), c2 = (-0.735, 0.1) -> delta c = 0.01
    // Both peaks have z_embed = 8.0 (sigma=8, rho = 0.1 * 2^-8 = 0.00039)
    const p1_x = -0.745;
    const p2_x = -0.735;
    const peakSigma = 8.0;

    const patch: TerrainPatch = {
      n: 3,
      center: [-0.74, 0.1],
      half: 0.01,
      positions: [
        p1_x, 0.1, peakSigma,
        -0.74, 0.1, 0.0,
        p2_x, 0.1, peakSigma,
        p1_x, 0.1, peakSigma,
        -0.74, 0.1, 0.0,
        p2_x, 0.1, peakSigma,
        p1_x, 0.1, peakSigma,
        -0.74, 0.1, 0.0,
        p2_x, 0.1, peakSigma,
      ],
      signed: [0.001, 0.1, 0.001, 0.001, 0.1, 0.001, 0.001, 0.1, 0.001],
      realm: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    };

    // Case 1: Player is low in the valley (rho0 = 0.1, sigma0 = 0)
    const snapValley = makeSnapshot(0.0, 0.1, -0.74, 0.1);
    const meshValley = buildTerrainMesh(patch);
    transformMeshToHyperbolic(meshValley, patch, snapValley, 0);
    const posV = meshValley.geometry.getAttribute('position') as THREE.BufferAttribute;
    const v1 = new THREE.Vector3(posV.getX(0), posV.getY(0), posV.getZ(0));
    const v2 = new THREE.Vector3(posV.getX(2), posV.getY(2), posV.getZ(2));
    const screenDistValley = Math.hypot(v1.x - v2.x, v1.y - v2.y);

    // Case 2: Player is high up on the mountain (rho0 = 0.001, sigma0 = 6.64)
    const snapHigh = makeSnapshot(6.64, 0.001, -0.74, 0.1);
    const meshHigh = buildTerrainMesh(patch);
    transformMeshToHyperbolic(meshHigh, patch, snapHigh, 0);
    const posH = meshHigh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const h1 = new THREE.Vector3(posH.getX(0), posH.getY(0), posH.getZ(0));
    const h2 = new THREE.Vector3(posH.getX(2), posH.getY(2), posH.getZ(2));
    const screenDistHigh = Math.hypot(h1.x - h2.x, h1.y - h2.y);

    console.log('Apparent distance between fixed peaks when player is low (rho0=0.1):', screenDistValley);
    console.log('Apparent distance between fixed peaks when player is high (rho0=0.001):', screenDistHigh);
    console.log('Ratio high / low:', screenDistHigh / screenDistValley);
  });
});
