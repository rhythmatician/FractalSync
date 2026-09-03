/**
 * Tests for the cockpit's exact coordinate charts (issue #111 review).
 *
 * The critique (2026-09-03) established that the rendered charts must be
 * LITERAL:
 *
 *   Q_physical  = S * (x, lambda*sigma, -y)          (one uniform S, all axes)
 *   Q_treadmill = S * ((x-x0)/rho0, lambda*(sigma-sigma0), -(y-y0)/rho0)
 *
 * Physical mode restores metric truth: its induced Euclidean metric is
 * S^2*G, so slopes/angles are honest, and height relates to potential by
 * U = (kappa/lambda) * Q_z — the skateboard-terrain metaphor is a surface
 * under uniform gravity. Treadmill mode is scale-stabilized: horizontal
 * magnification 1/rho0 with UNCHANGED vertical scale, so the rider sits at
 * the chart origin with zero height residue. The asinh surfaceY() is
 * retired from geometry (kept only as a labeled "compressed" presentation
 * option concept, not implemented here).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  SCENE_SCALE,
  LAMBDA,
  surfaceY,
  physicalChart,
  treadmillChart,
  physicalPitch,
  treadmillPitch,
  buildTerrainMesh,
  treadmillTransform,
  treadmillTrailTransform,
  physicalTransform,
  physicalTrailTransform,
} from '../cockpitScene';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import mockModule from './orbitSynthesizer.mock';
import { setWasmModuleForTesting, initOrbitSynth } from '../orbitSynthesizer';
import { baselineVariants } from '../shoreCrossingVariants';
import { CockpitRecorder } from '../debugCockpit';

/** Tiny stub mesh used for transform-only tests (geometry is irrelevant). */
function makeMesh(): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3)
  );
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
}

/**
 * Tiny TerrainPatch stub used for buildTerrainMesh Y-coordinate tests.
 * `positions` is a flat [x, y, sigma, x, y, sigma, ...] array; only the
 * sigma (z) slot is read for the Y coordinate in treadmill mode.
 */
function makePatch(positions: number[]): TerrainPatch {
  return {
    n: 2,
    center: [0.25, 0.0],
    half: 0.1,
    positions,
    signed: [0.001, 0.001],
    realm: [1, 1],
  };
}

/** Build a minimal snapshot-shaped fixture with explicit physics fields. */
function snapFixture(overrides: {
  c: [number, number];
  velocity?: [number, number];
  sigma?: number;
  rho?: number;
  sigmaDot?: number;
}): DebugSnapshot {
  return {
    version: 'debug-snapshot/1',
    timeSeconds: 0,
    action: null,
    map: { pyramidLoaded: false, shoreProximity: null, minimapWindow: null, extent: null },
    physics: {
      c: overrides.c,
      velocity: overrides.velocity ?? [0, 0],
      signedDistance: 0.001,
      realm: 1,
      rho: overrides.rho ?? Math.sqrt(0.001 ** 2 + 1e-8),
      sigma: overrides.sigma ?? 0,
      sigmaDot: overrides.sigmaDot ?? 0,
      scaleGradient: [0, 0],
      metric: [1, 0, 1],
      metricSpeed: 0,
      kinetic: 0,
      potential: 0,
      total: 0,
      geodesicAccel: [0, 0],
      potentialForce: [0, 0] as [number, number],
      netAccel: [0, 0] as [number, number],
      derivativeValid: true,
    },
    diagnostics: {
      derivativeStep: 1e-4,
      valid: true,
      lastError: null,
      lastDeltaTotal: null,
      crestPotential: Math.log2(0.1 / 1e-4),
    },
  };
}

describe('exact physical chart Q_phys = S*(x, lambda*sigma, -y)', () => {
  it('maps c to horizontal position with the uniform scene scale', () => {
    const s = snapFixture({ c: [0.3, 0.1], sigma: -1.0, rho: 0.126 });
    const p = physicalChart(s);
    expect(p.x).toBeCloseTo(0.3 * SCENE_SCALE, 9);
    expect(p.z).toBeCloseTo(-0.1 * SCENE_SCALE, 9);
  });

  it('vertical height is lambda*sigma with NO asinh compression', () => {
    // lambda = sqrt(lambda_sq) = 1 for the controller-default config, so
    // height must EQUAL sigma * S — metric truth, not a compressed view.
    const s = snapFixture({ c: [0.3, 0.0], sigma: 5.0, rho: 0.001 });
    const p = physicalChart(s);
    expect(p.y).toBeCloseTo(5.0 * SCENE_SCALE, 9);
  });

  it('height is an affine function of sigma (metric-exactness property)', () => {
    // U = kappa*sigma, Q_z = S*lambda*sigma => U = (kappa/(lambda*S)) * Q_z.
    // Equivalently Q_z(sigma)/Q_z(sigma') must equal sigma/sigma'.
    const a = snapFixture({ c: [0.3, 0.0], sigma: 2.0, rho: 0.025 });
    const b = snapFixture({ c: [0.3, 0.0], sigma: 6.0, rho: 0.0016 });
    const pa = physicalChart(a);
    const pb = physicalChart(b);
    // asinh would violate this ratio (asinh(2/1.5)/asinh(6/1.5) != 1/3).
    expect(pa.y / pb.y).toBeCloseTo(2.0 / 6.0, 9);
  });

  it('preserves the scene handedness x -> +X, y -> -Z', () => {
    const s = snapFixture({ c: [0.0, 0.5], sigma: 0.0, rho: 0.5 });
    const p = physicalChart(s);
    expect(p.x).toBeCloseTo(0, 9);
    expect(p.z).toBeLessThan(0);
  });
});

describe('exact treadmill chart Q_tread = S*((x-x0)/rho0, lambda*(sigma-sigma0), -(y-y0)/rho0)', () => {
  it('pins the rider at the chart origin with ZERO height residue', () => {
    // The old bug: uniform 1/rho scale + Y-translate left residue
    // (1/rho0 - 1) * h0, launching the rider 77k scene units high at the
    // crest. The exact chart leaves the rider's own point at exactly
    // (0, 0, 0) — the identity term of the transform.
    const s = snapFixture({ c: [0.25, 0.0], sigma: 9.96578, rho: 1e-4 });
    const p = treadmillChart(s);
    expect(p.x).toBeCloseTo(0, 9);
    expect(p.y).toBeCloseTo(0, 9);
    expect(p.z).toBeCloseTo(0, 9);
  });

  it('neighboring terrain keeps the vertical scale (lambda unchanged)', () => {
    // A point 0.001 c-units away with sigma LOWER by log2(rho0/rho) maps to
    // horizontal offset 0.001/rho0 and vertical offset lambda*(sigma-sigma0)
    // in scene units * S — NOT magnified by 1/rho0 vertically.
    const rho0 = 1e-3;
    const s0 = snapFixture({ c: [0.25, 0.0], sigma: 6.643856, rho: rho0 });
    // Neighbor at the same sigma: vertical offset must be exactly 0.
    const neighborSameHeight = treadmillChart(s0, 0.251, 0.0, 6.643856);
    expect(neighborSameHeight.y).toBeCloseTo(0, 9);
    expect(neighborSameHeight.x).toBeCloseTo(0.001 / rho0 * SCENE_SCALE, 9);
    // A neighbor one octave deeper in scale (sigma+1) sits exactly
    // lambda*S scene units higher — vertical is never magnified.
    const neighborDeeper = treadmillChart(s0, 0.251, 0.0, 6.643856 + 1.0);
    expect(neighborDeeper.y).toBeCloseTo(1.0 * SCENE_SCALE, 9);
  });

  it('uses the identity sigma - sigma0 = log2(rho0/rho)', () => {
    // For a point whose rho doubles relative to rho0, the vertical offset
    // is lambda * log2(1/2) = -lambda scene units (* S).
    const rho0 = 2e-3;
    const s0 = snapFixture({ c: [0.25, 0.0], sigma: 5.643856, rho: rho0 });
    const neighbor = treadmillChart(s0, 0.252, 0.0, 5.643856 - 1.0);
    expect(neighbor.y).toBeCloseTo(-1.0 * SCENE_SCALE, 9);
  });

  it('horizontal magnification is 1/rho0, vertical is 1', () => {
    const rho0 = 5e-3;
    const s0 = snapFixture({ c: [0.25, 0.1], sigma: 4.3219, rho: rho0 });
    const p = treadmillChart(s0, 0.26, 0.2, 4.3219);
    expect(p.x).toBeCloseTo((0.01 / rho0) * SCENE_SCALE, 9);
    expect(p.z).toBeCloseTo((-0.1 / rho0) * SCENE_SCALE, 9);
    expect(p.y).toBeCloseTo(0, 9);
  });
});

describe('rider pitch per camera chart', () => {
  it('physical pitch is atan2(lambda*sigmaDot, planarSpeed)', () => {
    // lambda=1: pitch = atan2(sigmaDot, |v|).
    const s = snapFixture({ c: [0.2, 0.0], sigma: 2.0, rho: 0.025, sigmaDot: 0.5, velocity: [1.0, 0.0] });
    expect(physicalPitch(s)).toBeCloseTo(Math.atan2(0.5, 1.0), 9);
  });

  it('treadmill pitch gains the rho0 factor: atan2(lambda*rho0*sigmaDot, |v|)', () => {
    // Without the rho0 factor the rider looks near-vertical near the Shore
    // even though the scale-normalized chart stays legible.
    const s = snapFixture({ c: [0.25, 0.0], sigma: 9.9, rho: 1e-4, sigmaDot: 200.0, velocity: [0.5, 0.0] });
    expect(treadmillPitch(s)).toBeCloseTo(Math.atan2(1.0 * 1e-4 * 200.0, 0.5), 9);
  });

  it('both pitches are zero when sigmaDot is zero', () => {
    const s = snapFixture({ c: [0.2, 0.0], sigma: 2.0, rho: 0.025, sigmaDot: 0, velocity: [1, 0] });
    expect(physicalPitch(s)).toBeCloseTo(0, 12);
    expect(treadmillPitch(s)).toBeCloseTo(0, 12);
  });

  it('pitch on a RECORDED trajectory is finite at every frame (both charts)', () => {
    setWasmModuleForTesting(mockModule as never);
    return initOrbitSynth().then(() => {
      const recorder = new CockpitRecorder();
      for (const variant of [...baselineVariants(), ...[4, 5, 6, 7].flatMap((i) => [baselineVariants()[i % baselineVariants().length]])]) {
        const trajectory = recorder.recordVariant(variant);
        for (const snap of trajectory.snapshots) {
          expect(Number.isFinite(physicalPitch(snap))).toBe(true);
          expect(Number.isFinite(treadmillPitch(snap))).toBe(true);
        }
      }
    });
  });
});

/**
 * Tests for the exported treadmill MESH/TRAIL transforms (the actual
 * Three.js objects the cockpit builds), not a re-derivation of their math.
 * These prove the anisotropic scale fix: 1/rho0 on X/Z, 1 on Y, and the
 * current surface height translated to Y=0 rather than multiplied by 1/rho0.
 */
describe('treadmill mesh + trail transforms (anisotropic 1/rho0)', () => {
  function makeMesh(): THREE.Mesh {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 0, 0, 0]), 3)
    );
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  }

  it('terrain X and Z scale by 1/rho0', () => {
    const s = snapFixture({ c: [0.25, 0.1], sigma: 4.3219, rho: 5e-3 });
    const mesh = makeMesh();
    treadmillTransform(mesh, s);
    expect(mesh.scale.x).toBeCloseTo(1.0 / 5e-3, 9);
    expect(mesh.scale.z).toBeCloseTo(1.0 / 5e-3, 9);
  });

  it('terrain Y scale remains 1 (not magnified by 1/rho0)', () => {
    const s = snapFixture({ c: [0.25, 0.1], sigma: 4.3219, rho: 5e-3 });
    const mesh = makeMesh();
    treadmillTransform(mesh, s);
    expect(mesh.scale.y).toBeCloseTo(1.0, 9);
    // The old uniform setScalar(1/rho0) would have made this 200.
    expect(mesh.scale.y).not.toBeCloseTo(1.0 / 5e-3, 6);
  });

  it('terrain position re-centers the rider own vertex to the origin', () => {
    const s = snapFixture({ c: [0.25, 0.0], sigma: 9.96578, rho: 1e-4 });
    const mesh = makeMesh();
    // A vertex sitting exactly at the rider's own LINEAR chart point
    // (cx*SCENE_SCALE, SCENE_SCALE*LAMBDA*sigma, -cy*SCENE_SCALE) must
    // map to (0,0,0). This is the identity term of the chart transform;
    // note the Y coordinate uses the LINEAR chart, NOT surfaceY(sigma)
    // — surfaceY carries the cosmetic asinh compression reserved for
    // physical mode.
    const gx = 0.25 * SCENE_SCALE;
    const gy = SCENE_SCALE * LAMBDA * 9.96578;
    const gz = 0.0;
    mesh.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([gx, gy, gz]), 3)
    );
    treadmillTransform(mesh, s);
    const p = mesh.geometry.getAttribute('position').array as Float32Array;
    // Apply the object transform: scale then translate. NOTE: positions
    // live in a Float32Array, so Y values of magnitude ~100 carry ~6e-6
    // absolute rounding error — precision 5 matches float32 storage.
    const tx = p[0] * mesh.scale.x + mesh.position.x;
    const ty = p[1] * mesh.scale.y + mesh.position.y;
    const tz = p[2] * mesh.scale.z + mesh.position.z;
    expect(tx).toBeCloseTo(0, 9);
    expect(ty).toBeCloseTo(0, 5);
    expect(tz).toBeCloseTo(0, 9);
  });

  it('terrain position.y subtracts the rider LINEAR chart Y (Y=0 recentering)', () => {
    const s = snapFixture({ c: [0.25, 0.0], sigma: 9.96578, rho: 1e-4 });
    const mesh = makeMesh();
    treadmillTransform(mesh, s);
    // Linear chart Y of the rider, NOT surfaceY(sigma) (the cosmetic
    // physical-mode compression). NOT -SCENE_SCALE * LAMBDA * sigma / rho0
    // either (the old magnified-residue bug).
    expect(mesh.position.y).toBeCloseTo(-SCENE_SCALE * LAMBDA * 9.96578, 9);
    expect(mesh.position.y).not.toBeCloseTo(-surfaceY(9.96578), 6);
    expect(mesh.position.y).not.toBeCloseTo(
      -SCENE_SCALE * LAMBDA * 9.96578 / 1e-4,
      6
    );
  });

  it('trail X and Z scale by 1/rho0', () => {
    const s = snapFixture({ c: [0.25, 0.1], sigma: 4.3219, rho: 5e-3 });
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
      new THREE.LineBasicMaterial()
    );
    treadmillTrailTransform(trail, s);
    expect(trail.scale.x).toBeCloseTo(1.0 / 5e-3, 9);
    expect(trail.scale.z).toBeCloseTo(1.0 / 5e-3, 9);
  });

  it('trail Y scale remains 1 (not magnified by 1/rho0)', () => {
    const s = snapFixture({ c: [0.25, 0.1], sigma: 4.3219, rho: 5e-3 });
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
      new THREE.LineBasicMaterial()
    );
    treadmillTrailTransform(trail, s);
    expect(trail.scale.y).toBeCloseTo(1.0, 9);
    expect(trail.scale.y).not.toBeCloseTo(1.0 / 5e-3, 6);
  });

  it('trail position.y subtracts the rider LINEAR chart Y (Y=0 recentering)', () => {
    const s = snapFixture({ c: [0.25, 0.0], sigma: 9.96578, rho: 1e-4 });
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
      new THREE.LineBasicMaterial()
    );
    treadmillTrailTransform(trail, s);
    // Linear chart Y of the rider — NOT surfaceY(sigma), and NOT the
    // magnified -SCENE_SCALE * LAMBDA * sigma / rho0 residue.
    expect(trail.position.y).toBeCloseTo(-SCENE_SCALE * LAMBDA * 9.96578, 9);
    expect(trail.position.y).not.toBeCloseTo(-surfaceY(9.96578), 6);
    expect(trail.position.y).not.toBeCloseTo(
      -SCENE_SCALE * LAMBDA * 9.96578 / 1e-4,
      6
    );
  });

  it('physical transforms reset scale to uniform 1', () => {
    const mesh = makeMesh();
    mesh.scale.set(200, 1, 200);
    mesh.position.set(5, 5, 5);
    physicalTransform(mesh);
    expect(mesh.scale.x).toBeCloseTo(1, 9);
    expect(mesh.scale.y).toBeCloseTo(1, 9);
    expect(mesh.scale.z).toBeCloseTo(1, 9);
    expect(mesh.position.x).toBe(0);
    expect(mesh.position.y).toBe(0);
    expect(mesh.position.z).toBe(0);

    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
      new THREE.LineBasicMaterial()
    );
    trail.scale.set(200, 1, 200);
    physicalTrailTransform(trail);
    expect(trail.scale.x).toBeCloseTo(1, 9);
    expect(trail.scale.y).toBeCloseTo(1, 9);
    expect(trail.scale.z).toBeCloseTo(1, 9);
  });
});

/**
 * Contract guard: the treadmill chart's vertical coordinate is the pure
 * LINEAR relative log-distance SCENE_SCALE * LAMBDA * (sigma - sigma0),
 * independent of the cosmetic `surfaceY()` asinh compression. The chart
 * is supposed to be a mathematically meaningful local-scale chart, not
 * a presentation curve of the compressed physical embedding.
 *
 * Conversely, physical mode keeps its asinh-compressed surfaceY() Y
 * coordinate unchanged — the cosmetic compression is reserved for
 * physical mode by design.
 */
describe('treadmill chart Y is independent of surfaceY() compression; physical mode unchanged', () => {
  it('treadmill chart Y equals SCENE_SCALE * LAMBDA * (sigma - sigma0) regardless of surfaceY', () => {
    // Two snapshots at the same rider c, with very different sigma values
    // so the old asinh compression (surfaceY) would produce visibly
    // different RATIOS than the linear chart. The new chart must collapse
    // to linear regardless of where on the asinh curve we sit.
    const sA = snapFixture({ c: [0.25, 0.0], sigma: -1.0, rho: 0.5 });
    const sB = snapFixture({ c: [0.25, 0.0], sigma: 5.0, rho: 1e-3 });
    // Same planar offset for both.
    const pA = treadmillChart(sA, 0.251, 0.0, sA.physics.sigma);
    const pB = treadmillChart(sB, 0.251, 0.0, sB.physics.sigma);
    // Treadmill Y at the SAME sigma as the rider is exactly 0 (identity).
    expect(pA.y).toBeCloseTo(0, 9);
    expect(pB.y).toBeCloseTo(0, 9);
    // Now nudge sigma by +0.5 (same delta for both): the Y response must
    // be IDENTICAL for both (linear) — the old asinh compression would
    // give a smaller response at sigma=5.0 than at sigma=-1.0.
    const pAd = treadmillChart(sA, 0.25, 0.0, sA.physics.sigma + 0.5);
    const pBd = treadmillChart(sB, 0.25, 0.0, sB.physics.sigma + 0.5);
    expect(pAd.y).toBeCloseTo(0.5 * SCENE_SCALE * LAMBDA, 9);
    expect(pBd.y).toBeCloseTo(0.5 * SCENE_SCALE * LAMBDA, 9);
    expect(pAd.y).toBeCloseTo(pBd.y, 9);
  });

  it('treadmill Y does NOT equal surfaceY(sigma) - surfaceY(sigma0)', () => {
    // The whole point: the old `treadmillTransform` produced Y =
    // surfaceY(sigma) - surfaceY(sigma0) (the cosmetic difference of
    // physical heights). The new chart Y must NOT equal that value at
    // any sigma where the asinh is nonlinear.
    const s = snapFixture({ c: [0.25, 0.0], sigma: 5.0, rho: 1e-3 });
    const neighbor = treadmillChart(s, 0.25, 0.0, s.physics.sigma + 1.0);
    const chartY = SCENE_SCALE * LAMBDA * 1.0;
    const oldCompressedY = surfaceY(s.physics.sigma + 1.0) - surfaceY(s.physics.sigma);
    // The two values must disagree at sigma=5.0 (well into the asinh
    // compression regime). If they happened to agree the chart would
    // have leaked surfaceY back in and the fix would be incomplete.
    expect(neighbor.y).toBeCloseTo(chartY, 9);
    expect(Math.abs(neighbor.y - oldCompressedY)).toBeGreaterThan(0.1);
  });

  it('buildTerrainMesh(mode=treadmill) sets vertex Y to SCENE_SCALE * LAMBDA * sigma', () => {
    // Build a tiny fake patch and read the position attribute directly —
    // proves the mesh is built in LINEAR chart coordinates for treadmill
    // mode, independent of surfaceY().
    const mesh = buildTerrainMesh(
      makePatch([0.24, 0.01, -1.0, 0.26, -0.01, 5.0]),
      'treadmill'
    );
    const pos = mesh.geometry.getAttribute('position').array as Float32Array;
    expect(pos[1]).toBeCloseTo(SCENE_SCALE * LAMBDA * -1.0, 9);
    expect(pos[4]).toBeCloseTo(SCENE_SCALE * LAMBDA * 5.0, 9);
  });

  it('buildTerrainMesh(mode=physical) keeps vertex Y = surfaceY(sigma) (unchanged)', () => {
    // The default mode is physical and must NOT have changed: vertex Y
    // equals surfaceY(z) for every vertex, exactly as before this PR.
    const mesh = buildTerrainMesh(
      makePatch([0.24, 0.01, -1.0, 0.26, -0.01, 5.0]),
      'physical'
    );
    const pos = mesh.geometry.getAttribute('position').array as Float32Array;
    // Float32Array storage rounds surfaceY values to ~7 significant
    // digits; for surfaceY(5) (~5.7567) that is ~1.3e-7 absolute —
    // precision 6 is the correct float32 storage tolerance here.
    expect(pos[1]).toBeCloseTo(surfaceY(-1.0), 6);
    expect(pos[4]).toBeCloseTo(surfaceY(5.0), 6);
  });

  it('treadmillTransform mesh.position.y is the LINEAR chart Y, not surfaceY(sigma)', () => {
    // A very large sigma would push surfaceY into deep asinh compression
    // (sigma=10 -> surfaceY ~ 6.49, while SCENE_SCALE * LAMBDA * 10 =
    // 100). If the chart recenter still followed surfaceY, the rider
    // would sit at Y=-6.49 rather than Y=-100, breaking the chart Y
    // contract. This test pins the difference.
    const s = snapFixture({ c: [0.25, 0.0], sigma: 10.0, rho: 1e-5 });
    const mesh = makeMesh();
    treadmillTransform(mesh, s);
    expect(mesh.position.y).toBeCloseTo(-SCENE_SCALE * LAMBDA * 10.0, 9);
    expect(Math.abs(mesh.position.y - (-surfaceY(10.0)))).toBeGreaterThan(1.0);
  });

  it('physicalTransform mesh.position.y is 0 (physical mode never recenters)', () => {
    // Physical mode never recenters the mesh; the rider sits at its
    // physical Y = surfaceY(sigma) at c0, and nearby vertices use their
    // own surfaceY(sigma). This must remain unchanged by this PR.
    const mesh = makeMesh();
    mesh.position.set(0, 0, 0);
    mesh.scale.setScalar(1.0);
    physicalTransform(mesh);
    expect(mesh.position.x).toBe(0);
    expect(mesh.position.y).toBe(0);
    expect(mesh.position.z).toBe(0);
    expect(mesh.scale.x).toBeCloseTo(1, 9);
    expect(mesh.scale.y).toBeCloseTo(1, 9);
    expect(mesh.scale.z).toBeCloseTo(1, 9);
  });
});
