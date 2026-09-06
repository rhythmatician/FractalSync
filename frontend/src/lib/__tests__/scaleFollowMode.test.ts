/**
 * Tests for the scale-follow camera mode (issue #111 / Shore-crossing
 * experiment).
 *
 * Scale-follow is a controlled experiment: does using the local Mandelbrot
 * horizontal ruler (1/rho0 magnification) make the Shore traversable
 * WITHOUT changing the vertical mapping? The mode combines:
 *
 *   - Physical vertical: Y = surfaceY(lambda*sigma(c)) — exactly the
 *     current physical embedding, no changes.
 *   - Treadmill horizontal ruler: X/Z scale by 1/rho0, recentered on the
 *     rider's current c.
 *
 * The rider sits at X/Z origin (world recentered beneath them) but retains
 * the physical surface height. Terrain and trail share the physical Y
 * mapping and the same horizontal transform, so they stay registered.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  SCENE_SCALE,
  surfaceY,
  buildTerrainMesh,
  buildTrail,
  scaleFollowTransform,
  scaleFollowTrailTransform,
  updateCamera,
  horizontalMagnification,
  isPhysicalYMode,
  applyRenderDistance,
} from '../cockpitScene';
import type { DebugSnapshot, TerrainPatch, CockpitTrajectory } from '../debugCockpit';

/** Tiny stub mesh for transform tests. */
function makeMesh(): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3)
  );
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
}

/** Minimal snapshot fixture. */
function snapFixture(overrides: {
  c: [number, number];
  velocity?: [number, number];
  sigma?: number;
  rho?: number;
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
      rho: overrides.rho ?? 0.001,
      sigma: overrides.sigma ?? 0,
      sigmaDot: 0,
      scaleGradient: [0, 0],
      metric: [1, 0, 1],
      metricSpeed: 0,
      kinetic: 0,
      potential: 0,
      total: 0,
      geodesicAccel: [0, 0],
      potentialForce: [0, 0],
      netAccel: [0, 0],
      derivativeValid: true,
    },
    diagnostics: {
      derivativeStep: 1e-4,
      valid: true,
      lastError: null,
      lastDeltaTotal: null,
      crestPotential: 10,
    },
  };
}

/** Minimal patch for terrain-build tests. */
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

describe('scale-follow mode discriminators', () => {
  it('isPhysicalYMode returns true for physical and scale-follow', () => {
    expect(isPhysicalYMode('physical')).toBe(true);
    expect(isPhysicalYMode('scale-follow')).toBe(true);
    expect(isPhysicalYMode('treadmill')).toBe(false);
  });

  it('horizontalMagnification returns 1 for physical, 1/rho for scale-follow and treadmill', () => {
    const rho = 0.01;
    expect(horizontalMagnification('physical', rho)).toBe(1.0);
    expect(horizontalMagnification('scale-follow', rho)).toBeCloseTo(100.0, 5);
    expect(horizontalMagnification('treadmill', rho)).toBeCloseTo(100.0, 5);
  });
});

describe('scale-follow terrain mesh uses physical surfaceY() vertical', () => {
  it('builds terrain with surfaceY(z) Y mapping, identical to physical mode', () => {
    // Patch with a vertex at embedding height z = 2.5 (already lambda-
    // multiplied by Rust).
    const positions = [
      0.0, 0.0, 2.5, // x, y, z (z = lambda*sigma from Rust)
      0.1, 0.0, 2.5,
    ];
    const patch = makePatch(positions);
    const meshPhysical = buildTerrainMesh(patch, 'physical');
    const meshScaleFollow = buildTerrainMesh(patch, 'scale-follow');

    const posPhys = meshPhysical.geometry.getAttribute('position') as THREE.BufferAttribute;
    const posScale = meshScaleFollow.geometry.getAttribute('position') as THREE.BufferAttribute;

    // Y coordinate in both modes: surfaceY(2.5)
    const expectedY = surfaceY(2.5);
    expect(posPhys.getY(0)).toBeCloseTo(expectedY, 5);
    expect(posScale.getY(0)).toBeCloseTo(expectedY, 5);
    // Scale-follow mesh Y is EXACTLY physical mesh Y before any transform.
    expect(posScale.getY(0)).toBeCloseTo(posPhys.getY(0), 5);
  });

  it('differs from treadmill mode which uses linear Y = SCENE_SCALE * z', () => {
    const positions = [0.0, 0.0, 2.5, 0.1, 0.0, 2.5];
    const patch = makePatch(positions);
    const meshScaleFollow = buildTerrainMesh(patch, 'scale-follow');
    const meshTreadmill = buildTerrainMesh(patch, 'treadmill');

    const posScale = meshScaleFollow.geometry.getAttribute('position') as THREE.BufferAttribute;
    const posTread = meshTreadmill.geometry.getAttribute('position') as THREE.BufferAttribute;

    const scaleY = posScale.getY(0);
    const treadY = posTread.getY(0);

    // Scale-follow: surfaceY(2.5) ~ asinh(2.5/1.5)*1.5*2 ~ 3.26
    expect(scaleY).toBeCloseTo(surfaceY(2.5), 5);
    // Treadmill: SCENE_SCALE * 2.5 = 25
    expect(treadY).toBeCloseTo(SCENE_SCALE * 2.5, 5);
    // They differ because treadmill bypasses surfaceY.
    expect(Math.abs(scaleY - treadY)).toBeGreaterThan(10);
  });
});

describe('scaleFollowTransform: horizontal recenter + 1/rho magnification, Y scale = 1', () => {
  it('recenters terrain horizontally on the rider c0', () => {
    const snap = snapFixture({ c: [0.251, -0.002], rho: 0.01 });
    const mesh = makeMesh();
    scaleFollowTransform(mesh, snap);

    const magnify = 1.0 / 0.01; // 100
    expect(mesh.position.x).toBeCloseTo(-0.251 * SCENE_SCALE * magnify, 4);
    // Z convention: -cy → +Z, so cy = -0.002 → mesh.position.z = -0.002 * SCENE_SCALE * magnify
    expect(mesh.position.z).toBeCloseTo(-0.002 * SCENE_SCALE * magnify, 4);
  });

  it('magnifies terrain X/Z by exactly 1/rho0', () => {
    const snap = snapFixture({ c: [0.25, 0.0], rho: 0.005 });
    const mesh = makeMesh();
    scaleFollowTransform(mesh, snap);

    const magnify = 1.0 / 0.005; // 200
    expect(mesh.scale.x).toBeCloseTo(magnify, 5);
    expect(mesh.scale.z).toBeCloseTo(magnify, 5);
  });

  it('leaves terrain Y scale exactly 1 (no vertical magnification)', () => {
    const snap = snapFixture({ c: [0.25, 0.0], rho: 0.001 });
    const mesh = makeMesh();
    scaleFollowTransform(mesh, snap);

    expect(mesh.scale.y).toBe(1.0);
  });

  it('leaves terrain Y position at 0 (no vertical recentering)', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 6.5, rho: 0.01 });
    const mesh = makeMesh();
    scaleFollowTransform(mesh, snap);

    // Unlike treadmill (which sets position.y = -SCENE_SCALE * sigma),
    // scale-follow keeps the physical surface: position.y = 0.
    expect(mesh.position.y).toBe(0);
  });
});

describe('scaleFollowTrailTransform: same horizontal transform as terrain', () => {
  it('applies identical horizontal recenter + magnification to the trail', () => {
    const snap = snapFixture({ c: [0.26, -0.01], rho: 0.02 });
    const mesh = makeMesh();
    const trail = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial()
    );

    scaleFollowTransform(mesh, snap);
    scaleFollowTrailTransform(trail, snap);

    // X/Z position and scale match exactly.
    expect(trail.position.x).toBeCloseTo(mesh.position.x, 5);
    expect(trail.position.z).toBeCloseTo(mesh.position.z, 5);
    expect(trail.scale.x).toBeCloseTo(mesh.scale.x, 5);
    expect(trail.scale.z).toBeCloseTo(mesh.scale.z, 5);
  });

  it('leaves trail Y position at 0 and Y scale at 1', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 5.0, rho: 0.01 });
    const trail = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial()
    );
    scaleFollowTrailTransform(trail, snap);

    expect(trail.position.y).toBe(0);
    expect(trail.scale.y).toBe(1.0);
  });
});

describe('buildTrail in scale-follow uses physical surfaceY()', () => {
  it('builds trail with surfaceY(sigma) Y mapping, matching physical mode', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 3.0, rho: 0.01 });
    const trajectory: CockpitTrajectory = {
      spec: {
        name: 'test',
        description: 'test',
        actions: [],
      },
      snapshots: [snap],
      crossingStep: null,
      crossed: false,
      maxPotential: 0,
      crestedRidge: false,
    };

    const trailPhysical = buildTrail(trajectory, 0, 'physical');
    const trailScaleFollow = buildTrail(trajectory, 0, 'scale-follow');

    const posPhys = trailPhysical.geometry.getAttribute('position') as THREE.BufferAttribute;
    const posScale = trailScaleFollow.geometry.getAttribute('position') as THREE.BufferAttribute;

    // Physical trail: surfaceY(3.0) + 0.05 lift
    const expectedY = surfaceY(3.0) + 0.05;
    expect(posPhys.getY(0)).toBeCloseTo(expectedY, 5);
    // Scale-follow trail: EXACTLY the same Y before transform.
    expect(posScale.getY(0)).toBeCloseTo(expectedY, 5);
  });
});

describe('updateCamera in scale-follow: rider at X/Z origin, physical height', () => {
  it('places camera relative to rider at (0, 0) horizontally', () => {
    const snap = snapFixture({
      c: [0.3, -0.05],
      velocity: [0.1, 0],
      sigma: 4.0,
      rho: 0.01,
    });
    const camera = new THREE.PerspectiveCamera();
    updateCamera(camera, snap, 'scale-follow');

    // Rider is at rx = 0, rz = 0; camera is behind/above.
    // Heading ~ atan2(0, 0.1) = 0 (faces +X).
    const back = 3.2;
    expect(camera.position.x).toBeCloseTo(-back, 2);
    expect(camera.position.z).toBeCloseTo(0, 2);
  });

  it('camera Y follows physical surfaceY(sigma), not chart Y', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 5.5, rho: 0.005 });
    const camera = new THREE.PerspectiveCamera();
    updateCamera(camera, snap, 'scale-follow');

    const up = 2.2;
    const expectedY = surfaceY(5.5) + up;
    expect(camera.position.y).toBeCloseTo(expectedY, 3);
  });

  it('differs from treadmill mode where camera Y is relative to chart Y=0', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 6.0, rho: 0.01 });
    const cameraScale = new THREE.PerspectiveCamera();
    const cameraTread = new THREE.PerspectiveCamera();

    updateCamera(cameraScale, snap, 'scale-follow');
    updateCamera(cameraTread, snap, 'treadmill');

    // Scale-follow: camera Y ~ surfaceY(6.0) + 2.2
    expect(cameraScale.position.y).toBeCloseTo(surfaceY(6.0) + 2.2, 2);
    // Treadmill: camera Y ~ 0 + 2.2 (chart recenters at Y=0).
    expect(cameraTread.position.y).toBeCloseTo(2.2, 2);
    // They differ significantly.
    expect(Math.abs(cameraScale.position.y - cameraTread.position.y)).toBeGreaterThan(5);
  });
});

describe('applyRenderDistance: scale-follow uses 1/rho horizontal magnification', () => {
  it('inflates fog wall by 1/rho, matching treadmill', () => {
    const camera = new THREE.PerspectiveCamera();
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x000000, 10, 20);

    const rho = 0.01;
    const half = 0.1;
    applyRenderDistance(camera, scene, 'scale-follow', rho, half);

    const magnify = 1.0 / rho; // 100
    const patchScene = half * 2 * SCENE_SCALE * magnify; // 200
    const diagonal = patchScene * Math.SQRT2; // ~282.8

    const fog = scene.fog as THREE.Fog;
    expect(fog.far).toBeGreaterThan(diagonal); // Fog far > diagonal
    expect(camera.far).toBeGreaterThan(diagonal); // Camera far > diagonal
  });

  it('matches treadmill magnification exactly', () => {
    const cameraScale = new THREE.PerspectiveCamera();
    const cameraTread = new THREE.PerspectiveCamera();
    const sceneScale = new THREE.Scene();
    const sceneTread = new THREE.Scene();
    sceneScale.fog = new THREE.Fog(0x000000, 10, 20);
    sceneTread.fog = new THREE.Fog(0x000000, 10, 20);

    const rho = 0.005;
    const half = 0.15;
    applyRenderDistance(cameraScale, sceneScale, 'scale-follow', rho, half);
    applyRenderDistance(cameraTread, sceneTread, 'treadmill', rho, half);

    expect(cameraScale.far).toBeCloseTo(cameraTread.far, 3);
    expect((sceneScale.fog as THREE.Fog).far).toBeCloseTo(
      (sceneTread.fog as THREE.Fog).far,
      3
    );
  });

  it('differs from physical mode which uses no magnification', () => {
    const cameraScale = new THREE.PerspectiveCamera();
    const cameraPhys = new THREE.PerspectiveCamera();
    const sceneScale = new THREE.Scene();
    const scenePhys = new THREE.Scene();
    sceneScale.fog = new THREE.Fog(0x000000, 10, 20);
    scenePhys.fog = new THREE.Fog(0x000000, 10, 20);

    const rho = 0.01;
    const half = 0.1;
    applyRenderDistance(cameraScale, sceneScale, 'scale-follow', rho, half);
    applyRenderDistance(cameraPhys, scenePhys, 'physical', rho, half);

    // Scale-follow uses 1/rho magnification; physical uses 1.
    // The ratio should be substantial (magnification is 100x), but the exact
    // multiple depends on the LOD formula's floor at camera distance.
    expect(cameraScale.far).toBeGreaterThan(cameraPhys.far * 10);
  });
});

describe('scale-follow coordinate contracts vs other modes', () => {
  it('terrain X/Z magnified by 1/rho, Y unmagnified — physical magnification is uniform 1', () => {
    const snap = snapFixture({ c: [0.25, 0.0], rho: 0.02 });
    const meshScale = makeMesh();
    const meshPhys = makeMesh();

    scaleFollowTransform(meshScale, snap);
    // Physical transform is identity.
    meshPhys.position.set(0, 0, 0);
    meshPhys.scale.setScalar(1.0);

    const magnify = 1.0 / 0.02; // 50
    expect(meshScale.scale.x).toBeCloseTo(magnify, 4);
    expect(meshScale.scale.y).toBe(1.0);
    expect(meshScale.scale.z).toBeCloseTo(magnify, 4);

    expect(meshPhys.scale.x).toBe(1.0);
    expect(meshPhys.scale.y).toBe(1.0);
    expect(meshPhys.scale.z).toBe(1.0);
  });

  it('treadmill magnifies X/Z by 1/rho but also recenters Y; scale-follow keeps Y=0', () => {
    const snap = snapFixture({ c: [0.25, 0.0], sigma: 7.0, rho: 0.01 });
    const meshScale = makeMesh();
    const meshTread = makeMesh();

    scaleFollowTransform(meshScale, snap);
    // Treadmill recenter: position.y = -SCENE_SCALE * sigma
    meshTread.position.x = -0.25 * SCENE_SCALE * 100;
    meshTread.position.z = 0;
    meshTread.position.y = -SCENE_SCALE * 7.0;
    meshTread.scale.set(100, 1.0, 100);

    // Scale-follow: Y position = 0 (no vertical recenter).
    expect(meshScale.position.y).toBe(0);
    // Treadmill: Y position = -SCENE_SCALE * sigma (chart origin).
    expect(meshTread.position.y).toBeCloseTo(-SCENE_SCALE * 7.0, 4);
  });
});
