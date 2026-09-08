import { describe, it, expect } from 'vitest';
import {
  computeHyperbolicCameraFrame,
  projectPointToCameraPoincare,
  poincareToSceneVector,
} from '../hyperbolicCamera';
import type { DebugSnapshot } from '../debugCockpit';
import * as THREE from 'three';
import { updateCamera, resetCameraSmoothing, getSmoothedCamHeading, placeRider } from '../cockpitScene';
import { transformRiderToHyperbolic, projectUpperHalfTangent } from '../hyperbolicCamera';

describe('hyperbolic camera lookAt target direction', () => {
  const dummySnapshot: DebugSnapshot = {
    version: 'debug-snapshot/2',
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
      upperHalf: { a: 1 / Math.LN2, z: 0.0026714 / Math.LN2, gradient: [0, 0], zDot: 0 },
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

  it.each([[1, 0], [-1, 0], [0, 1], [0, -1]])('chases behind velocity (%s, %s)', (vx, vy) => {
    const snap: DebugSnapshot = { ...dummySnapshot, physics: { ...dummySnapshot.physics, velocity: [vx, vy] } };
    resetCameraSmoothing();
    updateCamera(new THREE.PerspectiveCamera(), snap);
    const { cameraUpperHalf: camera } = computeHyperbolicCameraFrame(snap, getSmoothedCamHeading());
    expect((camera.x - snap.physics.c[0]) * vx + (camera.y - snap.physics.c[1]) * vy).toBeLessThan(0);
  });

  it('continues following turns while hyperbolic mode is active', () => {
    resetCameraSmoothing();
    const camera = new THREE.PerspectiveCamera();
    updateCamera(camera, { ...dummySnapshot, physics: { ...dummySnapshot.physics, velocity: [1, 0] } });
    const before = getSmoothedCamHeading();
    updateCamera(camera, { ...dummySnapshot, physics: { ...dummySnapshot.physics, velocity: [0, 1] } });
    expect(getSmoothedCamHeading()).not.toBe(before);
  });

  it('restores rider scale after the transient Euclidean staging pass', () => {
    const rider = new THREE.Group();
    transformRiderToHyperbolic(rider, dummySnapshot, 0);
    placeRider(rider, dummySnapshot, () => 0);
    expect(rider.scale.toArray()).toEqual([1, 1, 1]);
  });

  it.each([[1, 0], [-1, 0], [0, 1], [0, -1]])('aligns local +X with the projected sloped trajectory (%s, %s)', (vx, vy) => {
    const upper = { a: 2.3, z: 0.006, gradient: [0.4, -0.3] as [number, number], zDot: 0.4 * vx - 0.3 * vy };
    const snap: DebugSnapshot = { ...dummySnapshot, physics: { ...dummySnapshot.physics, velocity: [vx, vy], upperHalf: upper } };
    const frame = computeHyperbolicCameraFrame(snap, Math.atan2(vy, vx));
    const rider = new THREE.Group();
    transformRiderToHyperbolic(rider, snap, Math.atan2(vy, vx), frame);
    const p = { x: snap.physics.c[0], y: snap.physics.c[1], z: upper.z };
    // Independent centered-difference oracle for the displayed trajectory.
    const h = 1e-8;
    const project = (sign: number) => poincareToSceneVector(projectPointToCameraPoincare({
      x: p.x + sign * h * vx, y: p.y + sign * h * vy, z: p.z + sign * h * upper.zDot,
    }, frame.cameraUpperHalf)).applyMatrix4(frame.rotationMatrix.clone().invert());
    const trajectory = project(1).sub(project(-1)).normalize();
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(rider.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(rider.quaternion);
    const normal = projectUpperHalfTangent(p, new THREE.Vector3(...upper.gradient, -1), frame.cameraUpperHalf).transformDirection(frame.rotationMatrix.clone().invert());
    expect(forward.dot(trajectory)).toBeGreaterThan(0.999999);
    expect(up.dot(normal)).toBeGreaterThan(0.999999);
    expect(rider.quaternion.length()).toBeCloseTo(1, 12);
  });

  it('retains heading at rest and does not advance smoothing at dt=0', () => {
    resetCameraSmoothing();
    const camera = new THREE.PerspectiveCamera();
    const snap: DebugSnapshot = { ...dummySnapshot, physics: { ...dummySnapshot.physics, velocity: [0, 1] } };
    updateCamera(camera, snap);
    const heading = getSmoothedCamHeading();
    updateCamera(camera, { ...snap, physics: { ...snap.physics, velocity: [0, 0] } });
    expect(getSmoothedCamHeading()).toBe(heading);
    updateCamera(camera, { ...snap, physics: { ...snap.physics, velocity: [1, 0] } }, 0);
    expect(getSmoothedCamHeading()).toBe(heading);
  });

  it('pins the Three.js camera at the Poincaré ball origin with default orientation', () => {
    resetCameraSmoothing();
    const camera = new THREE.PerspectiveCamera();
    const snap: DebugSnapshot = { ...dummySnapshot, physics: { ...dummySnapshot.physics, c: [0, 0], velocity: [0, 1] } };
    updateCamera(camera, snap);
    expect(camera.position.toArray()).toEqual([0, 0, 0]);
    expect(camera.quaternion.toArray()).toEqual([0, 0, 0, 1]);
  });

  it('checks rider vector in camera space after applying inverse rotation', () => {
    const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(dummySnapshot, 0);
    const invRot = rotationMatrix.clone().invert();

    const [cx, cy] = dummySnapshot.physics.c;
    const zRider = dummySnapshot.physics.upperHalf!.z;

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
