import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { DebugSnapshot } from '../debugCockpit';
import {
  canonicalizeUpperHalf,
  computeHyperbolicCameraFrame,
  poincareToSceneVector,
  projectPointToCameraPoincare,
  transformRiderToHyperbolic,
  transportUpperHalfDirectionToCameraOrigin,
  type UpperHalfPoint,
} from '../hyperbolicCamera';

const baseSnapshot: DebugSnapshot = {
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
    velocity: [1.0, 0.35],
    signedDistance: 0.0026695,
    realm: 1,
    rho: 0.0026714,
    upperHalf: {
      a: 2.3,
      z: 0.0026714 * 2.3,
      gradient: [0, 0],
      zDot: 0,
    },
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

function snapshotWithGradient(gradient: [number, number]): DebugSnapshot {
  const [vx, vy] = baseSnapshot.physics.velocity;
  return {
    ...baseSnapshot,
    physics: {
      ...baseSnapshot.physics,
      upperHalf: {
        ...baseSnapshot.physics.upperHalf!,
        gradient,
        zDot: gradient[0] * vx + gradient[1] * vy,
      },
    },
  };
}

type MinkowskiVector = [number, number, number, number];

function minkowskiDot(a: MinkowskiVector, b: MinkowskiVector): number {
  return -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function upperHalfPointToHyperboloid(p: UpperHalfPoint): MinkowskiVector {
  const r2 = p.x * p.x + p.y * p.y;
  return [
    (r2 + p.z * p.z + 1) / (2 * p.z),
    p.x / p.z,
    p.y / p.z,
    (r2 + p.z * p.z - 1) / (2 * p.z),
  ];
}

function upperHalfTangentToHyperboloid(
  p: UpperHalfPoint,
  v: THREE.Vector3
): MinkowskiVector {
  const r2 = p.x * p.x + p.y * p.y;
  const z2 = p.z * p.z;
  return [
    (p.x / p.z) * v.x + (p.y / p.z) * v.y + ((z2 - r2 - 1) / (2 * z2)) * v.z,
    v.x / p.z - (p.x / z2) * v.z,
    v.y / p.z - (p.y / z2) * v.z,
    (p.x / p.z) * v.x + (p.y / p.z) * v.y + ((z2 - r2 + 1) / (2 * z2)) * v.z,
  ];
}

/** Independent hyperboloid-model oracle for transport along point -> camera. */
function hyperboloidTransportDirectionAtCameraOrigin(
  point: UpperHalfPoint,
  tangent: THREE.Vector3,
  camera: UpperHalfPoint
): THREE.Vector3 {
  const canonicalPoint = canonicalizeUpperHalf(point, camera);
  const canonicalTangent = tangent.clone().divideScalar(camera.z);
  const x = upperHalfPointToHyperboloid(canonicalPoint);
  const v = upperHalfTangentToHyperboloid(canonicalPoint, canonicalTangent);
  const y: MinkowskiVector = [1, 0, 0, 0];
  const scale = minkowskiDot(y, v) / (1 - minkowskiDot(x, y));
  const transported: MinkowskiVector = [0, 1, 2, 3].map(
    (i) => v[i] + scale * (x[i] + y[i])
  ) as MinkowskiVector;

  // At hyperboloid basepoint y=(1,0,0,0), the differential to the
  // Poincare-ball origin maps spatial tangent components by a common 1/2.
  // Direction is therefore just the scene-mapped spatial vector.
  return poincareToSceneVector(
    new THREE.Vector3(transported[1], transported[2], transported[3])
  ).normalize();
}

describe('transported hyperbolic camera level', () => {
  it('matches independent hyperboloid parallel transport to the camera', () => {
    const point = { x: -0.743, y: 0.131, z: 0.0062 };
    const camera = { x: -0.745, y: 0.1305, z: 0.0041 };
    const tangent = new THREE.Vector3(0.4, -0.3, -1).normalize();

    const actual = transportUpperHalfDirectionToCameraOrigin(point, tangent, camera);
    const expected = hyperboloidTransportDirectionAtCameraOrigin(point, tangent, camera);

    expect(actual.dot(expected)).toBeGreaterThan(1 - 1e-12);
  });

  it.each([
    ['flat control', [0, 0] as [number, number]],
    ['sloped Mandelbrot terrain', [0.4, -0.3] as [number, number]],
  ])('keeps the transported local-level horizon horizontal on %s', (_name, gradient) => {
    const snap = snapshotWithGradient(gradient);
    const [vx, vy] = snap.physics.velocity;
    const heading = Math.atan2(vy, vx);
    const frame = computeHyperbolicCameraFrame(snap, heading);
    const geometry = snap.physics.upperHalf!;
    const point = { x: snap.physics.c[0], y: snap.physics.c[1], z: geometry.z };
    const surfaceNormal = new THREE.Vector3(gradient[0], gradient[1], -1).normalize();
    const transportedUp = transportUpperHalfDirectionToCameraOrigin(
      point,
      surfaceNormal,
      frame.cameraUpperHalf
    );
    const invRot = frame.rotationMatrix.clone().invert();
    const cameraLevelNormal = transportedUp.clone().transformDirection(invRot);

    // The transported level normal has no camera-right component, so the
    // level plane intersects the image plane in a horizontal line. Pitch is
    // still free, hence the normal may retain a camera-forward component.
    expect(cameraLevelNormal.x).toBeCloseTo(0, 12);
    expect(cameraLevelNormal.y).toBeGreaterThan(0);

    const target = poincareToSceneVector(
      projectPointToCameraPoincare(frame.targetUpperHalf, frame.cameraUpperHalf)
    ).applyMatrix4(invRot);
    expect(target.x).toBeCloseTo(0, 12);
    expect(target.y).toBeCloseTo(0, 12);
    expect(target.z).toBeLessThan(0);

    const rider = new THREE.Group();
    transformRiderToHyperbolic(rider, snap, heading, frame);
    const riderUp = new THREE.Vector3(0, 1, 0).applyQuaternion(rider.quaternion);
    expect(riderUp.dot(cameraLevelNormal)).toBeGreaterThan(0.999999);
  });

  it('does not collapse sloped terrain back to the old fixed scene-up convention', () => {
    const snap = snapshotWithGradient([0.4, -0.3]);
    const [vx, vy] = snap.physics.velocity;
    const frame = computeHyperbolicCameraFrame(snap, Math.atan2(vy, vx));
    const geometry = snap.physics.upperHalf!;
    const point = { x: snap.physics.c[0], y: snap.physics.c[1], z: geometry.z };
    const transportedUp = transportUpperHalfDirectionToCameraOrigin(
      point,
      new THREE.Vector3(geometry.gradient[0], geometry.gradient[1], -1).normalize(),
      frame.cameraUpperHalf
    );

    expect(transportedUp.dot(new THREE.Vector3(0, 1, 0))).toBeLessThan(0.95);
  });
});
