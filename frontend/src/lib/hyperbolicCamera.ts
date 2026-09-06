/**
 * Hyperbolic camera projection for the FractalSync debug cockpit.
 *
 * Geometric background:
 * In upper half-space H^3 with metric ds^2 = a^2 (dx^2 + dy^2 + dz^2) / z^2,
 * where a = lambda / ln(2) and z = a * rho(x, y),
 * the cockpit views the 2D graph z(x, y) = a * rho(x, y).
 *
 * Pipeline:
 * 1. Upper-half-space point P = (x, y, z) with z = a * rho > 0.
 * 2. Hyperbolic isometry moving camera C = (cx, cy, cz) (with cz > 0) to canonical point (0, 0, 1):
 *      X = (x - cx) / cz
 *      Y = (y - cy) / cz
 *      Z = z / cz
 * 3. Cayley / Poincaré ball transform mapping (0, 0, 1) -> (0, 0, 0):
 *      d = X^2 + Y^2 + (Z + 1)^2
 *      bx = 2*X / d
 *      by = 2*Y / d
 *      bz = (X^2 + Y^2 + Z^2 - 1) / d
 * 4. Inverse camera orientation rotation:
 *      b_camera = R_camera^{-1} * b
 * 5. Feed b_camera into conventional perspective projection.
 *
 * Note on approximation:
 * This is a hyperbolic camera with conventional triangle rasterization,
 * not an exact geodesic surface intersection renderer.
 * Transformed vertices have exact hyperbolic viewing directions, but the GPU
 * still linearly rasterizes triangle interiors after projection.
 */

import * as THREE from 'three';
import type { DebugSnapshot, TerrainPatch } from './debugCockpit';

/** Upper-half-space metric coefficient a = lambda / ln(2). */
export function computeUpperHalfScaleA(lambda: number): number {
  return lambda / Math.LN2;
}

/**
 * Reconstruct physical rho(c) from the authoritative Rust embedding height
 * z_embed = lambda * sigma(c) = lambda * log2(d_ref / rho).
 *
 * Since z_embed = lambda * (ln(d_ref) - ln(rho)) / ln(2):
 *   z_embed / a = ln(d_ref) - ln(rho)
 *   ln(rho) = ln(d_ref) - z_embed / a
 *   rho = d_ref * 2^(-z_embed / lambda)
 *
 * Under default config (d_ref = 0.1, lambda = 1.0):
 *   rho = 0.1 * 2^(-z_embed)
 *
 * This recovers exact regularized rho(c) directly from the mesh vertex height
 * without recalculating or drifting from authoritative physics parameters.
 */
export function rhoFromEmbeddedHeight(
  zEmbed: number,
  dRef: number = 0.1,
  lambda: number = 1.0
): number {
  const exponent = -zEmbed / Math.max(lambda, 1e-9);
  return dRef * Math.pow(2, exponent);
}

export interface UpperHalfPoint {
  x: number;
  y: number;
  z: number;
}

export interface CameraHyperbolicState {
  /** Camera position in upper half space (cx, cy, cz) where cz > 0. */
  position: UpperHalfPoint;
  /** Camera orientation rotation (applied to world vectors). */
  rotation: THREE.Quaternion | THREE.Matrix4 | THREE.Euler;
}

/**
 * Hyperbolic isometry moving camera C = (cx, cy, cz) to canonical (0, 0, 1):
 *   X = (x - cx) / cz
 *   Y = (y - cy) / cz
 *   Z = z / cz
 */
export function canonicalizeUpperHalf(
  p: UpperHalfPoint,
  c: UpperHalfPoint
): UpperHalfPoint {
  const cz = Math.max(c.z, 1e-12);
  return {
    x: (p.x - c.x) / cz,
    y: (p.y - c.y) / cz,
    z: Math.max(p.z, 1e-12) / cz,
  };
}

/**
 * Map canonical upper-half-space point (X, Y, Z) to Poincaré ball b:
 *   d = X^2 + Y^2 + (Z + 1)^2
 *   bx = 2*X / d
 *   by = 2*Y / d
 *   bz = (X^2 + Y^2 + Z^2 - 1) / d
 *
 * singularity guard: since Z > 0, (Z + 1)^2 >= 1, so d >= 1 for all valid points.
 * We enforce a guard d > 1e-12.
 */
export function upperHalfToPoincareBall(
  canonical: UpperHalfPoint,
  epsilonDenom: number = 1e-12
): THREE.Vector3 {
  const { x: X, y: Y, z: Z } = canonical;
  const d = X * X + Y * Y + (Z + 1) * (Z + 1);
  const safeD = Math.max(d, epsilonDenom);
  const bx = (2 * X) / safeD;
  const by = (2 * Y) / safeD;
  const bz = (X * X + Y * Y + Z * Z - 1) / safeD;
  return new THREE.Vector3(bx, by, bz);
}

/**
 * Full transform from upper half space P=(x,y,z) to camera-centered Poincaré ball vector b_camera.
 *
 * Camera orientation R:
 * If R is the rotation transforming camera-local coordinates to world coordinates,
 * then the inverse rotation R^{-1} transforms world Poincaré ball coordinates b into
 * camera-local coordinates:
 *   b_camera = R^{-1} * b
 */
export function projectPointToCameraPoincare(
  p: UpperHalfPoint,
  cameraPos: UpperHalfPoint,
  cameraRotation?: THREE.Quaternion | THREE.Matrix4
): THREE.Vector3 {
  const canonical = canonicalizeUpperHalf(p, cameraPos);
  const b = upperHalfToPoincareBall(canonical);

  if (cameraRotation) {
    if (cameraRotation instanceof THREE.Quaternion) {
      const invQ = cameraRotation.clone().invert();
      b.applyQuaternion(invQ);
    } else if (cameraRotation instanceof THREE.Matrix4) {
      const invM = cameraRotation.clone().invert();
      b.applyMatrix4(invM);
    }
  }

  return b;
}

/**
 * Map upper-half space point (x, y, z) into Three.js scene coordinates for hyperbolic viewing.
 *
 * Scene conventions:
 * In the FractalSync cockpit:
 * - c-plane (x, y):
 *     horizontal X_scene = x * scale
 *     horizontal Z_scene = -y * scale
 *     vertical   Y_scene = height
 *
 * In upper half-space H^3:
 *   (x, y) are the c-plane coordinates (Re(c), Im(c)).
 *   z = a * rho(c) is the upper-half vertical coordinate (z > 0).
 *
 * When camera is at C = (cx, cy, cz) with cz = a * rho(C):
 * 1. P_H3 = (x, y, z)
 * 2. C_H3 = (cx, cy, cz)
 * 3. Map P_H3 and C_H3 to Poincaré ball:
 *      bx, by, bz
 *    where:
 *      bx corresponds to c-space Re (x),
 *      by corresponds to c-space Im (y),
 *      bz corresponds to upper-half vertical (z).
 *
 * 4. Align with Three.js scene / camera coordinate frame:
 *    In Three.js standard camera space:
 *      +X is Right
 *      +Y is Up
 *      -Z is Forward (into screen)
 *
 *    In our upper-half / Poincaré mapping:
 *      x is Re(c) -> scene X (Right)
 *      y is Im(c) -> in scene, -Im(c) was scene +Z.
 *      z is a*rho -> vertical upper-half coordinate.
 *
 *    To map Poincaré ball coordinates (bx, by, bz) to Three.js unrotated scene coordinates:
 *    Let:
 *      p_scene.x = bx
 *      p_scene.y = bz (vertical scale axis)
 *      p_scene.z = -by (so +Im(c) maps to -Z_scene, matching the existing scene convention)
 */
export function poincareToSceneVector(b: THREE.Vector3): THREE.Vector3 {
  // b.x = Re(c) direction
  // b.y = Im(c) direction
  // b.z = vertical z = a*rho direction
  return new THREE.Vector3(b.x, b.z, -b.y);
}

/**
 * Scale applied to Poincaré ball coordinates (which have norm < 1.0)
 * so that vertices map comfortably into the Three.js near/far clipping range
 * (e.g. near 0.1, far 200) without tiny sub-unit clipping or depth precision loss.
 */
export const HYPERBOLIC_VISUAL_SCALE = 20.0;

/**
 * Compute the camera's upper-half position and orientation rotation matrix
 * from the cockpit snapshot and heading.
 */
export function computeHyperbolicCameraFrame(
  snap: DebugSnapshot,
  smoothedHeading: number | null
): {
  cameraUpperHalf: UpperHalfPoint;
  rotationMatrix: THREE.Matrix4;
} {
  const [cx, cy] = snap.physics.c;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const a = computeUpperHalfScaleA(1.0); // lambda=1.0 default
  const riderZ = a * rho0;

  // In physical/scale-follow modes, camera sits:
  // back distance in horizontal plane along heading
  // and up distance above the rider.
  // In upper half-space H^3, the camera is positioned at a scale-relative offset:
  // camera c_x, c_y is offset backwards by back * rho0 along (cos heading, -sin heading),
  // and camera c_z is elevated above the manifold by a factor (1 + up_factor).
  const heading = smoothedHeading ?? 0;
  // c-space heading direction: (cos(heading), -sin(heading)) in scene corresponds to (cos, sin) in c-space
  // Since scene X = c_x * scale, scene Z = -c_y * scale:
  // camX = rx - cos(h)*back -> in c-space: cx - cos(h) * (back/scale)
  // camZ = rz + sin(h)*back -> -camY_c = -cy + sin(h)*(back/scale) -> camY_c = cy - sin(h)*(back/scale)
  // At local scale rho0:
  const backC = 0.48 * rho0;
  const upZMultiplier = 1.35; // camera flies at 1.35x rider's z height

  const camUpperX = cx - Math.cos(heading) * backC;
  const camUpperY = cy - Math.sin(heading) * backC;
  const camUpperZ = riderZ * upZMultiplier;

  const cameraUpperHalf: UpperHalfPoint = {
    x: camUpperX,
    y: camUpperY,
    z: Math.max(camUpperZ, 1e-9),
  };

  // Rider position in upper-half space
  const riderUpper: UpperHalfPoint = {
    x: cx,
    y: cy,
    z: riderZ,
  };

  // Map rider to camera-centered Poincaré ball to find the look-at target vector in the ball
  const bRider = projectPointToCameraPoincare(riderUpper, cameraUpperHalf);
  const vTargetScene = poincareToSceneVector(bRider);

  // We want a rotation matrix R such that in camera-local space, the target is along the look direction.
  // Camera sits at (0, 0, 0) and looks towards vTargetScene, with Up nominally along +Y.
  const eye = new THREE.Vector3(0, 0, 0);
  const target = vTargetScene.clone().normalize();
  const up = new THREE.Vector3(0, 1, 0);

  // In Three.js, camera.lookAt(target) sets the camera world matrix so that
  // camera's local -Z points to (target - eye).
  // An object transformed into camera-local space has position P_cam = M_world_to_cam * P_world = R^-1 * P_world.
  const rotMat = new THREE.Matrix4();
  if (target.lengthSq() > 1e-6) {
    rotMat.lookAt(eye, target, up);
  } else {
    rotMat.identity();
  }

  return {
    cameraUpperHalf,
    rotationMatrix: rotMat,
  };
}

/**
 * Transform an entire terrain patch into camera-local Poincaré ball coordinates.
 * Operates on the mesh's BufferGeometry position attribute in-place.
 */
export function transformMeshToHyperbolic(
  mesh: THREE.Mesh,
  patch: TerrainPatch,
  snap: DebugSnapshot,
  smoothedHeading: number | null
): void {
  const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(snap, smoothedHeading);
  const invRot = rotationMatrix.clone().invert();

  const posAttr = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
  if (!posAttr) return;
  const positions = posAttr.array as Float32Array;
  const n = patch.n;

  const a = computeUpperHalfScaleA(1.0); // lambda=1.0 default

  // Mesh itself sits at position (0, 0, 0), scale 1.0, rotation identity
  mesh.position.set(0, 0, 0);
  mesh.scale.set(HYPERBOLIC_VISUAL_SCALE, HYPERBOLIC_VISUAL_SCALE, HYPERBOLIC_VISUAL_SCALE);
  mesh.quaternion.set(0, 0, 0, 1);

  for (let i = 0; i < n * n; i++) {
    const x = patch.positions[i * 3];
    const y = patch.positions[i * 3 + 1];
    const zEmbed = patch.positions[i * 3 + 2]; // lambda * sigma(c)

    // z = a * rho(x, y)
    const rho = rhoFromEmbeddedHeight(zEmbed, 0.1, 1.0);
    const zUpper = a * rho;

    const pUpper: UpperHalfPoint = { x, y, z: zUpper };
    const b = projectPointToCameraPoincare(pUpper, cameraUpperHalf);

    // Unrotated scene vector (x -> X_scene, z -> Y_scene, y -> -Z_scene)
    const vScene = poincareToSceneVector(b);

    // Apply inverse camera orientation: v_camera = R_camera^{-1} * vScene
    vScene.applyMatrix4(invRot);

    positions[i * 3] = vScene.x;
    positions[i * 3 + 1] = vScene.y;
    positions[i * 3 + 2] = vScene.z;
  }

  posAttr.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
}

/**
 * Transform a recorded trail into camera-local Poincaré ball coordinates.
 */
export function transformTrailToHyperbolic(
  trail: THREE.Line,
  snapshots: DebugSnapshot[],
  snap: DebugSnapshot,
  smoothedHeading: number | null
): void {
  const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(snap, smoothedHeading);
  const invRot = rotationMatrix.clone().invert();

  trail.position.set(0, 0, 0);
  trail.scale.set(HYPERBOLIC_VISUAL_SCALE, HYPERBOLIC_VISUAL_SCALE, HYPERBOLIC_VISUAL_SCALE);
  trail.quaternion.set(0, 0, 0, 1);

  const posAttr = trail.geometry.getAttribute('position') as THREE.BufferAttribute;
  if (!posAttr) return;
  const positions = posAttr.array as Float32Array;

  const a = computeUpperHalfScaleA(1.0);
  const count = Math.min(snapshots.length, positions.length / 3);

  for (let i = 0; i < count; i++) {
    const s = snapshots[i];
    const [x, y] = s.physics.c;
    const rho = Math.max(s.physics.rho, 1e-9);
    const zUpper = a * rho;

    const pUpper: UpperHalfPoint = { x, y, z: zUpper };
    const b = projectPointToCameraPoincare(pUpper, cameraUpperHalf);
    const vScene = poincareToSceneVector(b);
    vScene.applyMatrix4(invRot);

    positions[i * 3] = vScene.x;
    positions[i * 3 + 1] = vScene.y;
    // Small lift along camera Y for visibility
    positions[i * 3 + 2] = vScene.z;
  }

  posAttr.needsUpdate = true;
}

/**
 * Transform rider position and orientation into camera-local Poincaré ball coordinates.
 */
export function transformRiderToHyperbolic(
  rider: THREE.Group,
  snap: DebugSnapshot,
  smoothedHeading: number | null
): void {
  const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(snap, smoothedHeading);
  const invRot = rotationMatrix.clone().invert();

  const [cx, cy] = snap.physics.c;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const a = computeUpperHalfScaleA(1.0);
  const zUpper = a * rho0;

  const pUpper: UpperHalfPoint = { x: cx, y: cy, z: zUpper };
  const b = projectPointToCameraPoincare(pUpper, cameraUpperHalf);
  const vScene = poincareToSceneVector(b);
  vScene.applyMatrix4(invRot);

  rider.position.set(
    vScene.x * HYPERBOLIC_VISUAL_SCALE,
    vScene.y * HYPERBOLIC_VISUAL_SCALE,
    vScene.z * HYPERBOLIC_VISUAL_SCALE
  );

  // Rider scale in Poincaré ball: local visual size
  rider.scale.setScalar(0.4);

  // Rider orientation: align with forward direction in camera space
  // Target direction is straight ahead in camera space (-Z)
  rider.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0);
}


/**
 * Hyperbolic distance in upper half-space / Poincaré ball:
 *   s = 2 * a * atanh(||b||)
 * For ||b|| < 1.
 */
export function poincareHyperbolicDistance(b: THREE.Vector3, a: number = 1.0): number {
  const r = Math.min(b.length(), 0.999999999);
  return 2 * a * Math.atanh(r);
}
