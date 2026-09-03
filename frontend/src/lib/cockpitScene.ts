/**
 * Three.js scene builder for the Mandelbrot-manifold debug cockpit
 * (issue #111 Phase A).
 *
 * Everything geometric here comes from the Rust DebugSnapshot/TerrainPatch
 * seam (see debugCockpit.ts) — this module contains NO manifold math. It
 * only converts authoritative samples into Three.js objects:
 *
 * - Terrain mesh: Q(c) = (x, y, lambda*sigma(c)) sampled by Rust;
 *   vertex colors encode realm (inside/outside/Shore).
 * - Rider: low-poly board + capsule, heading/pitch from authoritative
 *   c, velocity, and embedding geometry.
 * - Trail: the recorded c(t) polyline lifted onto the surface.
 * - Camera modes: physical embedding and scale-stabilized treadmill.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { DebugSnapshot, TerrainPatch, CockpitTrajectory } from './debugCockpit';

/** Camera presentation modes (issue #111). */
export type CameraMode = 'physical' | 'treadmill';

/**
 * Terrain-mesh build mode: 'physical' renders y = surfaceY(z) (the
 * asinh-compressed physical embedding); 'treadmill' renders y =
 * SCENE_SCALE * z so the chart Y is the exact relative embedding height,
 * with NO nonlinear compression. The two modes share the same Rust patch
 * input — only the Y mapping differs.
 */
export type TerrainMeshMode = CameraMode;

/** Scene scale: world units per c-space unit (visual magnification).
 *  Exported so the LOD planner (debugCockpit) plans in the same units. */
export const SCENE_SCALE = 10.0;

/**
 * NOTE on lambda: the canonical embedding height is lambda*sigma(c), and
 * Rust owns lambda (ManifoldConfig.lambda_sq). TerrainPatch.positions
 * arrive with z ALREADY multiplied by lambda (runtime-core debug.rs:
 * "z = lambda * sigma(c)"), so this module must not keep its own copy of
 * the constant — a second factor would silently square it. The
 * DebugSnapshot's physics.sigma is raw sigma; it coincides with the patch
 * height under the controller-default lambda^2 = 1 config that
 * `sampleTerrainPatch` samples with.
 */

/**
 * Vertical scale of the sigma axis (PHYSICAL MODE ONLY).
 *
 * Issue feedback: the raw lambda*sigma surface is far too steep — near the
 * Shore sigma reaches ~10 while valleys sit at ~-1.5, so linear scaling
 * turns hills into cliffs. surfaceY() applies an asinh compression
 * (logarithmic for large |sigma|, linear near 0) so the crest stays
 * dramatic but ridable, and EVERY scene height in physical mode (mesh,
 * rider, trail, camera) goes through this one function — no place
 * re-derives it. TREADMILL MODE does not use surfaceY (see the chart
 * helpers below).
 */
const Z_SCALE = 2.0;

/** asinh knee: below |sigma| ~ Z_COMPRESS_LINEAR the mapping is ~linear. */
const Z_COMPRESS_LINEAR = 1.5;

/**
 * Scene Y for a sigma value — the single physical-mode vertical authority.
 * asinh(sigma / k) * k * Z_SCALE: linear for |sigma| << k, logarithmic for
 * |sigma| >> k. Preserves sign and monotonicity (uphill stays uphill).
 * Treadmill mode bypasses this entirely (see treadmillChart).
 */
export function surfaceY(sigma: number): number {
  const s = sigma / Z_COMPRESS_LINEAR;
  return Math.asinh(s) * Z_COMPRESS_LINEAR * Z_SCALE;
}

// ---------------------------------------------------------------------------
// Treadmill chart (issue #111 / point 2)
// ---------------------------------------------------------------------------
//
// The scale-stabilized treadmill chart re-expresses the canonical embedding
// around the CURRENT rider point (x0, y0, sigma0), where sigma0 is the
// rider's own REGULARIZED embedding height rho(c0):
//
//   X = (x - x0) / rho0
//   Y = (lambda*sigma)(c) - (lambda*sigma)(c0)
//   Z = -(y - y0) / rho0
//
// Horizontal magnification 1/rho0 keeps local terrain resolvable as the
// rider descends into finer Mandelbrot scale; the vertical axis keeps scale
// 1 — height difference is never magnified. All three coordinates come
// from authoritative Rust data (embedding positions / snapshot physics);
// no lambda constant is duplicated here.
//
// The Y mapping is the pure linear relative embedding height, NOT
// surfaceY(sigma) - surfaceY(sigma0): the asinh compression in surfaceY is
// a presentation curve reserved for physical mode.

/**
 * Treadmill chart position for an arbitrary c-space point given a
 * snapshot defining the chart origin (c0, sigma0). The single authority
 * for "what scene position corresponds to a c-space point in treadmill
 * mode". `ySigma` is the point's REGULARIZED embedding height rho(c)
 * (the quantity Rust embeds; equals physics.sigma under the
 * controller-default lambda^2 = 1 config), NOT a raw sigma recomputed in
 * TypeScript.
 */
export function treadmillChart(
  snap: DebugSnapshot,
  x: number = snap.physics.c[0],
  y: number = snap.physics.c[1],
  ySigma: number = snap.physics.sigma
): { x: number; y: number; z: number } {
  const [cx, cy] = snap.physics.c;
  const sigma0 = snap.physics.sigma;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  return {
    x: ((x - cx) / rho0) * SCENE_SCALE,
    y: SCENE_SCALE * (ySigma - sigma0),
    z: -((y - cy) / rho0) * SCENE_SCALE,
  };
}

/**
 * Build (or rebuild) the terrain mesh from a Rust-sampled TerrainPatch.
 *
 * - mode = 'physical' (default): y = surfaceY(z) — the asinh-compressed
 *   physical embedding of the canonical Q(c) = (x, y, lambda*sigma(c))
 *   sampled by Rust; no invented heightfield (issue #111 mathematical
 *   basis).
 *
 * - mode = 'treadmill': y = SCENE_SCALE * z — the exact linear chart,
 *   consuming the patch's own embedding height z = lambda*sigma(c)
 *   (already lambda-multiplied by Rust). Combined with
 *   `treadmillTransform`'s recenter, the rider's own vertex lands at
 *   y = 0 and nearby terrain sits at SCENE_SCALE * (z - z0). No
 *   surfaceY() compression is applied: the treadmill chart is meant to
 *   be a mathematically meaningful local scale chart, not the cosmetic
 *   physical embedding.
 *
 * The patch input is identical in both modes — only the Y mapping
 * differs. Both modes go through this one function so there is no
 * parallel mesh builder.
 */
export function buildTerrainMesh(patch: TerrainPatch, mode: TerrainMeshMode = 'physical'): THREE.Mesh {
  const n = patch.n;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(n * n * 3);
  const colors = new Float32Array(n * n * 3);

  for (let i = 0; i < n * n; i++) {
    const x = patch.positions[i * 3];
    const y = patch.positions[i * 3 + 1];
    const z = patch.positions[i * 3 + 2]; // Rust-embedded lambda*sigma(c).
    positions[i * 3] = x * SCENE_SCALE;
    // Treadmill mode consumes z AS the embedding height — re-multiplying
    // by a TypeScript lambda constant would square it (hidden by lambda^2
    // = 1 in the controller-default config).
    positions[i * 3 + 1] = mode === 'treadmill' ? SCENE_SCALE * z : surfaceY(z);
    positions[i * 3 + 2] = -y * SCENE_SCALE;

    // Realm coloring from the authoritative signed distance: inside = deep
    // blue, outside = sand, Shore (|D| tiny) = bright band.
    const d = patch.signed[i];
    let r: number, g: number, b: number;
    if (Math.abs(d) < 0.002) {
      r = 1.0; g = 0.95; b = 0.4; // Shore band
    } else if (d < 0) {
      r = 0.12; g = 0.2; b = 0.45; // Inside M
    } else {
      r = 0.72; g = 0.62; b = 0.42; // Outside M
    }
    // Height shading: higher sigma = slightly lighter.
    const shade = Math.max(0.55, Math.min(1.15, 1.0 + z * 0.045));
    colors[i * 3] = r * shade;
    colors[i * 3 + 1] = g * shade;
    colors[i * 3 + 2] = b * shade;
  }

  const indices: number[] = [];
  for (let row = 0; row < n - 1; row++) {
    for (let col = 0; col < n - 1; col++) {
      const a = row * n + col;
      const b = row * n + col + 1;
      const c = (row + 1) * n + col;
      const dIdx = (row + 1) * n + col + 1;
      indices.push(a, c, b, b, c, dIdx);
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

/** Independent terrain overlay toggles (issue #111 terrain overlays). */
export interface TerrainOverlays {
  /** Highlight the Shore band D(c)=0. */
  shoreBand: boolean;
  /** Tint inside/outside realms. */
  realm: boolean;
  /** Shade by sigma(c) (Mandelbrot scale — distinct from Julia zoom). */
  sigma: boolean;
  /** Shade by U(c) = kappa*sigma (potential ramp). */
  potential: boolean;
  /** Tint vertices where derivative validity fails. */
  validity: boolean;
}

export const DEFAULT_OVERLAYS: TerrainOverlays = {
  shoreBand: true,
  realm: true,
  sigma: false,
  potential: false,
  validity: true,
};

/**
 * Repaint the terrain vertex colors from the patch + overlay toggles.
 * All quantities are already sampled by Rust in the TerrainPatch; this only
 * maps values to colors. sigma/U/validity overlays are mutually layered:
 * the last enabled one in the fixed order [validity, potential, sigma,
 * realm, shoreBand] wins for a vertex, keeping the mapping predictable.
 */
export function applyOverlays(mesh: THREE.Mesh, patch: TerrainPatch, overlays: TerrainOverlays): void {
  const colorAttr = mesh.geometry.getAttribute('color') as THREE.BufferAttribute;
  if (!colorAttr) return;
  const n = patch.n;
  const colors = colorAttr.array as Float32Array;

  // For the sigma/potential ramps: normalize against the patch min/max so
  // the ramp is legible regardless of the local scale range. Iterated (not
  // Math.min(...zs), which overflows the JS arg limit — and the call stack —
  // at LOD grid sizes above ~100k vertices).
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let i = 0; i < n * n; i++) {
    const z = patch.positions[i * 3 + 2];
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const span = Math.max(zMax - zMin, 1e-9);

  for (let i = 0; i < n * n; i++) {
    const d = patch.signed[i];
    const z = patch.positions[i * 3 + 2];
    const t = (z - zMin) / span;
    const nearShore = Math.abs(d) < 0.002;

    let r: number, g: number, b: number;
    if (overlays.validity && !Number.isFinite(d)) {
      // Derivative-validity overlay: non-finite D marks a failed sample.
      r = 1; g = 0.3; b = 1;
    } else if (overlays.potential) {
      // U ramp: violet (low U) -> red (high U).
      r = 0.25 + 0.75 * t;
      g = 0.1 + 0.15 * (1 - t);
      b = 0.6 - 0.45 * t;
    } else if (overlays.sigma) {
      // sigma ramp: deep teal (low scale) -> yellow (high scale).
      r = 0.1 + 0.9 * t * t;
      g = 0.75 - 0.15 * t;
      b = 0.55 - 0.4 * t;
    } else if (overlays.realm) {
      if (d < 0) {
        r = 0.12; g = 0.2; b = 0.45;
      } else {
        r = 0.72; g = 0.62; b = 0.42;
      }
    } else {
      r = 0.5; g = 0.5; b = 0.55;
    }

    if (overlays.shoreBand && nearShore) {
      r = 1.0; g = 0.95; b = 0.4;
    }

    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  colorAttr.needsUpdate = true;
}

/**
 * Load the animated rider (Meshy "Ant man" biped GLB) standing on the Meshy
 * skateboard GLB. Returns a group whose +X axis is the heading direction;
 * falls back to a capsule body if a model fails to load so the cockpit
 * degrades honestly instead of losing the rider.
 *
 * The rider GLB's Running animation is played through an AnimationMixer
 * owned by this module (see updateRiderAnimation) — the component only
 * advances time.
 */

/** Rider yaw: the GLB faces +Z in T-pose; +PI/2 turns it to face +X. */
const RIDER_MODEL_YAW = Math.PI / 2;

/** Rider scale: the GLB is ~1.6 units tall; the scene rider reads best ~1.4. */
const RIDER_MODEL_SCALE = 0.9;

/** Rider lift: feet rest on the skateboard deck top (deck spans y 0..0.15). */
const RIDER_MODEL_LIFT = 0.15;

/** Skateboard scale: the GLB deck is ~1.9 long; the scene board reads best ~1.3. */
const SKATEBOARD_SCALE = 0.7;

const riderMixers = new WeakMap<THREE.Group, THREE.AnimationMixer>();

/**
 * Advance the rider's animation mixer by dt, scaling playback speed with
 * the authoritative metric speed (clamped so slow rolls amble and fast
 * ones sprint). No-op for the fallback-capsule rider.
 */
export function updateRiderAnimation(rider: THREE.Group, dt: number, metricSpeed: number): void {
  const mixer = riderMixers.get(rider);
  if (!mixer) return;
  // Map metric speed to a legible gait: ~0.6x at a crawl, ~1.8x flat-out.
  mixer.timeScale = Math.max(0.6, Math.min(1.8, 0.6 + metricSpeed * 8.0));
  mixer.update(dt);
}

export async function buildRider(): Promise<THREE.Group> {
  const group = new THREE.Group();
  const loader = new GLTFLoader();

  // Skateboard GLB: already X-aligned (deck ~1.9 long on X, wheels at
  // y ~ -0.155), so it needs only scaling and a lift to put the wheels'
  // contact plane at y=0.
  try {
    const boardGltf = await loader.loadAsync('/models/skateboard.glb');
    const board = boardGltf.scene;
    board.scale.setScalar(SKATEBOARD_SCALE);
    // Wheels bottom at -0.155 * scale; lift so contact plane sits at y=0.
    board.position.y = 0.155 * SKATEBOARD_SCALE;
    group.add(board);
  } catch (error) {
    console.warn('[cockpitScene] skateboard GLB unavailable, using box fallback:', error);
    const boardGeometry = new THREE.BoxGeometry(1.1, 0.12, 0.42);
    const boardMaterial = new THREE.MeshLambertMaterial({ color: 0xcc4444 });
    const board = new THREE.Mesh(boardGeometry, boardMaterial);
    board.position.y = 0.12;
    group.add(board);
  }

  try {
    const gltf = await loader.loadAsync('/models/rider.glb');
    const model = gltf.scene;

    model.scale.setScalar(RIDER_MODEL_SCALE);
    model.position.y = RIDER_MODEL_LIFT;
    model.rotation.y = RIDER_MODEL_YAW;
    group.add(model);

    const clip =
      gltf.animations.find((a) => /run/i.test(a.name)) ?? gltf.animations[0];
    if (clip) {
      const mixer = new THREE.AnimationMixer(model);
      mixer.clipAction(clip).play();
      riderMixers.set(group, mixer);
    }
  } catch (error) {
    console.warn('[cockpitScene] rider GLB unavailable, using capsule fallback:', error);
    const bodyGeometry = new THREE.CapsuleGeometry(0.18, 0.5, 4, 8);
    const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x3388cc });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.62;
    group.add(body);
  }

  // Velocity arrow: thin cone along +X, visible when moving.
  const arrowGeometry = new THREE.ConeGeometry(0.09, 0.5, 6);
  const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0xffee44 });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  arrow.rotation.z = -Math.PI / 2;
  arrow.position.set(1.05, 0.35, 0);
  arrow.name = 'velocityArrow';
  group.add(arrow);

  return group;
}

/** Per-frame rider placement derived from authoritative state. */
export function placeRider(
  rider: THREE.Group,
  snap: DebugSnapshot,
  terrainHeightAt: (x: number, y: number) => number
): void {
  const [cx, cy] = snap.physics.c;
  const [vx, vy] = snap.physics.velocity;

  rider.position.set(cx * SCENE_SCALE, terrainHeightAt(cx, cy), -cy * SCENE_SCALE);

  // Heading from velocity; when nearly stationary keep last heading and
  // fall back to drive direction.
  const speed = Math.hypot(vx, vy);
  if (speed > 1e-7) {
    rider.rotation.y = Math.atan2(-vy, vx);
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    if (Math.hypot(dx, dy) > 1e-6) {
      rider.rotation.y = Math.atan2(-dy, dx);
    }
  }

  // Pitch from the surface slope along the heading (presentation derived
  // from authoritative geometry): the compressed rise rate ~ sigma_dot *
  // d(surfaceY)/dsigma evaluated at the current sigma.
  const pitchGain =
    (surfaceY(snap.physics.sigma + 0.01) - surfaceY(snap.physics.sigma - 0.01)) / 0.02;
  const pitch = Math.atan2(snap.physics.sigmaDot * pitchGain, Math.max(speed * SCENE_SCALE, 1e-9));
  rider.rotation.x = -pitch;

  // Velocity arrow visibility scales with speed.
  const arrow = rider.getObjectByName('velocityArrow');
  if (arrow) {
    arrow.visible = speed > 1e-5;
  }
}

/**
 * Build the trail line from a recorded trajectory up to `upTo` (inclusive).
 *
 * - mode = 'physical' (default): y = surfaceY(sigma) + 0.05 (small lift
 *   so the trail draws just above the surface) — matches the cosmetic
 *   compressed physical embedding.
 *
 * - mode = 'treadmill': y = SCENE_SCALE * sigma — the exact linear chart
 *   height (physics.sigma equals the Rust embedding height under the
 *   controller-default lambda^2 = 1 config; see the NOTE on lambda).
 *   Combined with `treadmillTrailTransform`'s recenter, the trail sits
 *   in the same chart as the treadmill terrain mesh so the trail stays
 *   glued to the surface in scale-stabilized mode.
 *
 * Building the trail in chart coordinates per-mode (rather than building
 * it once in physical coordinates and trying to compensate via a Y
 * affine transform) is the cleanest way to honor the "exact relative
 * embedding height" Y contract without leaking the asinh compression
 * into treadmill mode.
 */
export function buildTrail(
  trajectory: CockpitTrajectory,
  upTo: number,
  mode: CameraMode = 'physical'
): THREE.Line {
  const count = Math.min(upTo + 1, trajectory.snapshots.length);
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const [cx, cy] = trajectory.snapshots[i].physics.c;
    const z = trajectory.snapshots[i].physics.sigma;
    const yCoord =
      mode === 'treadmill'
        ? SCENE_SCALE * z
        : surfaceY(z) + 0.05;
    points.push(new THREE.Vector3(cx * SCENE_SCALE, yCoord, -cy * SCENE_SCALE));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x66ff99 });
  return new THREE.Line(geometry, material);
}

/**
 * Apply the treadmill chart transform to the trail (same chart as the
 * terrain so the trail stays glued to the surface in treadmill mode).
 *
 * The trail's Y coordinate was built in `buildTrail` with mode='treadmill'
 * as `SCENE_SCALE * sigma` (the embedding height). The post-transform
 * vertex Y is
 *   (SCENE_SCALE * sigma) * scale.y + position.y
 * = SCENE_SCALE * (sigma - sigma0)
 * which is exactly the chart Y (see `treadmillChart`).
 *
 * Previous fix (c3b6456) recentered with `position.y = -surfaceY(sigma)`,
 * which produced Y = surfaceY(sigma) - surfaceY(sigma0) + tiny — the
 * nonlinear asinh compression leaked into the treadmill chart, making it
 * a cosmetic view rather than a mathematically meaningful local scale.
 * Now `position.y = -SCENE_SCALE * sigma` (the rider's own value
 * in the LINEAR chart), and the trail's pre-built Y already matches the
 * chart's linear Y.
 */
export function treadmillTrailTransform(trail: THREE.Line, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const magnify = 1.0 / rho0;
  trail.position.x = -cx * SCENE_SCALE * magnify;
  trail.position.z = cy * SCENE_SCALE * magnify;
  // Vertical recentering subtracts the rider's CURRENT chart Y so the
  // trail at the rider's own sigma sits at Y=0 — the LINEAR chart Y,
  // NOT surfaceY(sigma) (the cosmetic physical-mode compression).
  trail.position.y = -SCENE_SCALE * sigma;
  trail.scale.set(magnify, 1.0, magnify);
}

/** Reset the trail to physical coordinates. */
export function physicalTrailTransform(trail: THREE.Line): void {
  trail.position.set(0, 0, 0);
  trail.scale.setScalar(1.0);
}

/**
 * Position the third-person camera behind and above the rider.
 *
 * - physical: raw (x, y, lambda*sigma) embedding — geometry-debug mode.
 * - treadmill: scale-stabilized chart X=(x-x0)/rho0, Y=(z(c)-z(c0))
 *   around the CURRENT rider position c0 — debug presentation ONLY; never
 *   feeds physics (guaranteed structurally: the camera only reads the
 *   snapshot, and this module has no path back into the recorder).
 */
export function updateCamera(
  camera: THREE.PerspectiveCamera,
  snap: DebugSnapshot,
  mode: CameraMode
): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;

  let rx: number, rz: number;
  if (mode === 'physical') {
    rx = cx * SCENE_SCALE;
    rz = -cy * SCENE_SCALE;
  } else {
    // Treadmill: the rider is pinned at the chart origin; the terrain mesh
    // carries the (x-x0)/rho0 transform (see treadmillTransform).
    rx = 0;
    rz = 0;
  }

  // Behind and above, biased along the rider's heading.
  const [vx, vy] = snap.physics.velocity;
  const speed = Math.hypot(vx, vy);
  const heading = speed > 1e-7 ? Math.atan2(-vy, vx) : Math.PI / 2;
  const back = 3.2;
  const up = 2.2;
  const targetY = (mode === 'treadmill' ? 0 : surfaceY(sigma)) + 0.6;

  const camX = rx - Math.cos(heading) * back;
  const camZ = rz + Math.sin(heading) * back;
  const camY = (mode === 'treadmill' ? 0 : surfaceY(sigma)) + up;

  camera.position.set(camX, camY, camZ);
  camera.lookAt(rx, targetY, rz);
}

/**
 * Scale-stabilized treadmill chart (issue #111):
 *   X = (x - x0) / rho0,
 *   Y = (lambda*sigma)(c) - (lambda*sigma)(c0),
 *   Z = -(y - y0) / rho0
 *
 * Implemented as a Three.js mesh transform on a `buildTerrainMesh(patch,
 * 'treadmill')` mesh:
 *   - Translate the patch horizontally by -c0 (scene units) and
 *     vertically by -SCENE_SCALE * sigma0 (the rider's own chart Y,
 *     NOT surfaceY(sigma0)).
 *   - Scale the patch by 1/rho0 on X/Z only (anisotropic).
 *
 * With the terrain mesh built using the patch's own embedding height
 * (see `buildTerrainMesh` with mode='treadmill'), the post-transform
 * vertex Y is the pure relative embedding height — exactly the intended
 * chart Y. There is NO surfaceY() compression anywhere in the treadmill
 * chart path: the cosmetic asinh curve is reserved for physical mode.
 *
 * Debug presentation ONLY — this module has no path back into physics
 * (the recorder never reads scene objects).
 */
export function treadmillTransform(mesh: THREE.Mesh, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const magnify = 1.0 / rho0;
  mesh.position.x = -cx * SCENE_SCALE * magnify;
  mesh.position.z = cy * SCENE_SCALE * magnify;
  // Vertical recentering subtracts the rider's CURRENT LINEAR chart Y
  // (SCENE_SCALE * sigma), NOT surfaceY(sigma). The terrain mesh built
  // in mode='treadmill' already uses the Rust embedding height for Y, so
  // after this recenter a vertex sits at exactly the relative chart Y.
  mesh.position.y = -SCENE_SCALE * sigma;
  mesh.scale.set(magnify, 1.0, magnify);
}

/** Reset the treadmill transform when returning to physical mode. */
export function physicalTransform(mesh: THREE.Mesh): void {
  mesh.position.set(0, 0, 0);
  mesh.scale.setScalar(1.0);
}

/** Lights + backdrop for the late-90s skate-game look. */
export function buildSceneDressing(scene: THREE.Scene): void {
  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff2cc, 0.9);
  sun.position.set(6, 10, 4);
  scene.add(sun);

  scene.background = new THREE.Color(0x0a0a14);
  // Fog distances are set per-terrain-rebuild by applyRenderDistance so the
  // mesh edge always hides inside the fog wall at every LOD level.
  scene.fog = new THREE.Fog(0x0a0a14, 18, 42);
}

/**
 * Apply the LOD render distance: camera far plane + fog wall track the
 * patch size so fidelity stays balanced with performance as scale shifts
 * (issue #111). Call on every terrain rebuild.
 *
 * The fog wall is floored at the camera-to-rider distance: the LOD patch
 * shrinks at deep scale, and a fog wall tighter than the camera distance
 * would swallow the whole scene (the "black viewport" failure mode). In
 * treadmill mode the 1/rho0 chart magnification inflates the effective
 * patch, so its fog wall inflates with it.
 */
export function applyRenderDistance(
  camera: THREE.PerspectiveCamera,
  scene: THREE.Scene,
  mode: CameraMode,
  rho: number,
  half: number
): void {
  const r = Math.max(rho, 1e-9);
  const magnify = mode === 'treadmill' ? 1.0 / r : 1.0;
  const patchScene = half * 2 * SCENE_SCALE * magnify;
  const diagonal = patchScene * Math.SQRT2;
  // updateCamera keeps the camera ~sqrt(3.2^2 + 2.2^2) ~ 3.9 scene units
  // from the rider; the fog must start beyond the subject.
  const cameraDist = 3.9;
  const fogNear = Math.max(cameraDist * 1.15, diagonal * 0.25);
  const fogFar = Math.max(diagonal * 1.15, cameraDist * 1.9);
  const far = Math.max(diagonal * 1.6, cameraDist * 2.4);

  if (camera.far !== far) {
    camera.far = far;
    camera.updateProjectionMatrix();
  }
  const fog = scene.fog;
  if (fog && 'near' in fog && 'far' in fog) {
    (fog as THREE.Fog).near = fogNear;
    (fog as THREE.Fog).far = fogFar;
  }
}