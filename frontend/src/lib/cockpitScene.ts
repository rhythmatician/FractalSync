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
 * - Camera modes: physical embedding, scale-follow (physical vertical +
 *   1/rho0 horizontal ruler — the controlled experiment), and the
 *   scale-stabilized treadmill chart.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import type { DebugSnapshot, TerrainPatch, CockpitTrajectory } from './debugCockpit';

/** Camera presentation modes (issue #111 + hyperbolic camera). */
export type CameraMode = 'physical' | 'scale-follow' | 'treadmill' | 'hyperbolic';

/**
 * Terrain-mesh build mode: 'physical', 'scale-follow', and 'hyperbolic' render y =
 * surfaceY(z) (the asinh-compressed physical embedding) — scale-follow
 * deliberately keeps the physical vertical presentation and only changes
 * the horizontal ruler at presentation time; 'hyperbolic' uses the hyperbolic
 * isometry + Poincaré ball projection to place vertices; 'treadmill' renders y =
 * SCENE_SCALE * z so the chart Y is the exact relative embedding height,
 * with NO nonlinear compression. All modes share the same Rust patch
 * input — only the Y mapping differs.
 */
export type TerrainMeshMode = CameraMode;

/**
 * True for modes whose TERRAIN MESH is built with the physical surfaceY()
 * vertical mapping. The terrain builder and any code that needs to know
 * "which vertical authority does this mode's mesh use" consults this —
 * 'scale-follow' keeps the physical vertical and only magnifies X/Z.
 */
export function isPhysicalYMode(mode: CameraMode): boolean {
  return mode !== 'treadmill';
}

/**
 * Horizontal magnification factor for a presentation mode at a given rho.
 * Single authority for "is this mode horizontally magnified, and by how
 * much": scale-follow and treadmill both use the local Mandelbrot ruler
 * 1/rho; physical and hyperbolic use 1.0 (hyperbolic maps into the Poincaré ball).
 */
export function horizontalMagnification(mode: CameraMode, rho: number): number {
  if (mode === 'physical' || mode === 'hyperbolic') return 1.0;
  return 1.0 / Math.max(rho, 1e-9);
}

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
 * - mode = 'physical' or 'scale-follow': y = surfaceY(z) — the
 *   asinh-compressed physical embedding of the canonical
 *   Q(c) = (x, y, lambda*sigma(c)) sampled by Rust; no invented heightfield
 *   (issue #111 mathematical basis). Scale-follow deliberately shares the
 *   physical vertical: the experiment changes ONLY the horizontal ruler
 *   (applied later by `scaleFollowTransform`), so the same mesh geometry
 *   serves both modes.
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
// Procedural grid texture generator for the fractal manifold terrain
let proceduralGridTexture: THREE.CanvasTexture | null = null;
function getGridTexture(): THREE.CanvasTexture | null {
  if (proceduralGridTexture) return proceduralGridTexture;
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  if (!canvas || typeof canvas.getContext !== 'function') return null;
  const ctx = canvas.getContext('2d');
  if (!ctx || typeof ctx.fillRect !== 'function') return null;

  canvas.width = 256;
  canvas.height = 256;
  // Clean neutral base that tints with vertexColors
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 256);
  // Subtle cyber/synthwave grid lines
  ctx.strokeStyle = 'rgba(150, 160, 190, 0.4)';
  ctx.lineWidth = 3;
  ctx.strokeRect(0, 0, 256, 256);
  // Secondary micro-grid
  ctx.strokeStyle = 'rgba(180, 190, 220, 0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(128, 0); ctx.lineTo(128, 256);
  ctx.moveTo(0, 128); ctx.lineTo(256, 128);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(32, 32);
  proceduralGridTexture = tex;
  return proceduralGridTexture;
}

export function buildTerrainMesh(patch: TerrainPatch, mode: TerrainMeshMode = 'physical'): THREE.Mesh {
  const n = patch.n;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(n * n * 3);
  const colors = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);

  for (let i = 0; i < n * n; i++) {
    const row = Math.floor(i / n);
    const col = i % n;
    const x = patch.positions[i * 3];
    const y = patch.positions[i * 3 + 1];
    const z = patch.positions[i * 3 + 2]; // Rust-embedded lambda*sigma(c).
    positions[i * 3] = x * SCENE_SCALE;
    positions[i * 3 + 1] = isPhysicalYMode(mode) ? surfaceY(z) : SCENE_SCALE * z;
    positions[i * 3 + 2] = -y * SCENE_SCALE;

    uvs[i * 2] = col / (n - 1);
    uvs[i * 2 + 1] = row / (n - 1);

    // Realm coloring from the authoritative signed distance: inside = deep
    // cosmic blue/indigo, outside = warm amber terrace, Shore (|D| tiny) = glowing electric gold.
    const d = patch.signed[i];
    let r: number, g: number, b: number;
    if (Math.abs(d) < 0.002) {
      r = 1.0; g = 0.96; b = 0.45; // Shore band
    } else if (d < 0) {
      r = 0.10; g = 0.18; b = 0.48; // Inside M (cosmic deep basin)
    } else {
      r = 0.68; g = 0.58; b = 0.40; // Outside M (warm sandy terrace)
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
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const gridTex = getGridTexture();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    ...(gridTex ? { map: gridTex } : {}),
    roughness: 0.72,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });
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
 * Load the animated rider (Meshy "Tiny Titan" biped GLB) standing on the Meshy
 * skateboard GLB. Returns a group whose +X axis is the heading direction;
 * falls back to a capsule body if a model fails to load so the cockpit
 * degrades honestly instead of losing the rider.
 *
 * The rider plays a Mixamo skateboarding animation retargeted onto the Meshy
 * biped skeleton via an AnimationMixer owned by this module (see updateRiderAnimation).
 */

/** Rider yaw: +PI/2 aligns the Mixamo skateboarding animation forward (+X) with feet spread along deck length. */
const RIDER_MODEL_YAW = Math.PI / 2;

/** Rider scale: the GLB is ~1.6 units tall; the scene rider reads best ~1.4. */
const RIDER_MODEL_SCALE = 0.9;

/** Rider lift: calibrated so shoe soles rest squarely and flush on the skateboard deck. */
const RIDER_MODEL_LIFT = -0.230;

/** Skateboard scale: scaled down to 0.38 (~0.72 long, ~0.22 wide) to fit Tiny Titan's smaller body/leg proportions. */
const SKATEBOARD_SCALE = 0.4;

interface RiderAnimationState {
  mixer: THREE.AnimationMixer;
  actionCoast: THREE.AnimationAction | null;
  actionPush: THREE.AnimationAction | null;
}

const riderAnimationStates = new WeakMap<THREE.Group, RiderAnimationState>();

/**
 * Retarget Mixamo animation tracks onto the Meshy biped skeleton.
 * Mixamo bone names start with 'mixamorig'; Meshy biped uses plain names
 * ('Hips', 'Spine', 'Spine01', 'Spine02', 'neck', 'Head', etc.).
 * Root motion on X/Z is locked so the character rides in place on the board.
 */
function retargetMixamoToMeshy(
  sourceClip: THREE.AnimationClip,
  targetName: string,
  targetSkeletonBones: Set<string>
): THREE.AnimationClip {
  const tracks: THREE.KeyframeTrack[] = [];
  for (const track of sourceClip.tracks) {
    const dotIdx = track.name.indexOf('.');
    if (dotIdx === -1) continue;
    const boneRaw = track.name.slice(0, dotIdx);
    const prop = track.name.slice(dotIdx);

    let targetBone = boneRaw.replace(/^mixamorig/, '');
    if (targetBone === 'Spine1') targetBone = 'Spine01';
    else if (targetBone === 'Spine2') targetBone = 'Spine02';
    else if (targetBone === 'Neck') targetBone = 'neck';

    if (targetSkeletonBones.has(targetBone)) {
      const cloned = track.clone();
      cloned.name = targetBone + prop;
      // Lock root horizontal motion so the skater stays planted on the deck
      if (cloned.name === 'Hips.position') {
        const x0 = cloned.values[0];
        const z0 = cloned.values[2];
        for (let i = 0; i < cloned.values.length; i += 3) {
          cloned.values[i] = x0;
          cloned.values[i + 2] = z0;
        }
      }
      tracks.push(cloned);
    }
  }
  return new THREE.AnimationClip(targetName, sourceClip.duration, tracks);
}

/**
 * Advance the rider's animation mixer by dt, blending between Skateboarding
 * (coasting / cruising) and PushOff (accelerating forward) according to
 * throttle / forward driving effort, and scaling cadence with metric speed.
 */
export function updateRiderAnimation(
  rider: THREE.Group,
  dt: number,
  metricSpeed: number,
  throttle: number = 0
): void {
  const state = riderAnimationStates.get(rider);
  if (!state) return;

  const { mixer, actionCoast, actionPush } = state;

  if (actionCoast && actionPush) {
    // When throttle > 0, blend in the PushOff animation; otherwise coast.
    const pushWeight = Math.min(1.0, Math.max(0.0, throttle));
    const coastWeight = 1.0 - pushWeight;
    actionCoast.setEffectiveWeight(coastWeight);
    actionPush.setEffectiveWeight(pushWeight);
  } else if (actionCoast) {
    actionCoast.setEffectiveWeight(1.0);
  } else if (actionPush) {
    actionPush.setEffectiveWeight(1.0);
  }

  // Map metric speed to a legible cadence: ~0.6x at a crawl, ~1.8x flat-out.
  mixer.timeScale = Math.max(0.6, Math.min(1.8, 0.6 + metricSpeed * 8.0));
  mixer.update(dt);
}

export async function buildRider(): Promise<THREE.Group> {
  const group = new THREE.Group();
  const gltfLoader = new GLTFLoader();
  const fbxLoader = new FBXLoader();

  // Skateboard GLB: already X-aligned (deck ~1.9 long on X, wheels at
  // y ~ -0.155), so it needs only scaling and a lift to put the wheels'
  // contact plane at y=0.
  try {
    const boardGltf = await gltfLoader.loadAsync('/models/skateboard.glb');
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
    // New player character without integrated duplicate skateboard/pedestal: Tiny Titan biped
    const titanPath = '/models/Meshy_AI_Tiny_Titan_biped/Meshy_AI_Tiny_Titan_biped_Character_output.glb';
    const gltf = await gltfLoader.loadAsync(titanPath);
    const model = gltf.scene;

    model.scale.setScalar(RIDER_MODEL_SCALE);
    model.position.y = RIDER_MODEL_LIFT;
    model.rotation.y = RIDER_MODEL_YAW;
    group.add(model);

    // Collect skeleton bone names for retargeting
    const boneNames = new Set<string>();
    model.traverse((o) => {
      if ((o as THREE.Bone).isBone) boneNames.add(o.name);
    });

    const mixer = new THREE.AnimationMixer(model);
    let actionCoast: THREE.AnimationAction | null = null;
    let actionPush: THREE.AnimationAction | null = null;

    // Load authentic skateboarding and push-off animations from Mixamo FBX
    try {
      const fbxSkate = await fbxLoader.loadAsync('/animations/Skateboarding.fbx');
      if (fbxSkate.animations.length > 0) {
        const skateClip = retargetMixamoToMeshy(fbxSkate.animations[0], 'Skateboarding', boneNames);
        actionCoast = mixer.clipAction(skateClip);
        actionCoast.play();
      }
    } catch (animError) {
      console.warn('[cockpitScene] skateboarding animation unavailable:', animError);
    }

    try {
      const fbxPush = await fbxLoader.loadAsync('/animations/PushOff.fbx');
      if (fbxPush.animations.length > 0) {
        const pushClip = retargetMixamoToMeshy(fbxPush.animations[0], 'PushOff', boneNames);
        actionPush = mixer.clipAction(pushClip);
        actionPush.play();
      }
    } catch (pushError) {
      console.warn('[cockpitScene] push-off animation unavailable:', pushError);
    }

    // Fallback if FBX animations fail
    if (!actionCoast && !actionPush && gltf.animations.length > 0) {
      const fallbackClip = gltf.animations[0];
      actionCoast = mixer.clipAction(fallbackClip);
      actionCoast.play();
    }

    riderAnimationStates.set(group, { mixer, actionCoast, actionPush });
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
  terrainHeightAt: (x: number, y: number) => number,
  mode: CameraMode = 'physical'
): void {
  rider.scale.setScalar(1);
  const [cx, cy] = snap.physics.c;
  const [vx, vy] = snap.physics.velocity;

  rider.position.set(cx * SCENE_SCALE, terrainHeightAt(cx, cy), -cy * SCENE_SCALE);

  // Heading direction in c-space.
  // In the scene: X = cx * SCENE_SCALE, Z = -cy * SCENE_SCALE.
  // When velocity is (vx, vy), the horizontal direction in scene space is (vx, -vy).
  // With rider group's forward axis along local +X:
  // Rotating (1, 0, 0) by theta_y around +Y gives:
  //   x' = cos(theta_y), z' = -sin(theta_y).
  // Matching (vx, -vy) requires cos(theta_y) ~ vx and -sin(theta_y) ~ -vy,
  // which means sin(theta_y) ~ vy.
  // Thus theta_y = Math.atan2(vy, vx).
  const speed = Math.hypot(vx, vy);
  let dirX = 1;
  let dirY = 0;
  let hasDir = false;

  if (speed > 1e-7) {
    dirX = vx / speed;
    dirY = vy / speed;
    hasDir = true;
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    const dSpeed = Math.hypot(dx, dy);
    if (dSpeed > 1e-6) {
      dirX = dx / dSpeed;
      dirY = dy / dSpeed;
      hasDir = true;
    }
  }

  // Consistent vertical-vs-horizontal scaling:
  // The terrain's visual horizontal scale is:
  //   H_scale = SCENE_SCALE * horizontalMagnification(mode, snap.physics.rho)
  //
  // The vertical rise rate in the scene is:
  // - In physical / scale-follow modes:
  //     y = surfaceY(sigma)
  //     dy/dsigma ~ d(surfaceY)/dsigma
  //     dy/dc = grad(sigma) * (d(surfaceY)/dsigma)
  // - In treadmill mode:
  //     y = SCENE_SCALE * sigma
  //     dy/dsigma = SCENE_SCALE
  //     dy/dc = grad(sigma) * SCENE_SCALE
  const magnify = horizontalMagnification(mode, snap.physics.rho);
  const hScale = SCENE_SCALE * magnify;

  const [gx, gy] = snap.physics.scaleGradient;
  const dYdSigma = isPhysicalYMode(mode)
    ? (surfaceY(snap.physics.sigma + 0.01) - surfaceY(snap.physics.sigma - 0.01)) / 0.02
    : SCENE_SCALE;

  // Scene vertical gradient: d(sceneY)/d(cx) and d(sceneY)/d(cy)
  const dY_dcx = gx * dYdSigma;
  const dY_dcy = gy * dYdSigma;

  // Slope along forward direction and lateral right direction (in scene units):
  // Forward c-space unit vector: (dirX, dirY).
  // Scene forward displacement: dX_scene = dirX * hScale, dZ_scene = -dirY * hScale.
  // Forward rise: dY_fwd = (dirX * dY_dcx + dirY * dY_dcy).
  // Slope forward = dY_fwd / hScale.
  //
  // Lateral right unit vector in c-space: (dirY, -dirX).
  // (In scene: dX_right = dirY * hScale, dZ_right = -(-dirX)*hScale = dirX * hScale,
  // which is perpendicular to scene forward (dirX, -dirY)).
  // Lateral rise: dY_lat = (dirY * dY_dcx - dirX * dY_dcy).
  // Slope lateral = dY_lat / hScale.
  const slopeFwd = hasDir ? (dirX * dY_dcx + dirY * dY_dcy) / Math.max(hScale, 1e-9) : 0;
  const slopeLat = hasDir ? (dirY * dY_dcx - dirX * dY_dcy) / Math.max(hScale, 1e-9) : 0;

  if (hasDir) {
    const thetaY = Math.atan2(dirY, dirX);
    const pitch = Math.atan(slopeFwd);
    const roll = Math.atan(slopeLat);

    // Forward tangent vector T (in scene coordinates):
    const T = new THREE.Vector3(
      Math.cos(pitch) * Math.cos(thetaY),
      Math.sin(pitch),
      -Math.cos(pitch) * Math.sin(thetaY)
    ).normalize();

    // Lateral right vector R_raw:
    const R_raw = new THREE.Vector3(
      Math.cos(roll) * Math.sin(thetaY),
      Math.sin(roll),
      Math.cos(roll) * Math.cos(thetaY)
    );

    // Surface normal N (points UP: R x T in right-handed coords):
    const N = new THREE.Vector3().crossVectors(R_raw, T).normalize();
    // Exact orthonormal lateral vector B = T x N:
    const B = new THREE.Vector3().crossVectors(T, N).normalize();

    // Construct basis matrix: local +X -> T (forward), local +Y -> N (up), local +Z -> B (right):
    const mat = new THREE.Matrix4().makeBasis(T, N, B);
    rider.quaternion.setFromRotationMatrix(mat);
  }

  // Velocity arrow visibility scales with speed.
  const arrow = rider.getObjectByName('velocityArrow');
  if (arrow) {
    arrow.visible = speed > 1e-5;
  }
}

/**
 * Build the trail line from a recorded trajectory up to `upTo` (inclusive).
 *
 * - mode = 'physical' or 'scale-follow': y = surfaceY(sigma) + 0.05
 *   (small lift so the trail draws just above the surface) — matches the
 *   cosmetic compressed physical embedding. Scale-follow keeps this
 *   physical vertical on purpose: the trail mesh is built exactly as in
 *   physical mode, and `scaleFollowTrailTransform` applies only the
 *   horizontal recenter + 1/rho0 magnification, so terrain and trail stay
 *   registered.
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
    const yCoord = isPhysicalYMode(mode)
      ? surfaceY(z) + 0.05
      : SCENE_SCALE * z;
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

// ---------------------------------------------------------------------------
// Scale-follow presentation (horizontal-ruler experiment)
// ---------------------------------------------------------------------------
//
// Controlled experiment (issue #111): the Shore reads as a near-vertical
// wall in physical mode possibly NOT because heights are wrong but because
// the FIXED horizontal ruler collapses valid terrain as rho(c) shrinks.
// Scale-follow tests that hypothesis by changing ONLY the horizontal ruler:
//
//   X = SCENE_SCALE * (x - x0) / rho0
//   Y = surfaceY(lambda*sigma(c))        <- EXACTLY physical mode's Y
//   Z = -SCENE_SCALE * (y - y0) / rho0
//
// The mesh is built with the physical surfaceY() mapping (identical
// geometry to physical mode) and only the PRESENTATION transform differs:
// translate by -c0 and scale by 1/rho0 on X/Z. Y scale stays exactly 1 —
// no vertical recentering, no asinh changes. Unlike treadmill, there is
// no relative-height chart; the rider's height comes from the same
// surfaceY() the rider stands on in physical mode.
//
// Debug presentation ONLY — no path back into physics.

/**
 * Apply the scale-follow horizontal transform to a terrain mesh whose
 * geometry was built in physical coordinates (same mesh `buildTerrainMesh`
 * produces for physical mode).
 *
 * Post-transform vertex position for the patch point at c = (x, y):
 *   X = SCENE_SCALE * (x - cx) / rho0
 *   Y = surfaceY(z)              (mesh Y scale is exactly 1)
 *   Z = -SCENE_SCALE * (y - cy) / rho0
 */
export function scaleFollowTransform(mesh: THREE.Mesh, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const magnify = horizontalMagnification('scale-follow', snap.physics.rho);
  mesh.position.x = -cx * SCENE_SCALE * magnify;
  mesh.position.z = cy * SCENE_SCALE * magnify;
  // Vertical: physical surface, NO recentering. Y scale stays exactly 1.
  mesh.position.y = 0;
  mesh.scale.set(magnify, 1.0, magnify);
}

/**
 * Apply the same horizontal transform to the trail so it stays registered
 * with the scale-follow terrain. The trail is built in physical
 * coordinates (`buildTrail` with a physical-Y mode), so the identical
 * recenter + 1/rho0 X/Z magnification glues it to the surface. Y is left
 * exactly as built.
 */
export function scaleFollowTrailTransform(trail: THREE.Line, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const magnify = horizontalMagnification('scale-follow', snap.physics.rho);
  trail.position.x = -cx * SCENE_SCALE * magnify;
  trail.position.z = cy * SCENE_SCALE * magnify;
  trail.position.y = 0;
  trail.scale.set(magnify, 1.0, magnify);
}

/**
 * Position the third-person camera behind and above the rider.
 *
 * - physical: raw (x, y, lambda*sigma) embedding — geometry-debug mode.
 * - scale-follow: the world is horizontally recentered/magnified beneath
 *   the rider, so the rider sits at X/Z origin, but the vertical follows
 *   the physical surface: camera Y tracks surfaceY(current sigma) exactly
 *   like physical mode. Feels like the normal third-person skateboard
 *   camera.
 * - treadmill: scale-stabilized chart X=(x-x0)/rho0, Y=(z(c)-z(c0))
 *   around the CURRENT rider position c0 — debug presentation ONLY; never
 *   feeds physics (guaranteed structurally: the camera only reads the
 *   snapshot, and this module has no path back into the recorder).
 */
// Persistent smoothed heading for the camera so it stays smoothly behind without whipping
let smoothedCamHeading: number | null = null;

export function getSmoothedCamHeading(): number | null {
  return smoothedCamHeading;
}

export function resetCameraSmoothing(): void {
  smoothedCamHeading = null;
}

export function updateSmoothedHeading(snap: DebugSnapshot, dt: number): number {
  // Behind and above, biased along the rider's heading or control direction.
  const [vx, vy] = snap.physics.velocity;
  const speed = Math.hypot(vx, vy);

  // Heading stays in c-space; scene placement explicitly maps y to -Z.
  let targetHeading = smoothedCamHeading ?? 0;
  if (speed > 0) {
    targetHeading = Math.atan2(vy, vx);
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    if (Math.hypot(dx, dy) > 1e-6) {
      targetHeading = Math.atan2(dy, dx);
    }
  }

  if (smoothedCamHeading === null) {
    smoothedCamHeading = targetHeading;
  } else {
    // Smooth angle interpolation handling wrap-around
    let diff = targetHeading - smoothedCamHeading;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    // Damped follow: ~3.5 rad/s keeps camera behind the board without jarring snaps
    const blend = 1 - Math.exp(-3.5 * Math.max(0, dt));
    smoothedCamHeading += diff * blend;
  }

  return smoothedCamHeading;
}

export const CAMERA_BACK_DISTANCE = 4.8;
export const CAMERA_UP_DISTANCE = 2.8;

export function updateCamera(
  camera: THREE.PerspectiveCamera,
  snap: DebugSnapshot,
  mode: CameraMode,
  dt: number = 0.016
): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;
  updateSmoothedHeading(snap, dt);

  let rx: number, rz: number;
  let followSurface: boolean;
  if (mode === 'physical') {
    rx = cx * SCENE_SCALE;
    rz = -cy * SCENE_SCALE;
    followSurface = true;
  } else if (mode === 'scale-follow') {
    // The terrain carries the (x-x0)/rho0 horizontal transform; the rider
    // is pinned at the horizontal origin, but height stays physical.
    rx = 0;
    rz = 0;
    followSurface = true;
  } else if (mode === 'hyperbolic') {
    // Hyperbolic camera: the terrain and trail vertices are projected into the
    // camera-centered Poincaré ball (origin = (0, 0, 0)), with inverse camera
    // orientation already applied. The Three.js camera sits at (0, 0, 0) with
    // default orientation (looking down -Z with +Y up).
    camera.position.set(0, 0, 0);
    camera.quaternion.set(0, 0, 0, 1);
    return;
  } else {
    // Treadmill: the rider is pinned at the chart origin; the terrain mesh
    // carries the (x-x0)/rho0 transform (see treadmillTransform).
    rx = 0;
    rz = 0;
    followSurface = false;
  }

  const heading = smoothedCamHeading ?? 0;
  // Placed further back and higher up to reveal more forward landscape
  const back = CAMERA_BACK_DISTANCE;
  const up = CAMERA_UP_DISTANCE;
  const riderY = followSurface ? surfaceY(sigma) : 0;
  const targetY = riderY + 0.8;

  const camX = rx - Math.cos(heading) * back;
  const camZ = rz + Math.sin(heading) * back;
  const camY = riderY + up;

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

/** Lights + atmospheric backdrop for an immersive game environment. */
export function buildSceneDressing(scene: THREE.Scene): void {
  const ambient = new THREE.AmbientLight(0xdde8ff, 0.65);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff4d6, 1.25);
  sun.position.set(12, 22, 10);
  scene.add(sun);

  // Rim / fill light for character silhouette
  const rimLight = new THREE.DirectionalLight(0x66aaff, 0.45);
  rimLight.position.set(-10, 8, -12);
  scene.add(rimLight);

  scene.background = new THREE.Color(0x070714);
  // Fog distances are set per-terrain-rebuild by applyRenderDistance so the
  // mesh edge always hides inside the fog wall at every LOD level.
  scene.fog = new THREE.Fog(0x070714, 20, 50);

  // Distant starfield particles for celestial atmosphere
  const starGeo = new THREE.BufferGeometry();
  const starCount = 350;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const r = 80 + Math.random() * 40;
    const theta = Math.random() * Math.PI * 2;
    const phi = (Math.random() * 0.4 + 0.1) * Math.PI; // Upper hemisphere
    starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = r * Math.cos(phi);
    starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  const starMat = new THREE.PointsMaterial({
    color: 0xccddff,
    size: 1.2,
    transparent: true,
    opacity: 0.75,
  });
  const stars = new THREE.Points(starGeo, starMat);
  scene.add(stars);
}

/**
 * Apply the LOD render distance: camera far plane + fog wall track the
 * patch size so fidelity stays balanced with performance as scale shifts
 * (issue #111). Call on every terrain rebuild.
 *
 * The fog wall is floored at the camera-to-rider distance: the LOD patch
 * shrinks at deep scale, and a fog wall tighter than the camera distance
 * would swallow the whole scene (the "black viewport" failure mode).
 * Horizontally magnified modes (treadmill AND scale-follow) inflate the
 * effective patch via `horizontalMagnification`, so their fog wall
 * inflates with it; physical mode is unchanged.
 */
export function applyRenderDistance(
  camera: THREE.PerspectiveCamera,
  scene: THREE.Scene,
  mode: CameraMode,
  rho: number,
  half: number
): void {
  if (mode === 'hyperbolic') {
    // In hyperbolic mode, coordinates live in the scaled Poincaré ball
    // with radius ~HYPERBOLIC_VISUAL_SCALE (20 units).
    camera.near = 0.1;
    camera.far = 100.0;
    camera.updateProjectionMatrix();
    const fog = scene.fog;
    if (fog && 'near' in fog && 'far' in fog) {
      (fog as THREE.Fog).near = 12.0;
      (fog as THREE.Fog).far = 19.0;
    }
    return;
  }
  const r = Math.max(rho, 1e-9);
  const magnify = horizontalMagnification(mode, r);
  const patchScene = half * 2 * SCENE_SCALE * magnify;
  const diagonal = patchScene * Math.SQRT2;
  // updateCamera keeps the camera ~sqrt(4.8^2 + 2.8^2) ~ 5.6 scene units
  // from the rider; the fog must start beyond the subject.
  const cameraDist = 5.6;
  const fogNear = Math.max(cameraDist * 1.15, diagonal * 0.25);
  const fogFar = Math.max(diagonal * 1.15, cameraDist * 2.2);
  const far = Math.max(diagonal * 1.8, cameraDist * 2.8);

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
