import * as THREE from 'three';
import type { TerrainPatch } from './debugCockpit';

// Rendering mirror of runtime-core/src/hyperbolic.rs, pinned by hyperbolicParity.test.ts.
export const HYPERBOLIC_VERTEX_PROJECTION = `
vec3 hyperbolicProject(vec3 point, vec3 camera) {
  vec3 p = vec3(point.xy - camera.xy, point.z) / camera.z;
  float d = dot(p.xy, p.xy) + (p.z + 1.0) * (p.z + 1.0);
  vec3 b = vec3(2.0 * p.xy, dot(p, p) - 1.0) / d;
  return vec3(b.x, -b.z, -b.y);
}
`;

interface ProjectionUniforms {
  hyperbolicCamera: THREE.IUniform<THREE.Vector3>;
  hyperbolicRotation: THREE.IUniform<THREE.Matrix3>;
  hyperbolicScale: THREE.IUniform<number>;
}

const projections = new WeakMap<THREE.Mesh, ProjectionUniforms>();

/** Upload once per terrain build. Camera motion only changes these uniforms. */
export function terrainProjectionUniforms(mesh: THREE.Mesh, patch: TerrainPatch): ProjectionUniforms {
  const existing = projections.get(mesh);
  if (existing) return existing;
  if (!patch.upperZ || patch.upperZ.length !== patch.n * patch.n) {
    throw new Error('Hyperbolic terrain requires Rust upperZ samples; rebuild wasm-orbit.');
  }
  if (!(mesh.material instanceof THREE.MeshStandardMaterial)) {
    throw new Error('Hyperbolic terrain requires MeshStandardMaterial.');
  }
  const vertices = new Float32Array(patch.n * patch.n * 3);
  for (let i = 0; i < patch.upperZ.length; i++) {
    // Subtract in JS double precision before upload. Absolute c values in a
    // float32 attribute lose local terrain detail at deep scales.
    vertices[3 * i] = patch.positions[3 * i] - patch.center[0];
    vertices[3 * i + 1] = patch.positions[3 * i + 1] - patch.center[1];
    vertices[3 * i + 2] = patch.upperZ[i];
  }
  mesh.geometry.setAttribute('upperHalfPosition', new THREE.BufferAttribute(vertices, 3));
  const uniforms: ProjectionUniforms = {
    hyperbolicCamera: { value: new THREE.Vector3() },
    hyperbolicRotation: { value: new THREE.Matrix3() },
    hyperbolicScale: { value: 20 },
  };
  const material = mesh.material;
  // Fragment derivatives compute normals from the projected triangle. No
  // camera-dependent CPU normal rebuild and no stale Euclidean lighting.
  material.flatShading = true;
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = `
attribute vec3 upperHalfPosition;
uniform vec3 hyperbolicCamera;
uniform mat3 hyperbolicRotation;
uniform float hyperbolicScale;
varying float vHyperbolicRadius;
varying vec2 vPatchUv;
${HYPERBOLIC_VERTEX_PROJECTION}
${shader.vertexShader}`.replace('#include <begin_vertex>', `
vec3 transformed = hyperbolicRotation * hyperbolicProject(upperHalfPosition, hyperbolicCamera) * hyperbolicScale;
vHyperbolicRadius = length(transformed);
vPatchUv = uv;
`);
    shader.fragmentShader = `varying float vHyperbolicRadius;\nvarying vec2 vPatchUv;\n${shader.fragmentShader}`
      .replace('#include <fog_fragment>', `
#ifdef USE_FOG
  float radialFog = smoothstep(fogNear, fogFar, vHyperbolicRadius);
  vec2 edge = min(vPatchUv, 1.0 - vPatchUv);
  float edgeFog = 1.0 - smoothstep(0.0, 0.08, min(edge.x, edge.y));
  gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, max(radialFog, edgeFog));
#endif
`);
  };
  material.customProgramCacheKey = () => 'hyperbolic-terrain/1';
  material.needsUpdate = true;
  // The CPU bounding sphere describes the source mesh, not the GPU projection.
  mesh.frustumCulled = false;
  projections.set(mesh, uniforms);
  return uniforms;
}
