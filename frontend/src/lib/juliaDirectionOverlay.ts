import * as THREE from 'three';
import { getWasmModule } from './orbitSynthesizer';
import type { TerrainPatch } from './debugCockpit';

const overlays = new WeakMap<THREE.MeshStandardMaterial, { value: THREE.Vector2 }>();

export function setJuliaContours(mesh: THREE.Mesh, patch: TerrainPatch, scale: boolean, turn: boolean) {
  const material = mesh.material;
  if (!(material instanceof THREE.MeshStandardMaterial)) return;
  let mode = overlays.get(material);
  if (!mode && !scale && !turn) return;
  if (!mode) {
    const samples = getWasmModule().juliaEffectContours(Float64Array.from(patch.positions));
    mesh.geometry.setAttribute('juliaContour', new THREE.BufferAttribute(samples, 4));
    const enabled = { value: new THREE.Vector2() };
    mode = enabled;
    overlays.set(material, enabled);
    const previous = material.onBeforeCompile.bind(material);
    const key = material.customProgramCacheKey();
    material.customProgramCacheKey = () => key + ':julia-contours';
    material.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      shader.uniforms.juliaContourMode = enabled;
      shader.vertexShader = 'attribute vec4 juliaContour; varying vec4 vJuliaContour;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <uv_vertex>',
        '#include <uv_vertex>\nvJuliaContour = juliaContour;');
      shader.fragmentShader = 'varying vec4 vJuliaContour; uniform vec2 juliaContourMode;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec2 levels = vec2(vJuliaContour.x * 12.0,
            atan(vJuliaContour.z, vJuliaContour.y) * (12.0 / 6.28318530718));
          // Use the phase's local differential across the angle wrap.
          vec2 phase = vJuliaContour.yz;
          float turnWidth = (abs(phase.x * dFdx(phase.y) - phase.y * dFdx(phase.x)) +
            abs(phase.x * dFdy(phase.y) - phase.y * dFdy(phase.x))) *
            (12.0 / 6.28318530718) / max(dot(phase, phase), 0.000001);
          vec2 width = max(vec2(fwidth(levels.x), turnWidth), vec2(0.00000001));
          // Nested subdivisions keep contours visible at close terrain zooms.
          vec2 detail = clamp(-log2(48.0 * width), vec2(0.0), vec2(16.0));
          vec2 subdivision = exp2(floor(detail));
          vec2 scaledWidth = width * subdivision;
          vec2 coarseDistance = abs(fract(levels * subdivision + 0.5) - 0.5);
          vec2 fineDistance = abs(fract(levels * subdivision * 2.0 + 0.5) - 0.5);
          vec2 coarse = 1.0 - smoothstep(scaledWidth * 0.55, scaledWidth * 1.55, coarseDistance);
          vec2 fine = 1.0 - smoothstep(scaledWidth * 1.1, scaledWidth * 3.1, fineDistance);
          vec2 line = max(coarse, fine * fract(detail)) *
            (1.0 - smoothstep(vec2(0.2), vec2(0.45), scaledWidth)) * juliaContourMode;
          vec3 contour = (vec3(0.02, 0.9, 1.0) * line.x + vec3(1.0, 0.16, 0.76) * line.y)
            / max(line.x + line.y, 0.000001);
          if (vJuliaContour.w > 0.99999 && length(phase) > 0.95) {
            diffuseColor.rgb = mix(diffuseColor.rgb, contour, max(line.x, line.y));
          }
        }
      `);
    };
    material.needsUpdate = true;
  }
  mode.value.set(Number(scale), Number(turn));
}
