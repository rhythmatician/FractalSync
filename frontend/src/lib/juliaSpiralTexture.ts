import * as THREE from 'three';
import { getWasmModule } from './orbitSynthesizer';
import type { TerrainPatch } from './debugCockpit';

export type SpiralLayer = 'off' | 'scale' | 'rotation' | 'residual';
export type SpiralBand = 'broad' | 'inner';

type Grid = {
  center: [number, number];
  half: number;
  size: number;
  samples: ([number, number, number][] | null)[];
};

let grid: Grid | undefined;
const textures = new Map<string, THREE.DataTexture>();
const empty = new THREE.DataTexture(new Uint8Array(4), 1, 1);
empty.needsUpdate = true;

function spiralGrid(): Grid {
  grid ??= JSON.parse(getWasmModule().juliaSpiralTextureJson()) as Grid;
  return grid;
}

export function spiralTextureAnchor(): [number, number] {
  return spiralGrid().center;
}

function texture(data: Grid, layer: Exclude<SpiralLayer, 'off'>, band: SpiralBand) {
  const key = layer + band;
  const cached = textures.get(key);
  if (cached) return cached;
  const pixels = new Uint8Array(data.size ** 2 * 4);
  const field = { scale: 0, rotation: 1, residual: 2 }[layer];
  const maximum = Math.max(1e-12, ...data.samples.flatMap(sample =>
    sample?.map(values => Math.abs(values[field])) ?? []));
  data.samples.forEach((sample, i) => {
    if (!sample) return;
    const value = sample[band === 'broad' ? 0 : 1][field];
    const strength = Math.min(1, Math.abs(value) / (layer === 'residual' ? 1 : maximum));
    const color = layer === 'residual' ? [234, 100, 255] : value < 0 ? [88, 162, 255] : [255, 104, 63];
    const offset = i * 4;
    pixels.set([...color, Math.round(245 * strength)], offset);
  });
  const result = new THREE.DataTexture(pixels, data.size, data.size);
  result.colorSpace = THREE.SRGBColorSpace;
  result.needsUpdate = true;
  textures.set(key, result);
  return result;
}

export function drawSpiralTexturePreview(canvas: HTMLCanvasElement, layer: Exclude<SpiralLayer, 'off'>, band: SpiralBand) {
  const data = spiralGrid();
  const pixels = texture(data, layer, band).image.data as Uint8Array;
  const topDown = new Uint8ClampedArray(pixels.length);
  const stride = data.size * 4;
  for (let row = 0; row < data.size; row++) {
    const start = (data.size - 1 - row) * stride;
    topDown.set(pixels.subarray(start, start + stride), row * stride);
  }
  canvas.width = canvas.height = data.size;
  canvas.getContext('2d')?.putImageData(new ImageData(topDown, data.size), 0, 0);
  return data.samples[Math.floor(data.samples.length / 2)]?.[band === 'broad' ? 0 : 1];
}

type Uniforms = {
  spiralMap: { value: THREE.Texture };
  spiralUv: { value: THREE.Vector4 };
};
const materials = new WeakMap<THREE.MeshStandardMaterial, Uniforms>();

/** Drape the diagnostic grid over the existing terrain without replacing its geometry. */
export function setSpiralTexture(mesh: THREE.Mesh, patch: TerrainPatch, layer: SpiralLayer, band: SpiralBand) {
  const material = mesh.material;
  if (!(material instanceof THREE.MeshStandardMaterial)) return;
  let uniforms = materials.get(material);
  if (layer === 'off') {
    if (uniforms) uniforms.spiralMap.value = empty;
    return;
  }
  const data = spiralGrid();
  if (!uniforms) {
    uniforms = { spiralMap: { value: empty }, spiralUv: { value: new THREE.Vector4() } };
    const values = uniforms;
    materials.set(material, values);
    const previous = material.onBeforeCompile.bind(material);
    const cacheKey = material.customProgramCacheKey();
    material.customProgramCacheKey = () => cacheKey + ':spiral-texture';
    material.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      Object.assign(shader.uniforms, values);
      shader.vertexShader = 'varying vec2 vSpiralUv;\nuniform vec4 spiralUv;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <uv_vertex>',
        '#include <uv_vertex>\nvSpiralUv = spiralUv.xy + uv * spiralUv.zw;');
      shader.fragmentShader = 'varying vec2 vSpiralUv;\nuniform sampler2D spiralMap;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
        if (all(greaterThanEqual(vSpiralUv, vec2(0.0))) && all(lessThanEqual(vSpiralUv, vec2(1.0)))) {
          vec4 spiralColor = texture2D(spiralMap, vSpiralUv);
          diffuseColor.rgb = mix(diffuseColor.rgb, spiralColor.rgb, spiralColor.a);
        }`);
    };
    material.needsUpdate = true;
  }
  const last = (patch.n ** 2 - 1) * 3;
  const span = 2 * data.half;
  const [x, y] = patch.positions;
  uniforms.spiralUv.value.set(
    (x - data.center[0] + data.half) / span,
    (y - data.center[1] + data.half) / span,
    (patch.positions[last] - x) / span,
    (patch.positions[last + 1] - y) / span,
  );
  uniforms.spiralMap.value = texture(data, layer, band);
}
