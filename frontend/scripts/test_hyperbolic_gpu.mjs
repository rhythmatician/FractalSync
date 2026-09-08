// Run with: npm --prefix frontend run test:hyperbolic-gpu
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const golden = JSON.parse(await readFile(resolve(root, '../shared/golden_vectors.json'), 'utf8'));
const server = await createServer({ root, server: { host: '127.0.0.1', port: 0 } });
await server.listen();
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  // Avoid mounting the app while importing the real render modules.
  await page.goto(`${server.resolvedUrls.local[0]}@vite/client`);
  const result = await page.evaluate(async cases => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { HYPERBOLIC_VERTEX_PROJECTION } = await import('/src/lib/hyperbolicTerrainMaterial.ts');
    const hyper = await import('/src/lib/hyperbolicCamera.ts');
    const cockpit = await import('/src/lib/cockpitScene.ts');
    const wasm = await import('/wasm/orbit_synth_wasm.js');
    await wasm.default();
    document.body.innerHTML = '';
    document.body.style.margin = '0';
    const canvas = document.createElement('canvas');
    document.body.append(canvas);
    const gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('WebGL2 unavailable');
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, `#version 300 es
precision highp float;
in vec3 point;
in vec3 camera;
out vec3 projected;
${HYPERBOLIC_VERTEX_PROJECTION}
void main() { projected = hyperbolicProject(point, camera); gl_Position = vec4(projected, 1.0); }`);
    const fragment = compile(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
out vec4 color;
void main() { color = vec4(1.0); }`);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.transformFeedbackVaryings(program, ['projected'], gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    for (const key of ['point', 'camera']) {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cases.flatMap(c => c[key])), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(program, key);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }
    const output = gl.createBuffer();
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, output);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, cases.length * 12, gl.STREAM_READ);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, output);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, cases.length);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    const actual = new Float32Array(cases.length * 3);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, actual);
    let maxError = 0;
    cases.forEach((c, i) => {
      const expected = [c.projected[0], -c.projected[2], -c.projected[1]];
      expected.forEach((v, j) => { maxError = Math.max(maxError, Math.abs(v - actual[3 * i + j])); });
    });
    if (!(maxError < 2e-6)) throw new Error(`GPU projection drift: ${maxError}`);
    if (gl.getError() !== gl.NO_ERROR) throw new Error('Transform feedback produced a GL error');
    // Use a fresh context so raw GL state cannot contaminate the Three renderer.
    canvas.remove();
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(1100, 800);
    document.body.append(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c18);
    scene.fog = new THREE.Fog(0x080c18, 12, 19);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404060, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(3, 6, 4);
    scene.add(light);
    const config = new wasm.ManifoldConfig(0.23, 0.0001, 2.25, 1, 1 / Math.PI);
    const snap = wasm.debugSnapshotFromState(-0.74, 0.132, 0.001, 0.001, undefined, 0, 0, 0, config, NaN, 0);
    if (!snap.physics.upperHalf) throw new Error('WASM upperHalf geometry missing');
    const patch = wasm.debugTerrainPatch(...snap.physics.c, snap.physics.rho * 6, 512, config);
    const mesh = cockpit.buildTerrainMesh(patch);
    scene.add(mesh);
    const rider = await cockpit.buildRider();
    scene.add(rider);
    const camera = new THREE.PerspectiveCamera(60, 1100 / 800, 0.1, 100);
    cockpit.resetCameraSmoothing();
    cockpit.updateCamera(camera, snap);
    const heading = cockpit.getSmoothedCamHeading();
    const frame = hyper.computeHyperbolicCameraFrame(snap, heading);
    hyper.transformMeshToHyperbolic(mesh, patch, snap, heading, frame);
    hyper.transformRiderToHyperbolic(rider, snap, heading, frame);
    const pos = mesh.geometry.getAttribute('position');
    const upper = mesh.geometry.getAttribute('upperHalfPosition');
    const initialVersions = [pos.version, upper.version];
    const start = performance.now();
    for (let i = 0; i < 1000; i++) hyper.transformMeshToHyperbolic(mesh, patch, snap, heading + i / 1000);
    const updateMs = (performance.now() - start) / 1000;
    if (pos.version !== initialVersions[0] || upper.version !== initialVersions[1]) throw new Error('Camera update mutated static buffers');
    hyper.transformMeshToHyperbolic(mesh, patch, snap, heading, frame);
    renderer.render(scene, camera);
    if (renderer.info.programs.some(p => p.diagnostics && !p.diagnostics.runnable)) throw new Error('Three.js shader failed');
    if (renderer.getContext().getError() !== 0) throw new Error('Three.js render produced a GL error');
    const triangles = renderer.info.render.triangles;
    if (triangles < 2 * 511 * 511) throw new Error(`Terrain not rendered: ${triangles} triangles`);
    // Keep the scene for screenshot capture.
    return { cases: cases.length, maxError, vertices: upper.count, triangles, updateMs };
  }, golden.hyperbolic_cases);
  await page.screenshot({ path: resolve(tmpdir(), 'fractalsync-hyperbolic-gpu.png') });
  assert.deepEqual(errors, [], `Browser errors: ${errors.join('\n')}`);
  console.log(JSON.stringify({
    ...result,
    screenshot: resolve(tmpdir(), 'fractalsync-hyperbolic-gpu.png'),
  }, null, 2));
} finally {
  await browser.close();
  await server.close();
}
