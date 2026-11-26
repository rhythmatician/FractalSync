// Run with: node frontend/scripts/check_julia_morphology.mjs
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(frontend, '..');
const output = resolve(process.argv[2] ?? resolve(root, 'tmp/julia-140/results.json'));
const python = process.env.FRACTALSYNC_PYTHON ?? resolve(root, process.platform === 'win32'
  ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const shader = await readFile(resolve(root, 'shared/shaders/julia.frag'), 'utf8');
const predicate = 'if (niter <= 0.0) return vec3(0.0);';
assert.equal(shader.split(predicate).length, 2, 'expected exactly one pre-palette predicate');
const maskShader = shader.replace(predicate, `
  float n128, n64, unusedStripe, unusedDem, unusedConfidence;
  vec2 unusedNormal;
  smoothIterJulia(z0, u_juliaSeed, 128, u_stripe_s, u_stripe_sig,
    n128, unusedStripe, unusedDem, unusedNormal, unusedConfidence);
  smoothIterJulia(z0, u_juliaSeed, 64, u_stripe_s, u_stripe_sig,
    n64, unusedStripe, unusedDem, unusedNormal, unusedConfidence);
  return vec3(niter <= 0.0 ? 1.0 : 0.0, n128 <= 0.0 ? 1.0 : 0.0, n64 <= 0.0 ? 1.0 : 0.0);
`);

const seeds = [
  { name: 'disk', c: [0, 0] },
  { name: 'cardioid', c: [-0.17, 0.09] },
  { name: 'period-two', c: [-0.98, 0.09] },
  { name: 'near-shore', c: [-0.74, 0.132] },
  { name: 'wide-exterior', c: [1.2, 0.25] },
];
const base = { zoom: 1, colorHue: 0.58, colorSat: 0.75, colorBright: 0.6, speed: 0, rim: 0.08 };
const variants = [
  { name: 'base', size: 128, params: base },
  { name: 'repeat', size: 128, params: base },
  { name: 'hue', size: 128, params: { ...base, colorHue: 0.08 } },
  { name: 'brightness', size: 128, params: { ...base, colorBright: 0.1 } },
  { name: 'rim', size: 128, params: { ...base, rim: 0.75 } },
  { name: 'zoom', size: 128, params: { ...base, zoom: 1.5 } },
  { name: 'small', size: 64, params: base },
  { name: 'large', size: 256, params: base },
];

const metricsSource = String.raw`
import base64
import importlib.metadata
import json
import sys
import runtime_core as rust

def measure(image, width, channels, c):
    return vars(rust.compute_runtime_visual_metrics([code / 255 for code in image],
                                                     width, width, channels, c, 500))

rows = []
for frame in json.load(sys.stdin):
    size = frame['size']
    c = complex(*frame['c'])
    rgb = base64.b64decode(frame.pop('rgb'))
    mask_rgb = base64.b64decode(frame.pop('mask'))
    assert len(rgb) == len(mask_rgb) == 3 * size * size
    mask500, mask128, mask64 = (mask_rgb[i::3] for i in range(3))
    assert set(mask_rgb) <= {0, 255}
    assert all(a <= b <= c for a, b, c in zip(mask500, mask128, mask64, strict=True))
    frames = ((500, mask500), (128, mask128), (64, mask64))
    masks = {n: measure(mask, size, 1, c) for n, mask in frames}
    assert all(abs(masks[n]['brightness_mean'] - mask.count(255) / size ** 2) < 1e-15
               for n, mask in frames)
    if c == 0:
        span = 2.7 / frame['params']['zoom']
        axis = [(i + 0.5 - size / 2) * span / size for i in range(size)]
        expected_disk = bytes(255 if x*x + y*y <= 1 else 0 for y in axis for x in axis)
        assert all(mask == expected_disk for _, mask in frames)
    rows.append({**frame, 'rgb_metrics': measure(rgb, size, 3, c), 'mask_metrics': masks})
json.dump({'installed_distribution': importlib.metadata.version('runtime-core'),
           'controller': rust.CONTROLLER_VERSION, 'rows': rows}, sys.stdout, allow_nan=False)
`;

const server = await createServer({
  root: frontend, configFile: false, appType: 'custom', server: { host: '127.0.0.1', port: 0 },
});
server.middlewares.use((request, response, next) => {
  if (request.url === '/__julia_morphology__.html') {
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><title>FractalSync Julia morphology probe</title>');
  } else if (request.url === '/api/shader/julia.frag') {
    response.setHeader('Content-Type', 'text/plain');
    response.end(shader);
  } else {
    next();
  }
});

let browser;
try {
  await server.listen();
  browser = await chromium.launch({
    headless: true, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
  });
  async function samplePage(mask) {
    const page = await browser.newPage({ viewport: { width: 400, height: 360 }, deviceScaleFactor: 1 });
    try {
      if (mask) {
        await page.route('**/api/shader/julia.frag', route => route.fulfill({
          body: maskShader, contentType: 'text/plain',
        }));
      }
      await page.goto(`${server.resolvedUrls.local[0]}__julia_morphology__.html`);
      return await page.evaluate(async ({ seeds, variants, base }) => {
        const { JuliaRenderer } = await import('/src/lib/juliaRenderer.ts');
        const canvas = document.createElement('canvas');
        document.body.append(canvas);
        const renderer = new JuliaRenderer(canvas);
        await renderer.init();
        const gl = canvas.getContext('webgl');
        if (!gl) throw new Error('WebGL unavailable');
        gl.disable(gl.DITHER);

        function render(c, size, params) {
          canvas.style.width = `${size}px`;
          canvas.style.height = `${size}px`;
          renderer.setRimIntensity(params.rim);
          renderer.renderFrozen({ ...params, juliaSeed: { real: c[0], imag: c[1] } });
          if (canvas.width !== size || canvas.height !== size) throw new Error('unexpected render size');
          const pixels = new Uint8Array(4 * size ** 2);
          gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          if (gl.getError() !== gl.NO_ERROR) throw new Error('WebGL readback failed');
          const rgb = new Uint8Array(3 * size ** 2);
          for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) rgb.set(pixels.subarray(i, i + 3), j);
          let binary = '';
          for (let i = 0; i < rgb.length; i += 8192) binary += String.fromCharCode(...rgb.subarray(i, i + 8192));
          return btoa(binary);
        }

        const rows = [];
        for (const seed of seeds) {
          for (const variant of variants) {
            if (variant.name === 'repeat') render([0.29, -0.04], variant.size, base);
            rows.push({ seed: seed.name, c: seed.c, variant: variant.name, size: variant.size,
              params: variant.params, rgb: render(seed.c, variant.size, variant.params) });
          }
        }
        const debug = gl.getExtension('WEBGL_debug_renderer_info');
        const program = gl.getParameter(gl.CURRENT_PROGRAM);
        const cap = gl.getUniformLocation(program, 'u_maxIter');
        return {
          rows,
          api: gl.getParameter(gl.VERSION),
          renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
          context: gl.getContextAttributes(),
          color_bits: [gl.RED_BITS, gl.GREEN_BITS, gl.BLUE_BITS].map(key => gl.getParameter(key)),
          dithering: gl.isEnabled(gl.DITHER), dpr: devicePixelRatio,
          active_iteration_uniform: gl.getUniform(program, cap),
        };
      }, { seeds, variants, base });
    } finally {
      await page.close();
    }
  }

  const { rows: rgbRows, ...renderer } = await samplePage(false);
  const { rows: maskRows, ...maskRenderer } = await samplePage(true);
  assert.equal(renderer.active_iteration_uniform, 500);
  assert.equal(maskRenderer.active_iteration_uniform, 500);
  const sha256 = text => createHash('sha256').update(text).digest('hex');
  const frames = rgbRows.map((row, i) => {
    const mask = maskRows[i];
    assert.equal(row.seed, mask.seed);
    assert.equal(row.variant, mask.variant);
    return { ...row, mask: mask.rgb, rgb_sha256: sha256(Buffer.from(row.rgb, 'base64')),
      mask_sha256: sha256(Buffer.from(mask.rgb, 'base64')) };
  });
  for (const seed of seeds) {
    const measured = new Map(frames.filter(row => row.seed === seed.name).map(row => [row.variant, row]));
    const original = measured.get('base');
    assert.equal(original.rgb_sha256, measured.get('repeat').rgb_sha256);
    for (const name of ['repeat', 'hue', 'brightness', 'rim']) {
      assert.equal(original.mask_sha256, measured.get(name).mask_sha256);
      if (name !== 'repeat') assert.notEqual(original.rgb_sha256, measured.get(name).rgb_sha256);
    }
  }
  const child = spawnSync(python, ['-c', metricsSource], {
    cwd: root, input: JSON.stringify(frames), encoding: 'utf8', timeout: 180_000, maxBuffer: 8 * 1024 * 1024,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) throw new Error(`Native metrics failed: ${child.stderr}`);
  const metrics = JSON.parse(child.stdout);
  const result = { schema: 'julia-morphology-diagnostic/1', shader_sha256: sha256(shader),
    tapped_shader_sha256: sha256(maskShader), chromium: browser.version(), renderer, maskRenderer, ...metrics };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  console.log(`Wrote ${result.rows.length} Julia morphology observations to ${output}`);
} finally {
  await browser?.close();
  await server.close();
}
