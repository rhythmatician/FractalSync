// Run with: node frontend/scripts/check_julia_sensitivity.mjs
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { calibrateJuliaSensitivity } from './julia_sensitivity_calibration.mjs';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(frontend, '..');
const heldOut = process.argv[2] === '--held-out';
const contours = process.argv[2] === '--contours';
const defaultOutput = heldOut ? 'tmp/julia-151/held-out.json'
  : contours ? 'tmp/navigation-185/julia-motion.json' : 'tmp/julia-151/results.json';
const outputArgument = heldOut ? process.argv[3] : contours ? process.argv[4] : process.argv[2];
const output = resolve(outputArgument ?? resolve(root, defaultOutput));
const replayFile = heldOut ? undefined : process.argv[3];
if (contours) assert.ok(replayFile, 'Expected the contour_controls_trial JSON path');
const replayBytes = replayFile ? await readFile(resolve(replayFile)) : null;
const replay = replayBytes ? JSON.parse(replayBytes) : null;
function sampleFrames(samples, stride) {
  return samples.filter((_, i) => i % stride === 0 || i === samples.length - 1)
    .map(({ mechanical_time: t, position: c }) => ({ t, c }));
}
const paths = contours
  ? [['cardioid', replay], ['period-two', replay.period_two]].flatMap(([chart, pair]) =>
    ['natural_multiplier', 'map_iso_scale'].map(profile => ({
      chart, profile, frames: sampleFrames(pair[profile].frames, 4),
    })))
  : replay ? replay.profiles.map(profile => ({
    profile: profile.profile, crossing_time: profile.fine.crossing.time,
    frames: sampleFrames(profile.fine.samples, 8),
  })) : [];
const shaderPath = 'shared/shaders/julia.frag';
const fragment = await readFile(resolve(root, shaderPath));
const resolution = heldOut ? 160 : 80;
const spacings = [0.02, 0.01, 0.005, 0.0025, 0.00125];
const presentation = { zoom: 1, colorHue: 0.58, colorSat: 0.75, colorBright: 0.6, speed: 0 };
const states = heldOut ? [
  { name: 'held-out-cardioid', c: [-0.31, 0.19] },
  { name: 'held-out-period-two', c: [-1.04, -0.13] },
  { name: 'held-out-attachment', c: [-0.75, 0] },
  { name: 'held-out-period-three', c: [-0.122561, -0.744862] },
  { name: 'held-out-exterior-collar', c: [0.27, 0.012] },
  { name: 'held-out-distant', c: [1.1, -0.31] },
] : [
  { name: 'real-axis-center', c: [0, 0] },
  { name: 'wide-cardioid', c: [-0.17, 0.09] },
  { name: 'period-two', c: [-0.98, 0.09] },
  { name: 'inside-near-shore', c: [-0.1, 0.85] },
  { name: 'outside-near-shore', c: [-0.74, 0.132] },
  { name: 'outside-wide', c: [1.2, 0.25] },
  { name: 'provider-cut-locus', c: [-1.001, 0] },
];

const python = process.env.FRACTALSYNC_PYTHON ?? resolve(root, process.platform === 'win32'
  ? '.venv/Scripts/python.exe' : '.venv/bin/python');

// The installed native binding is the geometry authority for this run.
// The browser only renders and reads the image; it does not respecify G.
const geometrySource = String.raw`
import importlib.metadata
import json
import sys
import runtime_core as rust

request = json.load(sys.stdin)
asset = rust.get_builtin_distance_field_py("mandelbrot_1024")
full = rust.ManifoldConfig()
scale = rust.ManifoldConfig()
scale.lambda_sq = 0.0
rows = []
for re, im in request["points"]:
    c = complex(re, im)
    try:
        jet = json.loads(rust.geometry_provider_query(c, full.epsilon))
        row = {"jet": jet}
        if request["metrics"] and jet["validity"] == "regular" and not jet["is_bridge"]:
            row.update(
                rho=rust.manifold_regularized_distance(c, full.epsilon),
                sigma=rust.manifold_mandelbrot_scale(c, full),
                G=rust.manifold_induced_metric(c, full),
                G0=rust.manifold_induced_metric(c, scale),
            )
    except (RuntimeError, ValueError) as error:
        row = {"error": str(error)}
    rows.append(row)
json.dump({
    "provider": rust.geometry_provider_version(),
    "installed_distribution": importlib.metadata.version("runtime-core"),
    "controller": rust.CONTROLLER_VERSION,
    "asset": {"name": "mandelbrot_1024", "dimensions_and_bounds": asset},
    "profile": {"epsilon": full.epsilon, "d_ref": full.d_ref, "lambda_sq": full.lambda_sq,
                "lambda_sq_scale_only": scale.lambda_sq},
    "points": rows,
}, sys.stdout, allow_nan=False)
`;

function queryGeometry(points, metrics) {
  const child = spawnSync(python, ['-c', geometrySource], {
    cwd: root, input: JSON.stringify({ points, metrics }), encoding: 'utf8', timeout: 180_000,
    maxBuffer: 8 * 1024 * 1024,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) throw new Error(`Native geometry failed: ${child.stderr}`);
  return JSON.parse(child.stdout);
}

function regular(row) {
  return row.jet?.validity === 'regular' && !row.jet.is_bridge && Number.isFinite(row.jet.D);
}

function basis(matrix) {
  const [[a, b], [b2, d]] = matrix;
  assert.ok(Math.abs(b - b2) <= 1e-12 * Math.max(1, Math.abs(b)));
  const schur = d - b * b / a;
  assert.ok(a > 0 && schur > 0);
  const e = [[1 / Math.sqrt(a), 0], [-b / (a * Math.sqrt(schur)), 1 / Math.sqrt(schur)]];
  const inner = (v, w) => v[0] * (a * w[0] + b * w[1]) + v[1] * (b * w[0] + d * w[1]);
  const residual = Math.max(...e.flatMap((v, i) =>
    e.map((w, j) => Math.abs(inner(v, w) - Number(i === j)))));
  assert.ok(residual < 1e-12, `basis residual ${residual}`);
  return { columns: e, max_orthonormality_error: residual };
}

const calibration = calibrateJuliaSensitivity(basis);
if (process.argv[2] === '--calibrate') {
  console.log(JSON.stringify(calibration, null, 2));
  process.exit(0);
}
const geometry = queryGeometry(states.map(({ c }) => c), true);
const centerRows = states.map((state, i) => ({ ...state, geometry: geometry.points[i], metrics: [] }));
const endpoints = [];
for (const state of centerRows) {
  const g = state.geometry;
  if (!regular(g)) continue;
  state.abs_D_over_epsilon = Math.abs(g.jet.D) / geometry.profile.epsilon;
  for (const [name, matrix] of [['I', [[1, 0], [0, 1]]], ['G0', g.G0], ['G', g.G]]) {
    const e = basis(matrix);
    const steps = spacings.map(h => {
      const index = endpoints.length;
      for (const direction of e.columns) {
        for (const sign of [-1, 1]) {
          endpoints.push(state.c.map((coordinate, j) => coordinate + sign * h * direction[j]));
        }
      }
      return { h, index };
    });
    state.metrics.push({ name, matrix, basis: e, steps });
  }
}
console.log(`Checking ${endpoints.length} stencil endpoints with native ${geometry.provider}`);
const perturbationGeometry = queryGeometry(endpoints, false);
for (const state of centerRows) {
  for (const metric of state.metrics) {
    for (const step of metric.steps) {
      step.endpoints = perturbationGeometry.points.slice(step.index, step.index + 4)
        .map((entry, j) => ({ c: endpoints[step.index + j], geometry: entry }));
      delete step.index;
      const invalid = step.endpoints.some(entry => !regular(entry.geometry));
      const crossing = step.endpoints.some(entry =>
        entry.geometry.jet && entry.geometry.jet.D * state.geometry.jet.D <= 0);
      step.status = invalid ? 'geometry-invalid' : crossing ? 'crosses-sampled-shore' : 'ready';
    }
  }
}

const server = await createServer({
  root: frontend, configFile: false, appType: 'custom', server: { host: '127.0.0.1', port: 0 },
});
server.middlewares.use((request, response, next) => {
  if (request.url === '/__julia_sensitivity__.html') {
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><title>FractalSync frozen Julia sensitivity</title>');
  } else if (request.url === '/api/shader/julia.frag') {
    response.setHeader('Content-Type', 'text/plain');
    response.end(fragment);
  } else {
    next();
  }
});

let browser;
try {
  await server.listen();
  browser = await chromium.launch({
    headless: true,
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
  });
  const page = await browser.newPage({
    viewport: { width: 320, height: 240 }, deviceScaleFactor: 1,
  });
  await page.goto(`${server.resolvedUrls.local[0]}__julia_sensitivity__.html`);
  const result = await page.evaluate(async ({ states, resolution, presentation, paths, contours }) => {
    const { JuliaRenderer } = await import('/src/lib/juliaRenderer.ts');
    const { dot, sensitivitySpectrum } = await import('/scripts/julia_sensitivity_spectrum.mjs');
    const canvas = document.createElement('canvas');
    canvas.width = resolution;
    canvas.height = resolution;
    canvas.style.width = `${resolution}px`;
    canvas.style.height = `${resolution}px`;
    document.body.append(canvas);
    const renderer = new JuliaRenderer(canvas);
    await renderer.init();
    const gl = canvas.getContext('webgl');
    if (!gl) throw new Error('WebGL unavailable');
    gl.disable(gl.DITHER);
    const components = 3 * resolution ** 2;
    const render = c => {
      renderer.renderFrozen({ ...presentation, juliaSeed: { real: c[0], imag: c[1] } });
      const rgba = new Uint8Array(resolution ** 2 * 4);
      gl.readPixels(0, 0, resolution, resolution, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      if (gl.getError() !== gl.NO_ERROR) throw new Error('WebGL readback error');
      return rgba;
    };
    const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    const before = render(states[0].c);
    render([0.29, -0.04]);
    const renderAgain = render(states[0].c);
    if (!before.every((byte, j) => byte === renderAgain[j])) throw new Error('Frozen frame changed');

    const program = gl.getParameter(gl.CURRENT_PROGRAM);
    const uniforms = {};
    for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
      const uniform = gl.getActiveUniform(program, i);
      if (!uniform || uniform.name === 'u_juliaSeed') continue;
      const value = gl.getUniform(program, gl.getUniformLocation(program, uniform.name));
      uniforms[uniform.name] = ArrayBuffer.isView(value) ? Array.from(value) : value;
    }
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererInfo = {
      api: gl.getParameter(gl.VERSION),
      shading_language: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
      renderer: gl.getParameter(gl.RENDERER),
      unmasked_renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
      color_bits: [gl.RED_BITS, gl.GREEN_BITS, gl.BLUE_BITS].map(key => gl.getParameter(key)),
      context: gl.getContextAttributes(),
      dithering: gl.isEnabled(gl.DITHER),
      device_pixel_ratio: window.devicePixelRatio,
      uniforms,
    };
    const normalizedScale = 1 / (255 * Math.sqrt(components));
    const rgbDistance = (a, b) => {
      let squaredCodes = 0;
      for (let i = 0; i < a.length; i += 4) {
        for (let k = 0; k < 3; k++) squaredCodes += (a[i + k] - b[i + k]) ** 2;
      }
      return Math.sqrt(squaredCodes) * normalizedScale;
    };
    const renders = paths.map(path => path.frames.map(({ c }) => render(c)));
    const motion = paths.map((path, index) => {
      const pixels = renders[index];
      const reference = contours && path.profile === 'map_iso_scale' ? renders[index - 1] : null;
      const frames = path.frames.map((frame, i) => {
        if (reference && frame.t !== paths[index - 1].frames[i].t) {
          throw new Error('Contour replay frame times differ');
        }
        return { ...frame, rms_from_previous: rgbDistance(pixels[i], pixels[Math.max(i - 1, 0)]),
          rms_from_initial: rgbDistance(pixels[i], pixels[0]),
          rms_from_natural: reference ? rgbDistance(pixels[i], reference[i]) : undefined };
      });
      return { ...path, frames };
    });
    const derivative = (minus, plus, h) => {
      const direction = new Float64Array(components);
      let changed = 0;
      let squaredCodes = 0;
      for (let i = 0, j = 0; i < plus.length; i += 4) {
        for (let k = 0; k < 3; k++) {
          const delta = plus[i + k] - minus[i + k];
          changed += Number(delta !== 0);
          squaredCodes += delta * delta;
          direction[j++] = delta * normalizedScale / (2 * h);
        }
      }
      return { direction, changed_component_fraction: changed / components,
        rms_code_difference: Math.sqrt(squaredCodes / components) };
    };
    const relative = (a, b) => Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) || 1);
    for (const state of states) {
      state.baseline_rgba_sha256 = await hash(render(state.c));
      for (const metric of state.metrics) {
        let previous;
        for (const step of metric.steps) {
          if (step.status !== 'ready') {
            previous = undefined;
            continue;
          }
          step.gpu_seed_f32 = step.endpoints.map(({ c }) => c.map(Math.fround));
          if ([0, 2].some(k => step.gpu_seed_f32[k].every((v, j) => v === step.gpu_seed_f32[k + 1][j]))) {
            step.status = 'gpu-seed-collision';
            previous = undefined;
            continue;
          }
          step.gpu_relative_displacement_error = [0, 2].map(k => {
            const delta = step.endpoints[k + 1].c.map((v, j) => v - step.endpoints[k].c[j]);
            const gpuDelta = step.gpu_seed_f32[k + 1].map((v, j) => v - step.gpu_seed_f32[k][j]);
            return Math.hypot(...gpuDelta.map((v, j) => v - delta[j])) / Math.hypot(...delta);
          });
          if (Math.max(...step.gpu_relative_displacement_error) > 0.05) {
            step.status = 'gpu-seed-imprecise';
            previous = undefined;
            continue;
          }
          const pixels = step.endpoints.map(({ c }) => render(c));
          const pairs = [0, 2].map(k => derivative(pixels[k], pixels[k + 1], step.h));
          const [v, w] = pairs.map(pair => pair.direction);
          step.status = 'measured';
          const spectrum = sensitivitySpectrum([v, w]);
          Object.assign(step, spectrum);
          const [minor, major] = spectrum.eigenvalues;
          const [, axis] = spectrum.eigenvectors_intrinsic;
          step.condition_gram_raw = minor > 0 ? major / minor : null;
          step.condition_jacobian_raw = minor > 0 ? Math.sqrt(major / minor) : null;
          step.quantization_operator_budget = Math.SQRT2 / (510 * step.h);
          step.rank_above_rgb8_budget = step.eigenvalues.filter(value =>
            value > step.quantization_operator_budget ** 2).length;
          if (step.rank_above_rgb8_budget === 2) {
            const bound = step.quantization_operator_budget;
            step.jacobian_condition_rgb8_interval = [
              Math.max(1, (Math.sqrt(major) - bound) / (Math.sqrt(minor) + bound)),
              (Math.sqrt(major) + bound) / (Math.sqrt(minor) - bound),
            ];
          }
          step.pair_diagnostics = pairs.map(({ direction, ...diagnostics }) => diagnostics);
          if (previous) {
            const eigenShift = step.eigenvalues.map((value, j) => relative(value, previous.step.eigenvalues[j]));
            const axisAngle = Math.acos(Math.min(1, Math.abs(dot(axis, previous.axis)))) * 180 / Math.PI;
            const directionShift = [v, w].map((column, j) => {
              const prior = previous.columns[j];
              const squaredError = column.reduce((sum, value, k) => sum + (value - prior[k]) ** 2, 0);
              return Math.sqrt(squaredError / (Math.max(dot(column, column), dot(prior, prior)) || 1));
            });
            const split = major > 0 && (major - minor) / major >= 0.1;
            const stable = step.rank_above_rgb8_budget === 2 && previous.step.rank_above_rgb8_budget === 2
              && Math.max(...eigenShift, ...directionShift) <= 0.15
              && (!split || !previous.split || axisAngle <= 10);
            step.refinement = { previous_h: previous.step.h, eigenvalue_relative_shift: eigenShift,
              column_relative_shift: directionShift, major_axis_angle_deg: axisAngle,
              axis_separated_in_both: split && previous.split, stable };
          }
          previous = { step, axis, columns: [v, w], split: major > 0 && (major - minor) / major >= 0.1 };
        }
      }
    }
    return { renderer: rendererInfo, repeatable_baseline: true, states,
      [contours ? 'contour_motion' : 'attachment_motion']: motion };
  }, { states: centerRows, resolution, presentation, paths, contours });
  const provenance = {
    panel: heldOut ? 'held-out' : 'initial',
    chromium: browser.version(),
    shader: { path: shaderPath, sha256: createHash('sha256').update(fragment).digest('hex'),
      bailout_squared: 1e10, supersampling: false },
    resolution: [resolution, resolution],
    zoom_one_span: 2.7,
    presentation: { ...presentation, center: [0, 0], rotation: 0, time: 0 },
    representation: 'mean-normalized RGB8 framebuffer/1; ignore alpha; bottom-up WebGL order',
    spacings,
    [contours ? 'contour_trial_sha256' : 'attachment_fixture_sha256']:
      replayBytes ? createHash('sha256').update(replayBytes).digest('hex') : undefined,
    geometry: { ...geometry, points: undefined },
  };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify({ provenance, calibration, ...result }, null, 2) + '\n');
  for (const state of result.states) {
    if (!state.metrics.length) console.log(`${state.name}: excluded ${state.geometry.jet?.validity ?? state.geometry.error}`);
    for (const metric of state.metrics) {
      const last = [...metric.steps].reverse().find(step => step.status === 'measured');
      console.log(`${state.name}/${metric.name}: ${metric.steps.map(step => step.status).join(', ')}`,
        last ? { h: last.h, eigenvalues: last.eigenvalues, rank: last.rank_above_rgb8_budget,
          refinement: metric.steps.map(step => step.refinement?.stable ?? null) } : '');
    }
  }
  console.log(`Evidence written to ${output}`);
} finally {
  await browser?.close();
  await server.close();
}
