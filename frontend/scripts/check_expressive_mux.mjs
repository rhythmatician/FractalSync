// Run with: node frontend/scripts/check_expressive_mux.mjs <ffmpeg-bin-directory>
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
const audioRoot = resolve(root, 'tmp/expressive-189');
const output = resolve(audioRoot, 'operator/mux-pilot');
const key = JSON.parse(await readFile(resolve(audioRoot, 'operator/key.json'), 'utf8'));
const shader = await readFile(resolve(root, 'shared/shaders/julia.frag'));
const sha256 = value => createHash('sha256').update(value).digest('hex');
const videoRate = 25;
const size = 96;
const samplesPerFrame = key.sample_rate / videoRate;
const frameCount = key.clip_samples / samplesPerFrame;
assert.equal(key.schema, 'expressive-stimuli/1');
assert.equal(key.sample_rate, 48000);
assert.equal(key.clip_samples, 491520);
assert.equal(key.hop_samples, 1024);
assert.ok(Number.isInteger(samplesPerFrame) && Number.isInteger(frameCount));

const selected = key.audio_assets.find(row => row.family === 1 && row.order === 'AAB' && row.fundamental_hz === 196);
const other = key.audio_assets.find(row => row.family === 1 && row.order === 'ABA' && row.fundamental_hz === 196);
assert.ok(selected && other);
const wav = resolve(audioRoot, 'playback', selected.file);
assert.equal(sha256(await readFile(wav)), selected.sha256);
function plan(starts) {
  const found = key.visual_timing_plans.find(row => JSON.stringify(row.starts_samples) === JSON.stringify(starts));
  assert.ok(found, 'timing plan missing from operator key');
  assert.equal(found.pulse_width_samples, 10 * key.hop_samples);
  return found;
}
const variants = [
  { label: 'pilot-matched', plan: plan(selected.accent_samples) },
  { label: 'pilot-other-order', plan: plan(other.accent_samples) },
  { label: 'pilot-delayed', plan: plan(selected.accent_samples.map(sample => sample + 8 * key.hop_samples)) },
];
const presentation = { zoom: 1, colorHue: 0.58, colorSat: 0.75, colorBright: 0.6, speed: 0 };
const anchor = [-0.17, 0.09];
const amount = 0.028;
const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];

function tool(name, args, options = {}) {
  const binary = process.argv[2] ? resolve(process.argv[2], name + (process.platform === 'win32' ? '.exe' : '')) : name;
  const result = spawnSync(binary, args, { cwd: root, maxBuffer: 16 * 1024 * 1024, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${name} exited ${result.status}: ${result.stderr.toString()}`);
  return result.stdout;
}
const ffmpeg = tool('ffmpeg', ['-version'], { encoding: 'utf8' }).split('\n')[0].trim();
const ffprobe = tool('ffprobe', ['-version'], { encoding: 'utf8' }).split('\n')[0].trim();
const sourcePcm = tool('ffmpeg', ['-v', 'error', '-i', wav, '-map', '0:a:0', '-f', 's16le', '-']);
assert.equal(sourcePcm.length, 2 * key.clip_samples);

const server = await createServer({
  root: frontend, configFile: false, appType: 'custom', server: { host: '127.0.0.1', port: 0 },
});
server.middlewares.use((request, response, next) => {
  if (request.url === '/__expressive_mux__.html') {
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><title>FractalSync frozen playback pilot</title>');
  } else if (request.url === '/api/shader/julia.frag') {
    response.setHeader('Content-Type', 'text/plain');
    response.end(shader);
  } else {
    next();
  }
});

function inspect(file) {
  const probe = (...args) => JSON.parse(tool('ffprobe', ['-v', 'error', ...args, '-of', 'json', file], { encoding: 'utf8' }));
  const streams = probe('-show_streams').streams;
  const video = streams.find(row => row.codec_type === 'video');
  const audio = streams.find(row => row.codec_type === 'audio');
  assert.ok(video && audio);
  for (const stream of [video, audio]) {
    assert.equal(stream.time_base, `1/${key.sample_rate}`);
    assert.equal(stream.start_pts, 0);
    assert.equal(stream.duration_ts, key.clip_samples);
  }
  const frames = probe('-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=pts').frames;
  assert.deepEqual(frames.map(frame => frame.pts), Array.from({ length: frameCount }, (_, i) => i * samplesPerFrame));
  const decoded = tool('ffmpeg', ['-v', 'error', '-i', file, '-map', '0:a:0', '-f', 's16le', '-']);
  assert.equal(decoded.length, sourcePcm.length);
  const audio_alignment_checks = [selected.accent_samples[0], selected.accent_samples[1], selected.accent_samples.at(-1)].map(start => {
    let peak;
    for (let lag = -128; lag <= 128; lag++) {
      let dot = 0;
      let sourceEnergy = 0;
      let decodedEnergy = 0;
      for (let t = start + 256; t < start + 3840; t++) {
        const a = sourcePcm.readInt16LE(2 * t);
        const b = decoded.readInt16LE(2 * (t + lag));
        dot += a * b;
        sourceEnergy += a * a;
        decodedEnergy += b * b;
      }
      const correlation = dot / Math.sqrt(sourceEnergy * decodedEnergy);
      if (!peak || correlation > peak.correlation) peak = { lag_samples: lag, correlation };
    }
    assert.equal(peak.lag_samples, 0);
    return { accent_sample: start, ...peak };
  });
  return { video, audio, video_pts_samples: frames.map(frame => frame.pts), decoded_audio_samples: decoded.length / 2,
    decoded_audio_pcm_sha256: sha256(decoded), audio_alignment_checks };
}

async function checkBrowserPlayback(page, bytes) {
  const playback = await page.evaluate(async base64 => {
    const video = document.createElement('video');
    video.muted = true;
    const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: 'video/mp4' }));
    const event = (name, subscribe) => new Promise((success, fail) => {
      const timer = setTimeout(() => fail(new Error(`Timed out at ${name}`)), 10000);
      subscribe(value => { clearTimeout(timer); success(value); });
    });
    try {
      document.body.append(video);
      const loaded = event('loadeddata', done => video.addEventListener('loadeddata', done, { once: true }));
      video.src = url;
      await loaded;
      const frame = event('first decoded frame', done => video.requestVideoFrameCallback((_, frame) => done(frame.mediaTime)));
      await video.play();
      return { duration: video.duration, width: video.videoWidth, height: video.videoHeight, media_time: await frame };
    } finally {
      video.pause();
      video.remove();
      URL.revokeObjectURL(url);
    }
  }, bytes.toString('base64'));
  assert.equal(playback.duration, key.clip_samples / key.sample_rate);
  assert.equal(playback.width, size);
  assert.equal(playback.height, size);
  assert.equal(playback.media_time, 0);
  return playback;
}

let browser;
try {
  await mkdir(output, { recursive: true });
  await server.listen();
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 320, height: 240 }, deviceScaleFactor: 1 });
  await page.goto(`${server.resolvedUrls.local[0]}__expressive_mux__.html`);
  await page.evaluate(async ({ size }) => {
    const { JuliaRenderer } = await import('/src/lib/juliaRenderer.ts');
    const canvas = document.createElement('canvas');
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    document.body.append(canvas);
    const renderer = new JuliaRenderer(canvas);
    await renderer.init();
    const gl = canvas.getContext('webgl');
    if (!gl) throw new Error('WebGL unavailable');
    gl.disable(gl.DITHER);
    window.renderPilot = params => {
      renderer.renderFrozen(params);
      if (canvas.width !== size || canvas.height !== size) throw new Error('Unexpected frame dimensions');
      const rgba = new Uint8Array(size * size * 4);
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      if (gl.getError() !== gl.NO_ERROR) throw new Error('Julia readback failed');
      return rgba;
    };
    window.pilotProvenance = () => {
      const program = gl.getParameter(gl.CURRENT_PROGRAM);
      const uniforms = {};
      for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
        const uniform = gl.getActiveUniform(program, i);
        if (uniform.name === 'u_juliaSeed') continue;
        const value = gl.getUniform(program, gl.getUniformLocation(program, uniform.name));
        uniforms[uniform.name] = ArrayBuffer.isView(value) ? Array.from(value) : value;
      }
      const debug = gl.getExtension('WEBGL_debug_renderer_info');
      return { uniforms, api: gl.getParameter(gl.VERSION), dithering: gl.isEnabled(gl.DITHER),
        context: gl.getContextAttributes(), renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null };
    };
  }, { size });

  const results = [];
  let renderer;
  for (const { label, plan: visual } of variants) {
    console.log(`Rendering ${label} (${frameCount} frames)`);
    const frames = Array.from({ length: frameCount }, (_, i) => {
      const sample = i * samplesPerFrame;
      const pulse = visual.starts_samples.findIndex(start => sample >= start && sample < start + visual.pulse_width_samples);
      const offset = pulse < 0 ? 0 : amount * Math.sin(Math.PI *
        (sample - visual.starts_samples[pulse]) / visual.pulse_width_samples) ** 2;
      const direction = pulse < 0 ? [0, 0] : directions[pulse % directions.length];
      return { sample, pulse: pulse < 0 ? null : pulse, c: anchor.map((v, j) => v + offset * direction[j]) };
    });
    const captured = await page.evaluate(({ frames, presentation, size }) => {
      const count = size * size;
      const params = c => ({ ...presentation, juliaSeed: { real: c[0], imag: c[1] } });
      const reference = window.renderPilot(params(frames[0].c));
      const raw = new Uint8Array(frames.length * count * 3);
      let previous;
      let lumaChange = 0;
      const luma = (r, g, b) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      for (let k = 0; k < frames.length; k++) {
        const rgba = window.renderPilot(params(frames[k].c));
        for (let p = 0; p < count; p++) {
          const i = 4 * p;
          const j = 3 * (count * k + p);
          raw[j] = rgba[i]; raw[j + 1] = rgba[i + 1]; raw[j + 2] = rgba[i + 2];
          if (previous) lumaChange += Math.abs(luma(rgba[i], rgba[i + 1], rgba[i + 2]) -
            luma(previous[i], previous[i + 1], previous[i + 2]));
        }
        previous = rgba;
      }
      const repeat = window.renderPilot(params(frames[0].c));
      if (!reference.every((byte, idx) => byte === repeat[idx])) throw new Error('Frozen repeat frame changed');
      let binary = '';
      for (let i = 0; i < raw.length; i += 8192) binary += String.fromCharCode(...raw.subarray(i, i + 8192));
      return { rgb: btoa(binary), renderer: window.pilotProvenance(),
        mean_absolute_interframe_luma: lumaChange / (count * (frames.length - 1)) };
    }, { frames, presentation, size });
    if (renderer) assert.deepEqual(captured.renderer, renderer);
    renderer = captured.renderer;
    const rgb = Buffer.from(captured.rgb, 'base64');
    const rawPath = resolve(output, `${label}.rgb`);
    const mp4Path = resolve(output, `${label}.mp4`);
    assert.equal(rgb.length, frameCount * size * size * 3);
    await writeFile(rawPath, rgb);
    tool('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', `${size}x${size}`,
      '-framerate', `${videoRate}`, '-i', rawPath, '-i', wav, '-map', '0:v:0', '-map', '1:a:0', '-vf', 'vflip',
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-bf', '0', '-threads:v', '1',
      '-c:a', 'aac', '-b:a', '96k', '-ar', `${key.sample_rate}`,
      '-video_track_timescale', `${key.sample_rate}`, '-movflags', '+faststart', mp4Path]);
    const mux = inspect(mp4Path);
    const mp4 = await readFile(mp4Path);
    const playback = await checkBrowserPlayback(page, mp4);
    const increments = frames.slice(1).map(({ c }, i) => Math.hypot(c[0] - frames[i].c[0], c[1] - frames[i].c[1]));
    results.push({ label, plan_id: visual.id, visual_starts: visual.starts_samples,
      pulse_width_samples: visual.pulse_width_samples, frames,
      coordinate_travel: increments.reduce((sum, delta) => sum + delta, 0),
      coordinate_rms_speed: videoRate * Math.sqrt(increments.reduce((sum, delta) => sum + delta ** 2, 0) / increments.length),
      active_frames: frames.filter(({ c }) => c.some((v, i) => v !== anchor[i])).length,
      mean_absolute_interframe_luma: captured.mean_absolute_interframe_luma,
      raw_rgb_sha256: sha256(rgb), mp4_sha256: sha256(mp4), mux, playback });
  }
  assert.equal(new Set(results.map(result => result.mux.decoded_audio_pcm_sha256)).size, 1);
  const report = { schema: 'expressive-frozen-mux-pilot/1', source_key_sha256: sha256(await readFile(resolve(audioRoot, 'operator/key.json'))),
    audio: { id: selected.id, wav_sha256: selected.sha256, source_pcm_sha256: sha256(sourcePcm),
      samples: key.clip_samples, sample_rate: key.sample_rate, accents: selected.accent_samples },
    seed_recipe: { anchor, peak_coordinate_offset: amount, directions, envelope: 'sin² and return to anchor' },
    shader_sha256: sha256(shader), chromium: browser.version(), ffmpeg, ffprobe, presentation, renderer,
    frame_rate: videoRate, frame_count: frameCount, size: [size, size], results };
  await writeFile(resolve(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Verified ${results.length} sample-aligned pilot MP4s in ${output}`);
} finally {
  await browser?.close();
  await server.close();
}
