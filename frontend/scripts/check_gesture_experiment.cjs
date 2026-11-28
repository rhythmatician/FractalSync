const { chromium } = require('playwright');

function modulatedWav() {
  const rate = 44100;
  const samples = rate * 22;
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write('RIFF');
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24);
  wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) {
    const t = 2 * Math.PI * i / rate;
    wav.writeInt16LE(Math.round(32767 * (0.55 + 0.4 * Math.cos(1.5 * t)) * Math.sin(330 * t)), 44 + i * 2);
  }
  return wav;
}

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${process.env.GESTURE_BASE_URL || 'http://127.0.0.1:3000'}/gestureExperiment.html`);
  await page.getByRole('button', { name: 'Start passage' }).waitFor({ state: 'visible', timeout: 30000 });
  await page.waitForFunction(() => !document.querySelector('button')?.disabled, { timeout: 30000 });
  const binding = await page.evaluate(async () => {
    const wasm = await import('/wasm/orbit_synth_wasm.js');
    await wasm.default();
    const matched = wasm.knownGesturePhases(1, false);
    const misaligned = wasm.knownGesturePhases(1, true);
    const point = wasm.twoSourceGestureTarget(-0.7269, 0.1889, matched[0], matched[1], 1, 1, 0.12, true);
    return { difference: Math.sin(matched[1] - misaligned[1]), point: Array.from(point) };
  });
  if (Math.abs(binding.difference) < 0.5 || !binding.point.every(Number.isFinite)) throw new Error(`Unexpected Wasm result: ${JSON.stringify(binding)}`);
  await page.getByRole('button', { name: 'Start passage' }).click();
  await page.getByRole('button', { name: 'Stop passage' }).waitFor({ timeout: 10000 });
  await page.waitForTimeout(2500);
  await page.getByLabel('Gain 2', { exact: true }).fill('0.8');
  await page.getByLabel('Geometry').selectOption('distinct');
  await page.getByLabel('Comparison').selectOption('misaligned');
  await page.waitForTimeout(500);
  await page.getByLabel('Comparison').selectOption('audio');
  await page.waitForTimeout(500);
  if (process.env.GESTURE_SCREENSHOT) await page.screenshot({ path: process.env.GESTURE_SCREENSHOT });
  const alert = await page.getByRole('alert').allTextContents();
  const text = await page.locator('main').innerText();
  if (alert.length || errors.length || /Passage: 0\.0 s/.test(text)) throw new Error(JSON.stringify({ alert, errors, text }));
  await page.getByRole('button', { name: 'Stop passage' }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export diagnostic record' }).click();
  const download = await downloadPromise;
  const record = JSON.parse(await require('node:fs/promises').readFile(await download.path(), 'utf8'));
  if (record.format !== 'fractal-sync-gesture-diagnostic/1' || !record.samples.length ||
      record.droppedEvents !== 0 || record.droppedSamples !== 0 ||
      !record.events.some(event => event.type === 'settings') || !record.events.some(event => event.type === 'stop')) {
    throw new Error('Incomplete diagnostic record');
  }
  console.log('Gesture page advanced, stopped, and exported its diagnostic record.');
  await page.setViewportSize({ width: 560, height: 350 });
  await page.getByLabel('Comparison').selectOption('matched');
  await page.getByLabel('Render Physics motion').check();
  await page.getByRole('button', { name: 'Start passage' }).click();
  await page.getByRole('button', { name: 'Stop passage' }).or(page.getByRole('alert')).first().waitFor();
  await page.getByRole('button', { name: 'Free navigation' }).click();
  await page.getByRole('button', { name: 'Up', exact: true }).click();
  await page.waitForTimeout(350);
  await page.getByRole('button', { name: 'Resume gesture tracking' }).click();
  const duration = process.env.GESTURE_FULL_PASSAGE ? 25 : 2;
  await page.waitForFunction(seconds => document.querySelector('[role=alert]') ||
    Number(document.querySelector('main span')?.textContent?.match(/([\d.]+) s/)?.[1]) >= seconds,
  duration, { timeout: (duration + 20) * 1000 }).catch(async error => {
    throw new Error(`${error.message}\n${await page.locator('main').innerText()}`);
  });
  if (await page.getByRole('alert').count()) throw new Error(await page.getByRole('alert').innerText());
  await page.getByRole('button', { name: 'Stop passage' }).click();
  const physicsDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export diagnostic record' }).click();
  const physics = JSON.parse(await require('node:fs/promises').readFile(await (await physicsDownload).path(), 'utf8'));
  const resumed = physics.samples.findIndex((sample, index) =>
    index > 0 && sample.physics.reference && physics.samples[index - 1].physics.reference === null);
  if ((physics.samples.at(-1)?.stepTime ?? 0) < duration - 0.1 ||
      !physics.samples.every(sample => Number.isFinite(sample.physics.energy)) ||
      !physics.samples.some(sample => sample.physics.position[0] !== sample.target[0]) || !physics.frames.length ||
      resumed < 1 || JSON.stringify(physics.samples[resumed].physics.start_velocity) !==
        JSON.stringify(physics.samples[resumed - 1].physics.velocity)) {
    throw new Error('Expected realized motion and energy through the requested duration');
  }
  await page.getByRole('button', { name: 'Replay Julia motion' }).click();
  await page.getByRole('button', { name: 'Stop replay' }).waitFor();
  const frameDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save Julia frame' }).click();
  const frame = await require('node:fs/promises').readFile(await (await frameDownload).path());
  if (frame.toString('hex', 0, 8) !== '89504e470d0a1a0a') throw new Error('Expected a PNG frame');
  console.log(`Physics preview recorded ${duration} seconds of realized steps, replayed the visuals, and exported a PNG.`);
  if (process.env.GESTURE_CYCLE_BANK) {
    await page.getByRole('button', { name: 'Replay Julia motion' }).waitFor();
    await page.getByLabel('Render Physics motion').uncheck();
    await page.getByLabel('Estimate fixture envelopes with CycleBank').check();
    await page.getByRole('button', { name: 'Start passage' }).click();
    const role = page.locator('[aria-label="Role 1 binding"]');
    const choice = role.locator('..').getByRole('button', { name: /^Bind \d+ \(1\.[3456]\d Hz/ }).first();
    await choice.click({ timeout: 25000 });
    await role.filter({ hasText: /\d+/ }).waitFor({ timeout: 5000 }).catch(async error => {
      throw new Error(`${error.message}\n${await page.locator('main').innerText()}`);
    });
    await page.getByRole('button', { name: 'Withdraw' }).first().click();
    await role.filter({ hasText: 'unbound' }).waitFor();
    await page.getByRole('button', { name: 'Stop passage' }).click();
    const exportPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export diagnostic record' }).click();
    const estimated = JSON.parse(await require('node:fs/promises').readFile(await (await exportPromise).path(), 'utf8'));
    const actions = estimated.events.filter(event => event.type === 'binding');
    if (!actions.some(event => Number.isInteger(event.id)) ||
        !actions.some(event => event.id === undefined) ||
        !estimated.samples.some(sample => sample.cycle?.modes.length)) {
      throw new Error('Expected observed modes, explicit binding and withdrawal');
    }
    console.log('CycleBank observed fixture modes and recorded explicit binding and withdrawal.');
  }
  if (process.env.GESTURE_AUDIO_FILE) {
    await page.getByRole('button', { name: 'Replay Julia motion' }).waitFor();
    await page.getByLabel('Render Physics motion').uncheck();
    await page.getByLabel('Local audio file').setInputFiles({
      name: 'pulse.wav', mimeType: 'audio/wav', buffer: modulatedWav(),
    });
    await page.getByRole('button', { name: 'Start passage' }).click();
    const role = page.locator('[aria-label="Role 1 binding"]');
    await role.locator('..').getByRole('button', { name: /^Bind \d+ \(1\.[3456]\d Hz/ }).first().click({ timeout: 24000 });
    await role.filter({ hasText: /\d+/ }).waitFor({ timeout: 4000 });
    await page.getByRole('button', { name: 'Stop passage' }).click();
    console.log('Decoded WAV playback acquired and bound the 1.5 Hz modulation.');
  }
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
