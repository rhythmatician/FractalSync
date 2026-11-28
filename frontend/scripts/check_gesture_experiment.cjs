const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:3000/gestureExperiment.html');
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
      !record.events.some(event => event.type === 'settings') || !record.events.some(event => event.type === 'stop')) {
    throw new Error('Incomplete diagnostic record');
  }
  console.log('Gesture page advanced, stopped, and exported its diagnostic record.');
  await page.setViewportSize({ width: 700, height: 450 });
  await page.getByLabel('Comparison').selectOption('matched');
  await page.getByLabel('Render Physics motion').check();
  await page.getByRole('button', { name: 'Start passage' }).click();
  await page.getByRole('button', { name: 'Stop passage' }).or(page.getByRole('alert')).first().waitFor();
  const duration = process.env.GESTURE_FULL_PASSAGE ? 25 : 2;
  await page.waitForFunction(seconds => document.querySelector('[role=alert]') ||
    Number(document.querySelector('main span')?.textContent?.match(/([\d.]+) s/)?.[1]) >= seconds,
  duration, { timeout: (duration + 15) * 1000 });
  if (await page.getByRole('alert').count()) throw new Error(await page.getByRole('alert').innerText());
  await page.getByRole('button', { name: 'Stop passage' }).click();
  const physicsDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export diagnostic record' }).click();
  const physics = JSON.parse(await require('node:fs/promises').readFile(await (await physicsDownload).path(), 'utf8'));
  if (physics.samples.at(-1)?.stepTime < duration - 0.1 || !physics.samples.every(sample => Number.isFinite(sample.physics.energy)) ||
      !physics.samples.some(sample => sample.physics.position[0] !== sample.target[0]) || !physics.frames.length) {
    throw new Error('Expected realized motion and energy through the requested duration');
  }
  await page.getByRole('button', { name: 'Replay Julia motion' }).click();
  await page.getByRole('button', { name: 'Stop replay' }).waitFor();
  const frameDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save Julia frame' }).click();
  const frame = await require('node:fs/promises').readFile(await (await frameDownload).path());
  if (frame.toString('hex', 0, 8) !== '89504e470d0a1a0a') throw new Error('Expected a PNG frame');
  console.log(`Physics preview recorded ${duration} seconds of realized steps, replayed the visuals, and exported a PNG.`);
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
