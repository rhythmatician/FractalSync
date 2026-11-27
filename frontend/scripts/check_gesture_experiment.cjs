const { chromium } = require('playwright');

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
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
