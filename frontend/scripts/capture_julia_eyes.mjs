import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

// Diagnostic capture with the same renderer, shader and presentation as the cockpit.
const folder = new URL('../../tmp/julia-eyes/', import.meta.url);
await mkdir(folder, { recursive: true });
const shader = await readFile('shared/shaders/julia.frag');
const server = await createServer({ root: 'frontend', configFile: false, appType: 'custom', server: { host: '127.0.0.1', port: 0 } });
server.middlewares.use((req, res, next) => {
  if (req.url === '/eye.html') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<!doctype html><canvas style="width:800px;height:600px"></canvas>');
  } else if (req.url === '/api/shader/julia.frag') {
    res.setHeader('Content-Type', 'text/plain');
    res.end(shader);
  } else {
    next();
  }
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 850, height: 650 }, deviceScaleFactor: 1 });
  await page.goto(server.resolvedUrls.local[0] + 'eye.html');
  await page.evaluate(async () => {
    const { JuliaRenderer } = await import('/src/lib/juliaRenderer.ts');
    const canvas = document.querySelector('canvas');
    const renderer = new JuliaRenderer(canvas);
    await renderer.init();
    window.readEyeFrame = (real, imag) => {
      renderer.renderFrozen({ juliaSeed: { real, imag }, zoom: 1, colorHue: .5, colorSat: .8, colorBright: .9, speed: 0 });
      return canvas.toDataURL().split(',')[1];
    };
  });
  const c = [-.7269, .1889];
  for (const [name, dx, dy] of [['base',0,0], ['repeat',0,0], ...[.00025,.0005,.001].flatMap(h => [
    [`xp-${h}`,h,0], [`xm-${h}`,-h,0], [`yp-${h}`,0,h], [`ym-${h}`,0,-h],
  ])]) {
    const data = await page.evaluate(([x,y]) => window.readEyeFrame(x,y), [c[0]+dx,c[1]+dy]);
    await writeFile(new URL(name + '.png', folder), Buffer.from(data, 'base64'));
  }
} finally {
  await browser?.close();
  await server.close();
}
