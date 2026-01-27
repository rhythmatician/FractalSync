import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

(async () => {
    const url = process.argv[2] || 'http://localhost:3000';
    const outDir = path.resolve(process.cwd(), 'tmp');
    try { fs.mkdirSync(outDir, { recursive: true }); } catch { }

    const browser = await chromium.launch({ headless: false });
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await context.newPage();

    const logs = [];
    const network = [];

    page.on('console', msg => {
        const text = msg.text();
        logs.push({ type: 'console', level: msg.type(), text, timestamp: Date.now() });
        console.log('[PAGE CONSOLE]', msg.type(), text);
    });

    page.on('pageerror', err => {
        logs.push({ type: 'pageerror', text: String(err), timestamp: Date.now() });
        console.error('[PAGEERROR]', String(err));
    });

    page.on('requestfailed', req => {
        network.push({ type: 'requestfailed', url: req.url(), method: req.method(), timestamp: Date.now() });
        console.warn('[REQUESTFAILED]', req.url(), req.failure()?.errorText);
    });

    page.on('response', async res => {
        const url = res.url();
        const ok = res.ok();
        const status = res.status();
        network.push({ type: 'response', url, ok, status, timestamp: Date.now() });
        // Save body for small-ish relevant assets
        try {
            if (/ort-?wasm|model|onnx|ort/.test(url)) {
                const text = await res.text();
                const fname = path.join(outDir, 'resp-' + encodeURIComponent(url).slice(0, 200) + '.txt');
                fs.writeFileSync(fname, text.slice(0, 20000), 'utf8');
                network.push({ type: 'response_body_saved', url, file: fname });
            }
        } catch (e) {
            // ignore
        }
    });

    console.log('Opening', url);
    try {
        await page.goto(url, { waitUntil: 'networkidle' });
    } catch (e) {
        console.error('Failed to open page:', e);
    }

    // Wait for model loaded or some indicator for up to 15s
    try {
        await page.waitForSelector('text=✓ Model loaded successfully', { timeout: 15000 });
        console.log('Model appears loaded');
    } catch (e) {
        console.warn('Model ready indicator not found within 15s');
    }

    // Click Start if present
    try {
        const start = page.getByRole('button', { name: /Start/ });
        await start.click();
        console.log('Clicked Start');
    } catch (e) {
        console.warn('Start button not found or click failed:', e);
    }

    // Capture 12s of runtime activity
    await page.waitForTimeout(12000);

    // screenshot
    const screenshotPath = path.join(outDir, 'headed_capture.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log('Saved screenshot to', screenshotPath);

    // write logs
    const out = { capturedAt: Date.now(), logs, network };
    const outFile = path.join(outDir, 'headed_capture.json');
    fs.writeFileSync(outFile, JSON.stringify(out, null, 2));
    console.log('Saved logs to', outFile);

    // Inspect logs for the specific overlay error or failed ort fetch
    const overlayErr = logs.find(l => /Failed to load url .* ort-wasm/.test(l.text) || /overlay/i.test(l.text));
    const ortRespFail = network.find(n => (n.url && /ort-wasm/.test(n.url) && (!n.ok || n.status >= 400)));

    if (overlayErr || ortRespFail) {
        console.error('Detected issues:', { overlayErr: overlayErr?.text, ortRespFail });
        await browser.close();
        process.exit(2);
    }

    await browser.close();
    console.log('No immediate overlay/errors detected in capture window.');
    process.exit(0);
})();
