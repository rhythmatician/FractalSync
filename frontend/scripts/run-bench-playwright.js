#!/usr/bin/env node
import { spawn } from 'child_process';
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';

// ESM-safe __dirname
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Use global fetch if available (Node 18+), otherwise dynamically import node-fetch
async function getFetch() {
    if (typeof globalThis.fetch === 'function') return globalThis.fetch.bind(globalThis);
    const mod = await import('node-fetch');
    return mod.default;
}

const FRONTEND_PORT = process.env.VITE_PORT || 5173;
const URL = `http://localhost:${FRONTEND_PORT}/`;

function startDev() {
    // Force a known port so the test can reliably connect. Pass through additional args after "--".
    const proc = spawn('npm', ['run', 'dev', '--', '--port', String(FRONTEND_PORT)], { cwd: __dirname + '/../', shell: true, stdio: ['pipe', 'pipe', 'pipe'] });
    // Create a promise that resolves when we see Vite 'ready' or 'Local' output
    const ready = new Promise((resolve) => {
        const onData = (d) => {
            const s = d.toString();
            process.stdout.write(`[vite] ${s}`);
            if (s.includes('ready') || s.includes('Local:')) {
                resolve(true);
            }
        };
        proc.stdout.on('data', onData);
        proc.stderr.on('data', onData);
        proc.on('exit', () => { /* noop */ });
    });
    return { proc, ready };
}

async function waitForServer(url, timeoutMs = 20000) {
    const fetchFn = await getFetch();
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        try {
            const r = await fetchFn(url);
        } catch (e) {
            // ignore
        }
        await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error('timed out waiting for server');
}

(async () => {
    const { proc: dev, ready } = startDev();
    try {
        console.log('waiting for dev server...');
        // Prefer the ready signal from the process output, fallback to health check
        await Promise.race([ready, waitForServer(URL, 20000)]);
        console.log('server ready, launching browser');

        const browser = await chromium.launch({ headless: true });
        const context = await browser.newContext();
        const page = await context.newPage();
        await page.goto(URL);

        // Click the nav button to open the bench view
        await page.locator('button', { hasText: 'wasm worker bench' }).click();

        // Click the Run benchmark button
        await page.locator('button', { hasText: 'Run benchmark' }).click();

        // Wait for the result pre to contain 'avg=' text
        const pre = page.locator('pre');
        await pre.waitFor({ timeout: 20000 });
        const txt = await pre.textContent();

        console.log('BENCH RESULT:', txt);

        await browser.close();
    } catch (err) {
        console.error('bench failed', err);
        dev.kill();
        process.exit(1);
    }
    dev.kill();
    process.exit(0);
})();