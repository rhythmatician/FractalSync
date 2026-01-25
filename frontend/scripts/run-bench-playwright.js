#!/usr/bin/env node
import { spawn } from 'child_process';
import fetch from 'node-fetch';
import { chromium } from 'playwright';

const FRONTEND_PORT = process.env.VITE_PORT || 5173;
const URL = `http://localhost:${FRONTEND_PORT}/`;

function startDev() {
    const proc = spawn('npm', ['run', 'dev'], { cwd: __dirname + '/../', shell: true, stdio: ['ignore', 'pipe', 'pipe'] });
    proc.stdout.on('data', (d) => process.stdout.write(`[vite] ${d}`));
    proc.stderr.on('data', (d) => process.stderr.write(`[vite] ${d}`));
    proc.on('exit', (c) => console.log('dev server exited', c));
    return proc;
}

async function waitForServer(url, timeoutMs = 20000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        try {
            const r = await fetch(url);
            if (r.ok) return true;
        } catch (e) {
            // ignore
        }
        await new Promise((r) => setTimeout(r, 200));
    }
    throw new Error('timed out waiting for server');
}

(async () => {
    const dev = startDev();
    try {
        console.log('waiting for dev server...');
        await waitForServer(URL, 20000);
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