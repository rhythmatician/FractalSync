#!/usr/bin/env node
// Cross-platform test runner that sets TEST_NO_NETWORK for the process (ESM)
import { spawn } from 'child_process';

process.env.TEST_NO_NETWORK = '1';
const cmd = process.platform === 'win32' ? 'npx.cmd' : 'npx';
const args = ['vitest', 'run', '-c', './frontend/vitest.fast.config.ts'];

const p = spawn(cmd, args, { stdio: 'inherit', env: process.env, shell: true });
p.on('exit', (code) => {
    process.exit(code);
});
