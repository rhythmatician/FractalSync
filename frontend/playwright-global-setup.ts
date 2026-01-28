import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import http from 'http';

const pidFile = path.join(__dirname, '.playwright_backend.pid');
const backendUrl = 'http://localhost:8000/api/model/latest';

function waitForUrl(url: string, timeout = 60_000, interval = 500) {
  const start = Date.now();
  return new Promise<void>((resolve, reject) => {
    (function check() {
      http.get(url, (res) => {
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 500) {
          resolve();
        } else if (Date.now() - start > timeout) {
          reject(new Error('Timeout waiting for ' + url));
        } else {
          setTimeout(check, interval);
        }
      }).on('error', () => {
        if (Date.now() - start > timeout) {
          reject(new Error('Timeout waiting for ' + url));
        } else {
          setTimeout(check, interval);
        }
      });
    })();
  });
}

export default async function globalSetup() {
  // If backend is already running, don't spawn another
  try {
    await waitForUrl(backendUrl, 2000);
    console.log('Backend already responding, skipping spawn.');
    return;
  } catch (e) {
    // proceed to spawn
  }

  const backendCwd = path.resolve(__dirname, '../backend');
  console.log('Starting backend server from', backendCwd);
  const proc = spawn('python', ['-m', 'api.server'], {
    cwd: backendCwd,
    stdio: ['ignore', 'inherit', 'inherit'],
    detached: true
  });

  fs.writeFileSync(pidFile, String(proc.pid));

  try {
    await waitForUrl(backendUrl, 60_000);
    console.log('Backend is up');
  } catch (err) {
    throw new Error('Backend did not become ready in time: ' + String(err));
  }
}
