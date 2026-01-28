const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

const pidFile = path.join(__dirname, '.playwright_backend.pid');
const backendUrl = 'http://localhost:8000/api/model/latest';

function waitForUrl(url, timeout = 60000, interval = 500) {
    const start = Date.now();
    return new Promise((resolve, reject) => {
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

module.exports = async function globalSetup() {
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
        await waitForUrl(backendUrl, 60000);
        console.log('Backend is up');
    } catch (err) {
        throw new Error('Backend did not become ready in time: ' + String(err));
    }
};
