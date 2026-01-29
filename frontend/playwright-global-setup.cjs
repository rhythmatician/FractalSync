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
        stdio: ['ignore', 'inherit', 'inherit']
    });

    fs.writeFileSync(pidFile, String(proc.pid));

    try {
        await waitForUrl(backendUrl, 60000);
        console.log('Backend is up');
    } catch (err) {
        throw new Error('Backend did not become ready in time: ' + String(err));
    }

    // Ensure a model exists. If /api/model/latest returns 200, we're done. If 404, train a tiny model and export it.
    const modelExists = await new Promise((resolve) => {
        http.get(backendUrl, (res) => {
            resolve(res.statusCode === 200);
        }).on('error', () => resolve(false));
    });

    if (modelExists) {
        console.log('Model already present.');
        return;
    }

    console.log('No model found; training a tiny model (1 epoch) for E2E tests...');

    // Prefer a committed ONNX model if present to avoid training in CI
    const committedModel = path.join(__dirname, '..', '..', 'models_i_like', 'model_orbit_control_20260128_002625.onnx');
    if (fs.existsSync(committedModel)) {
        const dest = path.join(backendCwd, 'checkpoints', 'model.onnx');
        console.log('Committed ONNX model found; copying to backend checkpoints:', committedModel);
        fs.mkdirSync(path.join(backendCwd, 'checkpoints'), { recursive: true });
        fs.copyFileSync(committedModel, dest);
        // Wait briefly for API to pick up model
        try {
            await waitForUrl(backendUrl, 20000);
            console.log('Copied committed model and API serves it.');
            return;
        } catch (err) {
            console.warn('Committed model copied but API not responding yet:', err);
            // continue with training fallback
        }
    }

    // Prefer committed audio fixture if present; otherwise generate a small audio file for training
    const fixturePath = path.join(__dirname, 'tests', 'fixtures', 'sample.wav');
    const audioDir = path.join(backendCwd, 'data', 'audio');
    fs.mkdirSync(audioDir, { recursive: true });
    const wavPath = path.join(audioDir, 'e2e_sample.wav');

    if (fs.existsSync(fixturePath)) {
        console.log('Copying committed sample fixture to backend audio folder');
        fs.copyFileSync(fixturePath, wavPath);
    } else {
        console.log('Fixture not found; generating sample WAV for backend training');
        // Generate a short WAV with silence (mono, 48kHz to match runtime-core SAMPLE_RATE)
        const durationSec = 0.5;
        const sampleRate = 48000;
        const numChannels = 1;
        const bytesPerSample = 2;
        const numSamples = Math.floor(durationSec * sampleRate);
        const byteRate = sampleRate * numChannels * bytesPerSample;
        const blockAlign = numChannels * bytesPerSample;
        const dataSize = numSamples * numChannels * bytesPerSample;
        const buffer = Buffer.alloc(44 + dataSize);
        buffer.write('RIFF', 0);
        buffer.writeUInt32LE(36 + dataSize, 4);
        buffer.write('WAVE', 8);
        buffer.write('fmt ', 12);
        buffer.writeUInt32LE(16, 16);
        buffer.writeUInt16LE(1, 20);
        buffer.writeUInt16LE(numChannels, 22);
        buffer.writeUInt32LE(sampleRate, 24);
        buffer.writeUInt32LE(byteRate, 28);
        buffer.writeUInt16LE(blockAlign, 32);
        buffer.writeUInt16LE(bytesPerSample * 8, 34);
        buffer.write('data', 36);
        buffer.writeUInt32LE(dataSize, 40);
        fs.writeFileSync(wavPath, buffer);
    }

    // Run a quick training: 1 epoch, limited files
    await new Promise((resolve, reject) => {
        const timeoutMs = 5 * 60 * 1000; // 5 minutes safety timeout
        let stderrData = '';
        let settled = false;
        const tproc = spawn('python', ['train.py', '--data-dir', 'data/audio', '--epochs', '1', '--max-files', '1'], {
            cwd: backendCwd,
            stdio: ['ignore', 'inherit', 'pipe'] // capture stderr for inspection
        });
        const cleanup = () => {
            if (timeoutId !== null) {
                clearTimeout(timeoutId);
            }
        };
        let timeoutId = setTimeout(() => {
            if (settled) {
                return;
            }
            settled = true;
            try {
                tproc.kill('SIGKILL');
            } catch (e) {
                // ignore kill errors
            }
            cleanup();
            reject(new Error('train.py did not complete within ' + timeoutMs + 'ms and was terminated'));
        }, timeoutMs);

        if (tproc.stderr) {
            tproc.stderr.on('data', (chunk) => {
                stderrData += chunk.toString();
            });
        }

        tproc.on('exit', (code) => {
            if (settled) {
                return;
            }
            settled = true;
            cleanup();
            if (code === 0) {
                resolve();
            } else {
                reject(new Error('train.py failed with code ' + code + (stderrData ? ' and stderr:\n' + stderrData : '')));
            }
        });

        tproc.on('error', (err) => {
            if (settled) {
                return;
            }
            settled = true;
            cleanup();
            reject(err);
        });
    });

    // Export checkpoint to ONNX using utils.checkpoint_loader
    // Note: checkpoint_loader infers metadata (window-frames, k-bands, delta flags) from the checkpoint.
    // If checkpoint metadata is incomplete, it will use default values which may not match the trained model.
    const checkpointPath = path.join(backendCwd, 'checkpoints', 'checkpoint_epoch_1.pt');
    const outputOnnx = path.join(backendCwd, 'checkpoints', 'model.onnx');
    await new Promise((resolve, reject) => {
        const eproc = spawn('python', ['-m', 'utils.checkpoint_loader', '--checkpoint', checkpointPath, '--output', outputOnnx], {
            cwd: backendCwd,
            stdio: ['ignore', 'inherit', 'inherit']
        });
        eproc.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('checkpoint export failed with code ' + code))));
        eproc.on('error', (err) => reject(err));
    });

    // Confirm model is now available via API
    try {
        await waitForUrl(backendUrl, 20000);
        console.log('Model exported and API reflects it.');
    } catch (err) {
        throw new Error('Model export completed but API did not serve it: ' + String(err));
    }
};
