#!/usr/bin/env node
// Run a command and forcibly terminate it if it exceeds the given timeout.
// Usage: node scripts/run-with-timeout.js --timeout 30 -- npm test --prefix frontend

const { spawn } = require('child_process');
const os = require('os');

function parseArgs() {
    const args = process.argv.slice(2);
    const res = { timeout: 30, cmd: [] };
    let i = 0;
    while (i < args.length) {
        if (args[i] === '--timeout' || args[i] === '-t') {
            res.timeout = Number(args[i + 1]);
            i += 2;
        } else if (args[i] === '--') {
            res.cmd = args.slice(i + 1);
            break;
        } else {
            // collect until '--'
            res.cmd.push(args[i]);
            i += 1;
        }
    }
    return res;
}

(async () => {
    const { timeout, cmd } = parseArgs();
    if (!cmd || cmd.length === 0) {
        console.error('Usage: run-with-timeout.js [--timeout seconds] -- <cmd> [args...]');
        process.exit(2);
    }

    // Use shell:true so platform PATH resolution (e.g., npm on Windows) works reliably
    const child = spawn(cmd[0], cmd.slice(1), { stdio: 'inherit', shell: true });

    let timedOut = false;
    const timer = setTimeout(() => {
        timedOut = true;
        console.error(`\nCommand exceeded timeout of ${timeout}s. Terminating (pid=${child.pid})...`);
        try {
            if (os.platform() === 'win32') {
                // taskkill to ensure child and its descendants are killed
                const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'inherit' });
                killer.on('exit', () => process.exit(124));
            } else {
                // POSIX: try SIGTERM then SIGKILL if necessary
                child.kill('SIGTERM');
                setTimeout(() => child.kill('SIGKILL'), 2000);
            }
        } catch (e) {
            console.error('Failed to terminate process:', e);
            process.exit(1);
        }
    }, timeout * 1000);

    child.on('exit', (code, signal) => {
        clearTimeout(timer);
        if (timedOut) {
            console.error(`Process terminated due to timeout (signal=${signal})`);
            process.exit(124);
        }
        if (signal) {
            console.error(`Process exited with signal: ${signal}`);
            process.exit(1);
        }
        process.exit(code);
    });

    child.on('error', (err) => {
        clearTimeout(timer);
        console.error('Failed to start child process:', err);
        process.exit(1);
    });
})();
