const fs = require('fs');
const path = require('path');
const pidFile = path.join(__dirname, '.playwright_backend.pid');

module.exports = async function globalTeardown() {
    try {
        if (fs.existsSync(pidFile)) {
            const pid = Number(fs.readFileSync(pidFile, 'utf-8'));
            try {
                process.kill(pid);
                console.log('Killed backend pid', pid);
            } catch (e) {
                console.warn('Failed to kill backend pid', pid, e);
            }
            fs.unlinkSync(pidFile);
        }
    } catch (e) {
        console.warn('Error in globalTeardown', e);
    }
};
