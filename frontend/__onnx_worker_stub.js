// Lightweight worker stub used in tests to satisfy onnxruntime-web worker creation in Node
// Works in both browser worker and Node worker_threads contexts.
try {
    // Node worker_threads
    const { parentPort } = require('worker_threads');
    if (parentPort) {
        parentPort.on('message', () => { });
    }
} catch (_) {
    // Browser worker environment
    try {
        self.onmessage = () => { };
    } catch (__e) {
        // Ignore if neither environment applies
    }
}
