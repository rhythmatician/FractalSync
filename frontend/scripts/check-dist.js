#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const distDir = path.resolve(__dirname, '..', 'dist');
const indexFile = path.join(distDir, 'index.html');

console.log('Checking frontend build output...');
if (!fs.existsSync(distDir)) {
    console.error(`ERROR: dist directory not found at ${distDir}`);
    process.exit(1);
}
if (!fs.existsSync(indexFile)) {
    console.error(`ERROR: index.html not found in dist`);
    process.exit(1);
}
console.log('OK: build artifacts present.');
process.exit(0);
