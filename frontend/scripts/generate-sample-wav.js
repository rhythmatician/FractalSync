const fs = require('fs');
const path = require('path');

// Generate a short silent WAV file in fixtures directory
const outDir = path.resolve(__dirname, '../../backend/data/audio');
fs.mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, 'sample.wav');

const durationSec = 0.5;
const sampleRate = 8000;
const numChannels = 1;
const bytesPerSample = 2;
const numSamples = Math.floor(durationSec * sampleRate);
const byteRate = sampleRate * numChannels * bytesPerSample;
const blockAlign = numChannels * bytesPerSample;
const dataSize = numSamples * numChannels * bytesPerSample;
const buffer = Buffer.alloc(44 + dataSize);
// WAV header
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

// Fill PCM 16-bit audio with a 440 Hz sine wave
const freq = 440;
const amplitude = 0.6 * 0x7fff; // 60% of max int16
for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    const sample = Math.round(amplitude * Math.sin(2 * Math.PI * freq * t));
    buffer.writeInt16LE(sample, 44 + i * 2);
}

fs.writeFileSync(outPath, buffer);
console.log('Generated', outPath, `(${dataSize} bytes data) with 440Hz tone`);
