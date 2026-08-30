/**
 * PCM tap AudioWorkletProcessor (issue #91).
 *
 * Single responsibility: observe newly rendered PCM from the Web Audio
 * sample clock and forward non-overlapping blocks to the main thread. It
 * does NOT schedule feature hops, resample, or run inference — that
 * deterministic logic lives in `analysisTimebase.ts` on the main thread so
 * it stays unit-testable without a browser.
 *
 * Contract with the main thread (`audioWorkletTransport.ts`):
 *   - Each posted message is `{ samples, startFrame, sampleRate }`.
 *   - `samples` is a mono Float32Array (multi-channel input is downmixed by
 *     averaging, matching AnalyserNode's downmix semantics).
 *   - `startFrame` is the source-frame position of `samples[0]` on this
 *     processor's render clock, monotonically non-decreasing.
 *   - Blocks are batched to `BATCH_FRAMES` to cut postMessage overhead;
 *     correctness never depends on the batch size (the timebase re-slices).
 *
 * The Web Audio render quantum is fixed at 128 frames; `process()` is called
 * by the audio rendering thread, never by requestAnimationFrame or a timer.
 */

const RENDER_QUANTUM = 128;
// Batch ~8 quanta (~2.7 ms at 48 kHz) per message. Purely an overhead
// optimization — the canonical timebase is invariant to this value.
const BATCH_FRAMES = 1024;

class PcmTapProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._frame = 0; // source frames rendered since processor start
    this._buf = new Float32Array(BATCH_FRAMES);
    this._bufStart = 0; // source-frame position of _buf[0]
    this._bufLen = 0;
  }

  _flush() {
    if (this._bufLen === 0) return;
    // Copy out the filled region; transfer the underlying buffer to avoid GC.
    const out = this._buf.slice(0, this._bufLen);
    this.port.postMessage(
      {
        samples: out,
        startFrame: this._bufStart,
        sampleRate, // global in AudioWorkletGlobalScope
      },
      [out.buffer]
    );
    this._bufLen = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channels = input;
      const frames = channels[0].length;
      const numCh = channels.length;

      for (let i = 0; i < frames; i++) {
        // Downmix to mono by averaging channels (AnalyserNode semantics).
        let s = 0;
        for (let c = 0; c < numCh; c++) s += channels[c][i];
        s /= numCh;

        if (this._bufLen === 0) this._bufStart = this._frame + i;
        this._buf[this._bufLen++] = s;
        if (this._bufLen >= BATCH_FRAMES) this._flush();
      }
      this._frame += frames;
    } else {
      // No input connected: still advance the clock so startFrame stays
      // monotonic if a source is attached later. Emit nothing (no PCM).
      this._frame += RENDER_QUANTUM;
    }
    return true; // keep the node alive
  }
}

registerProcessor('pcm-tap', PcmTapProcessor);
