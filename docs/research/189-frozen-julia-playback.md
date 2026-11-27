# Sample-aligned frozen Julia playback pilot

Three technical playback clips now render the same original synthetic WAV against distinct Julia seed schedules. Each passes frame timestamp, decoded audio length, and Chromium startup checks. The clips test the export needed by [#189](https://github.com/rhythmatician/FractalSync/issues/189) and [#191](https://github.com/rhythmatician/FractalSync/issues/191). They use prescribed seed queries at known times. Physics and Controls replay remain separate from this pilot.

## Inputs and render

The [stimulus generator](../../formal/checks/prepare_expressive_stimuli.py) supplies the operator's sample offsets and original PCM. This run uses the family-one 196 Hz AAB recording, `m-7b1aef6f5513020c`, for every clip. Its WAV SHA-256 is `bb87299b61035c148399a12c4f46e21269cc311946846c31c251f1abbb543a6e`. The three visual schedules use the same-order accents, the other phrase order's accents, or the same-order accents delayed by 8192 samples. Each contains 12 prescribed excursions of width 10,240 samples. The offset from the anchor $-0.17+0.09i$ has a $\sin^2$ envelope and peak 0.028 in coordinate units. The four coordinate headings repeat in the same order in each phrase. Each excursion returns to the anchor.

The [pilot script](../../frontend/scripts/check_expressive_mux.mjs) renders synchronously through the official [JuliaRenderer](../../frontend/src/lib/juliaRenderer.ts#L354) at each sample offset $1920j$ for $j=0,\dots,255$. It calls `renderFrozen` with shader time zero and reads raw RGB pixels before muxing. The square viewport has 96×96 pixels, DPR 1, antialiasing and dithering off, fixed zoom 1, fixed hue 0.58, saturation 0.75, brightness 0.6, and a 500-step escape budget. The report records active uniforms, raw clip hashes, frame sample offsets and seed coordinates, audio identity, and both tool versions. The baseline frame repeated byte for byte after each clip's intervening renders. The run used Chromium `153.0.8010.12`, WebGL1 via ANGLE/Vulkan SwiftShader and shader SHA-256 `c446d0d42789a45ea928fd93ceed700b76f3afbe39d698c04c35d90265786e7b`.

| Visual schedule | Non-anchor frames | Measured coordinate travel | Coordinate RMS speed | Mean absolute interframe luma change |
| --- | ---: | ---: | ---: | ---: |
| Matched | 61 | 0.657423 | 0.138340 | 0.0007581 |
| Other phrase order | 63 | 0.651492 | 0.138438 | 0.0007484 |
| Delayed | 63 | 0.654833 | 0.138367 | 0.0007536 |

The travel and speed columns are derived from consecutive displayed seeds. The final column averages absolute differences of normalized RGB luma over pixels and neighboring frames before encoding. The displayed travel, RMS speed, and mean luma-change measures differ by less than 2% across these three schedules. That is an empirical coordinate and image-change comparison for one patch. The motions have no certified intrinsic speeds, shape invariance, or guarantee of equal apparent activity.

## Mux and timing checks

FFmpeg writes H.264 video and mono AAC audio. The sound is encoded from the same WAV separately for each clip. FFprobe verifies that both streams start at timestamp zero, use timebase `1/48000`, and have exclusive end timestamp `491520`, exactly 10.24 s. The video has 256 decoded frames with presentation times `0, 1920, …, 489600` in audio-sample units. Decoding AAC to 16-bit PCM yields exactly 491,520 samples, and all three decoded audio hashes agree. The AAC output differs from the source PCM because the codec is lossy.

The normalized signed correlation of AAC-decoded PCM against the source peaks at lag zero over the searched integer lags from −128 through +128 samples at three sampled accents. The correlations at those peaks were 0.999768, 0.999998, and 0.999999. This check uses 3584 samples per window, starting 256 samples after the authored accent. Chromium reports 10.24 s and decodes the first 96×96 video frame at media time zero in every muted startup test. The timestamps and decode checks concern the file and browser pipeline. They leave speaker/display latency and actual listener-device delivery unmeasured.

## Reproduce

Generate the original audio and operator key with `formal/checks/prepare_expressive_stimuli.py`. Use a directory containing `ffmpeg` and `ffprobe`, or omit the argument if both are on `PATH`. This run verified and unpacked the publisher-linked Gyan FFmpeg 9.1 Essentials archive under ignored `tmp`. Its SHA-256 was `835b9a733445ba7a56a37a26cf7bde84e2f3ce7da2a8ec984276bb2abb1b922c`. The checkout needs the existing Vite, Playwright, and Chromium setup. [FFmpeg download references](https://ffmpeg.org/download.html), [Gyan Essentials builds](https://www.gyan.dev/ffmpeg/builds/).

```powershell
rtk proxy .venv/Scripts/python.exe formal/checks/prepare_expressive_stimuli.py
rtk proxy node frontend/scripts/check_expressive_mux.mjs tmp/expressive-189/tools/ffmpeg-9.1-essentials_build/bin
```

The three labelled MP4s, raw RGB, and `report.json` go to ignored `tmp/expressive-189/operator/mux-pilot/`. Those filenames reveal condition identity. A scored pilot still needs the blinded media pack for the full schedule, Controls/Physics replay with valid geometry and achieved intrinsic activity, calibrated playback, and held-out human trials. Keep both issues open for that evidence. [Experimental protocol](189-expressive-dynamics-protocol.md).
