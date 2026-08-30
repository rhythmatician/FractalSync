//! Authoritative sample-clock audio timebase (issue #91).
//!
//! This module owns the deterministic transport/timing/scheduling logic for
//! the canonical 48 kHz analysis timeline. It is the single source of truth
//! for:
//!
//!   - exactly-once, in-order PCM ingestion (monotonic source position)
//!   - stateful streaming resampling to the canonical 48 kHz timeline
//!   - exact HOP_LENGTH (1024) canonical-sample hop scheduling
//!   - timestamps derived from integer sample position (never wall clock)
//!   - stream-epoch reset semantics on discontinuity
//!
//! Per ADR 0001, this math lives in `runtime-core` only. The browser
//! consumes it through the wasm bindings (`wasm-orbit`); there is no
//! TypeScript mirror of the resampler, scheduler, or sample accounting.
//!
//! The browser's AudioWorklet render quantum, batching strategy, main-thread
//! stalls, and render FPS must not change which samples are new or where
//! hops occur: the same source PCM produces the same canonical stream and
//! the same tick sample indices regardless of how it is chunked.

use crate::controller::{HOP_LENGTH, N_FFT, SAMPLE_RATE, WINDOW_FRAMES};
use crate::features::FeatureExtractor;

/// Canonical analysis timeline sample rate (runtime-core authority).
pub const CANONICAL_SAMPLE_RATE: usize = SAMPLE_RATE;
/// Canonical hop length in canonical samples (runtime-core authority).
pub const CANONICAL_HOP_LENGTH: usize = HOP_LENGTH;

/// Canonical samples feeding one tick's feature window: exactly enough to
/// cover n_fft + (WINDOW_FRAMES - 1) hops. Tick windows are anchored to the
/// hop boundary (their END is fixed at the hop sample and their length is a
/// multiple of the hop), so any slice of this length or longer yields the
/// SAME last window — this is what makes tick features invariant to block
/// cadence (chunk invariance, issue #93 production-path parity).
pub const TICK_WINDOW_SAMPLES: usize = N_FFT + (WINDOW_FRAMES - 1) * HOP_LENGTH;

/// Reason a stream reset occurred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetReason {
    Start,
    Stop,
    SourceReplacement,
    DetectedGap,
}

/// A timestamped analysis event — the seam the future CycleBank consumes.
///
/// `sample_index` is the authoritative identity (integer canonical sample
/// position); `time_seconds` / `dt_seconds` are derived conveniences.
#[derive(Clone, Debug)]
pub struct AnalysisTick {
    /// Flattened frame-major feature window from the Rust extractor.
    pub features: Vec<f64>,
    /// Canonical 48 kHz sample index of this tick's hop boundary.
    pub sample_index: u64,
    /// Derived convenience: sample_index / CANONICAL_SAMPLE_RATE.
    pub time_seconds: f64,
    /// Derived convenience: canonical hop duration in seconds.
    pub dt_seconds: f64,
    /// Increments on every reset/discontinuity so consumers detect restarts.
    pub stream_epoch: u64,
}

/// Diagnostic snapshot for manual verification of the clock.
#[derive(Clone, Copy, Debug)]
pub struct TimebaseDiagnostics {
    pub source_sample_rate: usize,
    pub source_frames_ingested: u64,
    pub canonical_sample_index: u64,
    pub analysis_hop_count: u64,
    pub time_seconds: f64,
    pub stream_epoch: u64,
    pub detected_gaps: u64,
    pub detected_overlaps: u64,
    pub last_source_start_frame: u64,
    pub last_source_end_frame: u64,
}

/// Errors reported by the timebase (transport bugs, not discontinuities).
#[derive(Clone, Debug, PartialEq)]
pub enum TimebaseError {
    /// A PCM block overlapped already-ingested source frames.
    Overlap { start: u64, expected: u64 },
    /// The source sample rate changed mid-stream without a declared reset.
    RateChanged { from: usize, to: usize },
}

impl std::fmt::Display for TimebaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimebaseError::Overlap { start, expected } => write!(
                f,
                "overlapping PCM block (start {start} < expected {expected}) — transport bug"
            ),
            TimebaseError::RateChanged { from, to } => write!(
                f,
                "source rate changed mid-stream ({from} → {to}); declare a reset instead"
            ),
        }
    }
}

impl std::error::Error for TimebaseError {}

/// Stateful streaming linear resampler from an arbitrary source rate to the
/// canonical 48 kHz timeline.
///
/// Unlike a per-chunk resampler, this preserves fractional source position
/// across input blocks: the same source stream produces the same canonical
/// output regardless of how it is chunked. Correctness is invariant to input
/// block size within normal float rounding.
///
/// Model: a virtual source signal `v[j]` over all source frames. Output
/// sample `k` is linear interpolation of `v` at source position
/// `k * (from_rate / to_rate)`. We track the next output's absolute source
/// position and the previous block's tail so interpolation is continuous
/// across block boundaries.
pub struct StreamingResampler {
    from_rate: usize,
    to_rate: usize,
    /// Source frames advanced per output sample.
    step: f64,
    /// Absolute source position (source frames) of the next output sample.
    next_source_pos: f64,
    /// Absolute source index of the first frame of the current block.
    block_start: u64,
    /// Tail of the previous block (up to last 2 samples) for cross-block reads.
    prev_tail: Vec<f32>,
}

impl StreamingResampler {
    pub fn new(from_rate: usize, to_rate: usize) -> Self {
        assert!(from_rate > 0 && to_rate > 0, "invalid resample rates");
        Self {
            from_rate,
            to_rate,
            step: from_rate as f64 / to_rate as f64,
            next_source_pos: 0.0,
            block_start: 0,
            prev_tail: Vec::new(),
        }
    }

    /// Resample one block to the canonical rate.
    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            // Passthrough: still advance phase so a later rate change is coherent.
            self.block_start += block.len() as u64;
            self.next_source_pos = self.block_start as f64;
            return block.to_vec();
        }

        let mut out: Vec<f32> = Vec::new();
        let block_end = self.block_start + block.len() as u64; // absolute, exclusive

        // Resolve an absolute source index to a sample. Indices inside the
        // current block read directly; the single index just before the
        // block (block_start - 1) reads the previous block's last sample so
        // interpolation is continuous across the boundary.
        let sample_at = |abs_idx: i64| -> f32 {
            let rel = abs_idx - self.block_start as i64;
            if rel >= 0 && (rel as usize) < block.len() {
                block[rel as usize]
            } else if rel < 0 {
                // Previous block's tail: prev_tail[k] ↔ block_start - len + k.
                let tail_idx = (self.prev_tail.len() as i64 + rel).max(0) as usize;
                if tail_idx < self.prev_tail.len() {
                    self.prev_tail[tail_idx]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };

        // Emit every output whose interpolation is fully determined by
        // samples available now: we need v[i0] and v[i0+1], so require
        // i0 + 1 <= block_end - 1 (i.e. i0+1 is a real sample in this block)
        // OR i0+1 == block_start - 1 + 1 (the previous tail). Outputs whose
        // second tap would fall past this block's end are DEFERRED to the
        // next block (which will supply that sample), keeping the result
        // identical regardless of how the stream is chunked.
        //
        // The last available absolute index is block_end - 1.
        let last_avail = block_end as i64 - 1;
        while self.next_source_pos < block_end as f64 {
            let i0 = self.next_source_pos.floor() as i64;
            if i0 + 1 > last_avail {
                // Need a sample from the next block; stop and carry phase.
                break;
            }
            let frac = (self.next_source_pos - i0 as f64) as f32;
            let s0 = sample_at(i0);
            let s1 = sample_at(i0 + 1);
            out.push(s0 * (1.0 - frac) + s1 * frac);
            self.next_source_pos += self.step;
        }

        // Carry the tail forward for cross-block interpolation.
        if !block.is_empty() {
            self.prev_tail.clear();
            if block.len() > 1 {
                self.prev_tail.push(block[block.len() - 2]);
            }
            self.prev_tail.push(block[block.len() - 1]);
        }
        self.block_start = block_end;

        out
    }

    /// Flush any output deferred at the final block boundary, clamping its
    /// second interpolation tap to the last available sample. Call once at
    /// end-of-stream so a known-duration stream neither gains nor loses the
    /// final sample. Without a following block there is no "next sample", so
    /// clamping is the correct terminal semantics.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return Vec::new();
        }
        let mut out = Vec::new();
        let last_avail = self.block_start as i64 - 1;
        while self.next_source_pos < self.block_start as f64 {
            let i0 = self.next_source_pos.floor() as i64;
            if i0 > last_avail {
                break;
            }
            let frac = (self.next_source_pos - i0 as f64) as f32;
            let rel = i0 - self.block_start as i64;
            let s0 = if rel < 0 {
                let tail_idx = (self.prev_tail.len() as i64 + rel).max(0) as usize;
                if tail_idx < self.prev_tail.len() {
                    self.prev_tail[tail_idx]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            // Clamp the second tap to s0 (no future sample at end-of-stream).
            out.push(s0 * (1.0 - frac) + s0 * frac);
            self.next_source_pos += self.step;
        }
        out
    }

    /// Reset phase state (used on stream discontinuity).
    pub fn reset(&mut self) {
        self.next_source_pos = 0.0;
        self.block_start = 0;
        self.prev_tail.clear();
    }
}

/// The canonical analysis timebase.
///
/// Feed it non-overlapping PCM blocks from the AudioWorklet transport; it
/// validates monotonicity, resamples statefully to 48 kHz, accumulates the
/// rolling history the Rust FeatureExtractor needs, and emits AnalysisTicks
/// on exact 1024-sample canonical boundaries.
pub struct AnalysisTimebase {
    resampler: Option<StreamingResampler>,
    source_sample_rate: Option<usize>,
    extractor: FeatureExtractor,

    /// Canonical samples ingested since stream start (this epoch).
    canonical_pos: u64,
    /// Next hop boundary awaiting enough canonical samples.
    next_hop_sample: u64,
    /// Monotonic epoch counter; bumped on every reset.
    epoch: u64,
    /// Source-frame accounting for exactly-once validation.
    expected_source_frame: u64,
    last_source_start_frame: u64,
    last_source_end_frame: u64,
    detected_gaps: u64,
    detected_overlaps: u64,

    /// Rolling PCM history for the extractor (chronological, append-only).
    history: Vec<f32>,
    /// Cap on retained history. Must exceed TICK_WINDOW_SAMPLES plus the
    /// largest ingest block so a tick's anchored window is never truncated
    /// by draining (truncation would make features cadence-dependent).
    /// 2 s at 48 kHz supports ingest blocks up to ~82.7 k samples.
    history_cap: usize,
}

impl Default for AnalysisTimebase {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisTimebase {
    pub fn new() -> Self {
        Self {
            resampler: None,
            source_sample_rate: None,
            extractor: FeatureExtractor::default(),
            canonical_pos: 0,
            next_hop_sample: 0,
            epoch: 0,
            expected_source_frame: 0,
            last_source_start_frame: 0,
            last_source_end_frame: 0,
            detected_gaps: 0,
            detected_overlaps: 0,
            history: Vec::new(),
            history_cap: 2 * CANONICAL_SAMPLE_RATE,
        }
    }

    /// Ingest one non-overlapping PCM block.
    ///
    /// `source_start_frame` is the position of `samples[0]` on the source
    /// clock (source frames since stream start), monotonically
    /// non-decreasing. Returns the ticks whose hop boundaries completed
    /// within this block (zero or more; chunk size is an implementation
    /// detail).
    pub fn ingest(
        &mut self,
        samples: &[f32],
        source_sample_rate: usize,
        source_start_frame: u64,
    ) -> Result<Vec<AnalysisTick>, TimebaseError> {
        match self.source_sample_rate {
            None => {
                self.source_sample_rate = Some(source_sample_rate);
                self.resampler = Some(StreamingResampler::new(
                    source_sample_rate,
                    CANONICAL_SAMPLE_RATE,
                ));
            }
            Some(rate) if rate != source_sample_rate => {
                return Err(TimebaseError::RateChanged {
                    from: rate,
                    to: source_sample_rate,
                });
            }
            _ => {}
        }

        // Exactly-once / in-order validation on the source clock.
        if source_start_frame < self.expected_source_frame {
            self.detected_overlaps += 1;
            return Err(TimebaseError::Overlap {
                start: source_start_frame,
                expected: self.expected_source_frame,
            });
        }
        if source_start_frame > self.expected_source_frame {
            // A gap in the source clock. Do NOT fabricate PCM; declare a
            // reset so downstream consumers know the stream restarted.
            self.detected_gaps += 1;
            self.reset(ResetReason::DetectedGap);
            self.expected_source_frame = source_start_frame;
        }
        self.last_source_start_frame = source_start_frame;
        self.expected_source_frame = source_start_frame + samples.len() as u64;
        self.last_source_end_frame = self.expected_source_frame;

        // Stateful resample to canonical rate.
        let canonical = self.resampler.as_mut().unwrap().process(samples);
        self.append_history(&canonical);
        self.canonical_pos += canonical.len() as u64;

        // Emit every hop boundary fully covered by canonical samples so far.
        let mut ticks = Vec::new();
        while self.next_hop_sample + CANONICAL_HOP_LENGTH as u64 <= self.canonical_pos {
            self.next_hop_sample += CANONICAL_HOP_LENGTH as u64;
            ticks.push(self.emit_tick(self.next_hop_sample));
        }
        Ok(ticks)
    }

    /// Declare a stream discontinuity (start/stop/source replacement). Bumps
    /// the epoch and clears resampler phase + hop accounting. History is
    /// kept (the extractor needs warm-up PCM) but hop scheduling restarts.
    pub fn reset(&mut self, reason: ResetReason) {
        self.epoch += 1;
        self.canonical_pos = 0;
        self.next_hop_sample = 0;
        self.expected_source_frame = 0;
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
        if reason != ResetReason::DetectedGap {
            // A detected gap keeps the source rate; other resets may precede
            // a new source with a different rate.
            self.source_sample_rate = None;
            self.resampler = None;
        }
    }

    /// Flush end-of-stream: emit any resampler output deferred at the final
    /// block boundary and any hop boundary it completes. Call once when the
    /// source ends so a known-duration stream neither gains nor loses the
    /// final sample. Not used during normal streaming.
    pub fn flush(&mut self) -> Vec<AnalysisTick> {
        let mut ticks = Vec::new();
        if let Some(r) = self.resampler.as_mut() {
            let canonical = r.flush();
            if !canonical.is_empty() {
                self.append_history(&canonical);
                self.canonical_pos += canonical.len() as u64;
                while self.next_hop_sample + CANONICAL_HOP_LENGTH as u64 <= self.canonical_pos {
                    self.next_hop_sample += CANONICAL_HOP_LENGTH as u64;
                    ticks.push(self.emit_tick(self.next_hop_sample));
                }
            }
        }
        ticks
    }

    fn append_history(&mut self, samples: &[f32]) {
        self.history.extend_from_slice(samples);
        if self.history.len() > self.history_cap {
            let excess = self.history.len() - self.history_cap;
            self.history.drain(0..excess);
        }
    }

    fn emit_tick(&self, hop_sample: u64) -> AnalysisTick {
        // Anchor the feature window to the ABSOLUTE hop boundary: the
        // window ENDS at the hop sample and is TICK_WINDOW_SAMPLES long
        // (a multiple of the hop). The extractor's last window over any
        // slice of this length is then identical regardless of how much
        // extra history precedes it — so tick features are invariant to
        // ingest block cadence (production-path parity, issue #93).
        let window_len = TICK_WINDOW_SAMPLES;
        // Samples ingested past this tick's hop boundary (a later hop in
        // the same block): the slice must exclude them.
        let overshoot = (self.canonical_pos - hop_sample) as usize;
        let end = self.history.len() - overshoot;
        let start = end.saturating_sub(window_len);
        let slice = &self.history[start..end];
        let windows = self.extractor.extract_windowed_features(slice, WINDOW_FRAMES);
        let features = windows.into_iter().last().unwrap_or_default();
        AnalysisTick {
            features,
            sample_index: hop_sample,
            time_seconds: hop_sample as f64 / CANONICAL_SAMPLE_RATE as f64,
            dt_seconds: CANONICAL_HOP_LENGTH as f64 / CANONICAL_SAMPLE_RATE as f64,
            stream_epoch: self.epoch,
        }
    }

    pub fn diagnostics(&self) -> TimebaseDiagnostics {
        TimebaseDiagnostics {
            source_sample_rate: self.source_sample_rate.unwrap_or(0),
            source_frames_ingested: self.last_source_end_frame,
            canonical_sample_index: self.canonical_pos,
            analysis_hop_count: self.next_hop_sample / CANONICAL_HOP_LENGTH as u64,
            time_seconds: self.canonical_pos as f64 / CANONICAL_SAMPLE_RATE as f64,
            stream_epoch: self.epoch,
            detected_gaps: self.detected_gaps,
            detected_overlaps: self.detected_overlaps,
            last_source_start_frame: self.last_source_start_frame,
            last_source_end_frame: self.last_source_end_frame,
        }
    }
}
