//! Deterministic acceptance tests for the canonical sample-clock timebase
//! (issue #91). These prove the transport/timing/scheduling invariants
//! without any browser, AudioWorklet, or wall clock.

use runtime_core::controller::{HOP_LENGTH, SAMPLE_RATE};
use runtime_core::timebase::{
    AnalysisTimebase, ResetReason, StreamingResampler, TimebaseError, CANONICAL_HOP_LENGTH,
    CANONICAL_SAMPLE_RATE,
};

/// A deterministic test signal (ramp) so resampling output is predictable.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

#[test]
fn constants_match_runtime_authority() {
    assert_eq!(CANONICAL_SAMPLE_RATE, SAMPLE_RATE);
    assert_eq!(CANONICAL_HOP_LENGTH, HOP_LENGTH);
    assert_eq!(CANONICAL_SAMPLE_RATE, 48_000);
    assert_eq!(CANONICAL_HOP_LENGTH, 1_024);
}

#[test]
fn passthrough_resampler_is_identity() {
    let mut r = StreamingResampler::new(48_000, 48_000);
    let block = ramp(500);
    let out = r.process(&block);
    assert_eq!(out, block);
}

#[test]
fn resampler_is_chunk_partition_invariant() {
    // The same 44.1 kHz source stream, supplied as one chunk vs many small
    // chunks, must produce the same canonical output (within float rounding).
    let total = 4410; // 0.1 s at 44.1 kHz
    let source = ramp(total);

    let mut one = StreamingResampler::new(44_100, 48_000);
    let whole = one.process(&source);

    let mut many = StreamingResampler::new(44_100, 48_000);
    let mut stitched = Vec::new();
    let mut pos = 0;
    // Arbitrary, uneven partition.
    for chunk in [7usize, 128, 1, 1000, 333, 64, 2048] {
        let end = (pos + chunk).min(total);
        stitched.extend(many.process(&source[pos..end]));
        pos = end;
        if pos >= total {
            break;
        }
    }
    if pos < total {
        stitched.extend(many.process(&source[pos..]));
    }

    assert_eq!(
        whole.len(),
        stitched.len(),
        "chunk partitioning changed canonical output length"
    );
    for (a, b) in whole.iter().zip(stitched.iter()) {
        assert!((a - b).abs() < 1e-3, "resample mismatch {a} vs {b}");
    }
}

#[test]
fn resampler_preserves_duration_within_rounding() {
    // 1.0 s at 44.1 kHz → ~48000 canonical samples (±1 for phase rounding).
    let mut r = StreamingResampler::new(44_100, 48_000);
    let source = vec![0.0f32; 44_100];
    let mut out = r.process(&source);
    out.extend(r.flush());
    let expected = 48_000usize;
    assert!(
        (out.len() as i64 - expected as i64).abs() <= 1,
        "expected ~{expected} canonical samples, got {}",
        out.len()
    );
}

#[test]
fn flush_recovers_deferred_final_sample() {
    // Without flush, the final boundary output is deferred; flush recovers it.
    let mut r = StreamingResampler::new(44_100, 48_000);
    let source = vec![1.0f32; 44_100];
    let streamed = r.process(&source);
    let flushed = r.flush();
    assert!(
        !flushed.is_empty() || streamed.len() >= 47_999,
        "flush should recover the deferred tail (streamed={}, flushed={})",
        streamed.len(),
        flushed.len()
    );
    assert!(streamed.len() + flushed.len() >= 47_999);
}

#[test]
fn exactly_once_and_in_order() {
    let mut tb = AnalysisTimebase::new();
    // Feed 3 non-overlapping blocks at 48 kHz.
    for b in 0..3u64 {
        let start = b * 1024;
        tb.ingest(&vec![0.0f32; 1024], 48_000, start).unwrap();
    }
    let d = tb.diagnostics();
    assert_eq!(d.source_frames_ingested, 3072);
    assert_eq!(d.detected_gaps, 0);
    assert_eq!(d.detected_overlaps, 0);
}

#[test]
fn overlap_is_rejected_as_transport_bug() {
    let mut tb = AnalysisTimebase::new();
    tb.ingest(&vec![0.0f32; 1024], 48_000, 0).unwrap();
    // Re-ingest an overlapping block → error + overlap counter.
    let err = tb.ingest(&vec![0.0f32; 1024], 48_000, 512);
    assert!(matches!(err, Err(TimebaseError::Overlap { .. })));
    assert_eq!(tb.diagnostics().detected_overlaps, 1);
}

#[test]
fn gap_triggers_epoch_reset_without_fabricating_pcm() {
    let mut tb = AnalysisTimebase::new();
    tb.ingest(&vec![0.0f32; 1024], 48_000, 0).unwrap();
    let epoch_before = tb.diagnostics().stream_epoch;
    // Jump forward, leaving a gap of 2048 source frames.
    tb.ingest(&vec![0.0f32; 1024], 48_000, 3072).unwrap();
    let d = tb.diagnostics();
    assert_eq!(d.detected_gaps, 1);
    assert_eq!(d.stream_epoch, epoch_before + 1);
    // Canonical position restarts after the gap (no fabricated samples).
    assert_eq!(d.canonical_sample_index, 1024);
}

#[test]
fn hop_boundaries_are_exact_1024_canonical_samples() {
    let mut tb = AnalysisTimebase::new();
    // Feed 1 s of 48 kHz audio in irregular chunks; collect all ticks.
    let mut all_ticks = Vec::new();
    let mut pos = 0u64;
    let total = 48_000usize;
    let source = vec![0.0f32; total];
    for chunk in [1000usize, 128, 5000, 333, 4096, 7, 8192] {
        let end = (pos as usize + chunk).min(total);
        let ticks = tb
            .ingest(&source[pos as usize..end], 48_000, pos)
            .unwrap();
        all_ticks.extend(ticks);
        pos = end as u64;
        if pos as usize >= total {
            break;
        }
    }
    if (pos as usize) < total {
        all_ticks.extend(tb.ingest(&source[pos as usize..], 48_000, pos).unwrap());
    }

    assert!(!all_ticks.is_empty());
    // First tick at exactly HOP_LENGTH; each subsequent exactly +HOP_LENGTH.
    assert_eq!(all_ticks[0].sample_index, HOP_LENGTH as u64);
    for w in all_ticks.windows(2) {
        assert_eq!(w[1].sample_index - w[0].sample_index, HOP_LENGTH as u64);
    }
    // dt_seconds is the canonical hop duration.
    let expected_dt = HOP_LENGTH as f64 / SAMPLE_RATE as f64;
    for t in &all_ticks {
        assert!((t.dt_seconds - expected_dt).abs() < 1e-12);
        // Timestamps derive from sample_index.
        assert!((t.time_seconds - t.sample_index as f64 / SAMPLE_RATE as f64).abs() < 1e-12);
    }
}

#[test]
fn known_duration_stream_neither_gains_nor_loses_samples() {
    let mut tb = AnalysisTimebase::new();
    let total = 48_000usize; // exactly 1 s at 48 kHz
    let source = vec![0.0f32; total];
    tb.ingest(&source, 48_000, 0).unwrap();
    let d = tb.diagnostics();
    assert_eq!(d.canonical_sample_index, total as u64);
    assert!((d.time_seconds - 1.0).abs() < 1e-9);
}

#[test]
fn irregular_cadence_does_not_alter_tick_positions() {
    // Simulate irregular render/main-thread cadence by feeding the same
    // stream in two different chunkings; tick sample indices must match.
    let total = 20_000usize;
    let source = vec![0.0f32; total];

    let collect = |chunks: &[usize]| -> Vec<u64> {
        let mut tb = AnalysisTimebase::new();
        let mut idxs = Vec::new();
        let mut pos = 0usize;
        for &c in chunks {
            let end = (pos + c).min(total);
            for t in tb.ingest(&source[pos..end], 48_000, pos as u64).unwrap() {
                idxs.push(t.sample_index);
            }
            pos = end;
            if pos >= total {
                break;
            }
        }
        if pos < total {
            for t in tb.ingest(&source[pos..], 48_000, pos as u64).unwrap() {
                idxs.push(t.sample_index);
            }
        }
        idxs
    };

    let steady = collect(&[128, 128, 128, 128, 128, 128, 128, 128]);
    let bursty = collect(&[1, 4096, 13, 2048, 999, 128]);
    assert_eq!(steady, bursty, "cadence changed analysis tick positions");
}

#[test]
fn restart_bumps_epoch_and_resets_schedule() {
    let mut tb = AnalysisTimebase::new();
    tb.ingest(&vec![0.0f32; 4096], 48_000, 0).unwrap();
    let e0 = tb.diagnostics().stream_epoch;
    tb.reset(ResetReason::Stop);
    assert_eq!(tb.diagnostics().stream_epoch, e0 + 1);
    assert_eq!(tb.diagnostics().canonical_sample_index, 0);
    // After restart, the first new tick is again at HOP_LENGTH.
    let ticks = tb.ingest(&vec![0.0f32; 2048], 48_000, 0).unwrap();
    assert_eq!(ticks[0].sample_index, HOP_LENGTH as u64);
    assert_eq!(ticks[0].stream_epoch, e0 + 1);
}

#[test]
fn rate_change_mid_stream_is_rejected() {
    let mut tb = AnalysisTimebase::new();
    tb.ingest(&vec![0.0f32; 1024], 48_000, 0).unwrap();
    let err = tb.ingest(&vec![0.0f32; 1024], 44_100, 1024);
    assert!(matches!(err, Err(TimebaseError::RateChanged { .. })));
}

#[test]
fn resampled_stream_emits_ticks_on_canonical_boundaries() {
    // Feed 44.1 kHz audio; ticks must still land on exact 1024-canonical
    // boundaries and count must match the canonical duration.
    let mut tb = AnalysisTimebase::new();
    let total = 44_100usize; // 1.0 s at 44.1 kHz
    let source = vec![0.0f32; total];
    let ticks = tb.ingest(&source, 44_100, 0).unwrap();
    let d = tb.diagnostics();
    // ~48000 canonical samples → ~46 full hops (48000/1024 = 46.875).
    assert!((d.canonical_sample_index as i64 - 48_000).abs() <= 1);
    for w in ticks.windows(2) {
        assert_eq!(w[1].sample_index - w[0].sample_index, HOP_LENGTH as u64);
    }
    assert_eq!(ticks.len() as u64, d.analysis_hop_count);
}

#[test]
fn tick_features_are_chunk_invariant() {
    // PRODUCTION-PATH PARITY (issue #93): the same source stream fed in
    // different block cadences must produce identical tick FEATURES, not
    // just identical tick positions. The original emit_tick extracted from
    // the whole rolling history, so a large ingest block shifted the
    // extractor's window alignment relative to a small-block stream — a
    // divergence component tests could not see because they never fed the
    // same stream through two cadences and compared features.
    let total = 48_000usize;
    let source: Vec<f32> = (0..total)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        })
        .collect();

    let collect = |chunks: &[usize]| -> Vec<Vec<f64>> {
        let mut tb = AnalysisTimebase::new();
        let mut feats = Vec::new();
        let mut pos = 0usize;
        for &c in chunks {
            let end = (pos + c).min(total);
            for t in tb.ingest(&source[pos..end], 48_000, pos as u64).unwrap() {
                feats.push(t.features);
            }
            pos = end;
            if pos >= total {
                break;
            }
        }
        if pos < total {
            for t in tb.ingest(&source[pos..], 48_000, pos as u64).unwrap() {
                feats.push(t.features);
            }
        }
        feats
    };

    let steady = collect(&[4096; 12]);
    let bursty = collect(&[1, 8192, 13, 2048, 999, 128, 16384, 4096]);
    assert_eq!(steady.len(), bursty.len(), "cadence changed tick count");
    for (i, (a, b)) in steady.iter().zip(bursty.iter()).enumerate() {
        let max_err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-6,
            "tick {i} features diverged across cadences: max_err={max_err}"
        );
    }
}
