//! Slow-clock audio state bus.
//!
//! This module defines the low-frequency (hop-rate) state that is treated as
//! the source of truth for timing, beat tracking, and structure inference.

use serde::{Deserialize, Serialize};

use crate::features::FeatureExtractor;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeatClockState {
    pub t_sec: f32,
    pub spb: f32,
    pub phase: f32,
    pub beat_count: i64,
    pub conf: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureState {
    pub section_probs: Vec<f32>,
    pub hazard_probs: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlowState {
    pub beat: BeatClockState,
    pub structure: StructureState,
    pub features: Vec<f32>,
}

pub struct SlowClock {
    sr: f32,
    hop_size: usize,
    feature_extractor: FeatureExtractor,
    beat: BeatClockState,
    structure: StructureState,
}

impl SlowClock {
    pub fn new() -> Self {
        let feature_extractor = FeatureExtractor::default();
        let beat = BeatClockState {
            t_sec: 0.0,
            spb: 0.5,
            phase: 0.0,
            beat_count: 0,
            conf: 0.5,
        };
        let structure = StructureState {
            section_probs: vec![0.0; 4],
            hazard_probs: vec![0.0; 4],
        };
        Self {
            sr: feature_extractor.sr as f32,
            hop_size: feature_extractor.hop_length,
            feature_extractor,
            beat,
            structure,
        }
    }

    pub fn process_hop(&mut self, hop: &[f32]) -> SlowState {
        let dt = self.hop_size as f32 / self.sr;
        self.beat.t_sec += dt;
        let phase = self.beat.phase + dt / self.beat.spb;
        let beats_crossed = phase.floor() as i64;
        self.beat.phase = phase.rem_euclid(1.0);
        if beats_crossed > 0 {
            self.beat.beat_count += beats_crossed;
        }

        let features = self
            .feature_extractor
            .extract_features(hop)
            .iter()
            .map(|series| series.first().copied().unwrap_or(0.0) as f32)
            .collect::<Vec<f32>>();

        SlowState {
            beat: self.beat.clone(),
            structure: self.structure.clone(),
            features,
        }
    }
}

pub struct SlowStateStream {
    clock: SlowClock,
}

impl SlowStateStream {
    pub fn new() -> Self {
        Self {
            clock: SlowClock::new(),
        }
    }

    pub fn process_hop(&mut self, hop: &[f32]) -> SlowState {
        self.clock.process_hop(hop)
    }

    pub fn process_hop_json(&mut self, hop: &[f32]) -> serde_json::Result<String> {
        let state = self.clock.process_hop(hop);
        serde_json::to_string(&state)
    }

    pub fn process_hop_ndjson(&mut self, hop: &[f32]) -> serde_json::Result<String> {
        let mut json = self.process_hop_json(hop)?;
        json.push('\n');
        Ok(json)
    }
}
