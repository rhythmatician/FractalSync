//! Experimental two-source gesture in an intrinsic orthonormal frame.
//!
//! This computes an intended displacement, not a physically tracked position.
//! The caller supplies phases; it does not infer musical roles or confidence.

/// The contribution of one selected temporal mode.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    /// Phase in radians, including any deliberately chosen offset.
    pub phase: f64,
    /// Nonnegative contribution strength.
    pub gain: f64,
}

/// A frequency change in the known-phase listening passage.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PhaseSegment {
    pub start_seconds: f64,
    pub frequency_hz: f64,
}

/// The second pulse speeds up, slows down to recover its reference, then locks.
pub const KNOWN_A_HZ: f64 = 1.5;
pub const KNOWN_B_SEGMENTS: [PhaseSegment; 4] = [
    PhaseSegment {
        start_seconds: 0.0,
        frequency_hz: 2.0,
    },
    PhaseSegment {
        start_seconds: 8.0,
        frequency_hz: 2.25,
    },
    PhaseSegment {
        start_seconds: 16.0,
        frequency_hz: 1.75,
    },
    PhaseSegment {
        start_seconds: 24.0,
        frequency_hz: 2.0,
    },
];

/// Phases of the audible sine modulators, referenced to their common start.
pub fn known_phase(seconds: f64) -> Result<(f64, f64), &'static str> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("passage time must be finite and nonnegative");
    }
    let mut cycles_b = 0.0;
    for (index, segment) in KNOWN_B_SEGMENTS.iter().enumerate() {
        let end = KNOWN_B_SEGMENTS
            .get(index + 1)
            .map_or(seconds, |next| next.start_seconds.min(seconds));
        cycles_b += (end - segment.start_seconds).max(0.0) * segment.frequency_hz;
    }
    let phase = |cycles: f64| std::f64::consts::TAU * cycles - std::f64::consts::FRAC_PI_2;
    let result = (phase(KNOWN_A_HZ * seconds), phase(cycles_b));
    if result.0.is_finite() && result.1.is_finite() {
        Ok(result)
    } else {
        Err("passage phase overflow")
    }
}

/// Comparison with B's departure and recovery occurring eight seconds early.
/// At time zero the phases coincide modulo a full cycle.
pub fn known_phase_misaligned(seconds: f64) -> Result<(f64, f64), &'static str> {
    let (a, _) = known_phase(seconds)?;
    let (_, b) = known_phase(seconds + 8.0)?;
    Ok((a, b))
}

/// One passage-scoped causal estimator. IDs never carry across passages.
/// The fixture supplies its two envelope modulations at canonical hop times.
pub struct PassageCycles {
    bank: crate::cycle_bank::CycleBank,
    sample_index: u64,
    modes: Vec<crate::cycle_bank::CycleMode>,
    roles: [Option<u64>; 2],
}

#[derive(serde::Serialize)]
pub struct PassageCycleFrame<'a> {
    pub sample_index: u64,
    pub modes: &'a [crate::cycle_bank::CycleMode],
    pub roles: [Option<u64>; 2],
    pub phases: [Option<f64>; 2],
}

impl Default for PassageCycles {
    fn default() -> Self {
        Self {
            bank: crate::cycle_bank::CycleBank::new(Default::default()),
            sample_index: 0,
            modes: Vec::new(),
            roles: [None; 2],
        }
    }
}

#[cfg(test)]
mod turnover {
    use super::PassageCycles;
    use std::f64::consts::TAU;

    #[test]
    fn a_new_ridge_requires_deliberate_rebinding_after_source_loss() {
        let mut passage = PassageCycles::default();
        let evidence = |time: f64| {
            let a = if time <= 18.0 {
                (TAU * 1.5 * time).cos()
            } else if time >= 36.0 {
                (TAU * 3.0 * (time - 36.0)).cos()
            } else {
                0.0
            };
            Ok([a, 0.0])
        };
        for second in 1..=18 {
            passage
                .advance_with_envelopes(second as f64, evidence)
                .unwrap();
        }
        let frame = passage.advance_with_envelopes(18.0, evidence).unwrap();
        let old = frame
            .modes
            .iter()
            .find(|mode| (mode.frequency_hz - 1.5).abs() < 0.15)
            .unwrap()
            .id;
        passage.bind(0, Some(old)).unwrap();
        assert!(passage
            .advance_with_envelopes(18.0, evidence)
            .unwrap()
            .phases[0]
            .is_some());

        for second in 19..=54 {
            passage
                .advance_with_envelopes(second as f64, evidence)
                .unwrap();
        }
        let frame = passage.advance_with_envelopes(54.0, evidence).unwrap();
        let replacement = frame
            .modes
            .iter()
            .find(|mode| (mode.frequency_hz - 3.0).abs() < 0.15)
            .unwrap()
            .id;
        assert_ne!(replacement, old);
        assert_eq!(frame.roles[0], Some(old));
        assert_eq!(frame.phases[0], None);
        passage.bind(0, Some(replacement)).unwrap();
        let frame = passage.advance_with_envelopes(54.0, evidence).unwrap();
        assert_eq!(frame.roles[0], Some(replacement));
        assert!(frame.phases[0].is_some());
    }
}

impl PassageCycles {
    pub fn bind(&mut self, role: usize, id: Option<u64>) -> Result<(), String> {
        let slot = self.roles.get_mut(role).ok_or("unknown gesture role")?;
        if let Some(id) = id {
            if !self
                .modes
                .iter()
                .any(|mode| mode.id == id && mode.missing_observations == 0)
            {
                return Err("that mode is no longer observed".into());
            }
        }
        *slot = id;
        Ok(())
    }

    pub fn advance(&mut self, seconds: f64) -> Result<PassageCycleFrame<'_>, String> {
        self.advance_with_envelopes(seconds, |time| {
            let (a, b) = known_phase(time)?;
            Ok([a.cos(), b.cos()])
        })
    }

    fn advance_with_envelopes(
        &mut self,
        seconds: f64,
        envelopes: impl Fn(f64) -> Result<[f64; 2], &'static str>,
    ) -> Result<PassageCycleFrame<'_>, String> {
        use crate::controller::{HOP_LENGTH, SAMPLE_RATE};
        use crate::cycle_bank::{CycleEvidenceChannel, CycleObservation};
        let rate = SAMPLE_RATE as f64;
        let last_seconds = self.sample_index as f64 / rate;
        if !seconds.is_finite() || seconds < last_seconds || seconds - last_seconds > 2.0 {
            return Err("invalid CycleBank passage clock".into());
        }
        let mut advanced = false;
        while (self.sample_index + HOP_LENGTH as u64) as f64 / rate <= seconds {
            let index = self.sample_index + HOP_LENGTH as u64;
            let [a, b] = envelopes(index as f64 / rate)?;
            self.bank
                .observe(&CycleObservation {
                    sample_index: index,
                    dt_seconds: HOP_LENGTH as f64 / rate,
                    stream_epoch: 1,
                    channels: vec![
                        CycleEvidenceChannel::new("fixture-envelope-a", a),
                        CycleEvidenceChannel::new("fixture-envelope-b", b),
                    ],
                })
                .map_err(|error| error.to_string())?;
            self.sample_index = index;
            advanced = true;
        }
        if advanced {
            self.modes = self.bank.modes();
        }
        let delta = seconds - self.sample_index as f64 / rate;
        let phases = self.roles.map(|id| {
            let id = id?;
            self.modes
                .iter()
                .find(|mode| mode.id == id && mode.missing_observations == 0)
                .map(|mode| mode.phase_at(delta))
        });
        Ok(PassageCycleFrame {
            sample_index: self.sample_index,
            modes: &self.modes,
            roles: self.roles,
            phases,
        })
    }
}

/// The two interpretations used in the initial perceptual comparison.
#[derive(Clone, Copy, Debug)]
pub enum Geometry {
    /// Each mode moves along its own axis.
    Distinct,
    /// Both modes rotate in the same plane and can interfere.
    Rotating,
}

#[derive(Clone, Copy, Debug)]
pub struct Gesture {
    pub geometry: Geometry,
    /// Scale of the combined intrinsic displacement.
    pub excursion: f64,
}

/// Experimental bounded pursuit of a reference through the normal Controls seam.
pub const TRACKING_STEP_SECONDS: f64 = 0.02;

pub struct GestureTracker {
    position: num_complex::Complex64,
    velocity: (f64, f64),
    config: crate::manifold::ManifoldConfig,
}

#[derive(Debug, serde::Serialize)]
pub struct TrackingSample {
    pub reference: Option<[f64; 2]>,
    pub start_position: [f64; 2],
    pub start_velocity: [f64; 2],
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    pub metric_lag: Option<f64>,
    pub throttle: f64,
    pub saturated: bool,
    pub energy: f64,
    pub control_work: f64,
    pub friction_work: f64,
    pub energy_residual: f64,
    pub drive_covector: [f64; 2],
    pub signed_distance: f64,
    pub shore_crossing: bool,
    pub ledger_valid: bool,
}

impl GestureTracker {
    pub fn new(position: [f64; 2], velocity: [f64; 2]) -> Result<Self, String> {
        if !position
            .iter()
            .chain(velocity.iter())
            .all(|v| v.is_finite())
            || position[0].hypot(position[1]) >= 2.0
        {
            return Err("invalid initial tracking state".into());
        }
        Ok(Self {
            position: num_complex::Complex64::new(position[0], position[1]),
            velocity: (velocity[0], velocity[1]),
            config: Default::default(),
        })
    }

    /// One fixed-time Physics step. A failed geometry query leaves state untouched.
    pub fn step(&mut self, reference: [f64; 2], dt: f64) -> Result<TrackingSample, String> {
        if !reference.iter().all(|v| v.is_finite())
            || reference[0].hypot(reference[1]) >= 2.0
            || !dt.is_finite()
            || dt <= 0.0
            || dt > TRACKING_STEP_SECONDS
        {
            return Err("invalid reference or tracking step".into());
        }
        let error = [
            reference[0] - self.position.re,
            reference[1] - self.position.im,
        ];
        let command = [
            4.0 * error[0] - 2.0 * self.velocity.0,
            4.0 * error[1] - 2.0 * self.velocity.1,
        ];
        let metric = crate::manifold::induced_metric(self.position, &self.config)?;
        let norm = |v: [f64; 2]| {
            (v[0] * (metric[0][0] * v[0] + metric[0][1] * v[1])
                + v[1] * (metric[1][0] * v[0] + metric[1][1] * v[1]))
                .sqrt()
        };
        let effort = norm(command);
        if !effort.is_finite() {
            return Err("invalid tracking metric".into());
        }
        let direction_norm = command[0].hypot(command[1]);
        let direction = if direction_norm > 0.0 {
            [command[0] / direction_norm, command[1] / direction_norm]
        } else {
            [0.0; 2]
        };
        let controls = crate::controls::MotionControls {
            direction,
            throttle: effort.min(1.0),
            brake: 0.0,
            grip: 0.5,
            impulse: 0.0,
        };
        self.integrate(Some(reference), controls, effort > 1.0, dt)
    }

    pub fn navigate(&mut self, direction: [f64; 2], dt: f64) -> Result<TrackingSample, String> {
        if !direction.iter().all(|value| value.is_finite())
            || !dt.is_finite()
            || dt <= 0.0
            || dt > TRACKING_STEP_SECONDS
        {
            return Err("invalid navigation step".into());
        }
        let throttle = direction[0].hypot(direction[1]).min(1.0) * 0.25;
        let controls = crate::controls::MotionControls {
            direction,
            throttle,
            ..Default::default()
        };
        self.integrate(None, controls, false, dt)
    }

    fn integrate(
        &mut self,
        reference: Option<[f64; 2]>,
        controls: crate::controls::MotionControls,
        saturated: bool,
        dt: f64,
    ) -> Result<TrackingSample, String> {
        let (position, velocity, info) = crate::controls::integrate_motion_controls(
            self.position,
            self.velocity,
            &controls,
            dt,
            &self.config,
        )?;
        let metric_lag = if let Some(reference) = reference {
            let metric = crate::manifold::induced_metric(position, &self.config)?;
            let [x, y] = [reference[0] - position.re, reference[1] - position.im];
            let lag = (x * (metric[0][0] * x + metric[0][1] * y)
                + y * (metric[1][0] * x + metric[1][1] * y))
                .sqrt();
            if !lag.is_finite() {
                return Err("invalid tracking lag".into());
            }
            Some(lag)
        } else {
            None
        };
        self.position = position;
        self.velocity = velocity;
        Ok(TrackingSample {
            reference,
            position: [position.re, position.im],
            velocity: [velocity.0, velocity.1],
            start_position: info.ledger.start_position,
            start_velocity: info.ledger.start_velocity,
            metric_lag,
            throttle: controls.throttle,
            saturated,
            energy: info.total,
            control_work: info.ledger.control_work,
            friction_work: info.ledger.friction_work,
            energy_residual: info.ledger.residual,
            drive_covector: info.ledger.control_covector,
            signed_distance: info.ledger.end_geometry.d,
            shore_crossing: info.ledger.start_geometry.d * info.ledger.end_geometry.d < 0.0,
            ledger_valid: info.ledger.valid,
        })
    }
}

impl Gesture {
    /// Intended displacement in a metric-orthonormal tangent frame.
    pub fn displacement(&self, a: Source, b: Source) -> Result<[f64; 2], &'static str> {
        if !self.excursion.is_finite()
            || self.excursion < 0.0
            || !a.phase.is_finite()
            || !b.phase.is_finite()
            || !a.gain.is_finite()
            || a.gain < 0.0
            || !b.gain.is_finite()
            || b.gain < 0.0
        {
            return Err("gesture inputs must be finite and gains and excursion nonnegative");
        }
        let result = match self.geometry {
            Geometry::Distinct => [a.gain * a.phase.cos(), b.gain * b.phase.cos()],
            Geometry::Rotating => [
                a.gain * a.phase.cos() + b.gain * b.phase.cos(),
                a.gain * a.phase.sin() + b.gain * b.phase.sin(),
            ],
        };
        let result = [result[0] * self.excursion, result[1] * self.excursion];
        if result.iter().all(|x| x.is_finite()) {
            Ok(result)
        } else {
            Err("gesture displacement overflow")
        }
    }

    /// First-order local map using the metric at `anchor`.
    ///
    /// The frame is world-axis aligned by Cholesky decomposition. This is
    /// not an exponential map and makes no claim about large displacements.
    pub fn local_target(
        &self,
        anchor: [f64; 2],
        metric: [[f64; 2]; 2],
        a: Source,
        b: Source,
    ) -> Result<[f64; 2], &'static str> {
        if !anchor.iter().all(|v| v.is_finite())
            || !metric.iter().flatten().all(|v| v.is_finite())
            || metric[0][1] != metric[1][0]
            || metric[0][0] <= 0.0
        {
            return Err("anchor and symmetric positive definite metric required");
        }
        let l00 = metric[0][0].sqrt();
        let l10 = metric[1][0] / l00;
        let remainder = metric[1][1] - l10 * l10;
        if !remainder.is_finite() || remainder <= 0.0 {
            return Err("anchor and symmetric positive definite metric required");
        }
        let [u, v] = self.displacement(a, b)?;
        let y = v / remainder.sqrt();
        let x = (u - l10 * y) / l00;
        let target = [anchor[0] + x, anchor[1] + y];
        if target.iter().all(|v| v.is_finite()) {
            Ok(target)
        } else {
            Err("gesture target overflow")
        }
    }
}
