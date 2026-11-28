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
    pub reference: [f64; 2],
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    pub metric_lag: f64,
    pub throttle: f64,
    pub saturated: bool,
    pub energy: f64,
    pub control_work: f64,
    pub friction_work: f64,
    pub energy_residual: f64,
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
        let (position, velocity, info) = crate::controls::integrate_motion_controls(
            self.position,
            self.velocity,
            &controls,
            dt,
            &self.config,
        )?;
        let end_metric = crate::manifold::induced_metric(position, &self.config)?;
        let end_error = [reference[0] - position.re, reference[1] - position.im];
        let lag = (end_error[0]
            * (end_metric[0][0] * end_error[0] + end_metric[0][1] * end_error[1])
            + end_error[1] * (end_metric[1][0] * end_error[0] + end_metric[1][1] * end_error[1]))
            .sqrt();
        if !lag.is_finite() {
            return Err("invalid tracking lag".into());
        }
        self.position = position;
        self.velocity = velocity;
        Ok(TrackingSample {
            reference,
            position: [position.re, position.im],
            velocity: [velocity.0, velocity.1],
            metric_lag: lag,
            throttle: controls.throttle,
            saturated: effort > 1.0,
            energy: info.total,
            control_work: info.ledger.control_work,
            friction_work: info.ledger.friction_work,
            energy_residual: info.ledger.residual,
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
