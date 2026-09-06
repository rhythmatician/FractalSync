//! Read-only DebugSnapshot seam (issue #111 Phase A).
//!
//! One canonical, versioned, read-only diagnostic snapshot of authoritative
//! runtime state for the debug cockpit. Rust owns ALL semantics (ADR 0001):
//! every field is computed by the same canonical functions the physics kernel
//! and the minimap use — nothing here re-derives geometry, and nothing here
//! mutates runtime state.
//!
//! Contract (issue #111):
//! 1. Snapshot creation cannot mutate runtime state.
//! 2. Rust owns deterministic Map/geometry/Physics/control semantics.
//! 3. WASM and PyO3 expose equivalent snapshot semantics.
//! 4. TypeScript owns only UI/camera/render interpolation.
//! 5. DebugSnapshot must not become a back door that silently expands
//!    PlayerObservation — the `observation` section is deliberately absent
//!    until #108 defines the versioned contract.
//!
//! Wire format is camelCase (serde) so the browser and the trainer read the
//! same keys, matching the AnalysisTick parity convention (issue #93).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Version of the DebugSnapshot contract. Bump on any field/grouping change,
/// in the same commit as binding + UI updates.
pub const DEBUG_SNAPSHOT_VERSION: &str = "debug-snapshot/1";

/// Canonical analysis-tick cadence (issue #91): HOP_LENGTH / SAMPLE_RATE.
/// Derived from the timebase authority — not restated (ADR 0001).
pub const CANONICAL_DT: f64 =
    crate::controller::HOP_LENGTH as f64 / crate::controller::SAMPLE_RATE as f64;

// ---------------------------------------------------------------------------
// Action section (Controls v2, issue #107)
// ---------------------------------------------------------------------------

/// Raw motion controls exactly as the policy emitted them (pre-clamp).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawMotionControls {
    pub direction: [f64; 2],
    pub throttle: f64,
    pub brake: f64,
    pub grip: f64,
    pub impulse: f64,
}

/// The effective (clamped) motion controls physics actually applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectiveMotionControls {
    pub direction: [f64; 2],
    pub throttle: f64,
    pub brake: f64,
    pub grip: f64,
    pub impulse: f64,
}

/// Action section: raw policy output, effective applied action, and the
/// generalized force quantities physics derived from them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionSnapshot {
    pub raw: RawMotionControls,
    pub effective: EffectiveMotionControls,
    /// Metric-consistent generalized drive covector Q_drive actually used.
    pub drive_covector: [f64; 2],
    /// Effective friction coefficient beta = GRIP_BASE + grip*GRIP_COEFF + brake*BRAKE_COEFF.
    pub friction_beta: f64,
    /// Frictional power P = v^T Q_friction <= 0 (PSD dissipation evidence).
    pub friction_power: f64,
}

// ---------------------------------------------------------------------------
// Physics section (manifold, issue #106)
// ---------------------------------------------------------------------------

/// Physics section: authoritative world truth at c.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsSnapshot {
    /// Configuration point c = (x, y).
    pub c: [f64; 2],
    /// Planar velocity v = (vx, vy).
    pub velocity: [f64; 2],
    /// Signed distance D(c): <0 inside M, >0 outside, 0 on The Shore.
    pub signed_distance: f64,
    /// Realm: -1 inside, +1 outside, 0 on the boundary.
    pub realm: i8,
    /// Regularized distance rho = sqrt(D^2 + epsilon^2).
    pub rho: f64,
    /// Mandelbrot scale sigma(c) = log2(d_ref / rho). Distinct from Julia zoom.
    pub sigma: f64,
    /// sigma_dot = grad(sigma) . v (no independent v_sigma state exists).
    pub sigma_dot: f64,
    /// Scale gradient grad(sigma) = (gx, gy).
    pub scale_gradient: [f64; 2],
    /// Induced metric G = rho^-2 I + lambda^2 grad(sigma) grad(sigma)^T, flat [g11, g12, g22].
    pub metric: [f64; 3],
    /// Metric speed sqrt(v^T G v).
    pub metric_speed: f64,
    /// Kinetic energy K = 1/2 v^T G v.
    pub kinetic: f64,
    /// Shore potential U_sigma = kappa * sigma(c), used for crest diagnostics.
    pub potential: f64,
    /// Total mechanical energy E = K + U_sigma + U_wall.
    pub total: f64,
    /// Geodesic (curvature) acceleration -Gamma(v,v) as coordinate acceleration.
    pub geodesic_accel: [f64; 2],
    /// Shore force covector Q_sigma = -kappa grad(sigma).
    pub potential_force: [f64; 2],
    /// Net coordinate acceleration applied last step (diagnostic).
    pub net_accel: [f64; 2],
    /// Physics validity: sampled derivatives and displayed dynamics are finite at c.
    pub derivative_valid: bool,
}

// ---------------------------------------------------------------------------
// Map section (canonical minimap, issue #88)
// ---------------------------------------------------------------------------

/// Map section: read-only view of the canonical mip pyramid. No new map math.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MapSnapshot {
    /// Whether a mip pyramid is currently loaded.
    pub pyramid_loaded: bool,
    /// Shore proximity S(c) in [0,1] at the current c (sensitivity, NOT distance).
    pub shore_proximity: Option<f32>,
    /// The Player's canonical 9x9 minimap window at c (row-major, row 0 = north).
    pub minimap_window: Option<Vec<f32>>,
    /// Pyramid extent [re_min, re_max, im_min, im_max].
    pub extent: Option<[f64; 4]>,
}

// ---------------------------------------------------------------------------
// Diagnostics section
// ---------------------------------------------------------------------------

/// Diagnostics section: integrator/derivative health evidence for #82-style
/// Shore-crossing diagnosis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticsSnapshot {
    /// Finite-difference step derived from the SDF provider's pixel spacing.
    pub derivative_step: f64,
    /// Whether the last manifold step succeeded (fail-closed evidence).
    pub valid: bool,
    /// The most recent manifold integration error, if any.
    pub last_error: Option<String>,
    /// Total-energy change of the last step (energy-ledger evidence).
    pub last_delta_total: Option<f64>,
    /// The regularized crest potential U = kappa * log2(d_ref / epsilon):
    /// the mechanical ceiling of the Shore ridge. Rust-owned so consumers
    /// never restate the crest value (issue #111).
    pub crest_potential: f64,
}

// ---------------------------------------------------------------------------
// The snapshot itself
// ---------------------------------------------------------------------------

/// One read-only diagnostic snapshot of authoritative runtime state.
///
/// The `observation` section is deliberately ABSENT until #108 defines the
/// versioned PlayerObservation contract (Phase B). This struct must not grow
/// observation-shaped fields before then.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugSnapshot {
    /// Contract version.
    pub version: &'static str,
    /// Authoritative step time in seconds (destination step clock).
    pub time_seconds: f64,
    /// Action section (None before the first step).
    pub action: Option<ActionSnapshot>,
    /// Map section (canonical minimap state).
    pub map: MapSnapshot,
    /// Physics section (authoritative world truth).
    pub physics: PhysicsSnapshot,
    /// Diagnostics section.
    pub diagnostics: DiagnosticsSnapshot,
}

/// Inputs to [`snapshot_from_state`] describing the last applied action.
#[derive(Clone, Copy, Debug)]
pub struct LastAction {
    /// Raw controls as emitted (pre-clamp).
    pub raw: crate::controls::MotionControls,
    /// Effective friction beta used by the last step.
    pub friction_beta: f64,
    /// Frictional power of the last step.
    pub friction_power: f64,
}

/// Build a read-only snapshot from explicit authoritative state.
///
/// This is the pure core: it takes plain values (no references to mutable
/// runtime objects) so it is trivially non-mutating. Bindings adapt their
/// runtime objects into these arguments.
#[allow(clippy::too_many_arguments)]
pub fn snapshot_from_state(
    c: Complex64,
    v: (f64, f64),
    last_action: Option<LastAction>,
    manifold_drag: Option<f64>,
    config: &crate::manifold::ManifoldConfig,
    last_delta_total: Option<f64>,
) -> Result<DebugSnapshot, String> {
    // ---- Physics: every value from the canonical manifold functions ----
    let signed_distance = crate::manifold::signed_distance(c)?;
    let realm: i8 = if signed_distance < 0.0 {
        -1
    } else if signed_distance > 0.0 {
        1
    } else {
        0
    };
    let rho = crate::manifold::regularized_distance(c, config.epsilon)?;
    let sigma = crate::manifold::mandelbrot_scale(c, config)?;
    let (gx, gy) = crate::manifold::scale_gradient(c, config)?;
    let sigma_dot = crate::manifold::sigma_dot(c, v, config)?;
    let g = crate::manifold::induced_metric(c, config)?;
    let gv0 = g[0][0] * v.0 + g[0][1] * v.1;
    let gv1 = g[1][0] * v.0 + g[1][1] * v.1;
    let metric_speed = (v.0 * gv0 + v.1 * gv1).sqrt();
    let kinetic = crate::manifold::kinetic_energy(v, c, config)?;
    let potential = crate::manifold::potential_energy(c, config)?;
    let total = crate::manifold::total_energy(v, c, config)?;
    let geodesic = crate::manifold::geodesic_acceleration(v, c, config)?;
    let q_potential = crate::manifold::potential_force(c, config)?;
    let q_wall = crate::manifold::wall_force(c, config)?;

    // Net coordinate acceleration of the last step, reconstructed from the
    // same covector sum the kernel uses (potential + wall + drive + drag ->
    // G^-1). MUST mirror manifold::integrate_step's force sum exactly — if
    // the kernel gains a term, this reconstruction gains it too.
    let (q_drive, beta_used) = match last_action {
        Some(a) => {
            let q = a.raw.clamped().drive_covector(c, config)?;
            (q, a.friction_beta)
        }
        None => ((0.0, 0.0), manifold_drag.unwrap_or(0.0)),
    };
    let q_drag = crate::manifold::drag_force(v, c, beta_used, config)?;
    let q_total = (
        q_potential.0 + q_wall.0 + q_drive.0 + q_drag.0,
        q_potential.1 + q_wall.1 + q_drive.1 + q_drag.1,
    );
    let a_force = crate::manifold::apply_generalized_force(q_total, c, config)?;
    let net_accel = (-geodesic.0 + a_force.0, -geodesic.1 + a_force.1);

    // Physics validity: every sampled derivative and displayed dynamic must be finite.
    let hess = crate::manifold::scale_hessian(c, config)?;
    let derivative_valid = c.re.is_finite()
        && c.im.is_finite()
        && v.0.is_finite()
        && v.1.is_finite()
        && signed_distance.is_finite()
        && rho.is_finite()
        && sigma.is_finite()
        && gx.is_finite()
        && gy.is_finite()
        && sigma_dot.is_finite()
        && g.iter().all(|row| row.iter().all(|x| x.is_finite()))
        && metric_speed.is_finite()
        && kinetic.is_finite()
        && potential.is_finite()
        && total.is_finite()
        && geodesic.0.is_finite()
        && geodesic.1.is_finite()
        && q_potential.0.is_finite()
        && q_potential.1.is_finite()
        && net_accel.0.is_finite()
        && net_accel.1.is_finite()
        && hess.iter().all(|row| row.iter().all(|x| x.is_finite()));

    let physics = PhysicsSnapshot {
        c: [c.re, c.im],
        velocity: [v.0, v.1],
        signed_distance,
        realm,
        rho,
        sigma,
        sigma_dot,
        scale_gradient: [gx, gy],
        metric: [g[0][0], g[0][1], g[1][1]],
        metric_speed,
        kinetic,
        potential,
        total,
        geodesic_accel: [geodesic.0, geodesic.1],
        potential_force: [q_potential.0, q_potential.1],
        net_accel: [net_accel.0, net_accel.1],
        derivative_valid,
    };

    // ---- Action section ----
    let action = last_action.map(|a| {
        let clamped = a.raw.clamped();
        ActionSnapshot {
            raw: RawMotionControls {
                direction: a.raw.direction,
                throttle: a.raw.throttle,
                brake: a.raw.brake,
                grip: a.raw.grip,
                impulse: a.raw.impulse,
            },
            effective: EffectiveMotionControls {
                direction: clamped.direction,
                throttle: clamped.throttle,
                brake: clamped.brake,
                grip: clamped.grip,
                impulse: clamped.impulse,
            },
            drive_covector: [q_drive.0, q_drive.1],
            friction_beta: a.friction_beta,
            friction_power: a.friction_power,
        }
    });

    // ---- Map section: canonical pyramid only ----
    let map = crate::minimap::with_pyramid(|pyr| match pyr {
        None => MapSnapshot {
            pyramid_loaded: false,
            shore_proximity: None,
            minimap_window: None,
            extent: None,
        },
        Some(p) => MapSnapshot {
            pyramid_loaded: true,
            shore_proximity: p.shore_proximity_at(c, 0),
            minimap_window: p.minimap(c, 0, 4),
            extent: Some([p.re_min, p.re_max, p.im_min, p.im_max]),
        },
    });

    // ---- Diagnostics section ----
    let diagnostics = DiagnosticsSnapshot {
        derivative_step: crate::manifold::derivative_step(),
        valid: true,
        last_error: None,
        last_delta_total,
        crest_potential: config.kappa * (config.d_ref / config.epsilon).log2(),
    };

    Ok(DebugSnapshot {
        version: DEBUG_SNAPSHOT_VERSION,
        time_seconds: 0.0,
        action,
        map,
        physics,
        diagnostics,
    })
}

// ---------------------------------------------------------------------------
// Terrain patch: the skate park from authoritative geometry
// ---------------------------------------------------------------------------

/// A sampled terrain patch of the canonical embedding Q(c) = (x, y, lambda*sigma(c)).
///
/// The height visualizes canonical scale. The full Physics metric also weights
/// horizontal motion by rho^-2, so this Euclidean patch is a diagnostic view of
/// the graph rather than an isometric embedding of the scale-relative manifold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainPatch {
    /// Grid dimension (n x n vertices).
    pub n: usize,
    /// Patch center in c-space.
    pub center: [f64; 2],
    /// Patch half-extent in c-space units.
    pub half: f64,
    /// Flat vertex positions, row-major, row 0 = north (im = center + half):
    /// [x0, y0, z0, x1, y1, z1, ...] with z = lambda * sigma(c).
    pub positions: Vec<f64>,
    /// Signed distance D(c) per vertex (row-major).
    pub signed: Vec<f64>,
    /// Realm per vertex: -1 inside, +1 outside, 0 on the boundary.
    pub realm: Vec<i8>,
}

impl TerrainPatch {
    /// Grid dimension (n x n vertices).
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Sample an n x n terrain patch of the canonical embedding centered at
/// (cx, cy) with half-extent `half` in c-space units.
///
/// Every vertex height is lambda * sigma(c) from the canonical scale function;
/// every signed distance comes from the canonical SDF authority.
pub fn terrain_patch(
    cx: f64,
    cy: f64,
    half: f64,
    n: usize,
    config: &crate::manifold::ManifoldConfig,
) -> Result<TerrainPatch, String> {
    if !(2..=512).contains(&n) {
        return Err(format!("terrain patch grid size {n} out of range [2, 512]"));
    }
    if !(half.is_finite() && half > 0.0) {
        return Err(format!("terrain patch half-extent {half} must be positive"));
    }
    let lambda = config.lambda_sq.sqrt();
    let mut positions = Vec::with_capacity(n * n * 3);
    let mut signed = Vec::with_capacity(n * n);
    let mut realm = Vec::with_capacity(n * n);
    // Row 0 is the north edge (im = cy + half); column increases with Re.
    for row in 0..n {
        let im = cy + half - 2.0 * half * (row as f64) / ((n - 1) as f64);
        for col in 0..n {
            let re = cx - half + 2.0 * half * (col as f64) / ((n - 1) as f64);
            let c = Complex64::new(re, im);
            let d = crate::manifold::signed_distance(c)?;
            let sigma = crate::manifold::mandelbrot_scale(c, config)?;
            positions.push(re);
            positions.push(im);
            positions.push(lambda * sigma);
            let r: i8 = if d < 0.0 {
                -1
            } else if d > 0.0 {
                1
            } else {
                0
            };
            signed.push(d);
            realm.push(r);
        }
    }
    Ok(TerrainPatch {
        n,
        center: [cx, cy],
        half,
        positions,
        signed,
        realm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dt_matches_hop_cadence() {
        assert!((CANONICAL_DT - 1024.0 / 48000.0).abs() < 1e-15);
    }

    #[test]
    fn snapshot_total_includes_wall_while_potential_remains_shore_specific() {
        let _lock = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let config = crate::manifold::ManifoldConfig::default();
        let c = Complex64::new(1.5, 0.0);
        let v = (0.03, -0.02);
        let snapshot = snapshot_from_state(c, v, None, None, &config, None).unwrap();
        let wall = crate::manifold::wall_potential(c, &config).unwrap();

        assert!(wall > 0.0);
        assert!((snapshot.physics.total
            - (snapshot.physics.kinetic + snapshot.physics.potential + wall))
            .abs()
            < 1e-10);
        assert!(snapshot.physics.potential < snapshot.physics.total);
    }

    #[test]
    fn snapshot_fails_closed_when_wall_energy_is_not_finite() {
        let _lock = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let config = crate::manifold::ManifoldConfig::default();
        let error = snapshot_from_state(
            Complex64::new(2.0, 0.0),
            (0.0, 0.0),
            None,
            None,
            &config,
            None,
        )
        .expect_err("a state on the open-disk wall must not produce a valid snapshot");
        assert!(error.contains("wall potential unstable"));
    }
}
