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
pub const DEBUG_SNAPSHOT_VERSION: &str = "debug-snapshot/5";

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
    pub drive_covector: Option<[f64; 2]>,
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
    pub signed_distance: Option<f64>,
    /// Realm: -1 inside, +1 outside, 0 on the boundary.
    pub realm: Option<i8>,
    /// Regularized distance rho = sqrt(D^2 + epsilon^2).
    pub rho: Option<f64>,
    /// Authoritative H3 rendering data, using this snapshot's manifold config.
    pub upper_half: Option<UpperHalfGeometry>,
    /// Mandelbrot scale sigma(c) = log2(d_ref / rho). Distinct from Julia zoom.
    pub sigma: Option<f64>,
    /// sigma_dot = grad(sigma) . v (no independent v_sigma state exists).
    pub sigma_dot: Option<f64>,
    /// Scale gradient grad(sigma) = (gx, gy).
    pub scale_gradient: Option<[f64; 2]>,
    /// Induced metric G = rho^-2 I + lambda^2 grad(sigma) grad(sigma)^T, flat [g11, g12, g22].
    pub metric: Option<[f64; 3]>,
    /// Metric speed sqrt(v^T G v).
    pub metric_speed: Option<f64>,
    /// Kinetic energy K = 1/2 v^T G v.
    pub kinetic: Option<f64>,
    /// Shore potential U_sigma = kappa * sigma(c), used for crest diagnostics.
    pub potential: Option<f64>,
    /// Total mechanical energy E = K + U_sigma + U_wall.
    pub total: Option<f64>,
    /// Geodesic (curvature) acceleration -Gamma(v,v) as coordinate acceleration.
    pub geodesic_accel: Option<[f64; 2]>,
    /// Shore force covector Q_sigma = -kappa grad(sigma).
    pub potential_force: Option<[f64; 2]>,
    /// Net coordinate acceleration applied last step (diagnostic).
    pub net_accel: Option<[f64; 2]>,
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

/// Geometry section: scale-aware provider diagnostics (ADR 0004, issue #145).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySnapshot {
    /// Provider version (e.g. "geometry-provider/1").
    pub provider_version: String,
    /// Provider name ("scale-aware" or "raster-bridge").
    pub provider_name: String,
    /// Deterministic tile/cache identity at c.
    pub tile_id: String,
    /// Requested local scale: alpha * max(rho, epsilon).
    pub requested_scale: Option<f64>,
    /// Actual resolved scale / error capability the provider achieved.
    pub resolved_scale: Option<f64>,
    /// Measured cross-level jet error, in distance units.
    pub estimated_error: Option<f64>,
    /// Whether this jet came from the temporary bridge (true) or destination provider.
    pub is_bridge: bool,
    /// Validity classification: regular / unresolved / singular / outside_provider / provider_failure.
    pub validity: String,
    /// Singularity classification where known: none / cut_locus / high_curvature / etc.
    pub singularity: String,
    /// Signed distance D(c).
    pub d: Option<f64>,
    /// Gradient norm |grad D| (eikonal: ~1 where smooth).
    pub grad_d_norm: Option<f64>,
    /// Hessian Frobenius norm |H_D|.
    pub hessian_norm: Option<f64>,
    /// Hessian eigenvalues where valid.
    pub hessian_eigenvalues: Option<[f64; 2]>,
}

/// Diagnostics section: integrator/derivative health evidence for #82-style
/// Shore-crossing diagnosis, now with scale-aware geometry provider evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticsSnapshot {
    /// Scale-aware evaluation step (alpha * epsilon) for diagnostics.
    /// Legacy `pixel/24` tuning no longer drives Physics.
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
    /// Scale-aware geometry provider diagnostics (ADR 0004).
    pub geometry: GeometrySnapshot,
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
    validate_snapshot_inputs(c, v, last_action, manifold_drag, config, last_delta_total)?;

    // The provider may return a useful value jet while explicitly refusing
    // derivative semantics at a cut locus or unresolved location. Preserve
    // scalar values there, but only evaluate ordinary manifold dynamics for a
    // Regular jet; never turn best-effort derivatives into apparent physics.
    let jet = crate::manifold::geometry_jet(c, config)?;
    let signed_distance = jet.d.is_finite().then_some(jet.d);
    let realm = signed_distance.map(|d| if d < 0.0 { -1 } else if d > 0.0 { 1 } else { 0 });
    let rho = crate::geometry_provider::rho_from_jet(&jet, config.epsilon);
    let rho = rho.is_finite().then_some(rho);
    let sigma_value = crate::geometry_provider::sigma_from_jet(&jet, config);
    let sigma = sigma_value.is_finite().then_some(sigma_value);
    let potential = sigma.map(|s| config.kappa * s).filter(|x| x.is_finite());

    let regular = jet.validity == crate::geometry_provider::GeometryValidity::Regular;
    let mut q_drive: Option<(f64, f64)> = None;
    let mut derivative_error: Option<String> = None;
    let (upper_half, sigma_dot, scale_gradient, metric, metric_speed, kinetic, total,
        geodesic_accel, potential_force, net_accel) = if regular {
        let evaluate = || -> Result<_, String> {
            let rho = rho.ok_or_else(|| "geometry provider returned no finite rho".to_string())?;
            let (gx, gy) = crate::manifold::scale_gradient(c, config)?;
            let sd = crate::manifold::sigma_dot(c, v, config)?;
            let g = crate::manifold::induced_metric(c, config)?;
            let gv0 = g[0][0] * v.0 + g[0][1] * v.1;
            let gv1 = g[1][0] * v.0 + g[1][1] * v.1;
            let speed = (v.0 * gv0 + v.1 * gv1).sqrt();
            let k = crate::manifold::kinetic_energy(v, c, config)?;
            let energy = crate::manifold::total_energy(v, c, config)?;
            let geodesic = crate::manifold::geodesic_acceleration(v, c, config)?;
            let q_potential = crate::manifold::potential_force(c, config)?;
            let q_wall = crate::manifold::wall_force(c, config)?;
            let (drive, beta) = match last_action {
                Some(a) => (a.raw.clamped().drive_covector(c, config)?, a.friction_beta),
                None => ((0.0, 0.0), manifold_drag.unwrap_or(0.0)),
            };
            let q_drag = crate::manifold::drag_force(v, c, beta, config)?;
            let q_total = (q_potential.0 + q_wall.0 + drive.0 + q_drag.0,
                           q_potential.1 + q_wall.1 + drive.1 + q_drag.1);
            let acceleration = crate::manifold::apply_generalized_force(q_total, c, config)?;
            let net = (acceleration.0 - geodesic.0, acceleration.1 - geodesic.1);
            let hess = crate::manifold::scale_hessian(c, config)?;
            let finite = [gx, gy, sd, speed, k, energy, geodesic.0, geodesic.1,
                q_potential.0, q_potential.1, net.0, net.1]
                .iter().all(|x| x.is_finite())
                && g.iter().flatten().all(|x| x.is_finite())
                && hess.iter().flatten().all(|x| x.is_finite());
            if !finite { return Err("regular geometry produced non-finite derivatives".into()); }
            let upper = UpperHalfGeometry::new(rho, [gx, gy], sd, config);
            if !upper.a.is_finite()
                || !upper.z.is_finite()
                || upper.gradient.iter().any(|x| !x.is_finite())
                || !upper.z_dot.is_finite()
            {
                return Err("regular geometry produced non-finite upper-half values".into());
            }
            Ok((upper, gx, gy, sd, g, speed, k, energy, geodesic, q_potential, net, drive))
        };
        match evaluate() {
            Ok((upper, gx, gy, sd, g, speed, k, energy, geo, force, net, drive)) => {
                q_drive = Some(drive);
                (Some(upper),
                 Some(sd), Some([gx, gy]), Some([g[0][0], g[0][1], g[1][1]]),
                 Some(speed), Some(k), Some(energy), Some([geo.0, geo.1]),
                 Some([force.0, force.1]), Some([net.0, net.1]))
            }
            Err(error) => {
                derivative_error = Some(error);
                (None, None, None, None, None, None, None, None, None, None)
            }
        }
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };
    let derivative_valid = regular && derivative_error.is_none();
    let physics = PhysicsSnapshot {
        c: [c.re, c.im], velocity: [v.0, v.1], signed_distance, realm, rho,
        upper_half, sigma, sigma_dot, scale_gradient, metric, metric_speed,
        kinetic, potential, total, geodesic_accel, potential_force, net_accel,
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
            drive_covector: q_drive.map(|q| [q.0, q.1]),
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

    // ---- Diagnostics section (ADR 0004: scale-aware geometry provider) ----
    let validity = jet.validity.as_str().to_string();
    let singularity = jet.singularity.as_str().to_string();
    let last_error = derivative_error.or_else(|| {
        (!derivative_valid).then(|| format!("geometry not regular: {validity} (singularity={singularity})"))
    });
    let diagnostics = DiagnosticsSnapshot {
        derivative_step: crate::manifold::derivative_step(),
        valid: derivative_valid,
        last_error,
        last_delta_total,
        crest_potential: config.kappa * (config.d_ref / config.epsilon).log2(),
        geometry: GeometrySnapshot {
            provider_version: jet.provider_version.clone(),
            provider_name: if jet.is_bridge { "raster-bridge".to_string() } else { "scale-aware".to_string() },
            tile_id: jet.tile_id.clone(),
            requested_scale: jet.requested_scale.is_finite().then_some(jet.requested_scale),
            resolved_scale: jet.resolved_scale.is_finite().then_some(jet.resolved_scale),
            estimated_error: jet.estimated_error.is_finite().then_some(jet.estimated_error),
            is_bridge: jet.is_bridge,
            validity,
            singularity,
            d: jet.d.is_finite().then_some(jet.d),
            grad_d_norm: (regular && jet.grad_norm().is_finite()).then(|| jet.grad_norm()),
            hessian_norm: (regular && jet.hessian_norm().is_finite()).then(|| jet.hessian_norm()),
            hessian_eigenvalues: (regular && jet.hessian_eigenvalues().iter().all(|x| x.is_finite()))
                .then(|| jet.hessian_eigenvalues()),
        },
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

fn validate_snapshot_inputs(
    c: Complex64,
    v: (f64, f64),
    last_action: Option<LastAction>,
    manifold_drag: Option<f64>,
    config: &crate::manifold::ManifoldConfig,
    last_delta_total: Option<f64>,
) -> Result<(), String> {
    if !c.re.is_finite() || !c.im.is_finite() || !v.0.is_finite() || !v.1.is_finite() {
        return Err("debug snapshot state c and velocity must be finite".into());
    }
    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err("debug snapshot config epsilon must be finite and positive".into());
    }
    if !config.d_ref.is_finite() || config.d_ref <= 0.0 {
        return Err("debug snapshot config d_ref must be finite and positive".into());
    }
    if !config.lambda_sq.is_finite() || config.lambda_sq < 0.0 {
        return Err("debug snapshot config lambda_sq must be finite and nonnegative".into());
    }
    if !config.kappa.is_finite() || !config.mu.is_finite() {
        return Err("debug snapshot config kappa and mu must be finite".into());
    }
    let scale_ratio = config.d_ref / config.epsilon;
    if !scale_ratio.is_finite() || scale_ratio <= 0.0 {
        return Err("debug snapshot config d_ref/epsilon must be finite and positive".into());
    }
    let crest = config.kappa * scale_ratio.log2();
    if !crest.is_finite() {
        return Err("debug snapshot config produces a non-finite crest potential".into());
    }
    if manifold_drag.is_some_and(|x| !x.is_finite())
        || last_delta_total.is_some_and(|x| !x.is_finite())
    {
        return Err("debug snapshot drag and energy delta must be finite when present".into());
    }
    if let Some(action) = last_action {
        let raw = action.raw;
        let effective = raw.clamped();
        if raw.direction.iter().any(|x| !x.is_finite())
            || !raw.throttle.is_finite()
            || !raw.brake.is_finite()
            || !raw.grip.is_finite()
            || !raw.impulse.is_finite()
            || !action.friction_beta.is_finite()
            || !action.friction_power.is_finite()
            || effective.direction.iter().any(|x| !x.is_finite())
            || !effective.throttle.is_finite()
            || !effective.brake.is_finite()
            || !effective.grip.is_finite()
            || !effective.impulse.is_finite()
        {
            return Err("debug snapshot action values must be finite".into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Terrain patch: the skate park from authoritative geometry
// ---------------------------------------------------------------------------

/// Upper-half-space geometry derived only from the active manifold configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpperHalfGeometry {
    pub a: f64,
    pub z: f64,
    /// dz_H/dc. The presentation's outward normal points toward decreasing z_H.
    pub gradient: [f64; 2],
    pub z_dot: f64,
}

impl UpperHalfGeometry {
    fn new(
        rho: f64,
        gradient: [f64; 2],
        sigma_dot: f64,
        config: &crate::manifold::ManifoldConfig,
    ) -> Self {
        let lambda = config.lambda_sq.sqrt();
        Self {
            a: lambda / std::f64::consts::LN_2,
            z: lambda / std::f64::consts::LN_2 * rho,
            gradient: [-lambda * rho * gradient[0], -lambda * rho * gradient[1]],
            z_dot: -lambda * rho * sigma_dot,
        }
    }
}

/// A sampled terrain patch of the canonical embedding Q(c) = (x, y, lambda*sigma(c)).
///
/// The height visualizes canonical scale. The full Physics metric also weights
/// horizontal motion by rho^-2, so this Euclidean patch is a diagnostic view of
/// the graph rather than an isometric embedding of the scale-relative manifold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
    /// Authoritative z_H = a*rho per vertex. Never invert embedding heights in clients.
    pub upper_z: Vec<f64>,
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
    let mut upper_z = Vec::with_capacity(n * n);
    let mut signed = Vec::with_capacity(n * n);
    let mut realm = Vec::with_capacity(n * n);
    // Row 0 is the north edge (im = cy + half); column increases with Re.
    for row in 0..n {
        let im = cy + half - 2.0 * half * (row as f64) / ((n - 1) as f64);
        for col in 0..n {
            let re = cx - half + 2.0 * half * (col as f64) / ((n - 1) as f64);
            let c = Complex64::new(re, im);
            let jet = crate::manifold::geometry_jet(c, config)?;
            let d = jet.d;
            if !d.is_finite() { return Err(format!("terrain geometry has no finite distance: {:?}", jet.validity)); }
            let sigma = crate::geometry_provider::sigma_from_jet(&jet, config);
            positions.push(re);
            positions.push(im);
            positions.push(lambda * sigma);
            upper_z.push(
                lambda / std::f64::consts::LN_2
                    * crate::geometry_provider::rho_from_jet(&jet, config.epsilon),
            );
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
        upper_z,
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
        assert!(
            (snapshot.physics.total.unwrap()
                - (snapshot.physics.kinetic.unwrap() + snapshot.physics.potential.unwrap() + wall))
                .abs()
                < 1e-10
        );
        assert!(snapshot.physics.potential.unwrap() < snapshot.physics.total.unwrap());
    }

    #[test]
    fn snapshot_reports_outside_domain_without_derivatives() {
        let _lock = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let config = crate::manifold::ManifoldConfig::default();
        let snapshot = snapshot_from_state(
            Complex64::new(2.0, 0.0),
            (0.0, 0.0),
            None,
            None,
            &config,
            None,
        )
        .expect("outside-domain geometry should remain inspectable");
        assert_eq!(snapshot.diagnostics.geometry.validity, "outside_provider");
        assert!(!snapshot.diagnostics.valid);
        assert!(snapshot.diagnostics.last_error.is_some());
        assert!(!snapshot.physics.derivative_valid);
        assert!(snapshot.physics.metric.is_none());
        assert!(snapshot.physics.kinetic.is_none());
        assert!(snapshot.physics.total.is_none());
        assert!(snapshot.physics.geodesic_accel.is_none());
    }
}
