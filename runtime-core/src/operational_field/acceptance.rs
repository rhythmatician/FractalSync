//! Executable ADR 0005 contract, limited to complete, immutable prepared assets.
use super::PreparedField;
use crate::manifold::{self, ManifoldConfig};
use num_complex::Complex64;
use serde::Serialize;
use std::{cell::RefCell, sync::Arc};

pub const EXPERIMENT_MANIFEST: &str =
    include_str!("../../../docs/validation/geometry-145-operational-experiment.json");

#[derive(serde::Deserialize)]
pub struct Perturbation {
    pub name: String,
    pub offset: [f64; 2],
    pub speed_scale: f64,
}
#[derive(serde::Deserialize)]
pub struct Preparation {
    pub origin: [f64; 2],
    pub spacing: f64,
    pub shape: [usize; 2],
}
#[derive(serde::Deserialize)]
pub struct Experiment {
    pub preparation: Preparation,
    pub perturbations: Vec<Perturbation>,
    pub crossing_start: [f64; 2],
    pub ridge_start: [f64; 2],
    pub crossing_timesteps: Vec<f64>,
    pub crossing_horizon_seconds: f64,
    pub launch_energy_factors: [f64; 2],
    pub ridge_dt: f64,
    pub ridge_steps: usize,
}

pub fn frozen_experiment_report(field: Arc<PreparedField>) -> Result<serde_json::Value, String> {
    frozen_experiment_report_with_manifest(field, EXPERIMENT_MANIFEST, &[])
}

/// Inherit the fixed mechanical test contract; a coverage declaration may only
/// supply its prepared domain. It cannot change launches, tolerances or targets.
pub fn resolve_experiment_manifest(text: &str) -> Result<serde_json::Value, String> {
    let declared: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut base: serde_json::Value =
        serde_json::from_str(EXPERIMENT_MANIFEST).map_err(|e| e.to_string())?;
    if declared == base {
        return Ok(base);
    }
    if declared["parent_experiment"] != "geometry-145-operational-experiment.json"
        || declared["schema"] != "operational-coverage-experiment/1"
    {
        return Err("unsupported experiment parent or schema".into());
    }
    let preparation = declared
        .get("preparation")
        .ok_or("missing prepared domain")?
        .clone();
    let _: Preparation = serde_json::from_value(preparation.clone()).map_err(|e| e.to_string())?;
    base["preparation"] = preparation;
    base["coverage_declaration"] = declared;
    Ok(base)
}

pub fn frozen_experiment_report_with_manifest(
    field: Arc<PreparedField>,
    manifest: &str,
    cuts: &[ReferenceCut],
) -> Result<serde_json::Value, String> {
    frozen_experiment_report_with_integrator(field, manifest, cuts, Integrator::Euler)
}

pub fn frozen_experiment_report_with_integrator(
    field: Arc<PreparedField>,
    manifest: &str,
    cuts: &[ReferenceCut],
    method: Integrator,
) -> Result<serde_json::Value, String> {
    validate_reference_cuts(cuts)?;
    let resolved = resolve_experiment_manifest(manifest)?;
    if let Some(declared) = resolved["coverage_declaration"].get("selected_integrator") {
        if declared != &serde_json::to_value(method).map_err(|e| e.to_string())? {
            return Err("declared and selected integrator differ".into());
        }
    }
    if let Some(declared) = resolved["coverage_declaration"].get("field_id") {
        if declared != &serde_json::Value::String(format!("{:016x}", field.identity())) {
            return Err("declared and loaded field differ".into());
        }
    }
    let spec: Experiment = serde_json::from_value(resolved.clone()).map_err(|e| e.to_string())?;
    if field.origin != spec.preparation.origin
        || field.spacing != spec.preparation.spacing
        || field.shape != spec.preparation.shape
    {
        return Err("asset does not match frozen experiment coverage".into());
    }
    let mut runs = Vec::new();
    let mut energy_runs = Vec::new();
    for (name, start, steps) in [
        ("high_curvature", [0.4, 0.05], 150),
        ("near_shore", [0.3, 0.05], 100),
    ] {
        let frozen = FrozenRollout::new(
            field.clone(),
            PhysicsProfile::new(ManifoldConfig::default(), 0.002, 0.0, method)?,
            cuts.to_vec(),
        )?;
        let mut state = frozen.initial_state(start, [0.0, 0.02])?;
        let initial = state;
        let mut excursion = 0.0f64;
        for _ in 0..steps {
            frozen.attempt_step(&mut state, [0.0; 2])?;
            excursion = excursion.max((state.energy - initial.energy).abs());
        }
        let drift = (state.energy - initial.energy).abs();
        energy_runs.push(serde_json::json!({"name":name,"field_id":format!("{:016x}",frozen.field_id()),"physics_profile":frozen.profile().manifest(),"initial":initial,"final":state,"maximum_excursion":excursion,"absolute_drift":drift,"unchanged_limit":0.05,"passed":drift<0.05 && excursion<0.05}));
    }
    let mut join_gradient = 0.0f64;
    let mut join_hessian = 0.0f64;
    for x in [
        0.3205 - 1e-9,
        0.3205,
        0.3205 + 1e-9,
        0.4 - 1e-9,
        0.4,
        0.4 + 1e-9,
    ] {
        let p = [x, 0.0475];
        let j = field.query(p).map_err(|_| "join unavailable")?;
        for delta in [1e-6, 1e-7] {
            for axis in 0..2 {
                let mut plus = p;
                let mut minus = p;
                plus[axis] += delta;
                minus[axis] -= delta;
                let a = field.query(plus).map_err(|_| "join stencil unavailable")?;
                let b = field.query(minus).map_err(|_| "join stencil unavailable")?;
                join_gradient = join_gradient
                    .max(((a.value - b.value) / (2.0 * delta) - j.gradient[axis]).abs());
                for component in 0..2 {
                    let expected = j.hessian[component][axis];
                    join_hessian = join_hessian.max(
                        ((a.gradient[component] - b.gradient[component]) / (2.0 * delta)
                            - expected)
                            .abs()
                            / expected.abs().max(1.0),
                    );
                }
            }
        }
    }
    let config = ManifoldConfig {
        kappa: 0.1,
        ..ManifoldConfig::default()
    };
    let locate = FrozenRollout::new(
        field.clone(),
        PhysicsProfile::new(config.clone(), 0.001, 0.0, method)?,
        cuts.to_vec(),
    )?;
    let ((barrier, launch), _) = locate.evaluate(|config| {
        let (mut lo, mut hi) = (0.2, 0.5);
        for _ in 0..80 {
            let mid = (lo + hi) * 0.5;
            let j = crate::geometry_provider::query_geometry(
                Complex64::new(mid, 0.05),
                config.epsilon,
            )?;
            if j.d < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let crest = manifold::potential_energy(Complex64::new((lo + hi) / 2.0, 0.05), config)?;
        let c = Complex64::new(spec.crossing_start[0], spec.crossing_start[1]);
        let barrier = crest - manifold::potential_energy(c, config)?;
        let g = manifold::induced_metric(c, config)?;
        Ok((
            barrier,
            (2.0 * spec.launch_energy_factors[1] * barrier / g[0][0]).sqrt(),
        ))
    })?;
    for perturb in &spec.perturbations {
        let start = [
            spec.crossing_start[0] + perturb.offset[0],
            spec.crossing_start[1] + perturb.offset[1],
        ];
        for dt in &spec.crossing_timesteps {
            let rollout = FrozenRollout::new(
                field.clone(),
                PhysicsProfile::new(config.clone(), *dt, 0.0, method)?,
                cuts.to_vec(),
            )?;
            runs.push(replay(
                &rollout,
                start,
                [-launch * perturb.speed_scale, 0.0],
                (spec.crossing_horizon_seconds / dt).round() as usize,
                true,
                &perturb.name,
            )?);
        }
        let low = launch * (spec.launch_energy_factors[0] / spec.launch_energy_factors[1]).sqrt();
        let rollout = FrozenRollout::new(
            field.clone(),
            PhysicsProfile::new(config.clone(), 0.001, 0.0, method)?,
            cuts.to_vec(),
        )?;
        runs.push(replay(
            &rollout,
            start,
            [-low * perturb.speed_scale, 0.0],
            (spec.crossing_horizon_seconds / 0.001).round() as usize,
            false,
            &perturb.name,
        )?);
        let rollout = FrozenRollout::new(
            field.clone(),
            PhysicsProfile::new(ManifoldConfig::default(), spec.ridge_dt, 0.0, method)?,
            cuts.to_vec(),
        )?;
        runs.push(replay(
            &rollout,
            [
                spec.ridge_start[0] + perturb.offset[0],
                spec.ridge_start[1] + perturb.offset[1],
            ],
            [-0.03 * perturb.speed_scale, 0.0],
            spec.ridge_steps,
            false,
            &format!("ridge_{}", perturb.name),
        )?);
    }
    let mut parity = Vec::new();
    for p in [
        [0.4, 0.05],
        [0.3205, 0.0475],
        [0.3205 - 1e-9, 0.0475],
        [0.3205 + 1e-9, 0.0475],
        [0.3, 0.05],
        [0.6, 0.15],
        [-0.1, 0.0],
        [0.8, 0.0],
    ] {
        parity.push(match field.query(p) {Ok(j)=>serde_json::json!({"point":p,"availability":"available","jet":[j.value,j.gradient[0],j.gradient[1],j.hessian[0][0],j.hessian[0][1],j.hessian[1][1]]}),Err(_)=>serde_json::json!({"point":p,"availability":"GeometryUnavailable"})});
    }
    let neighboring = super::verification::fixed_neighbor_report(&field)
        .map_err(|e| format!("Neighbor differential verification refused: {e:?}"))?;
    Ok(
        serde_json::json!({"experiment_manifest":resolved,"selected_integrator":method,"field_id":format!("{:016x}",field.identity()),"barrier":barrier,"launch_speed":launch,"runs":runs,"parity_samples":parity,
        "reference_scalar_fidelity":{"status":"unknown","certified_regions":0,"measured_raster_discrepancy_is_not_a_bound":true},
        "energy_runs":energy_runs,
        "neighbor_differential_self_consistency":neighboring,
        "join_differential_self_consistency":{"gradient_absolute_error":join_gradient,"hessian_normalized_error":join_hessian,"tolerance":1e-4,"passed":join_gradient<1e-4 && join_hessian<1e-4},
        "reference_differential_fidelity":"unknown",
        "reference_singularity_policy":"authoritative evidence refuses before operational query; no reference-cut coverage certificate for this corridor",
        "continuous_trajectory":"piecewise-linear reconstruction of successful committed states; no claim of exact continuous ODE solution"}),
    )
}

fn replay(
    rollout: &FrozenRollout,
    start: [f64; 2],
    velocity: [f64; 2],
    steps: usize,
    expect_crossing: bool,
    name: &str,
) -> Result<serde_json::Value, String> {
    let mut state = rollout.initial_state(start, velocity)?;
    let original = state;
    let mut observed = false;
    let mut event = None;
    let mut failure = None;
    let mut work = 0usize;
    let mut max_work = 0usize;
    let mut points = vec![state];
    let mut committed = 0;
    for _ in 0..steps {
        match rollout.attempt_step(&mut state, [0.0; 2]) {
            Ok(count) => {
                work += count.geometry_evaluations;
                max_work = max_work.max(count.geometry_evaluations);
                committed += 1;
            }
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
        let (d, _) = rollout.evaluate(|config| {
            crate::geometry_provider::query_geometry(
                Complex64::new(state.position[0], state.position[1]),
                config.epsilon,
            )
            .map(|j| j.d)
        })?;
        if committed % 25 == 0 {
            points.push(state);
        }
        if d < 0.0 {
            observed = true;
            event = Some(state.mechanical_time);
            points.push(state);
            break;
        }
    }
    if points.last().map(|p| p.mechanical_time) != Some(state.mechanical_time) {
        points.push(state);
    }
    // Reference work occurs only after the complete trajectory is frozen.
    // Existing conservative interval membership is reused without escalation.
    let mut reference_pair = None;
    let mut outside = None;
    let evidence:Vec<_>=points.iter().map(|p| {
        let realm=reference_realm(p.position);
        if realm==Realm::Outside && outside.is_none() {outside=Some(*p);}
        if realm==Realm::Inside && reference_pair.is_none() {if let Some(before)=outside {if before.mechanical_time<p.mechanical_time {reference_pair=Some([before,*p]);}}}
        serde_json::json!({"time":p.mechanical_time,"position":p.position,"realm":realm,"kind":if realm==Realm::Unknown {"unknown"} else {"certified"},"source":"existing outward-rounded uniform_box::membership(point, width=0)"})
    }).collect();
    Ok(
        serde_json::json!({"name":name,"field_id":format!("{:016x}",rollout.field_id()),"physics_profile":rollout.profile.manifest(),"reference_cuts":rollout.reference_cuts(),"initial":original,"final":state,
        "committed_steps":committed,"expected_crossing":expect_crossing,"operational_crossing":if observed {"passed"}else{"not_observed"},"crossing_time":event,
        "mechanical_expectation":if failure.is_none() && observed==expect_crossing {"passed"}else{"failed"},
        "reference_crossing":if reference_pair.is_some() {"passed"}else{"unverified"},"reference_pair":reference_pair,"reference_evidence":evidence,
        "availability":if failure.as_ref().is_some_and(|e|e.contains("GeometryUnavailable")) {"GeometryUnavailable"}else if failure.is_none(){"complete"}else{"refused"},"failure":failure,
        "geometry_evaluations_in_steps":work,"coefficient_reads_in_steps_upper_bound":work*36,"maximum_geometry_evaluations_per_step":max_work,
        "maximum_prepared_query_coefficient_reads":36,"field_and_profile_fixed":state.field_id==original.field_id && state.physics_profile_id==original.physics_profile_id}),
    )
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Integrator {
    Euler,
    Rk4,
    Rk4FourSubsteps,
}

impl Integrator {
    pub fn parse_name(name: &str) -> Result<Self, String> {
        match name {
            "Euler" => Ok(Self::Euler),
            "Rk4" => Ok(Self::Rk4),
            "Rk4FourSubsteps" => Ok(Self::Rk4FourSubsteps),
            _ => Err("unknown experimental integrator".into()),
        }
    }
}

#[derive(Clone)]
pub struct PhysicsProfile {
    config: ManifoldConfig,
    dt: f64,
    beta: f64,
    method: Integrator,
    id: u64,
}

fn fingerprint(words: impl IntoIterator<Item = u64>) -> u64 {
    words
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        })
}

fn config_bits(c: &ManifoldConfig) -> [u64; 5] {
    [
        c.d_ref.to_bits(),
        c.epsilon.to_bits(),
        c.lambda_sq.to_bits(),
        c.kappa.to_bits(),
        c.mu.to_bits(),
    ]
}

impl PhysicsProfile {
    pub fn new(
        config: ManifoldConfig,
        dt: f64,
        beta: f64,
        method: Integrator,
    ) -> Result<Self, String> {
        if ![
            config.d_ref,
            config.epsilon,
            config.lambda_sq,
            config.kappa,
            config.mu,
            dt,
            beta,
        ]
        .iter()
        .all(|v| v.is_finite())
            || config.d_ref <= 0.0
            || config.epsilon < 1e-4
            || config.lambda_sq < 0.0
            || dt <= 0.0
            || beta < 0.0
        {
            return Err("invalid experimental Physics profile".into());
        }
        // Version 1 pins manifold metric/potential/force semantics, explicit
        // covector inputs, fixed beta/dt and the selected integrator/work rule.
        let id = fingerprint(config_bits(&config).into_iter().chain([
            1,
            dt.to_bits(),
            beta.to_bits(),
            method as u64,
        ]));
        Ok(Self {
            config,
            dt,
            beta,
            method,
            id,
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn config(&self) -> &ManifoldConfig {
        &self.config
    }
    pub fn integrator(&self) -> Integrator {
        self.method
    }
    pub fn manifest(&self) -> serde_json::Value {
        serde_json::json!({"physics_profile_id":format!("{:016x}",self.id),"semantics":"operational-mechanics/1;current-reference-refusal-policy;explicit-control-covector;fixed-dt-beta", "integrator":self.method,"dt":self.dt,"beta":self.beta,
            "d_ref":self.config.d_ref,"epsilon":self.config.epsilon,"lambda_sq":self.config.lambda_sq,"kappa":self.config.kappa,"mu":self.config.mu,
            "substeps":match self.method {Integrator::Rk4FourSubsteps=>4,_=>1},
            "work_rule":match self.method {Integrator::Euler=>"left-endpoint control-plus-drag power",Integrator::Rk4=>"RK4 control-plus-drag power quadrature",Integrator::Rk4FourSubsteps=>"four fixed RK4 substeps; held control covector; summed RK4 work quadrature; one outer commit"}})
    }
}

/// Authoritative evidence supplied before replay, scoped to a closed box.
/// Synthetic analytic controls must identify themselves, never as Map proofs.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ReferenceCut {
    pub bounds: [f64; 4], // xmin, ymin, xmax, ymax
    pub authority: String,
}

#[derive(serde::Deserialize)]
struct ReferencePolicy {
    field_id: String,
    refusals: Vec<ReferenceCut>,
}

pub fn reference_policy_for_field(
    field: &PreparedField,
    json: &str,
) -> Result<Vec<ReferenceCut>, String> {
    let policy: ReferencePolicy = serde_json::from_str(json).map_err(|e| e.to_string())?;
    if policy.field_id != format!("{:016x}", field.identity()) {
        return Err("reference policy belongs to another field".into());
    }
    validate_reference_cuts(&policy.refusals)?;
    Ok(policy.refusals)
}

pub fn validate_reference_cuts(cuts: &[ReferenceCut]) -> Result<(), String> {
    if cuts.len() > 64
        || cuts.iter().any(|c| {
            !c.bounds.iter().all(|v| v.is_finite())
                || c.bounds[0] > c.bounds[2]
                || c.bounds[1] > c.bounds[3]
                || c.authority.is_empty()
                || c.authority.len() > 512
        })
    {
        return Err("invalid frozen reference evidence".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct MechanicalState {
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    pub mechanical_time: f64,
    pub integrated_work: f64,
    pub energy: f64,
    #[serde(serialize_with = "serialize_identity")]
    pub field_id: u64,
    #[serde(serialize_with = "serialize_identity")]
    pub physics_profile_id: u64,
}

fn serialize_identity<S: serde::Serializer>(id: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{id:016x}"))
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct QueryWork {
    pub geometry_evaluations: usize,
    pub coefficient_reads_upper_bound: usize,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Realm {
    Inside,
    Outside,
    Unknown,
}

/// Pointwise error floor for a returned scalar with sign opposite a conclusive
/// reference realm. This needs no nearest-boundary or differentiability claim.
/// Unknown or sign agreement supplies no useful error floor.
pub fn certified_realm_scalar_error_floor(value: f64, realm: Realm) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    match realm {
        Realm::Inside if value > 0.0 => Some(value),
        Realm::Outside if value < 0.0 => Some(-value),
        _ => None,
    }
}

/// A point-scoped distance claim. The measured and unknown variants cannot
/// authorize a reference realm verdict, regardless of how small delta is.
pub enum DistanceClaim {
    Certified { point: [f64; 2], delta: f64 },
    Measured { delta: f64 },
    Unknown,
}
pub fn realm_from_claim(point: [f64; 2], value: f64, claim: &DistanceClaim) -> Realm {
    if let DistanceClaim::Certified {
        point: scope,
        delta,
    } = claim
    {
        if point == *scope && value.is_finite() && delta.is_finite() && *delta >= 0.0 {
            if value > *delta {
                return Realm::Outside;
            }
            if value < -*delta {
                return Realm::Inside;
            }
        }
    }
    Realm::Unknown
}

/// Offline reference evidence. Never invoked by the field or Physics step.
pub fn reference_realm(point: [f64; 2]) -> Realm {
    match crate::geometry_provider::operational_reference_membership(Complex64::new(
        point[0], point[1],
    )) {
        Some(true) => Realm::Inside,
        Some(false) => Realm::Outside,
        None => Realm::Unknown,
    }
}

struct Context {
    profile: PhysicsProfile,
    cuts: Arc<[ReferenceCut]>,
    queries: usize,
}
thread_local! { static CONTEXT:RefCell<Option<Context>>=const{RefCell::new(None)}; }
pub(super) fn is_pinned() -> bool {
    CONTEXT.with(|c| c.borrow().is_some())
}

pub fn check_config(config: &ManifoldConfig) -> Result<(), String> {
    CONTEXT.with(|context| {
        if let Some(context) = context.borrow().as_ref() {
            if config_bits(config) != config_bits(&context.profile.config) {
                return Err("FrozenIdentity: Physics profile changed".into());
            }
        }
        Ok(())
    })
}

pub fn check_drag(beta: f64) -> Result<(), String> {
    CONTEXT.with(|context| {
        if context
            .borrow()
            .as_ref()
            .is_some_and(|c| c.profile.beta.to_bits() != beta.to_bits())
        {
            return Err("FrozenIdentity: drag parameter changed".into());
        }
        Ok(())
    })
}
pub fn check_euler_step(beta: f64, dt: f64) -> Result<(), String> {
    check_step(beta, dt, Integrator::Euler)
}
fn check_step(beta: f64, dt: f64, method: Integrator) -> Result<(), String> {
    check_drag(beta)?;
    CONTEXT.with(|context| {
        if context
            .borrow()
            .as_ref()
            .is_some_and(|c| c.profile.dt.to_bits() != dt.to_bits() || c.profile.method != method)
        {
            return Err("FrozenIdentity: timestep or integrator changed".into());
        }
        Ok(())
    })
}

/// Only explicit experimental RK4 profiles override the legacy Euler seam.
/// Raw tuple callers receive one checked transaction; persistent time/work
/// ledgers remain the responsibility of an owned FrozenRollout.
pub(crate) fn experimental_step(
    c: Complex64,
    v: (f64, f64),
    q: (f64, f64),
    beta: f64,
    dt: f64,
    config: &ManifoldConfig,
) -> Option<Result<(Complex64, (f64, f64), manifold::EnergyInfo), String>> {
    let pinned = CONTEXT.with(|ctx| ctx.borrow().as_ref().map(|ctx| ctx.profile.method));
    let method = pinned.unwrap_or_else(|| super::ACTIVE_INTEGRATOR.with(|m| m.get()));
    if method == Integrator::Euler {
        return None;
    }
    let attempt = |config: &ManifoldConfig| {
        check_config(config)?;
        check_step(beta, dt, method)?;
        if ![c.re, c.im, v.0, v.1, q.0, q.1]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err("invalid mechanical state or input".into());
        }
        if c.norm_sqr() >= 4.0 {
            return Err("Hard invariant violated: initial |c| >= 2".into());
        }
        let old_energy = manifold::total_energy(v, c, config)?;
        let old_kinetic = manifold::kinetic_energy(v, c, config)?;
        let (end, velocity, work) = rk4_profile_advance(method, c, v, q, beta, dt, config)?;
        let energy = checked_endpoint(end, velocity, config)?;
        let kinetic = manifold::kinetic_energy(velocity, end, config)?;
        if ![
            old_energy,
            old_kinetic,
            energy,
            kinetic,
            work,
            energy - kinetic,
            energy - old_energy,
            kinetic - old_kinetic,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err("nonfinite proposed mechanical ledger".into());
        }
        Ok((
            end,
            velocity,
            manifold::EnergyInfo {
                kinetic,
                potential: energy - kinetic,
                total: energy,
                delta_total: energy - old_energy,
                delta_kinetic: kinetic - old_kinetic,
            },
        ))
    };
    Some(if pinned.is_some() {
        attempt(config)
    } else {
        (|| {
            let field = super::ACTIVE
                .with(|a| a.borrow().clone())
                .ok_or("RK4 replay requires a prepared field")?;
            let cuts = super::ACTIVE_REFERENCE.with(|a| a.borrow().clone());
            let profile = PhysicsProfile::new(config.clone(), dt, beta, method)?;
            let frame = FrozenRollout::new(field, profile, cuts)?;
            frame.evaluate(attempt).map(|(result, _)| result)
        })()
    })
}

pub(super) fn check_query(c: Complex64, epsilon: f64) -> Result<(), String> {
    CONTEXT.with(|context| {
        if let Some(context) = context.borrow_mut().as_mut() {
            context.queries += 1;
            if epsilon.to_bits() != context.profile.config.epsilon.to_bits() {
                return Err("FrozenIdentity: epsilon changed".into());
            }
            for cut in context.cuts.iter() {
                let b = cut.bounds;
                if c.re >= b[0] && c.re <= b[2] && c.im >= b[1] && c.im <= b[3] {
                    return Err(format!("ReferenceSingularity: {}", cut.authority));
                }
            }
        }
        Ok(())
    })
}

pub struct FrozenRollout {
    field: Arc<PreparedField>,
    profile: PhysicsProfile,
    cuts: Arc<[ReferenceCut]>,
    last_work: std::cell::Cell<QueryWork>,
}

impl FrozenRollout {
    pub fn new(
        field: Arc<PreparedField>,
        mut profile: PhysicsProfile,
        cuts: Vec<ReferenceCut>,
    ) -> Result<Self, String> {
        validate_reference_cuts(&cuts)?;
        let evidence = cuts.iter().flat_map(|c| {
            c.bounds
                .map(f64::to_bits)
                .into_iter()
                .chain(c.authority.as_bytes().iter().map(|b| *b as u64))
        });
        profile.id = fingerprint([profile.id, 1].into_iter().chain(evidence));
        Ok(Self {
            field,
            profile,
            cuts: cuts.into(),
            last_work: std::cell::Cell::new(QueryWork {
                geometry_evaluations: 0,
                coefficient_reads_upper_bound: 0,
            }),
        })
    }
    pub fn field_id(&self) -> u64 {
        self.field.identity()
    }
    pub fn profile(&self) -> &PhysicsProfile {
        &self.profile
    }
    pub fn reference_cuts(&self) -> &[ReferenceCut] {
        &self.cuts
    }
    pub fn last_evaluation_work(&self) -> QueryWork {
        self.last_work.get()
    }

    pub fn evaluate<T>(
        &self,
        f: impl FnOnce(&ManifoldConfig) -> Result<T, String>,
    ) -> Result<(T, QueryWork), String> {
        if is_pinned() {
            return Err("FrozenIdentity: nested rollout activation".into());
        }
        let previous = super::ACTIVE.with(|a| a.replace(Some(self.field.clone())));
        CONTEXT.with(|c| {
            *c.borrow_mut() = Some(Context {
                profile: self.profile.clone(),
                cuts: self.cuts.clone(),
                queries: 0,
            })
        });
        struct Restore(Option<Arc<PreparedField>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CONTEXT.with(|c| *c.borrow_mut() = None);
                super::ACTIVE.with(|a| *a.borrow_mut() = self.0.take());
            }
        }
        let _restore = Restore(previous);
        let result = f(&self.profile.config);
        let queries = CONTEXT.with(|c| c.borrow().as_ref().unwrap().queries);
        let work = QueryWork {
            geometry_evaluations: queries,
            coefficient_reads_upper_bound: queries * 36,
        };
        self.last_work.set(work);
        Ok((result?, work))
    }

    pub fn initial_state(
        &self,
        position: [f64; 2],
        velocity: [f64; 2],
    ) -> Result<MechanicalState, String> {
        if !position
            .iter()
            .chain(velocity.iter())
            .all(|v| v.is_finite())
            || position[0] * position[0] + position[1] * position[1] >= 4.0
        {
            return Err("invalid initial mechanical state".into());
        }
        let (energy, _) = self.evaluate(|config| {
            manifold::total_energy(
                (velocity[0], velocity[1]),
                Complex64::new(position[0], position[1]),
                config,
            )
        })?;
        if !energy.is_finite() {
            return Err("nonfinite initial mechanical energy".into());
        }
        Ok(MechanicalState {
            position,
            velocity,
            energy,
            mechanical_time: 0.0,
            integrated_work: 0.0,
            field_id: self.field_id(),
            physics_profile_id: self.profile.id(),
        })
    }

    /// A transaction over all mechanical state. No caller-owned field is written
    /// until stage, complete differential endpoint and energy checks succeed.
    pub fn attempt_step(
        &self,
        state: &mut MechanicalState,
        control: [f64; 2],
    ) -> Result<QueryWork, String> {
        self.last_work.set(QueryWork {
            geometry_evaluations: 0,
            coefficient_reads_upper_bound: 0,
        });
        if state.field_id != self.field_id() || state.physics_profile_id != self.profile.id() {
            return Err("FrozenIdentity: state belongs to another field/profile".into());
        }
        if !control
            .iter()
            .chain(state.position.iter())
            .chain(state.velocity.iter())
            .chain([
                &state.mechanical_time,
                &state.integrated_work,
                &state.energy,
            ])
            .all(|v| v.is_finite())
        {
            return Err("invalid mechanical state or input".into());
        }
        let ((position, velocity, energy, work), counts) = self.evaluate(|config| {
            let c = Complex64::new(state.position[0], state.position[1]);
            let v = (state.velocity[0], state.velocity[1]);
            let dt = self.profile.dt;
            let beta = self.profile.beta;
            let q = (control[0], control[1]);
            let (position, velocity, work) = match self.profile.method {
                Integrator::Euler => {
                    let drag = manifold::drag_force(v, c, beta, config)?;
                    let work = dt * (v.0 * (q.0 + drag.0) + v.1 * (q.1 + drag.1));
                    let (c, v, _) = manifold::integrate_step(c, v, q, beta, dt, config)
                        .map_err(|e| format!("Euler attempt: {e}"))?;
                    (c, v, work)
                }
                Integrator::Rk4 | Integrator::Rk4FourSubsteps => {
                    rk4_profile_advance(self.profile.method, c, v, q, beta, dt, config)?
                }
            };
            let energy = checked_endpoint(position, velocity, config)?;
            Ok((
                [position.re, position.im],
                [velocity.0, velocity.1],
                energy,
                work,
            ))
        })?;
        let next = MechanicalState {
            position,
            velocity,
            energy,
            mechanical_time: state.mechanical_time + self.profile.dt,
            integrated_work: state.integrated_work + work,
            ..*state
        };
        if ![next.energy, next.mechanical_time, next.integrated_work]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err("nonfinite proposed mechanical ledger".into());
        }
        *state = next;
        Ok(counts)
    }
}

fn rk4_profile_advance(
    method: Integrator,
    mut c: Complex64,
    mut v: (f64, f64),
    q: (f64, f64),
    beta: f64,
    dt: f64,
    config: &ManifoldConfig,
) -> Result<(Complex64, (f64, f64), f64), String> {
    let count = if method == Integrator::Rk4FourSubsteps {
        4
    } else {
        1
    };
    let mut work = 0.0;
    for step in 0..count {
        let (next, velocity, increment) = rk4_advance(c, v, q, beta, dt / count as f64, config)?;
        // Internal endpoints are temporary, but require the same complete
        // differential and energy checks as the final outer endpoint.
        if step + 1 < count {
            checked_endpoint(next, velocity, config)
                .map_err(|e| format!("RK substep {} endpoint: {e}", step + 1))?;
        }
        c = next;
        v = velocity;
        work += increment;
    }
    Ok((c, v, work))
}

fn rk4_advance(
    c: Complex64,
    v: (f64, f64),
    q: (f64, f64),
    beta: f64,
    dt: f64,
    config: &ManifoldConfig,
) -> Result<(Complex64, (f64, f64), f64), String> {
    let initial = [c.re, c.im, v.0, v.1];
    let k1 = rhs(initial, q, beta, config).map_err(|e| format!("RK stage 1: {e}"))?;
    let k2 = rhs(advance(initial, k1, dt / 2.0), q, beta, config)
        .map_err(|e| format!("RK stage 2: {e}"))?;
    let k3 = rhs(advance(initial, k2, dt / 2.0), q, beta, config)
        .map_err(|e| format!("RK stage 3: {e}"))?;
    let k4 =
        rhs(advance(initial, k3, dt), q, beta, config).map_err(|e| format!("RK stage 4: {e}"))?;
    let mut end = initial;
    for i in 0..4 {
        end[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    Ok((
        Complex64::new(end[0], end[1]),
        (end[2], end[3]),
        dt / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]),
    ))
}
fn checked_endpoint(
    position: Complex64,
    velocity: (f64, f64),
    config: &ManifoldConfig,
) -> Result<f64, String> {
    // This point generally differs from the fourth RK stage position.
    let endpoint = crate::geometry_provider::query_geometry(position, config.epsilon)
        .map_err(|e| format!("Final endpoint: {e}"))?;
    if endpoint.validity != crate::geometry_provider::GeometryValidity::Regular {
        return Err("Final endpoint: geometry not regular".into());
    }
    if position.norm_sqr() >= 4.0 || !position.re.is_finite() || !position.im.is_finite() {
        return Err("Hard invariant violated at endpoint".into());
    }
    let energy = manifold::total_energy(velocity, position, config)
        .map_err(|e| format!("Final endpoint energy: {e}"))?;
    Ok(energy)
}
fn advance(s: [f64; 4], k: [f64; 5], dt: f64) -> [f64; 4] {
    std::array::from_fn(|i| s[i] + dt * k[i])
}
fn rhs(s: [f64; 4], q: (f64, f64), beta: f64, config: &ManifoldConfig) -> Result<[f64; 5], String> {
    let c = Complex64::new(s[0], s[1]);
    let v = (s[2], s[3]);
    if !s.iter().all(|v| v.is_finite()) || c.norm_sqr() >= 4.0 {
        return Err("Hard invariant violated at RK stage".into());
    }
    let geometry = crate::geometry_provider::query_geometry(c, config.epsilon)?;
    if geometry.validity != crate::geometry_provider::GeometryValidity::Regular {
        return Err("RK geometry not regular".into());
    }
    let curvature = manifold::geodesic_acceleration(v, c, config)?;
    let potential = manifold::potential_force(c, config)?;
    let wall = manifold::wall_force(c, config)?;
    let drag = manifold::drag_force(v, c, beta, config)?;
    let force = manifold::apply_generalized_force(
        (
            q.0 + potential.0 + wall.0 + drag.0,
            q.1 + potential.1 + wall.1 + drag.1,
        ),
        c,
        config,
    )?;
    Ok([
        v.0,
        v.1,
        force.0 - curvature.0,
        force.1 - curvature.1,
        v.0 * (q.0 + drag.0) + v.1 * (q.1 + drag.1),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_substeps_improve_drag_accuracy_and_roll_back_internal_refusals() {
        let method = Integrator::Rk4FourSubsteps;
        let frame = FrozenRollout::new(constant(None), profile(method), vec![]).unwrap();
        let mut state = frame.initial_state([0.4, 0.0], [0.1, 0.0]).unwrap();
        let initial = state;
        frame.attempt_step(&mut state, [0.0; 2]).unwrap();
        let exact_v = 0.1 * (-0.4_f64).exp();
        assert!((state.velocity[0] - exact_v).abs() < 1e-7);
        assert!((state.position[0] - (0.4 + (0.1 - exact_v) / 2.0)).abs() < 1e-7);
        assert!((state.energy - initial.energy - state.integrated_work).abs() < 1e-7);
        assert_eq!(state.mechanical_time, 0.2);
        // A required stage in substep two, after substep one has completed.
        let blocked = FrozenRollout::new(constant(Some(0.40702)), profile(method), vec![]).unwrap();
        let mut state = blocked.initial_state([0.4, 0.0], [0.1, 0.0]).unwrap();
        let before = serde_json::to_string(&state).unwrap();
        assert!(blocked
            .attempt_step(&mut state, [0.0; 2])
            .unwrap_err()
            .contains("RK stage 2"));
        assert_eq!(serde_json::to_string(&state).unwrap(), before);
    }
    #[test]
    fn replay_declaration_rejects_a_different_integrator_or_field_before_evaluation() {
        let manifest =
            include_str!("../../../docs/validation/geometry-145-operational-rk4-experiment.json");
        let field = constant(None);
        assert!(frozen_experiment_report_with_integrator(
            field.clone(),
            manifest,
            &[],
            Integrator::Euler
        )
        .unwrap_err()
        .contains("integrator"));
        assert!(
            frozen_experiment_report_with_integrator(field, manifest, &[], Integrator::Rk4)
                .unwrap_err()
                .contains("field")
        );
    }
    #[test]
    fn explicit_rk4_legacy_step_matches_owned_transaction_and_keeps_euler_default() {
        let field = constant(None);
        let owned = FrozenRollout::new(field.clone(), profile(Integrator::Rk4), vec![]).unwrap();
        let config = owned.profile.config.clone();
        let mut expected = owned.initial_state([0.4, 0.0], [0.1, 0.0]).unwrap();
        owned.attempt_step(&mut expected, [0.0; 2]).unwrap();
        super::super::activate_with_reference_and_integrator(
            Some(field.clone()),
            vec![],
            Integrator::Rk4,
        )
        .unwrap();
        let (c, v, energy) = manifold::integrate_step(
            Complex64::new(0.4, 0.0),
            (0.1, 0.0),
            (0.0, 0.0),
            2.0,
            0.2,
            &config,
        )
        .unwrap();
        assert_eq!([c.re, c.im], expected.position);
        assert_eq!([v.0, v.1], expected.velocity);
        assert_eq!(energy.total, expected.energy);
        assert!(manifold::integrate_step(
            Complex64::new(2.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            2.0,
            0.2,
            &config
        )
        .unwrap_err()
        .contains("Hard invariant"));
        let pinned = owned
            .evaluate(|cfg| {
                manifold::integrate_step(
                    Complex64::new(0.4, 0.0),
                    (0.1, 0.0),
                    (0.0, 0.0),
                    2.0,
                    0.1,
                    cfg,
                )
            })
            .unwrap_err();
        assert!(pinned.contains("FrozenIdentity"));
        super::super::activate_with_reference(Some(field), vec![]).unwrap();
        let (euler, _, _) = manifold::integrate_step(
            Complex64::new(0.4, 0.0),
            (0.1, 0.0),
            (0.0, 0.0),
            2.0,
            0.2,
            &config,
        )
        .unwrap();
        assert!((euler.re - c.re).abs() > 0.001);
        super::super::activate(None);
    }
    #[test]
    fn coverage_manifest_preserves_declared_mechanics() {
        let coverage = include_str!(
            "../../../docs/validation/geometry-145-operational-coverage-experiment.json"
        );
        let resolved = resolve_experiment_manifest(coverage).unwrap();
        let base: serde_json::Value = serde_json::from_str(EXPERIMENT_MANIFEST).unwrap();
        let declared: serde_json::Value = serde_json::from_str(coverage).unwrap();
        assert_eq!(resolved["preparation"], declared["preparation"]);
        for key in [
            "perturbations",
            "crossing_timesteps",
            "crossing_horizon_seconds",
            "launch_energy_factors",
            "parity_scalar_abs_tolerance",
            "parity_hessian_relative_tolerance",
        ] {
            assert_eq!(resolved[key], base[key]);
        }
        assert!(resolve_experiment_manifest(
            r#"{"parent_experiment":"unapproved-parent.json","preparation":{}}"#
        )
        .is_err());
    }
    fn constant(missing: Option<f64>) -> Arc<PreparedField> {
        let (shape, h, origin) = ([5005, 20], 1e-5, [0.39, -0.0001]);
        let mut c = vec![Some(1.0); shape[0] * shape[1]];
        if let Some(x) = missing {
            let ix = ((x - origin[0]) / h).floor() as usize;
            c[10 * shape[0] + ix] = None;
        }
        Arc::new(PreparedField::new_sparse(origin, h, shape, c).unwrap())
    }
    fn profile(method: Integrator) -> PhysicsProfile {
        let config = ManifoldConfig {
            lambda_sq: 0.0,
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        PhysicsProfile::new(config, 0.2, 2.0, method).unwrap()
    }
    #[test]
    fn measured_and_unknown_distance_claims_do_not_certify_realms() {
        let p = [0.3, 0.05];
        for claim in [
            DistanceClaim::Measured { delta: 0.0 },
            DistanceClaim::Unknown,
        ] {
            assert_eq!(realm_from_claim(p, 1.0, &claim), Realm::Unknown);
            assert_eq!(realm_from_claim(p, -1.0, &claim), Realm::Unknown);
        }
        let certified = DistanceClaim::Certified {
            point: p,
            delta: 0.1,
        };
        assert_eq!(realm_from_claim(p, 0.2, &certified), Realm::Outside);
        assert_eq!(realm_from_claim(p, -0.2, &certified), Realm::Inside);
        assert_eq!(realm_from_claim(p, 0.1, &certified), Realm::Unknown);
        assert_eq!(
            realm_from_claim([0.31, 0.05], 0.2, &certified),
            Realm::Unknown
        );
        assert_eq!(reference_realm([0.0, 0.0]), Realm::Inside);
        assert_eq!(reference_realm([1.0, 0.5]), Realm::Outside);
    }
    #[test]
    fn opposite_certified_realm_bounds_returned_scalar_error() {
        assert_eq!(
            certified_realm_scalar_error_floor(0.01, Realm::Inside),
            Some(0.01)
        );
        assert_eq!(
            certified_realm_scalar_error_floor(-0.02, Realm::Outside),
            Some(0.02)
        );
        assert_eq!(
            certified_realm_scalar_error_floor(0.01, Realm::Unknown),
            None
        );
        assert_eq!(
            certified_realm_scalar_error_floor(-0.01, Realm::Inside),
            None
        );
        assert_eq!(
            certified_realm_scalar_error_floor(f64::NAN, Realm::Inside),
            None
        );
    }
    #[test]
    fn asset_roundtrip_and_cache_stress_preserve_jets_and_refusals() {
        let f = constant(Some(0.42));
        let bytes = f.asset_bytes();
        let restored = PreparedField::from_asset_bytes(&bytes).unwrap();
        assert_eq!(f.identity(), restored.identity());
        let points = [[0.4, 0.0], [0.42, 0.0], [0.43, 0.0], [1.5, 0.0]];
        let expected = points.map(|p| f.query(p));
        let mut eviction = vec![0u64; 8 * 1024 * 1024];
        for (i, v) in eviction.iter_mut().enumerate().step_by(8) {
            *v = i as u64;
        }
        std::hint::black_box(eviction);
        for order in [[3, 1, 0, 2], [2, 0, 1, 3]] {
            for i in order {
                assert_eq!(expected[i], restored.query(points[i]));
            }
        }
        assert!(PreparedField::from_asset_bytes(&bytes[..bytes.len() - 1]).is_err());
    }
    #[test]
    fn rk_intermediate_and_distinct_weighted_endpoint_refuse_transactionally() {
        let complete =
            FrozenRollout::new(constant(None), profile(Integrator::Rk4), vec![]).unwrap();
        let mut state = complete.initial_state([0.4, 0.0], [0.2, 0.0]).unwrap();
        complete.attempt_step(&mut state, [0.0; 2]).unwrap();
        assert!((state.position[0] - 0.43296).abs() < 1e-8);
        for (missing, phase) in [(0.416, "RK stage 3"), (0.43296, "Final endpoint")] {
            let rollout =
                FrozenRollout::new(constant(Some(missing)), profile(Integrator::Rk4), vec![])
                    .unwrap();
            let mut state = rollout.initial_state([0.4, 0.0], [0.2, 0.0]).unwrap();
            state.mechanical_time = 7.0;
            state.integrated_work = 3.0;
            let before = serde_json::to_vec(&state).unwrap();
            let error = rollout.attempt_step(&mut state, [0.0; 2]).unwrap_err();
            assert!(error.contains(phase), "{error}");
            assert!(error.contains("GeometryUnavailable"), "{error}");
            assert_eq!(before, serde_json::to_vec(&state).unwrap());
        }
    }
    #[test]
    fn full_differential_stencil_is_required_even_when_central_value_exists() {
        let f = constant(Some(0.42));
        assert!(f.query([0.420015, 0.0]).is_err());
        assert!(f.query([0.43, 0.0]).is_ok());
    }
    #[test]
    fn pinned_profile_and_field_replacement_are_rejected() {
        let rollout =
            FrozenRollout::new(constant(None), profile(Integrator::Euler), vec![]).unwrap();
        assert!(rollout
            .evaluate(|_| super::super::try_activate(None))
            .unwrap_err()
            .contains("FrozenIdentity"));
        let error = rollout
            .evaluate(|config| {
                let mut changed = config.clone();
                changed.lambda_sq = 1.0;
                manifold::total_energy((0.1, 0.0), Complex64::new(0.4, 0.0), &changed)
            })
            .unwrap_err();
        assert!(error.contains("FrozenIdentity"), "{error}");
        for (beta, dt) in [(0.1, 0.2), (2.0, 0.1)] {
            let error = rollout
                .evaluate(|config| {
                    manifold::integrate_step(
                        Complex64::new(0.4, 0.0),
                        (0.01, 0.0),
                        (0.0, 0.0),
                        beta,
                        dt,
                        config,
                    )
                })
                .unwrap_err();
            assert!(error.contains("FrozenIdentity"), "{error}");
        }
    }
    #[test]
    fn authoritative_reference_refusal_precedes_smooth_operational_sample() {
        let field = constant(None);
        assert!(field.query([0.4, 0.0]).is_ok());
        let cuts = vec![ReferenceCut {
            bounds: [0.399, -0.00001, 0.401, 0.00001],
            authority: "synthetic analytic reference-control cut; not a Mandelbrot certificate"
                .into(),
        }];
        let rollout = FrozenRollout::new(field, profile(Integrator::Euler), cuts).unwrap();
        assert!(rollout
            .initial_state([0.4, 0.0], [0.2, 0.0])
            .unwrap_err()
            .contains("ReferenceSingularity"));
    }

    #[test]
    fn canonical_reference_singularity_is_not_overridden_by_prepared_field() {
        let point = Complex64::new(-1.0, 0.0);
        let reference = crate::geometry_provider::query_geometry(point, 1e-4).unwrap();
        assert_eq!(
            reference.validity,
            crate::geometry_provider::GeometryValidity::Singular
        );
        let field = Arc::new(
            PreparedField::new([-1.5, -0.5], 0.01, [101; 2], vec![1.0; 101 * 101]).unwrap(),
        );
        assert!(field.query([-1.0, 0.0]).is_ok());
        let cuts = vec![ReferenceCut {
            bounds: [-1.0, 0.0, -1.0, 0.0],
            authority: format!(
                "current canonical reference refusal: {} {:?}",
                reference.provider_version, reference.singularity
            ),
        }];
        let rollout = FrozenRollout::new(field, profile(Integrator::Euler), cuts).unwrap();
        assert!(rollout
            .initial_state([-1.0, 0.0], [0.0; 2])
            .unwrap_err()
            .contains("ReferenceSingularity"));
    }
    #[test]
    fn plane_control_reaches_regularization_metric_connection_and_integration() {
        let mut c = Vec::new();
        for _ in 0..101 {
            for x in 0..101 {
                c.push(-0.5 + x as f64 * 0.01);
            }
        }
        let field = Arc::new(PreparedField::new([-0.5; 2], 0.01, [101; 2], c).unwrap());
        let config = ManifoldConfig {
            epsilon: 0.02,
            lambda_sq: 0.7,
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        let rollout = FrozenRollout::new(
            field,
            PhysicsProfile::new(config.clone(), 0.0001, 0.0, Integrator::Rk4).unwrap(),
            vec![],
        )
        .unwrap();
        for x in [-0.07, -0.02, 0.0, 0.02, 0.07] {
            rollout
                .evaluate(|config| {
                    use crate::geometry_provider::{
                        grad_rho_from_jet, hessian_rho_from_jet, query_geometry, rho_from_jet,
                    };
                    let p = Complex64::new(x, 0.0);
                    let jet = query_geometry(p, config.epsilon)?;
                    let r = x.hypot(config.epsilon);
                    assert!((rho_from_jet(&jet, config.epsilon) - r).abs() < 1e-12);
                    let grad = grad_rho_from_jet(&jet, config.epsilon);
                    assert!((grad[0] - x / r).abs() < 1e-10 && grad[1].abs() < 1e-10);
                    let h = hessian_rho_from_jet(&jet, config.epsilon);
                    assert!((r * h[0][0] - config.epsilon.powi(2) / r.powi(2)).abs() < 1e-9);
                    assert!((r * h[0][1]).abs() < 1e-9 && (r * h[1][1]).abs() < 1e-9);
                    let a = 1.0 / r.powi(2)
                        + config.lambda_sq * x * x / (r.powi(4) * std::f64::consts::LN_2.powi(2));
                    let b = 1.0 / r.powi(2);
                    let da = -2.0 * x / r.powi(4)
                        + config.lambda_sq / std::f64::consts::LN_2.powi(2)
                            * (2.0 * x / r.powi(4) - 4.0 * x.powi(3) / r.powi(6));
                    let db = -2.0 * x / r.powi(4);
                    let g = manifold::induced_metric(p, config)?;
                    assert!((g[0][0] - a).abs() < 1e-8 * a && (g[1][1] - b).abs() < 1e-8 * b);
                    let gamma = manifold::christoffel_symbols(p, config)?;
                    let expected = [
                        [[da / (2.0 * a), 0.0], [0.0, -db / (2.0 * a)]],
                        [[0.0, db / (2.0 * b)], [db / (2.0 * b), 0.0]],
                    ];
                    for i in 0..2 {
                        for j in 0..2 {
                            for k in 0..2 {
                                assert!(
                                    (gamma[i][j][k] - expected[i][j][k]).abs()
                                        < 1e-4 * expected[i][j][k].abs().max(1.0)
                                );
                            }
                        }
                    }
                    Ok(())
                })
                .unwrap();
            let mut state = rollout.initial_state([x, 0.0], [0.01, 0.0]).unwrap();
            rollout.attempt_step(&mut state, [0.0; 2]).unwrap();
            assert!(state.energy.is_finite() && state.mechanical_time == 0.0001);
        }
    }
}
