use crate::manifold::{self, ManifoldConfig};
use crate::operational_field::PreparedField;
use num_complex::Complex64;
use serde::Serialize;
use std::cell::RefCell;
use std::sync::Arc;

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

struct Context {
    profile: PhysicsProfile,
    cuts: Arc<[ReferenceCut]>,
    queries: usize,
}
thread_local! { static CONTEXT:RefCell<Option<Context>>=const{RefCell::new(None)}; }
pub(super) fn is_pinned() -> bool {
    CONTEXT.with(|c| c.borrow().is_some())
}

pub(super) fn check_query(c: Complex64, epsilon: f64) -> Result<(), String> {
    CONTEXT.with(|context| {
        if let Some(context) = context.borrow_mut().as_mut() {
            context.queries += 1;
            if epsilon.to_bits() != context.profile.config.epsilon.to_bits() {
                return Err("FrozenIdentity: epsilon changed".into());
            }
            // Retain the canonical exact-point refusals even if an asset's
            // additional reference snapshot omits them. Other cut coverage is
            // explicitly unknown unless supplied by that frozen snapshot.
            if (c.re == -1.0 || c.re == 0.25) && c.im == 0.0 {
                return Err("ReferenceSingularity: canonical cut or cusp".into());
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
        profile.id = fingerprint([profile.id, 2].into_iter().chain(evidence));
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

    /// Evaluate the frozen field and downstream mechanics without changing state.
    /// The scalar is operational; no reference-distance accuracy is implied.
    pub fn probe(&self, position: [f64; 2]) -> Result<serde_json::Value, String> {
        let c = Complex64::new(position[0], position[1]);
        let ((d, metric, potential), _) = self.evaluate(|config| {
            let jet = crate::geometry_provider::query_geometry(c, config.epsilon)?;
            Ok((
                jet.d,
                manifold::induced_metric(c, config)?,
                manifold::potential_energy(c, config)?,
            ))
        })?;
        Ok(
            serde_json::json!({"field_id":format!("{:016x}", self.field_id()), "physics_profile_id":format!("{:016x}", self.profile.id()), "operational_value":d, "metric":metric, "potential":potential, "reference_distance_accuracy":"unknown"}),
        )
    }

    /// Convert the existing Controls drive to a force under this frozen field.
    pub fn drive_covector(
        &self,
        position: [f64; 2],
        controls: &crate::controls::MotionControls,
    ) -> Result<[f64; 2], String> {
        let (q, _) = self.evaluate(|config| {
            controls.drive_covector(Complex64::new(position[0], position[1]), config)
        })?;
        Ok([q.0, q.1])
    }

    fn evaluate<T>(
        &self,
        f: impl FnOnce(&ManifoldConfig) -> Result<T, String>,
    ) -> Result<(T, QueryWork), String> {
        if is_pinned() {
            return Err("FrozenIdentity: nested rollout activation".into());
        }
        let previous =
            crate::operational_field::ACTIVE.with(|a| a.replace(Some(self.field.clone())));
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
                crate::operational_field::ACTIVE.with(|a| *a.borrow_mut() = self.0.take());
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
    fn canonical_reference_cusp_and_cut_cannot_be_omitted_from_policy() {
        let field = Arc::new(
            PreparedField::new([-1.5, -0.5], 0.01, [201, 101], vec![1.0; 201 * 101]).unwrap(),
        );
        let rollout = FrozenRollout::new(field, profile(Integrator::Rk4), vec![]).unwrap();
        for point in [[-1.0, 0.0], [0.25, 0.0]] {
            assert!(rollout
                .initial_state(point, [0.0; 2])
                .unwrap_err()
                .contains("ReferenceSingularity"));
        }
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
