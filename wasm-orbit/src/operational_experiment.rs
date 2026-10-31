//! Thin experimental bindings. All field/mechanical math stays in Rust core.
use runtime_core::operational_field::{
    acceptance::{
        frozen_experiment_report_with_integrator, reference_policy_for_field, FrozenRollout,
        Integrator, MechanicalState, PhysicsProfile, ReferenceCut, EXPERIMENT_MANIFEST,
    },
    PreparedField,
};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ExperimentalOperationalField {
    field: Arc<PreparedField>,
    rollout: FrozenRollout,
    state: Option<MechanicalState>,
}

#[wasm_bindgen]
impl ExperimentalOperationalField {
    #[wasm_bindgen(constructor)]
    pub fn new(asset: &[u8]) -> Result<ExperimentalOperationalField, JsValue> {
        let field = Arc::new(PreparedField::from_asset_bytes(asset).map_err(JsValue::from_str)?);
        Self::from_parts(field, vec![], Integrator::Euler)
    }
    #[wasm_bindgen(js_name=withPolicy)]
    pub fn with_policy(
        asset: &[u8],
        policy: &str,
    ) -> Result<ExperimentalOperationalField, JsValue> {
        Self::with_policy_and_integrator(asset, policy, "Euler")
    }
    #[wasm_bindgen(js_name=withPolicyAndIntegrator)]
    pub fn with_policy_and_integrator(
        asset: &[u8],
        policy: &str,
        integrator: &str,
    ) -> Result<ExperimentalOperationalField, JsValue> {
        let method = Integrator::parse_name(integrator).map_err(|e| JsValue::from_str(&e))?;
        let field = Arc::new(PreparedField::from_asset_bytes(asset).map_err(JsValue::from_str)?);
        let cuts = reference_policy_for_field(&field, policy).map_err(|e| JsValue::from_str(&e))?;
        Self::from_parts(field, cuts, method)
    }
    pub fn profile_json(&self) -> String {
        self.rollout.profile().manifest().to_string()
    }
    pub fn state_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.state).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl ExperimentalOperationalField {
    fn from_parts(
        field: Arc<PreparedField>,
        cuts: Vec<ReferenceCut>,
        method: Integrator,
    ) -> Result<Self, JsValue> {
        let profile = PhysicsProfile::new(
            runtime_core::manifold::ManifoldConfig::default(),
            0.001,
            0.0,
            method,
        )
        .map_err(|e| JsValue::from_str(&e))?;
        let rollout =
            FrozenRollout::new(field.clone(), profile, cuts).map_err(|e| JsValue::from_str(&e))?;
        Ok(Self {
            field,
            rollout,
            state: None,
        })
    }
}

#[wasm_bindgen]
impl ExperimentalOperationalField {
    pub fn identity(&self) -> String {
        format!("{:016x}", self.field.identity())
    }
    pub fn query(&self, x: f64, y: f64) -> Vec<f64> {
        match self.field.query([x, y]) {
            Ok(j) => vec![
                j.value,
                j.gradient[0],
                j.gradient[1],
                j.hessian[0][0],
                j.hessian[0][1],
                j.hessian[1][1],
            ],
            Err(_) => Vec::new(),
        }
    }
    pub fn reset(&mut self, x: f64, y: f64, vx: f64, vy: f64) -> Result<(), JsValue> {
        self.state = Some(
            self.rollout
                .initial_state([x, y], [vx, vy])
                .map_err(|e| JsValue::from_str(&e))?,
        );
        Ok(())
    }
    pub fn step(&mut self) -> Result<String, JsValue> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("explicit initial state required"))?;
        let counts = self
            .rollout
            .attempt_step(state, [0.0; 2])
            .map_err(|e| JsValue::from_str(&e))?;
        serde_json::to_string(&serde_json::json!({"state":state,"work":counts}))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    pub fn report(&self) -> Result<String, JsValue> {
        self.report_with_manifest(EXPERIMENT_MANIFEST)
    }
    pub fn report_with_manifest(&self, manifest: &str) -> Result<String, JsValue> {
        serde_json::to_string(
            &frozen_experiment_report_with_integrator(
                self.field.clone(),
                manifest,
                self.rollout.reference_cuts(),
                self.rollout.profile().integrator(),
            )
            .map_err(|e| JsValue::from_str(&e))?,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
