//! Thin experimental bindings. All field/mechanical math stays in Rust core.
use runtime_core::operational_field::{
    acceptance::{
        frozen_experiment_report, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
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
        let profile = PhysicsProfile::new(
            runtime_core::manifold::ManifoldConfig::default(),
            0.001,
            0.0,
            Integrator::Euler,
        )
        .map_err(|e| JsValue::from_str(&e))?;
        let rollout = FrozenRollout::new(field.clone(), profile, vec![])
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(Self {
            field,
            rollout,
            state: None,
        })
    }
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
        serde_json::to_string(
            &frozen_experiment_report(self.field.clone()).map_err(|e| JsValue::from_str(&e))?,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
