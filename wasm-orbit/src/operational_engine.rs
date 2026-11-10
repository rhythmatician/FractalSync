use runtime_core::manifold::ManifoldConfig;
use runtime_core::operational_engine::{
    reference_policy_for_field, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
};
use runtime_core::operational_field::PreparedField;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BoundedOperationalRollout {
    rollout: FrozenRollout,
    state: Option<MechanicalState>,
}

#[wasm_bindgen]
impl BoundedOperationalRollout {
    pub fn asset_field_id(asset: &[u8]) -> Result<String, JsValue> {
        let field = PreparedField::from_asset_bytes(asset).map_err(JsValue::from_str)?;
        Ok(format!("{:016x}", field.identity()))
    }

    pub fn field_id(&self) -> String {
        format!("{:016x}", self.rollout.field_id())
    }

    pub fn physics_profile_id(&self) -> String {
        format!("{:016x}", self.rollout.profile().id())
    }

    pub fn probe(&self, x: f64, y: f64) -> Result<String, JsValue> {
        let result = self
            .rollout
            .probe([x, y])
            .map_err(|e| JsValue::from_str(&e))?;
        serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(constructor)]
    pub fn new(
        asset: &[u8],
        reference_policy: &str,
        d_ref: f64,
        epsilon: f64,
        lambda_sq: f64,
        kappa: f64,
        mu: f64,
        dt: f64,
        beta: f64,
    ) -> Result<BoundedOperationalRollout, JsValue> {
        let field = Arc::new(PreparedField::from_asset_bytes(asset).map_err(JsValue::from_str)?);
        let cuts = reference_policy_for_field(&field, reference_policy)
            .map_err(|e| JsValue::from_str(&e))?;
        let profile = PhysicsProfile::new(
            ManifoldConfig {
                d_ref,
                epsilon,
                lambda_sq,
                kappa,
                mu,
            },
            dt,
            beta,
            Integrator::Rk4FourSubsteps,
        )
        .map_err(|e| JsValue::from_str(&e))?;
        let rollout =
            FrozenRollout::new(field, profile, cuts).map_err(|e| JsValue::from_str(&e))?;
        Ok(Self {
            rollout,
            state: None,
        })
    }

    pub fn start(&mut self, x: f64, y: f64, vx: f64, vy: f64) -> Result<String, JsValue> {
        let state = self
            .rollout
            .initial_state([x, y], [vx, vy])
            .map_err(|e| JsValue::from_str(&e))?;
        let json = serde_json::to_string(&state).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.state = Some(state);
        Ok(json)
    }

    pub fn step(&mut self, qx: f64, qy: f64) -> Result<String, JsValue> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("rollout not started"))?;
        self.rollout
            .attempt_step(state, [qx, qy])
            .map_err(|e| JsValue::from_str(&e))?;
        serde_json::to_string(state).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn state(&self) -> Result<String, JsValue> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| JsValue::from_str("rollout not started"))?;
        serde_json::to_string(state).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
