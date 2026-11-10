//! Explicit opt-in binding for a frozen experimental rollout.
use crate::manifold::ManifoldConfig;
use crate::operational_engine::{
    reference_policy_for_field, FrozenRollout, Integrator, MechanicalState, PhysicsProfile,
};
use crate::operational_field::PreparedField;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(unsendable, name = "BoundedOperationalRollout")]
pub struct BoundedOperationalRollout {
    rollout: FrozenRollout,
    state: Option<MechanicalState>,
}

#[pymethods]
impl BoundedOperationalRollout {
    #[staticmethod]
    fn asset_field_id(asset: &[u8]) -> PyResult<String> {
        let field = PreparedField::from_asset_bytes(asset).map_err(PyValueError::new_err)?;
        Ok(format!("{:016x}", field.identity()))
    }

    fn field_id(&self) -> String {
        format!("{:016x}", self.rollout.field_id())
    }

    fn physics_profile_id(&self) -> String {
        format!("{:016x}", self.rollout.profile().id())
    }

    fn probe(&self, x: f64, y: f64) -> PyResult<String> {
        let result = self
            .rollout
            .probe([x, y])
            .map_err(PyRuntimeError::new_err)?;
        serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[new]
    #[pyo3(signature = (asset, reference_policy, d_ref, epsilon, lambda_sq, kappa, mu, dt, beta))]
    fn new(
        asset: &[u8],
        reference_policy: &str,
        d_ref: f64,
        epsilon: f64,
        lambda_sq: f64,
        kappa: f64,
        mu: f64,
        dt: f64,
        beta: f64,
    ) -> PyResult<Self> {
        let field =
            Arc::new(PreparedField::from_asset_bytes(asset).map_err(PyValueError::new_err)?);
        let cuts =
            reference_policy_for_field(&field, reference_policy).map_err(PyValueError::new_err)?;
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
        .map_err(PyValueError::new_err)?;
        let rollout = FrozenRollout::new(field, profile, cuts).map_err(PyValueError::new_err)?;
        Ok(Self {
            rollout,
            state: None,
        })
    }

    fn start(&mut self, x: f64, y: f64, vx: f64, vy: f64) -> PyResult<String> {
        let state = self
            .rollout
            .initial_state([x, y], [vx, vy])
            .map_err(PyRuntimeError::new_err)?;
        let json =
            serde_json::to_string(&state).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.state = Some(state);
        Ok(json)
    }

    fn step(&mut self, qx: f64, qy: f64) -> PyResult<String> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("rollout not started"))?;
        self.rollout
            .attempt_step(state, [qx, qy])
            .map_err(PyRuntimeError::new_err)?;
        serde_json::to_string(state).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn state(&self) -> PyResult<String> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("rollout not started"))?;
        serde_json::to_string(state).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
