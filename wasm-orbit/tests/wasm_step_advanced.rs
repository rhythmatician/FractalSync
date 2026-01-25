use wasm_bindgen_test::*;

use orbit_synth_wasm::OrbitState;
use orbit_synth_wasm::ResidualParams;
use orbit_synth_wasm::DistanceField;
use wasm_bindgen::JsValue;

#[wasm_bindgen_test]
fn smoke_step_advanced_with_distance_field() {
    // Construct a small DF
    let resolution = 3usize;
    let field = vec![
        1.0_f32, 1.0_f32, 1.0_f32,
        1.0_f32, 0.0_f32, 1.0_f32,
        1.0_f32, 1.0_f32, 1.0_f32,
    ];
    let df = DistanceField::new(field, resolution, -1.0, 1.0, -1.0, 1.0, 1.0, 0.1);

    // Create an orbit state and ensure step_advanced does not panic when passed a JS-side DF wrapper
    let mut st = OrbitState::new(0, 0, 0.0, 0.0, 0.0, 0.0, 3, 1.0);
    let params = ResidualParams::new(3, 1.0, 1.0);

    // Convert the DF to a JsValue (this mirrors the JS wrapper object passed by frontend)
    let js_df = JsValue::from(df);

    // If this call panics it will fail the test. This mirrors the JS call path that previously attempted ownership
    let _ = st.step_advanced(0.01, &params, None, 0.0, None, None, Some(js_df));
}
