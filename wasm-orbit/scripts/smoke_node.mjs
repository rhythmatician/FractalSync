async function main() {
    const pkg = await import('../pkg/orbit_synth_wasm.js');
    const { OrbitState, ResidualParams, DistanceField } = pkg;

    const resolution = 3;
    const field = new Float32Array([1, 1, 1, 1, 0, 1, 1, 1, 1]);
    const df = new DistanceField(field, resolution, -1.0, 1.0, -1.0, 1.0, 1.0, 0.1);

    const st = OrbitState.newWithSeed ? OrbitState.newWithSeed(0, 0, 0.0, 0.0, 0.0, 0.0, 3, 1.0, 0n) : new OrbitState(0, 0, 0.0, 0.0, 0.0, 0.0, 3, 1.0);
    const params = new ResidualParams(3, 1.0, 1.0);

    try {
        const c = st.step_advanced(0.01, params, undefined, 0.0, undefined, undefined, df);
        console.log('step_advanced returned', c);
    } catch (e) {
        console.error('smoke test failed:', e);
        process.exit(2);
    }

    console.log('smoke test OK');
}

main().catch(e => { console.error(e); process.exit(1); });