/* @ts-self-types="./orbit_synth_wasm.d.ts" */

/**
 * Authoritative sample-clock audio timebase (issue #91).
 *
 * The browser's AudioWorklet transport feeds non-overlapping PCM blocks
 * here; the Rust timebase validates monotonicity, resamples statefully to
 * the canonical 48 kHz timeline, schedules exact 1024-sample hops, and runs
 * the canonical FeatureExtractor — all in Rust, so there is no TypeScript
 * mirror of the timing/scheduling math (ADR 0001).
 */
export class AnalysisTimebase {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AnalysisTimebaseFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_analysistimebase_free(ptr, 0);
    }
    /**
     * Diagnostic snapshot for verifying the clock manually.
     * See ``ingest`` for the typing note about the .d.ts return type.
     * @returns {any}
     */
    diagnostics() {
        const ret = wasm.analysistimebase_diagnostics(this.__wbg_ptr);
        return ret;
    }
    /**
     * Flush end-of-stream (recovers the deferred final sample/tick).
     * See ``ingest`` for the typing note about the .d.ts return type.
     * @returns {any}
     */
    flush() {
        const ret = wasm.analysistimebase_flush(this.__wbg_ptr);
        return ret;
    }
    /**
     * Ingest one non-overlapping PCM block. Returns a JS array of ticks
     * (possibly empty). Throws on overlap / mid-stream rate change.
     *
     * Type note: the generated .d.ts types this as `any` because the
     * function returns a `JsValue` (it serializes via
     * ``serde_wasm_bindgen``). The TS adapter in
     * ``frontend/src/lib/analysisTimebase.ts`` re-types this signature
     * as ``AnalysisTick[]`` and the ``AnalysisTick`` interface itself
     * is provided by the ``TS_TYPES`` custom section above — so the
     * Rust source remains the single authority for the wire shape.
     * @param {Float32Array} samples
     * @param {number} source_sample_rate
     * @param {bigint} source_start_frame
     * @returns {any}
     */
    ingest(samples, source_sample_rate, source_start_frame) {
        const ptr0 = passArrayF32ToWasm0(samples, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.analysistimebase_ingest(this.__wbg_ptr, ptr0, len0, source_sample_rate, source_start_frame);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    constructor() {
        const ret = wasm.analysistimebase_new();
        this.__wbg_ptr = ret >>> 0;
        AnalysisTimebaseFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Declare a stream discontinuity. `reason` is informational; the epoch
     * always bumps and the schedule resets.
     */
    reset() {
        wasm.analysistimebase_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) AnalysisTimebase.prototype[Symbol.dispose] = AnalysisTimebase.prototype.free;

export class ColorIntent {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ColorIntent.prototype);
        obj.__wbg_ptr = ptr;
        ColorIntentFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ColorIntentFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_colorintent_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get accent_weight() {
        const ret = wasm.colorintent_accent_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get anchor_hue() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get chroma() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get harmony() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.colorintent_harmony(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get lightness() {
        const ret = wasm.colorintent_lightness(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} anchor_hue
     * @param {number} chroma
     * @param {number} lightness
     * @param {string} harmony
     * @param {number} accent_weight
     */
    constructor(anchor_hue, chroma, lightness, harmony, accent_weight) {
        const ptr0 = passStringToWasm0(harmony, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.colorintent_new(anchor_hue, chroma, lightness, ptr0, len0, accent_weight);
        this.__wbg_ptr = ret >>> 0;
        ColorIntentFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} v
     */
    set accent_weight(v) {
        wasm.colorintent_set_accent_weight(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set anchor_hue(v) {
        wasm.colorintent_set_anchor_hue(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set chroma(v) {
        wasm.colorintent_set_chroma(this.__wbg_ptr, v);
    }
    /**
     * @param {string} v
     */
    set harmony(v) {
        const ptr0 = passStringToWasm0(v, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.colorintent_set_harmony(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} v
     */
    set lightness(v) {
        wasm.colorintent_set_lightness(this.__wbg_ptr, v);
    }
}
if (Symbol.dispose) ColorIntent.prototype[Symbol.dispose] = ColorIntent.prototype.free;

/**
 * Wrapper for complex number to/from JavaScript
 */
export class Complex {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Complex.prototype);
        obj.__wbg_ptr = ptr;
        ComplexFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ComplexFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_complex_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get imag() {
        const ret = wasm.__wbg_get_complex_imag(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get real() {
        const ret = wasm.__wbg_get_complex_real(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set imag(arg0) {
        wasm.__wbg_set_complex_imag(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set real(arg0) {
        wasm.__wbg_set_complex_real(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Complex.prototype[Symbol.dispose] = Complex.prototype.free;

export class ControlsV2 {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ControlsV2.prototype);
        obj.__wbg_ptr = ptr;
        ControlsV2Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ControlsV2Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_controlsv2_free(ptr, 0);
    }
    /**
     * @returns {ControlsV2}
     */
    clamped() {
        const ret = wasm.controlsv2_clamped(this.__wbg_ptr);
        return ControlsV2.__wrap(ret);
    }
    /**
     * @param {Float64Array} output
     * @returns {ControlsV2}
     */
    static fromModelOutput(output) {
        const ptr0 = passArrayF64ToWasm0(output, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.controlsv2_fromModelOutput(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ControlsV2.__wrap(ret[0]);
    }
    /**
     * @returns {any[]}
     */
    static modelOutputOrder() {
        const ret = wasm.controlsv2_modelOutputOrder();
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {MotionControls}
     */
    get motion() {
        const ret = wasm.controlsv2_motion(this.__wbg_ptr);
        return MotionControls.__wrap(ret);
    }
    /**
     * @param {MotionControls} motion
     * @param {JuliaViewControls} view
     */
    constructor(motion, view) {
        _assertClass(motion, MotionControls);
        var ptr0 = motion.__destroy_into_raw();
        _assertClass(view, JuliaViewControls);
        var ptr1 = view.__destroy_into_raw();
        const ret = wasm.controlsv2_new(ptr0, ptr1);
        this.__wbg_ptr = ret >>> 0;
        ControlsV2Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {MotionControls} v
     */
    set motion(v) {
        _assertClass(v, MotionControls);
        var ptr0 = v.__destroy_into_raw();
        wasm.controlsv2_set_motion(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {JuliaViewControls} v
     */
    set view(v) {
        _assertClass(v, JuliaViewControls);
        var ptr0 = v.__destroy_into_raw();
        wasm.controlsv2_set_view(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Float64Array}
     */
    to_model_output() {
        const ret = wasm.controlsv2_to_model_output(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {JuliaViewControls}
     */
    get view() {
        const ret = wasm.controlsv2_view(this.__wbg_ptr);
        return JuliaViewControls.__wrap(ret);
    }
}
if (Symbol.dispose) ControlsV2.prototype[Symbol.dispose] = ControlsV2.prototype.free;

/**
 * Canonical observed-ridge CycleBank (issue #92), browser surface.
 *
 * The browser feeds one canonical `AnalysisTick` per authoritative hop and
 * reads the currently observed modes / relations. It never interprets the
 * rolling feature window's offsets itself.
 */
export class CycleBank {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CycleBankFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_cyclebank_free(ptr, 0);
    }
    /**
     * Rational relations among the currently observed modes (latest batch).
     * @returns {any}
     */
    latest_relations() {
        const ret = wasm.cyclebank_latest_relations(this.__wbg_ptr);
        return ret;
    }
    /**
     * Current confirmed observed modes (`CycleMode[]`).
     * @returns {any}
     */
    modes() {
        const ret = wasm.cyclebank_modes(this.__wbg_ptr);
        return ret;
    }
    /**
     * Construct with the canonical defaults (no config). Config overrides
     * are a Rust-side concern; the browser runs the canonical pipeline.
     */
    constructor() {
        const ret = wasm.cyclebank_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        CycleBankFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Number of currently confirmed modes.
     * @returns {number}
     */
    num_modes() {
        const ret = wasm.cyclebank_num_modes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Feed one explicit observation of named scalar evidence channels
     * (diagnostic entry point; the production path is `observe_tick`).
     * @param {bigint} sample_index
     * @param {number} dt_seconds
     * @param {bigint} stream_epoch
     * @param {any} channels
     * @returns {any}
     */
    observe(sample_index, dt_seconds, stream_epoch, channels) {
        const ret = wasm.cyclebank_observe(this.__wbg_ptr, sample_index, dt_seconds, stream_epoch, channels);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Feed one canonical analysis tick (the `AnalysisTick` produced by the
     * wasm `AnalysisTimebase.ingest`/`flush`). Returns the current observed
     * `CycleMode[]`. The newest-frame extraction is done in Rust.
     * @param {any} tick
     * @returns {any}
     */
    observe_tick(tick) {
        const ret = wasm.cyclebank_observe_tick(this.__wbg_ptr, tick);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Deterministic discontinuity reset.
     */
    reset() {
        wasm.cyclebank_reset(this.__wbg_ptr);
    }
    /**
     * The Rust-owned contract version (`CYCLE_BANK_VERSION`).
     * @returns {string}
     */
    get version() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.cyclebank_version(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) CycleBank.prototype[Symbol.dispose] = CycleBank.prototype.free;

/**
 * Audio feature extractor — the SAME Rust implementation the trainer uses.
 *
 * The browser feeds a rolling window of PCM samples (from
 * AnalyserNode.getFloatTimeDomainData) and receives feature windows laid
 * out identically to training inputs. This eliminates the entire class of
 * browser-vs-trainer extraction drift (FFT size, smoothing, dB conversion,
 * per-file normalization, window layout) by construction: there is only
 * one implementation, executed in two places.
 */
export class FeatureExtractor {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FeatureExtractorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_featureextractor_free(ptr, 0);
    }
    /**
     * Extract the MOST RECENT flattened feature window from `audio`
     * (frame-major).
     *
     * `audio` is the rolling PCM history in chronological order; the
     * returned window covers the latest `window_frames` STFT frames,
     * matching what live inference needs. Short input is padded by
     * repeating the last frame, matching training behavior for short
     * files.
     * @param {Float32Array} audio
     * @param {number} window_frames
     * @returns {Float64Array}
     */
    extract_window(audio, window_frames) {
        const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.featureextractor_extract_window(this.__wbg_ptr, ptr0, len0, window_frames);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * Create an extractor with the shared runtime defaults (48 kHz,
     * hop 1024, n_fft 4096). Callers must resample browser audio to the
     * runtime sample rate before feeding PCM here.
     */
    constructor() {
        const ret = wasm.featureextractor_new();
        this.__wbg_ptr = ret >>> 0;
        FeatureExtractorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get num_features_per_frame() {
        const ret = wasm.featureextractor_num_features_per_frame(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) FeatureExtractor.prototype[Symbol.dispose] = FeatureExtractor.prototype.free;

export class JuliaViewControls {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JuliaViewControls.prototype);
        obj.__wbg_ptr = ptr;
        JuliaViewControlsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JuliaViewControlsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_juliaviewcontrols_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get accent_delta() {
        const ret = wasm.juliaviewcontrols_accent_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get chroma_delta() {
        const ret = wasm.colorintent_accent_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {JuliaViewControls}
     */
    clamped() {
        const ret = wasm.juliaviewcontrols_clamped(this.__wbg_ptr);
        return JuliaViewControls.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get harmony_shift() {
        const ret = wasm.juliaviewcontrols_harmony_shift(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get hue_delta() {
        const ret = wasm.colorintent_lightness(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get lightness_delta() {
        const ret = wasm.juliaviewcontrols_lightness_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} zoom_delta
     * @param {number} rotation_delta
     * @param {number} hue_delta
     * @param {number} chroma_delta
     * @param {number} lightness_delta
     * @param {number} accent_delta
     * @param {number} harmony_shift
     */
    constructor(zoom_delta, rotation_delta, hue_delta, chroma_delta, lightness_delta, accent_delta, harmony_shift) {
        const ret = wasm.juliaviewcontrols_new(zoom_delta, rotation_delta, hue_delta, chroma_delta, lightness_delta, accent_delta, harmony_shift);
        this.__wbg_ptr = ret >>> 0;
        JuliaViewControlsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get rotation_delta() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} v
     */
    set accent_delta(v) {
        wasm.juliaviewcontrols_set_accent_delta(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set chroma_delta(v) {
        wasm.colorintent_set_accent_weight(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set harmony_shift(v) {
        wasm.juliaviewcontrols_set_harmony_shift(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set hue_delta(v) {
        wasm.colorintent_set_lightness(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set lightness_delta(v) {
        wasm.juliaviewcontrols_set_lightness_delta(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set rotation_delta(v) {
        wasm.colorintent_set_chroma(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set zoom_delta(v) {
        wasm.colorintent_set_anchor_hue(this.__wbg_ptr, v);
    }
    /**
     * @returns {number}
     */
    get zoom_delta() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) JuliaViewControls.prototype[Symbol.dispose] = JuliaViewControls.prototype.free;

export class JuliaViewState {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JuliaViewState.prototype);
        obj.__wbg_ptr = ptr;
        JuliaViewStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JuliaViewStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_juliaviewstate_free(ptr, 0);
    }
    /**
     * @param {JuliaViewControls} controls
     */
    applyControls(controls) {
        _assertClass(controls, JuliaViewControls);
        wasm.juliaviewstate_applyControls(this.__wbg_ptr, controls.__wbg_ptr);
    }
    /**
     * @returns {JuliaViewState}
     */
    clamped() {
        const ret = wasm.juliaviewstate_clamped(this.__wbg_ptr);
        return JuliaViewState.__wrap(ret);
    }
    /**
     * @returns {ColorIntent}
     */
    get color() {
        const ret = wasm.juliaviewstate_color(this.__wbg_ptr);
        return ColorIntent.__wrap(ret);
    }
    /**
     * @returns {boolean}
     */
    get harmony_armed() {
        const ret = wasm.juliaviewstate_harmony_armed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get harmony_cooldown() {
        const ret = wasm.juliaviewstate_harmony_cooldown(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} zoom
     * @param {number} rotation
     * @param {ColorIntent | null | undefined} color
     * @param {number} harmony_cooldown
     * @param {boolean} harmony_armed
     */
    constructor(zoom, rotation, color, harmony_cooldown, harmony_armed) {
        let ptr0 = 0;
        if (!isLikeNone(color)) {
            _assertClass(color, ColorIntent);
            ptr0 = color.__destroy_into_raw();
        }
        const ret = wasm.juliaviewstate_new(zoom, rotation, ptr0, harmony_cooldown, harmony_armed);
        this.__wbg_ptr = ret >>> 0;
        JuliaViewStateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get rotation() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ColorIntent} v
     */
    set color(v) {
        _assertClass(v, ColorIntent);
        var ptr0 = v.__destroy_into_raw();
        wasm.juliaviewstate_set_color(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {boolean} v
     */
    set harmony_armed(v) {
        wasm.juliaviewstate_set_harmony_armed(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set harmony_cooldown(v) {
        wasm.juliaviewstate_set_harmony_cooldown(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set rotation(v) {
        wasm.colorintent_set_chroma(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set zoom(v) {
        wasm.colorintent_set_anchor_hue(this.__wbg_ptr, v);
    }
    /**
     * @returns {number}
     */
    get zoom() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) JuliaViewState.prototype[Symbol.dispose] = JuliaViewState.prototype.free;

/**
 * Manifold configuration for the browser (issue #106).
 */
export class ManifoldConfig {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ManifoldConfig.prototype);
        obj.__wbg_ptr = ptr;
        ManifoldConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ManifoldConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_manifoldconfig_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get d_ref() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
    /**
     * Use the same defaults as OrbitController, without a second browser authority.
     * @returns {ManifoldConfig}
     */
    static defaults() {
        const ret = wasm.manifoldconfig_defaults();
        return ManifoldConfig.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get epsilon() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get kappa() {
        const ret = wasm.colorintent_accent_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get lambda_sq() {
        const ret = wasm.colorintent_lightness(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get mu() {
        const ret = wasm.juliaviewcontrols_lightness_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} d_ref
     * @param {number} epsilon
     * @param {number} lambda_sq
     * @param {number} kappa
     * @param {number} mu
     */
    constructor(d_ref, epsilon, lambda_sq, kappa, mu) {
        const ret = wasm.manifoldconfig_new(d_ref, epsilon, lambda_sq, kappa, mu);
        this.__wbg_ptr = ret >>> 0;
        ManifoldConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ManifoldConfig.prototype[Symbol.dispose] = ManifoldConfig.prototype.free;

export class MotionControls {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MotionControls.prototype);
        obj.__wbg_ptr = ptr;
        MotionControlsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MotionControlsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_motioncontrols_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get brake() {
        const ret = wasm.colorintent_accent_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {MotionControls}
     */
    clamped() {
        const ret = wasm.motioncontrols_clamped(this.__wbg_ptr);
        return MotionControls.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get direction_x() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get direction_y() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    drive_magnitude() {
        const ret = wasm.motioncontrols_drive_magnitude(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    friction_beta() {
        const ret = wasm.motioncontrols_friction_beta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get grip() {
        const ret = wasm.juliaviewcontrols_lightness_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get impulse() {
        const ret = wasm.juliaviewcontrols_accent_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} direction_x
     * @param {number} direction_y
     * @param {number} throttle
     * @param {number} brake
     * @param {number} grip
     * @param {number} impulse
     */
    constructor(direction_x, direction_y, throttle, brake, grip, impulse) {
        const ret = wasm.motioncontrols_new(direction_x, direction_y, throttle, brake, grip, impulse);
        this.__wbg_ptr = ret >>> 0;
        MotionControlsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} v
     */
    set brake(v) {
        wasm.colorintent_set_accent_weight(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set direction_x(v) {
        wasm.colorintent_set_anchor_hue(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set direction_y(v) {
        wasm.colorintent_set_chroma(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set grip(v) {
        wasm.juliaviewcontrols_set_lightness_delta(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set impulse(v) {
        wasm.juliaviewcontrols_set_accent_delta(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set throttle(v) {
        wasm.colorintent_set_lightness(this.__wbg_ptr, v);
    }
    /**
     * @returns {number}
     */
    get throttle() {
        const ret = wasm.colorintent_lightness(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) MotionControls.prototype[Symbol.dispose] = MotionControls.prototype.free;

/**
 * --- May-proven OrbitController bindings (restored baseline) ---
 */
export class OrbitController {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OrbitControllerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_orbitcontroller_free(ptr, 0);
    }
    /**
     * Apply model-predicted control signals (s, alpha).
     * @param {number} s
     * @param {number} alpha
     */
    apply_controls(s, alpha) {
        wasm.orbitcontroller_apply_controls(this.__wbg_ptr, s, alpha);
    }
    /**
     * Authoritative player position c in the complex plane.
     * Read/write so test harnesses and the debug cockpit can seed a
     * non-default starting point (e.g. "approach from outside M" trajectories
     * that begin at a seahorse-basin c without paying the launch cost of
     * crossing the cardioid ridge).
     * @returns {Complex}
     */
    get c() {
        const ret = wasm.orbitcontroller_c(this.__wbg_ptr);
        return Complex.__wrap(ret);
    }
    /**
     * Read-only DebugSnapshot of the current authoritative state.
     * @returns {any}
     */
    debugSnapshot() {
        const ret = wasm.orbitcontroller_debugSnapshot(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get the current manifold configuration.
     * @returns {ManifoldConfig}
     */
    manifold_config() {
        const ret = wasm.orbitcontroller_manifold_config(this.__wbg_ptr);
        return ManifoldConfig.__wrap(ret);
    }
    /**
     * Get the drag coefficient for manifold physics.
     * @returns {number}
     */
    get manifold_drag() {
        const ret = wasm.orbitcontroller_manifold_drag(this.__wbg_ptr);
        return ret;
    }
    /**
     * The most recent manifold-physics failure, if any. When manifold mode is
     * selected and the integrator fails, the controller fails closed (holds
     * the last valid state) and records the error here.
     * @returns {string | undefined}
     */
    get manifold_error() {
        const ret = wasm.orbitcontroller_manifold_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Whether manifold physics is currently enabled.
     * @returns {boolean}
     */
    get manifold_physics() {
        const ret = wasm.orbitcontroller_manifold_physics(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} s
     * @param {number} alpha
     * @param {number} omega
     */
    constructor(s, alpha, omega) {
        const ret = wasm.orbitcontroller_new(s, alpha, omega);
        this.__wbg_ptr = ret >>> 0;
        OrbitControllerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Seed the authoritative player position from (re, im) parts. The next
     * step_with_controls call advances from this point. Parts (not a
     * Complex instance) so callers never need to construct wasm objects.
     * @param {number} re
     * @param {number} im
     */
    setC(re, im) {
        wasm.orbitcontroller_setC(this.__wbg_ptr, re, im);
    }
    /**
     * Seed the planar velocity from (vx, vy) parts. The next
     * step_with_controls call applies Q_drive and drag from this velocity.
     * @param {number} vx
     * @param {number} vy
     */
    setVelocity(vx, vy) {
        wasm.orbitcontroller_setVelocity(this.__wbg_ptr, vx, vy);
    }
    /**
     * Target shore proximity for the shore-bias servo.
     * @param {number} d_star
     */
    set d_star(d_star) {
        wasm.orbitcontroller_set_d_star(this.__wbg_ptr, d_star);
    }
    /**
     * Friction for momentum refinement (default 0.90).
     * @param {number} drag
     */
    set drag(drag) {
        wasm.orbitcontroller_set_drag(this.__wbg_ptr, drag);
    }
    /**
     * Audio energy in [0, 1]: raises the servo's target shore-proximity
     * (loud audio pulls c toward the Shore).
     * @param {number} energy
     */
    set energy(energy) {
        wasm.orbitcontroller_set_energy(this.__wbg_ptr, energy);
    }
    /**
     * Set the manifold configuration (used only when manifold_physics is on).
     * @param {ManifoldConfig} config
     */
    set_manifold_config(config) {
        _assertClass(config, ManifoldConfig);
        wasm.orbitcontroller_set_manifold_config(this.__wbg_ptr, config.__wbg_ptr);
    }
    /**
     * Set the drag coefficient for manifold physics (beta in Q_drag = -beta*G*v).
     * @param {number} drag
     */
    set manifold_drag(drag) {
        wasm.orbitcontroller_set_manifold_drag(this.__wbg_ptr, drag);
    }
    /**
     * Enable or disable manifold physics. When on, step() routes through a
     * LEGACY ADAPTER that translates the old (s, alpha, energy) servo into a
     * generalized force covector for the musically-ignorant manifold kernel.
     * Transitional; not destination Controls v2 (issue #107).
     * @param {boolean} on
     */
    set manifold_physics(on) {
        wasm.orbitcontroller_set_manifold_physics(this.__wbg_ptr, on);
    }
    /**
     * Max world-space step per frame for shore bias.
     * @param {number} max_step
     */
    set max_step(max_step) {
        wasm.orbitcontroller_set_max_step(this.__wbg_ptr, max_step);
    }
    /**
     * Refinement 1 toggle: momentum (persistent velocity + drag).
     * @param {boolean} on
     */
    set momentum(on) {
        wasm.orbitcontroller_set_momentum(this.__wbg_ptr, on);
    }
    /**
     * Refinement 2 toggle: shore bias via minimap contour stepping.
     * @param {boolean} on
     */
    set shore_bias(on) {
        wasm.orbitcontroller_set_shore_bias(this.__wbg_ptr, on);
    }
    /**
     * Audio thrust for momentum: sustained energy builds inertia.
     * @param {number} thrust
     */
    set thrust(thrust) {
        wasm.orbitcontroller_set_thrust(this.__wbg_ptr, thrust);
    }
    /**
     * Advance one frame; returns the new c. `h` is the transient signal
     * in [0, 1] — near 1 opens the Shore wall for boundary crossing.
     * @param {number} dt
     * @param {number} h
     * @param {Float64Array | null} [band_gates]
     * @returns {Complex}
     */
    step(dt, h, band_gates) {
        var ptr0 = isLikeNone(band_gates) ? 0 : passArrayF64ToWasm0(band_gates, wasm.__wbindgen_malloc);
        var len0 = WASM_VECTOR_LEN;
        const ret = wasm.orbitcontroller_step(this.__wbg_ptr, dt, h, ptr0, len0);
        return Complex.__wrap(ret);
    }
    /**
     * Destination manifold step driven by Controls v2 (issue #107/#106).
     * @param {number} dt
     * @param {MotionControls} motion
     * @returns {Complex}
     */
    stepWithControls(dt, motion) {
        _assertClass(motion, MotionControls);
        const ret = wasm.orbitcontroller_stepWithControls(this.__wbg_ptr, dt, motion.__wbg_ptr);
        return Complex.__wrap(ret);
    }
    /**
     * Wobble phase (diagnostic).
     * @returns {number}
     */
    get theta() {
        const ret = wasm.orbitcontroller_theta(this.__wbg_ptr);
        return ret;
    }
    /**
     * Authoritative planar velocity (vx, vy) used by the destination
     * manifold integrator.
     * @returns {Complex}
     */
    get velocity() {
        const ret = wasm.orbitcontroller_velocity(this.__wbg_ptr);
        return Complex.__wrap(ret);
    }
}
if (Symbol.dispose) OrbitController.prototype[Symbol.dispose] = OrbitController.prototype.free;

/**
 * Orbit state wrapper for WASM
 */
export class OrbitState {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OrbitState.prototype);
        obj.__wbg_ptr = ptr;
        OrbitStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OrbitStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_orbitstate_free(ptr, 0);
    }
    /**
     * Advance state by dt seconds
     * @param {number} dt
     */
    advance(dt) {
        wasm.orbitstate_advance(this.__wbg_ptr, dt);
    }
    /**
     * Get alpha (residual amplitude)
     * @returns {number}
     */
    get alpha() {
        const ret = wasm.colorintent_accent_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get lobe
     * @returns {number}
     */
    get lobe() {
        const ret = wasm.orbitstate_lobe(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create new orbit state with optional seed
     * @param {number} lobe
     * @param {number} sub_lobe
     * @param {number} theta
     * @param {number} omega
     * @param {number} s
     * @param {number} alpha
     * @param {number} k_residuals
     * @param {number} residual_omega_scale
     * @param {bigint | null} [seed]
     */
    constructor(lobe, sub_lobe, theta, omega, s, alpha, k_residuals, residual_omega_scale, seed) {
        const ret = wasm.orbitstate_new(lobe, sub_lobe, theta, omega, s, alpha, k_residuals, residual_omega_scale, !isLikeNone(seed), isLikeNone(seed) ? BigInt(0) : seed);
        this.__wbg_ptr = ret >>> 0;
        OrbitStateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create deterministic orbit with default parameters and seed
     * @param {bigint} seed
     * @returns {OrbitState}
     */
    static newDefault(seed) {
        const ret = wasm.orbitstate_newDefault(seed);
        return OrbitState.__wrap(ret);
    }
    /**
     * Get s (radius scaling)
     * @returns {number}
     */
    get s() {
        const ret = wasm.colorintent_lightness(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set alpha (residual amplitude)
     * @param {number} alpha
     */
    set alpha(alpha) {
        wasm.colorintent_set_accent_weight(this.__wbg_ptr, alpha);
    }
    /**
     * Set lobe
     * @param {number} lobe
     */
    set lobe(lobe) {
        wasm.orbitstate_set_lobe(this.__wbg_ptr, lobe);
    }
    /**
     * Set omega (base angular velocity)
     * @param {number} omega
     */
    set omega(omega) {
        wasm.colorintent_set_chroma(this.__wbg_ptr, omega);
    }
    /**
     * Set s (radius scaling)
     * @param {number} s
     */
    set s(s) {
        wasm.colorintent_set_lightness(this.__wbg_ptr, s);
    }
    /**
     * Set sub_lobe
     * @param {number} sub_lobe
     */
    set sub_lobe(sub_lobe) {
        wasm.orbitstate_set_sub_lobe(this.__wbg_ptr, sub_lobe);
    }
    /**
     * Get sub_lobe
     * @returns {number}
     */
    get sub_lobe() {
        const ret = wasm.orbitstate_sub_lobe(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get theta
     * @returns {number}
     */
    get theta() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) OrbitState.prototype[Symbol.dispose] = OrbitState.prototype.free;

/**
 * Player c-space integrator wrapper for WASM (issue #88, Q2).
 *
 * Holds `c` as persistent state and moves it toward a model-driven target
 * point on the Mandelbrot boundary, biased along the Shore's contours via
 * the minimap. This replaces the closed-loop carrier for audio-driven
 * wandering.
 */
export class PlayerState {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PlayerStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_playerstate_free(ptr, 0);
    }
    /**
     * Apply model-predicted control signals.
     * @param {number} s
     * @param {number} alpha
     * @param {number} omega_scale
     */
    apply_controls(s, alpha, omega_scale) {
        wasm.playerstate_apply_controls(this.__wbg_ptr, s, alpha, omega_scale);
    }
    /**
     * Current c (imaginary part).
     * @returns {number}
     */
    get c_im() {
        const ret = wasm.colorintent_chroma(this.__wbg_ptr);
        return ret;
    }
    /**
     * Current c (real part).
     * @returns {number}
     */
    get c_re() {
        const ret = wasm.colorintent_anchor_hue(this.__wbg_ptr);
        return ret;
    }
    /**
     * Create a PlayerState starting on the boundary at (s, alpha).
     * @param {number} lobe
     * @param {number} sub_lobe
     * @param {number} s
     * @param {number} alpha
     */
    constructor(lobe, sub_lobe, s, alpha) {
        const ret = wasm.playerstate_new(lobe, sub_lobe, s, alpha);
        this.__wbg_ptr = ret >>> 0;
        PlayerStateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set the target shore-proximity distance the servo pulls toward.
     * @param {number} d_star
     */
    set d_star(d_star) {
        wasm.playerstate_set_d_star(this.__wbg_ptr, d_star);
    }
    /**
     * Set the audio energy in [0, 1] (loudness). Raises the servo's
     * target shore-proximity: loud audio pulls c toward the Shore.
     * @param {number} energy
     */
    set energy(energy) {
        wasm.playerstate_set_energy(this.__wbg_ptr, energy);
    }
    /**
     * Set the mip level used for the contour step.
     * @param {number} level
     */
    set level(level) {
        wasm.playerstate_set_level(this.__wbg_ptr, level);
    }
    /**
     * Switch the active Mandelbrot lobe.
     * @param {number} lobe
     * @param {number} sub_lobe
     */
    set_lobe(lobe, sub_lobe) {
        wasm.playerstate_set_lobe(this.__wbg_ptr, lobe, sub_lobe);
    }
    /**
     * Set the maximum world-space step per frame.
     * @param {number} max_step
     */
    set max_step(max_step) {
        wasm.playerstate_set_max_step(this.__wbg_ptr, max_step);
    }
    /**
     * Current c-space velocity magnitude (Momentum diagnostic).
     * @returns {number}
     */
    get speed() {
        const ret = wasm.playerstate_speed(this.__wbg_ptr);
        return ret;
    }
    /**
     * Advance the Player by dt, moving c toward the model-driven target,
     * biased along the Shore's contours. Returns the new c.
     * @param {number} dt
     * @param {number} h
     * @param {Float64Array | null} [band_gates]
     * @returns {Complex}
     */
    step(dt, h, band_gates) {
        var ptr0 = isLikeNone(band_gates) ? 0 : passArrayF64ToWasm0(band_gates, wasm.__wbindgen_malloc);
        var len0 = WASM_VECTOR_LEN;
        const ret = wasm.playerstate_step(this.__wbg_ptr, dt, h, ptr0, len0);
        return Complex.__wrap(ret);
    }
}
if (Symbol.dispose) PlayerState.prototype[Symbol.dispose] = PlayerState.prototype.free;

/**
 * Residual parameters
 */
export class ResidualParams {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ResidualParamsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_residualparams_free(ptr, 0);
    }
    /**
     * Create default residual parameters
     * @param {number} k_residuals
     * @param {number} residual_cap
     * @param {number} radius_scale
     */
    constructor(k_residuals, residual_cap, radius_scale) {
        const ret = wasm.residualparams_new(k_residuals, residual_cap, radius_scale);
        this.__wbg_ptr = ret >>> 0;
        ResidualParamsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ResidualParams.prototype[Symbol.dispose] = ResidualParams.prototype.free;

/**
 * @param {Float64Array} values
 * @param {number} features_per_frame
 * @returns {string}
 */
export function audioFeatureAveragesJson(values, features_per_frame) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.audioFeatureAveragesJson(ptr0, len0, features_per_frame);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Shared constants exposed to JavaScript
 * @returns {any}
 */
export function constants() {
    const ret = wasm.constants();
    return ret;
}

/**
 * Contour-biased integrator step for Physics. Returns [new_real, new_imag].
 * @param {number} real
 * @param {number} imag
 * @param {number} u_real
 * @param {number} u_imag
 * @param {number} h
 * @param {number} d_star
 * @param {number} max_step
 * @param {number} level
 * @param {number} energy
 * @returns {Float64Array}
 */
export function contour_biased_step(real, imag, u_real, u_imag, h, d_star, max_step, level, energy) {
    const ret = wasm.contour_biased_step(real, imag, u_real, u_imag, h, d_star, max_step, level, energy);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @param {number} c_re
 * @param {number} c_im
 * @param {number} vx
 * @param {number} vy
 * @param {MotionControls} motion
 * @param {number} dt
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function controlsIntegrateStep(c_re, c_im, vx, vy, motion, dt, config) {
    _assertClass(motion, MotionControls);
    _assertClass(config, ManifoldConfig);
    const ret = wasm.controlsIntegrateStep(c_re, c_im, vx, vy, motion.__wbg_ptr, dt, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @returns {string}
 */
export function controlsV2SchemaJson() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.controlsV2SchemaJson();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * @param {number} c_re
 * @param {number} c_im
 * @param {string} controls_json
 * @param {string | null} [presentation_json]
 * @returns {string}
 */
export function controlsV2VisualParametersJson(c_re, c_im, controls_json, presentation_json) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(controls_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(presentation_json) ? 0 : passStringToWasm0(presentation_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.controlsV2VisualParametersJson(c_re, c_im, ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Build a read-only DebugSnapshot from explicit authoritative state.
 *
 * `motion_raw` is the last raw (pre-clamp) MotionControls, or null before
 * the first step. `last_delta_total` is the last step's total-energy change
 * (NaN = none). Never mutates runtime state.
 * @param {number} c_re
 * @param {number} c_im
 * @param {number} vx
 * @param {number} vy
 * @param {MotionControls | null | undefined} motion_raw
 * @param {number} friction_beta
 * @param {number} friction_power
 * @param {number} manifold_drag
 * @param {ManifoldConfig} config
 * @param {number} last_delta_total
 * @param {number} time_seconds
 * @returns {any}
 */
export function debugSnapshotFromState(c_re, c_im, vx, vy, motion_raw, friction_beta, friction_power, manifold_drag, config, last_delta_total, time_seconds) {
    let ptr0 = 0;
    if (!isLikeNone(motion_raw)) {
        _assertClass(motion_raw, MotionControls);
        ptr0 = motion_raw.__destroy_into_raw();
    }
    _assertClass(config, ManifoldConfig);
    const ret = wasm.debugSnapshotFromState(c_re, c_im, vx, vy, ptr0, friction_beta, friction_power, manifold_drag, config.__wbg_ptr, last_delta_total, time_seconds);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * The DebugSnapshot contract version and canonical step cadence.
 * @returns {any}
 */
export function debugSnapshotMeta() {
    const ret = wasm.debugSnapshotMeta();
    return ret;
}

/**
 * Sample an n x n terrain patch of the canonical embedding
 * Q(c) = (x, y, lambda*sigma(c)) centered at (cx, cy) with half-extent
 * `half` in c-space. Returns a camelCase JSON object:
 * { n, center, half, positions, signed, realm }.
 * @param {number} cx
 * @param {number} cy
 * @param {number} half
 * @param {number} n
 * @param {ManifoldConfig} config
 * @returns {any}
 */
export function debugTerrainPatch(cx, cy, half, n, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.debugTerrainPatch(cx, cy, half, n, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {Float64Array} values
 * @returns {string}
 */
export function decodeControlsV2Json(values) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decodeControlsV2Json(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {Float64Array} values
 * @param {boolean} audio_reactive
 * @param {number} rms
 * @param {number} onset
 * @returns {string}
 */
export function decodeLegacyVisualJson(values, audio_reactive, rms, onset) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decodeLegacyVisualJson(ptr0, len0, audio_reactive, rms, onset);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {Float64Array} values
 * @param {number} k_bands
 * @returns {string}
 */
export function decodeOrbitControlJson(values, k_bands) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decodeOrbitControlJson(ptr0, len0, k_bands);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Deep-zoom unsigned distance field for the minimap (issue #111 feedback:
 * the minimap is a Mandelbrot deep zoom whose zoom level follows the
 * player). Resolution-unlimited escape-iteration estimator — resolves
 * structure where the baked mip pyramid runs out of texels. Returns one
 * unsigned distance per input point (0 inside the set).
 * @param {Float64Array} re
 * @param {Float64Array} im
 * @returns {Float32Array}
 */
export function deepZoomField(re, im) {
    const ptr0 = passArrayF64ToWasm0(re, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(im, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.deepZoomField(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {Float64Array} values
 * @returns {string}
 */
export function legacyAudioFeatureAveragesJson(values) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.legacyAudioFeatureAveragesJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {number} rms
 * @param {number} onset
 * @returns {string}
 */
export function legacyOrbitDriveInputsJson(rms, onset) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.legacyOrbitDriveInputsJson(rms, onset);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * @returns {string}
 */
export function legacyVisualExportRangesJson() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.legacyVisualExportRangesJson();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * @returns {string}
 */
export function legacyVisualSchemaJson() {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.legacyVisualSchemaJson();
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Convert a generalized force covector to coordinate acceleration: a = G^{-1} Q.
 * Returns [ax, ay]. This is the single place G^{-1} maps a covector to acceleration.
 * @param {number} qx
 * @param {number} qy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_apply_generalized_force(qx, qy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_apply_generalized_force(qx, qy, real, imag, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Christoffel symbols Gamma^i_jk. Returns a flat JS array of 8 values:
 * [Gamma^0_00, Gamma^0_01, Gamma^0_10, Gamma^0_11, Gamma^1_00, Gamma^1_01, Gamma^1_10, Gamma^1_11].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_christoffel_symbols(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_christoffel_symbols(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Metric-consistent isotropic drag covector: Q_drag = -beta G v. Returns [Qx, Qy].
 * This is a covector, not a coordinate acceleration; its power P = v^T Q_drag <= 0.
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {number} beta
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_drag_force(vx, vy, real, imag, beta, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_drag_force(vx, vy, real, imag, beta, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Embedding q(c) = (x, y, sigma(c)). Returns [x, y, sigma].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_embedding(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_embedding(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Geodesic acceleration term: Gamma^i_jk v^j v^k. Returns [ax, ay].
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_geodesic_acceleration(vx, vy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_geodesic_acceleration(vx, vy, real, imag, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Scale-relative induced metric
 * G(c) = rho^-2 I + lambda^2 * grad_sigma * grad_sigma^T.
 * Returns a flat JS array [g11, g12, g12, g22].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_induced_metric(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_induced_metric(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Semi-implicit Euler integration step for manifold dynamics.
 *
 * Integrates: r_ddot + Gamma(r_dot, r_dot) = -G^{-1}∇U + G^{-1}Q
 *
 * Returns [new_re, new_im, new_vx, new_vy, kinetic, potential, total, delta_total, delta_kinetic].
 * @param {number} c_re
 * @param {number} c_im
 * @param {number} vx
 * @param {number} vy
 * @param {number} qx
 * @param {number} qy
 * @param {number} beta
 * @param {number} dt
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_integrate_step(c_re, c_im, vx, vy, qx, qy, beta, dt, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_integrate_step(c_re, c_im, vx, vy, qx, qy, beta, dt, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Jacobian J_q(c) = ∂q/∂(x,y) as a 3×2 matrix. Returns flat array [1,0,0,1,sigma_x,sigma_y].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_jacobian(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_jacobian(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Kinetic energy K = 1/2 v^T G v.
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {number}
 */
export function manifold_kinetic_energy(vx, vy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_kinetic_energy(vx, vy, real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Mandelbrot scale sigma(c) = log2(d_ref / rho(c)).
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {number}
 */
export function manifold_mandelbrot_scale(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_mandelbrot_scale(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Native potential U = kappa * sigma(c).
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {number}
 */
export function manifold_potential_energy(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_potential_energy(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
 * Returns [Qx, Qy]. This is a covector, not a coordinate acceleration; convert
 * with `manifold_apply_generalized_force`.
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_potential_force(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_potential_force(real, imag, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Embedded velocity q_dot = J_q(c) v. Returns [vx, vy, sigma_dot].
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_q_dot(vx, vy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_q_dot(vx, vy, real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Regularized distance rho(c) = sqrt(D^2 + epsilon^2).
 * @param {number} real
 * @param {number} imag
 * @param {number} epsilon
 * @returns {number}
 */
export function manifold_regularized_distance(real, imag, epsilon) {
    const ret = wasm.manifold_regularized_distance(real, imag, epsilon);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Scale gradient ∇sigma(c) = (∂sigma/∂x, ∂sigma/∂y). Returns [gx, gy].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function manifold_scale_gradient(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_scale_gradient(real, imag, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Scale Hessian [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
 * Returns a flat JS array [xx, xy, xy, yy].
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {Array<any>}
 */
export function manifold_scale_hessian(real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_scale_hessian(real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Time derivative of Mandelbrot scale: sigma_dot = ∇sigma·v. No independent v_sigma.
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {number}
 */
export function manifold_sigma_dot(vx, vy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_sigma_dot(vx, vy, real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Signed distance to the Mandelbrot boundary. Positive outside, negative inside.
 * @param {number} real
 * @param {number} imag
 * @returns {number}
 */
export function manifold_signed_distance(real, imag) {
    const ret = wasm.manifold_signed_distance(real, imag);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Total mechanical energy E = K + U_sigma + U_wall.
 * @param {number} vx
 * @param {number} vy
 * @param {number} real
 * @param {number} imag
 * @param {ManifoldConfig} config
 * @returns {number}
 */
export function manifold_total_energy(vx, vy, real, imag, config) {
    _assertClass(config, ManifoldConfig);
    const ret = wasm.manifold_total_energy(vx, vy, real, imag, config.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Unsigned geometric distance d(c) = |D(c)|. Distinct from S sensitivity.
 * @param {number} real
 * @param {number} imag
 * @returns {number}
 */
export function manifold_unsigned_distance(real, imag) {
    const ret = wasm.manifold_unsigned_distance(real, imag);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Batch shore-proximity (S field) sampling over the canonical mip pyramid
 * (issue #111 minimap panel). Same field/level/rounding as the single-point
 * sampler; one lock for the whole batch. Returns a flat Float32Array.
 * @param {Float64Array} re
 * @param {Float64Array} im
 * @param {number} level
 * @returns {Float32Array}
 */
export function minimapShoreProximityBatch(re, im, level) {
    const ptr0 = passArrayF64ToWasm0(re, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(im, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.minimapShoreProximityBatch(ptr0, len0, ptr1, len1, level);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * Slope of the shore-proximity field at c on a mip level. Returns [gx, gy].
 * @param {number} real
 * @param {number} imag
 * @param {number} level
 * @returns {Float64Array}
 */
export function minimap_slope(real, imag, level) {
    const ret = wasm.minimap_slope(real, imag, level);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @param {string | null} [model_type]
 * @param {string | null} [controls_version]
 * @returns {string}
 */
export function modelOutputKind(model_type, controls_version) {
    let deferred3_0;
    let deferred3_1;
    try {
        var ptr0 = isLikeNone(model_type) ? 0 : passStringToWasm0(model_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(controls_version) ? 0 : passStringToWasm0(controls_version, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.modelOutputKind(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * @param {number} c_re
 * @param {number} c_im
 * @param {MotionControls} motion
 * @param {ManifoldConfig} config
 * @returns {Float64Array}
 */
export function motionDriveCovector(c_re, c_im, motion, config) {
    _assertClass(motion, MotionControls);
    _assertClass(config, ManifoldConfig);
    const ret = wasm.motionDriveCovector(c_re, c_im, motion.__wbg_ptr, config.__wbg_ptr);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @param {number} k_bands
 * @returns {string}
 */
export function orbitControlSchemaJson(k_bands) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ret = wasm.orbitControlSchemaJson(k_bands);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * @param {number} c_re
 * @param {number} c_im
 * @param {string} controls_json
 * @param {number} rms
 * @param {number} onset
 * @returns {string}
 */
export function orbitVisualParametersJson(c_re, c_im, controls_json, rms, onset) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(controls_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.orbitVisualParametersJson(c_re, c_im, ptr0, len0, rms, onset);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * The Player's full observation at c: 4x81 greys + 8 slope values = 332.
 * @param {number} real
 * @param {number} imag
 * @returns {Float32Array}
 */
export function player_observation(real, imag) {
    const ret = wasm.player_observation(real, imag);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Set the mip pyramid from host-provided flat planes (row-major, per level).
 * @param {Float32Array} f_flat
 * @param {Float32Array} s_flat
 * @param {Uint32Array} widths
 * @param {Uint32Array} heights
 * @param {number} re_min
 * @param {number} re_max
 * @param {number} im_min
 * @param {number} im_max
 */
export function set_mip_pyramid(f_flat, s_flat, widths, heights, re_min, re_max, im_min, im_max) {
    const ptr0 = passArrayF32ToWasm0(f_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(s_flat, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(widths, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray32ToWasm0(heights, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.set_mip_pyramid(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, re_min, re_max, im_min, im_max);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * Step the orbit forward and synthesize
 * @param {OrbitState} state
 * @param {number} dt
 * @param {ResidualParams} residual_params
 * @param {Float64Array | null} [band_gates]
 * @returns {Complex}
 */
export function step(state, dt, residual_params, band_gates) {
    _assertClass(state, OrbitState);
    _assertClass(residual_params, ResidualParams);
    var ptr0 = isLikeNone(band_gates) ? 0 : passArrayF64ToWasm0(band_gates, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.step(state.__wbg_ptr, dt, residual_params.__wbg_ptr, ptr0, len0);
    return Complex.__wrap(ret);
}

/**
 * Synthesize Julia parameter from orbit state
 * @param {OrbitState} state
 * @param {ResidualParams} residual_params
 * @param {Float64Array | null} [band_gates]
 * @returns {Complex}
 */
export function synthesize(state, residual_params, band_gates) {
    _assertClass(state, OrbitState);
    _assertClass(residual_params, ResidualParams);
    var ptr0 = isLikeNone(band_gates) ? 0 : passArrayF64ToWasm0(band_gates, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.synthesize(state.__wbg_ptr, residual_params.__wbg_ptr, ptr0, len0);
    return Complex.__wrap(ret);
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_Number_04624de7d0e8332d: function(arg0) {
            const ret = Number(arg0);
            return ret;
        },
        __wbg_String_8f0eb39a4a4c2f66: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_bigint_get_as_i64_8fcf4ce7f1ca72a2: function(arg0, arg1) {
            const v = arg1;
            const ret = typeof(v) === 'bigint' ? v : undefined;
            getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_boolean_get_bbbb1c18aa2f5e25: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_0bc8482c6e3508ae: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_in_47fa6863be6f2f25: function(arg0, arg1) {
            const ret = arg0 in arg1;
            return ret;
        },
        __wbg___wbindgen_is_bigint_31b12575b56f32fc: function(arg0) {
            const ret = typeof(arg0) === 'bigint';
            return ret;
        },
        __wbg___wbindgen_is_function_0095a73b8b156f76: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_5ae8e5880f2c1fbd: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_cd444516edc5b180: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_9e4d92534c42d778: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_jsval_eq_11888390b0186270: function(arg0, arg1) {
            const ret = arg0 === arg1;
            return ret;
        },
        __wbg___wbindgen_jsval_loose_eq_9dd77d8cd6671811: function(arg0, arg1) {
            const ret = arg0 == arg1;
            return ret;
        },
        __wbg___wbindgen_number_get_8ff4255516ccad3e: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_72fb696202c56729: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_389efe28435a9388: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_call_4708e0c13bdc8e95: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_crypto_86f2631e91b51511: function(arg0) {
            const ret = arg0.crypto;
            return ret;
        },
        __wbg_done_57b39ecd9addfe81: function(arg0) {
            const ret = arg0.done;
            return ret;
        },
        __wbg_getRandomValues_b3f15fcbfabb0f8b: function() { return handleError(function (arg0, arg1) {
            arg0.getRandomValues(arg1);
        }, arguments); },
        __wbg_get_9b94d73e6221f75c: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_b3ed3ad4be2bc8ac: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_with_ref_key_1dc361bd10053bfe: function(arg0, arg1) {
            const ret = arg0[arg1];
            return ret;
        },
        __wbg_instanceof_ArrayBuffer_c367199e2fa2aa04: function(arg0) {
            let result;
            try {
                result = arg0 instanceof ArrayBuffer;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Uint8Array_9b9075935c74707c: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isArray_d314bb98fcf08331: function(arg0) {
            const ret = Array.isArray(arg0);
            return ret;
        },
        __wbg_isSafeInteger_bfbc7332a9768d2a: function(arg0) {
            const ret = Number.isSafeInteger(arg0);
            return ret;
        },
        __wbg_iterator_6ff6560ca1568e55: function() {
            const ret = Symbol.iterator;
            return ret;
        },
        __wbg_length_32ed9a279acd054c: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_35a7bace40f36eac: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_msCrypto_d562bbe83e0d4b91: function(arg0) {
            const ret = arg0.msCrypto;
            return ret;
        },
        __wbg_new_361308b2356cecd0: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_3eb36ae241fe6f44: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_dd2b680c8bf6ae29: function(arg0) {
            const ret = new Uint8Array(arg0);
            return ret;
        },
        __wbg_new_no_args_1c7c842f08d00ebb: function(arg0, arg1) {
            const ret = new Function(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_a2c39cbe88fd8ff1: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_next_3482f54c49e8af19: function() { return handleError(function (arg0) {
            const ret = arg0.next();
            return ret;
        }, arguments); },
        __wbg_next_418f80d8f5303233: function(arg0) {
            const ret = arg0.next;
            return ret;
        },
        __wbg_node_e1f24f89a7336c2e: function(arg0) {
            const ret = arg0.node;
            return ret;
        },
        __wbg_process_3975fd6c72f520aa: function(arg0) {
            const ret = arg0.process;
            return ret;
        },
        __wbg_prototypesetcall_bdcdcc5842e4d77d: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_8ffdcb2063340ba5: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_randomFillSync_f8c153b79f285817: function() { return handleError(function (arg0, arg1) {
            arg0.randomFillSync(arg1);
        }, arguments); },
        __wbg_require_b74f47fc2d022fd6: function() { return handleError(function () {
            const ret = module.require;
            return ret;
        }, arguments); },
        __wbg_set_3f1d0b984ed272ed: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_f43e577aea94465b: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_static_accessor_GLOBAL_12837167ad935116: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_e628e89ab3b1c95f: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_a621d3dfbb60d0ce: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_f8727f0cf888e0bd: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_a96e1fef17ed23cb: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_value_0546255b415e96c1: function(arg0) {
            const ret = arg0.value;
            return ret;
        },
        __wbg_versions_4e31226f5e8dc909: function(arg0) {
            const ret = arg0.versions;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./orbit_synth_wasm_bg.js": import0,
    };
}

const AnalysisTimebaseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_analysistimebase_free(ptr >>> 0, 1));
const ColorIntentFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_colorintent_free(ptr >>> 0, 1));
const ComplexFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_complex_free(ptr >>> 0, 1));
const ControlsV2Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_controlsv2_free(ptr >>> 0, 1));
const CycleBankFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_cyclebank_free(ptr >>> 0, 1));
const FeatureExtractorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_featureextractor_free(ptr >>> 0, 1));
const JuliaViewControlsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_juliaviewcontrols_free(ptr >>> 0, 1));
const JuliaViewStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_juliaviewstate_free(ptr >>> 0, 1));
const ManifoldConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_manifoldconfig_free(ptr >>> 0, 1));
const MotionControlsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_motioncontrols_free(ptr >>> 0, 1));
const OrbitControllerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_orbitcontroller_free(ptr >>> 0, 1));
const OrbitStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_orbitstate_free(ptr >>> 0, 1));
const PlayerStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_playerstate_free(ptr >>> 0, 1));
const ResidualParamsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_residualparams_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('orbit_synth_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
