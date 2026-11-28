# Two-source gesture: review procedure

The known-phase preview offers a direct target and a bounded Physics comparison. Enable **Render Physics motion** before starting the second passage and keep the starting anchor, source settings and Julia presentation fixed for the comparison. The first-order mapping is local, and perceptibility remains a question for the live review.

The experiment uses the existing model compatibility version.

Build the Wasm bundle from `wasm-orbit` with `wasm-pack build --target web --out-dir ../frontend/src/wasm --out-name orbit_synth_wasm`. Start the shader API using `.venv\Scripts\python.exe -m uvicorn api.server:app --app-dir backend --host 127.0.0.1 --port 8000` and the frontend using `npm --prefix frontend run dev`. Open `http://localhost:3000/gestureExperiment.html`, or follow the link in the debug cockpit. The shader API must be running. Audio starts only after the Start passage button is pressed.

Two synthetic amplitude-modulated tones share the browser audio clock with the Rust-generated visual phases. They begin at 1.5 Hz and 2 Hz. Pulse B changes to 2.25 Hz at 8 seconds, to 1.75 Hz at 16 seconds, and returns to 2 Hz at 24 seconds. Its accumulated phase returns to the initial 3:4 reference at 24 seconds. The passage continues until stopped; restarting returns it to the beginning.

For an initial manual review:

1. Keep the anchor, zoom, palette, and phase offset fixed. Start with source 1 on pulse A and source 2 on pulse B, with gain 2 at zero.
2. Start the passage and establish A's motion. Increase gain 2 deliberately. Repeat with each geometry and adjust excursion conservatively. Source changes and phase offset changes take effect immediately and may cause a jump.
3. Use **Show Julia only** to give a viewer the audience image without the controls or passage clock. Listen for, and mark, the perceived departure and return. Escape exits fullscreen.
4. Repeat with **Audio with fixed Julia** and **Visual changes early**. The latter uses the same audio, while B's visual departure and recovery occur eight seconds early. The tones and visual remain periodic, but this is a particular mismatched comparison, not an independently validated perceptual control. Randomize condition order and keep the settings and gain changes consistent if collecting observations.

The raw sum's extent can cycle even while the generalized 3:4 relationship is locked. Do not score every expansion or contraction as evidence of a phase slip. This manual procedure is exploratory; it does not by itself measure performance against chance, blind participants, or prove that the Julia output improves detection. Test known phases first, then separately investigate physical tracking and CycleBank uncertainty and role reassociation.

The default anchor is only a starting point. If a target leaves the Map or geometry cannot be evaluated, playback stops with an error. An accepted target is not a guarantee of geometric fidelity or useful visual sensitivity. The comparison does not yet include an unrelated visual matched for activity, and there is no recording or automated blinded trial.


Use **Export diagnostic record** during or after each passage. In Physics mode, the record includes the intended target and realized position for each 20 ms step, along with metric lag, throttle saturation, energy, control work, friction work and residual. Settings changes take effect at the first step whose audio time reaches the recorded change. If integration falls more than one second behind audio, the passage stops and records the error.

During a Physics passage, choose **Free navigation** and steer with the direction buttons or **Coast**. **Resume gesture tracking** resumes pursuit of the configured anchor. **Resume from here** uses the latest realized position as the new anchor. Both preserve position, velocity and source bindings. The record includes the start and end state of each step and flags changes in the signed distance across the Shore at the sampled endpoints.

**Replay Julia motion** plays back the most recent passage's visual parameters in memory with the original timing. Replay is silent and uses the current viewport. **Save Julia frame** exports the current preview or replay frame as a PNG. Save the trace and representative frames before starting another passage. Record the human observations and renderer environment alongside them.

The trace retains at most 36,000 samples, 36,000 frames and 2,048 events. The export counts omitted entries in `droppedSamples`, `droppedFrames` and `droppedEvents`.

To try causal estimation, enable **Estimate fixture envelopes with CycleBank** before playback. Wait for confirmed modes, then choose a mode for each role. Each binding holds the mode ID within the current passage. An unavailable mode retains its role label and contributes zero gain. Use **Withdraw** to release a role, or explicitly bind an observed mode to change it. The trace records mode IDs, estimated phase, strength, confidence, missing observations and binding events. This path feeds the two synthetic modulation envelopes into CycleBank at the canonical hop cadence. It does not analyze microphone or file audio. Binding changes take effect at the next eligible step and may create a reference discontinuity.

The browser smoke check uses a 560 × 350 viewport for the Physics comparison and software WebGL. Set `GESTURE_FULL_PASSAGE=1` when running `node frontend/scripts/check_gesture_experiment.cjs` to exercise the departure and return over 25 seconds. Set `GESTURE_CYCLE_BANK=1` to check causal estimation, explicit binding and withdrawal. A Domain Expert still needs to make the listening and playability judgment. A deliberate source-loss and replacement trial remains open.
