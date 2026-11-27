# Two-source gesture: review procedure

This is a known-phase experiment, not the Player or CycleBank. It shows a local first-order intended target directly in the Julia renderer. It does not run the target through Physics, prove that the local mapping is accurate at large excursions, or establish that the relationship is perceptible.

The repository's canonical-math version rule advances `CONTROLLER_VERSION` to `orbit-controller/7` for this addition, although the existing controller step is unchanged. Models stamped with an older controller version will be rejected until they are updated under the project's model contract.

Build the Wasm bundle from `wasm-orbit` with `wasm-pack build --target web --out-dir ../frontend/src/wasm --out-name orbit_synth_wasm`. Start the shader API using `.venv\Scripts\python.exe -m uvicorn api.server:app --app-dir backend --host 127.0.0.1 --port 8000` and the frontend using `npm --prefix frontend run dev`. Open `http://localhost:3000/gestureExperiment.html`, or follow the link in the debug cockpit. The shader API must be running. Audio starts only after the Start passage button is pressed.

Two synthetic amplitude-modulated tones share the browser audio clock with the Rust-generated visual phases. They begin at 1.5 Hz and 2 Hz. Pulse B changes to 2.25 Hz at 8 seconds, to 1.75 Hz at 16 seconds, and returns to 2 Hz at 24 seconds. Its accumulated phase returns to the initial 3:4 reference at 24 seconds. The passage continues until stopped; restarting returns it to the beginning.

For an initial manual review:

1. Keep the anchor, zoom, palette, and phase offset fixed. Start with source 1 on pulse A and source 2 on pulse B, with gain 2 at zero.
2. Start the passage and establish A's motion. Increase gain 2 deliberately. Repeat with each geometry and adjust excursion conservatively. Source changes and phase offset changes take effect immediately and may cause a jump.
3. Use **Show Julia only** to give a viewer the audience image without the controls or passage clock. Listen for, and mark, the perceived departure and return. Escape exits fullscreen.
4. Repeat with **Audio with fixed Julia** and **Visual changes early**. The latter uses the same audio, while B's visual departure and recovery occur eight seconds early. The tones and visual remain periodic, but this is a particular mismatched comparison, not an independently validated perceptual control. Randomize condition order and keep the settings and gain changes consistent if collecting observations.

The raw sum's extent can cycle even while the generalized 3:4 relationship is locked. Do not score every expansion or contraction as evidence of a phase slip. This manual procedure is exploratory; it does not by itself measure performance against chance, blind participants, or prove that the Julia output improves detection. Test known phases first, then separately investigate physical tracking and CycleBank uncertainty and role reassociation.

The default anchor is only a starting point. If a target leaves the Map or geometry cannot be evaluated, playback stops with an error. An accepted target is not a guarantee of geometric fidelity or useful visual sensitivity. The comparison does not yet include an unrelated visual matched for activity, and there is no recording or automated blinded trial.


Use **Export diagnostic record** during or after each passage to save a JSON record before starting another. It contains the synthetic phases, intended target, settings and audio-clock time for each rendered frame, along with settings changes, stop and error events. Save successful and failed trials separately. This is a diagnostic trace, not a replay or a recording of rendered pixels; it contains no realized Physics trajectory, forces, energy, CycleBank identities or confidence. Record the human observations and renderer environment alongside it.
