"""Compare native Python and browser WASM tether traces."""
import json
import math
from pathlib import Path
import subprocess

from runtime_core import (
    CockpitRope, ManifoldConfig, MotionControls, controls_integrate_step, tether_rho,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rope = CockpitRope()
    drive: list[list[float]] = [
        [
            math.cos(i / 80), math.sin(i / 80), 0.2, 0.0, 0.5, 0.0,
            math.cos(i / 8),
            math.sin(i / 8),
            (1.0, 0.4, 0.0, 0.0, 1.0)[i // 40],
            (0.0, 0.0, 0.0, 0.35, 1.0)[i // 40],
            1024 / 48000,
        ]
        for i in range(200)
    ]
    actions = [*drive, None, *drive]
    native = []
    config = ManifoldConfig()
    rider = (-0.744, 0.132, 0.0, 0.0)
    for action in actions:
        if action is None:
            rope.reset()
            rider = (-0.744, 0.132, 0.0, 0.0)
        else:
            dt = action[-1]
            x, y, vx, vy, _ = controls_integrate_step(
                *rider, MotionControls(*action[:6]), dt, config
            )
            rider = (x, y, vx, vy)
            seed = rope.step_analog(x, y, tether_rho(complex(x, y), 1e-4), *action[6:])
            native.append([*rider, *seed, *rope.state()])

    result = subprocess.run(
        ["node", "--input-type=module", "-e", """
import { readFileSync } from 'node:fs';
import { initSync, CockpitRope, ManifoldConfig, MotionControls, controlsIntegrateStep, manifold_regularized_distance } from './frontend/src/wasm/orbit_synth_wasm.js';
initSync({module: readFileSync('./frontend/src/wasm/orbit_synth_wasm_bg.wasm')});
const rope = new CockpitRope();
const trace = [];
const config = ManifoldConfig.defaults();
const start = [-0.744, 0.132, 0, 0];
let rider = start;
for (const action of JSON.parse(readFileSync(0, 'utf8'))) {
    if (action === null) {
        rider = start;
        rope.reset();
    } else {
        const motion = new MotionControls(...action.slice(0, 6));
        const dt = action.at(-1);
        rider = Array.from(controlsIntegrateStep(...rider, motion, dt, config).slice(0, 4));
        motion.free();
        const [x, y] = rider;
        const rho = manifold_regularized_distance(x, y, 1e-4);
        trace.push([...rider, ...rope.step_analog(x, y, rho, ...action.slice(6)), ...rope.state()]);
    }
}
rope.free();
config.free();
console.log(JSON.stringify(trace));
"""],
        input=json.dumps(actions),
        capture_output=True,
        check=True,
        text=True,
        cwd=root,
    )
    wasm = json.loads(result.stdout)
    for tick, (left, right) in enumerate(zip(native, wasm, strict=True)):
        for channel, (a, b) in enumerate(zip(left, right, strict=True)):
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(f"tick {tick}, channel {channel}: {a} != {b}")
    assert native[:200] == native[200:]
    assert wasm[:200] == wasm[200:]
    print(f"Matched {len(native)} skate-and-tether ticks across Python and WASM, including reset.")


if __name__ == "__main__":
    main()
