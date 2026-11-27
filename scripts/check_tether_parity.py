"""Compare native Python and browser WASM tether traces."""
import json
import math
from pathlib import Path
import subprocess

from runtime_core import CockpitRope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rope = CockpitRope()
    drive: list[list[float]] = [
        [
            -0.744 + 1e-6 * i,
            0.132 + 2e-5 * math.sin(i / 20),
            0.0027 + i * 1e-6,
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
    for action in actions:
        if action is None:
            rope.reset()
        else:
            native.append([*rope.step_analog(*action), *rope.state()])

    result = subprocess.run(
        ["node", "--input-type=module", "-e", """
import { readFileSync } from 'node:fs';
import { initSync, CockpitRope } from './frontend/src/wasm/orbit_synth_wasm.js';
initSync({module: readFileSync('./frontend/src/wasm/orbit_synth_wasm_bg.wasm')});
const rope = new CockpitRope();
const trace = [];
for (const action of JSON.parse(readFileSync(0, 'utf8'))) {
    if (action === null) rope.reset();
    else trace.push([...rope.step_analog(...action), ...rope.state()]);
}
rope.free();
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
    print(f"Matched {len(native)} tether ticks across Python and WASM, including reset.")


if __name__ == "__main__":
    main()
