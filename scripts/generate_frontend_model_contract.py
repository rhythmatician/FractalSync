"""Generate frontend TS modelContract from contracts/model_io_contract.json"""

from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
contract_path = root / "contracts" / "model_io_contract.json"
out_path = root / "frontend" / "src" / "lib" / "modelContract.ts"

with open(contract_path, "r", encoding="utf-8") as f:
    cj = json.load(f)

inp = cj["input"]
out = cj["output"]

input_names = []
for frame in range(inp["window_frames"]):
    for feat in inp["feature_names"]:
        input_names.append(f"frame_{frame}_{feat}")

out_lines = []
out_lines.append(f"export const MODEL_INPUT_NAME = \"{inp['name']}\";")
out_lines.append(f"export const MODEL_OUTPUT_NAME = \"{out['name']}\";")
out_lines.append("")
out_lines.append("export const INPUT_NAMES = [")
for name in input_names:
    out_lines.append(f'  "{name}",')
out_lines.append("];\n")

out_lines.append(
    "export function buildOutputNames(kBands = {}) {{".format(out["k_bands"])
)
out_lines.append(
    '  const names = ["{}", "{}", "{}"];'.format(*out["base_output_names"])
)
out_lines.append(
    "  for (let i = 0; i < kBands; i += 1) { names.push(`"
    + out["parameter_prefix"]
    + "${i}`); }"
)
out_lines.append("  return names;")
out_lines.append("}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

print("Wrote", out_path)
