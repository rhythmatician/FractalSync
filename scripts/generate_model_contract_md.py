"""Generate human-readable MODEL_IO_CONTRACT.md from contracts/model_io_contract.json"""

from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
contract_path = root / "contracts" / "model_io_contract.json"
md_path = root / "docs" / "MODEL_IO_CONTRACT.md"

with open(contract_path, "r", encoding="utf-8") as f:
    cj = json.load(f)

inp = cj["input"]
out = cj["output"]

lines = []
lines.append("# Model I/O Contract (Generated)")
lines.append("")
lines.append("> Auto-generated from `contracts/model_io_contract.json`")
lines.append("")
lines.append("## Input")
lines.append(f"- name: `{inp['name']}`")
lines.append(f"- window_frames: {inp['window_frames']}")
lines.append(f"- features_per_frame: {inp['features_per_frame']}")
lines.append("")
lines.append("### Feature names (per-frame, oldest->newest)")
for i, feat in enumerate(inp["feature_names"]):
    lines.append(f"- `{feat}`")

lines.append("")
lines.append("## Output")
lines.append(f"- name: `{out['name']}`")
lines.append(f"- k_bands: {out['k_bands']}")
lines.append("")
lines.append("### Output elements (order)")
for name in out["base_output_names"]:
    lines.append(f"- `{name}`")
for i in range(out["k_bands"]):
    lines.append(f"- `{out['parameter_prefix']}{i}`")

lines.append("")
lines.append("---")
lines.append(
    "This document is generated; update `contracts/model_io_contract.json` instead."
)

md_path.parent.mkdir(exist_ok=True, parents=True)
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote", md_path)
