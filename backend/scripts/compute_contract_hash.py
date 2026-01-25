from pathlib import Path
import sys
# shim to root-level scripts/compute_contract_hash.py for tests that expect it in backend/scripts
ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "scripts" / "compute_contract_hash.py"
if not SHIM.exists():
    print("Missing shim target:", SHIM, file=sys.stderr)
    sys.exit(1)
# execute the real script
with open(SHIM, "r", encoding="utf-8") as f:
    code = f.read()
# run in a fresh globals dict
globals_dict = {"__name__": "__main__", "__file__": str(SHIM)}
exec(compile(code, str(SHIM), "exec"), globals_dict)
