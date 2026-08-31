# AGENTS.md

Short, opinionated contract for any AI agent working in this repo. Read this,
then read `CONTEXT.md` and the ADRs you need.

## Core principle

Apply YAGNI and DRY, and respect the seam defined by ADR 0001: anything the
Player sees, decides, or has done to them belongs in `runtime-core` (Rust),
not in Python or TypeScript. Authority: [docs/adr/0001-rust-first-parity.md](docs/adr/0001-rust-first-parity.md).

## Seams at a glance

- `runtime-core/` — Rust. **Authoritative** for synthesis, geometry, feature
  extraction, minimap, visual metrics, control ranges, timebase, and any
  state shared between training and runtime.
- `backend/` — Python. **Training only.** Models, optimizers, data, training
  loops, ONNX export, model-serving API. No domain math; delegate to
  `runtime-core`. Mirrors are exceptions, must pass parity.
- `frontend/` — TypeScript / React. **Browser only.** WebAudio, ONNX Runtime
  Web, UI, lifecycle. No domain math; delegate to `wasm-orbit`/`runtime-core`.
- `wasm-orbit/` — `wasm-bindgen` bindings. Thin glue between Rust core and
  the browser. Should be small.
- Parity gate: `python scripts/preflight_parity.py` must pass before
  `backend/train.py` will run. Golden vectors live in
  `shared/golden_vectors.json`.

If you find yourself re-implementing behavior accross Python and TypeScript,
stop and add a Rust implementation and binding instead.  Rust should carry as much of the shared logic as possible, to minimize the need for parity tests between Python and TypeScript.

## Working agreements

- **Token safety:** prefer RTK for supported commands (`rtk git status`,
  `rtk cargo test`, `rtk npm test`). Don't wrap PowerShell built-ins,
  aliases, or syntax; use RTK-native equivalents where they exist
  (`rtk read`, not `Get-Content`).
- **Type safety:** prefer syntax that allows static code analysis to catch real
  mistakes early.  For example, in Python, this means prefering explicit type-hints
  over "Any" or "object", while avoiding getattr() or cast(,Any) or anything else
  that pyright cannot explicitly check. Done right, static code analysis can often
  catch errors before tests.
- **Multi-interpreter pitfall:** Python work goes through the project venv
  (`.venv\Scripts\python.exe` on Windows) and reuses `sys.executable` in
  subprocesses. CI enforces this.
- **Build paths:** prefer explicit prefixes (`npm --prefix frontend run build`,
  `maturin develop --release`, `wasm-pack build --target web`) over `cd`-and-run.
  Use `Push-Location`/`Pop-Location` when you must change directories.
- **Canonical-math changes** require regenerating
  `shared/golden_vectors.json` and updating all mirrors in the same commit.
  Bump `CONTROLLER_VERSION` or `FEATURE_VERSION` alongside.
- **Issue tracker, triage, and domain context:** see `docs/agents/`.

## What we don't do here

- We don't restate architecture, workflows, or debugging tips in this file —
  the ADRs and `CONTEXT.md` own that. Keep this file short.
- We don't maintain a parallel `.github/copilot-instructions.md`. That file
  is retired; the AGENTS.md convention replaces it.

If anything in this repo contradicts ADR 0001, the ADR wins.
