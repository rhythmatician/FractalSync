# AGENTS.md

Instructions for AI coding agents working in this repository.

## Agent skills

## Coding

Use YAGNI and DRY principles.

Runtime-consumed math lives in runtime-core (Rust) only; mirrors must pass `python scripts/preflight_parity.py` before training. Authority: docs/adr/0001-rust-first-parity.md.

### Issue tracker

Issues are tracked in GitHub Issues (`rhythmatician/FractalSync`) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role triage vocabulary (label strings equal to role names). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

## Token safety

Use RTK for supported commands and potentially large command output. Do not blindly prefix PowerShell cmdlets, aliases, shell built-ins, or shell syntax with rtk. Prefer RTK-native equivalents when available, e.g. rtk read instead of Get-Content. For unsupported PowerShell operations, invoke PowerShell normally and explicitly bound potentially large output.
