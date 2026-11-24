# Agent-owned mathematical verification

Read the parent `AGENTS.md`, this package's `README.md`, and the relevant research
note before editing. Rust remains runtime authority; Lean is a development-time
specification and proof tool. No proof-completion gate is added to training.

- Keep the pinned Lean/Mathlib environment and committed Lake manifest coherent.
  Use Mathlib's precompiled cache. Agents handle setup and proof maintenance.
- Use the official `leanprover/skills` proof guidance when available: check each
  active proof step with Lean and resolve diagnostics before continuing.
- State exact hypotheses and conventional mathematical meaning. Search Mathlib
  before adding machinery. Preserve the distinction between proofs, numerical
  checks, hypotheses, and originality claims.
- Import every accepted module in `FractalSync.lean`. Run `lake build --wfail`
  and `lake env lean Audit.lean`. Do not accept placeholders or expand the audit
  allowlist to make a proof pass.
- Update the theorem inventory and the corresponding research note. A scalar
  algebra lemma must not be reported as a proof of its full geometric or analytic
  application. A proof over reals must not be reported as verified Rust floats.
- Keep numerical research checks outside the runtime/training import graph.
  They exercise analytic models and counterexamples and have no domain authority.
- Extend this package for #157/#159. Aeneas/Charon work remains separate in #158.
