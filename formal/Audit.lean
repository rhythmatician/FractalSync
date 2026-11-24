import FractalSync
import Lean.Util.CollectAxioms
import Lean.Elab.Command

/-!
Inspect transitive dependencies in Lean's checked environment. Roots include
every declaration originating in a FractalSync module, even when a backported
lemma uses an upstream namespace. Only the usual
classical foundations used by Mathlib are allowed; sorryAx, native evaluation
axioms, and additional project axioms are rejected.
-/

open Lean Elab Command

run_cmd do
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let env ← getEnv
  let mut names : Array Name := #[]
  for (name, _) in env.constants.toList do
    let fromProjectModule := match env.getModuleIdxFor? name with
      | some idx => (`FractalSync).isPrefixOf env.header.moduleNames[idx]!
      | none => false
    if (`FractalSync).isPrefixOf name || fromProjectModule then
      names := names.push name
  if names.isEmpty then
    throwError "No project declarations were imported; audit would be empty"
  -- Use Lean's own kernel-environment traversal with one shared visited set.
  -- This computes the same union as separate `collectAxioms` calls while
  -- avoiding repeated traversal of the large shared Mathlib dependency graph.
  let (_, audit) := ((names.forM Lean.CollectAxioms.collect).run env).run {}
  for axiomName in audit.axioms do
    unless allowed.contains axiomName do
      throwError "Unapproved axiom {axiomName} reachable from FractalSync declarations"
  logInfo m!"Transitive axioms: {audit.axioms}"
  logInfo m!"Axiom audit passed for {names.size} project declarations"
