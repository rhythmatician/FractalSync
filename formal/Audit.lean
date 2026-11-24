import FractalSync
import Lean.Util.CollectAxioms
import Lean.Elab.Command

/-!
Inspect transitive dependencies in Lean's checked environment. Only the usual
classical foundations used by Mathlib are allowed; sorryAx, native evaluation
axioms, and additional project axioms are rejected.
-/

open Lean Elab Command

run_cmd do
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let env ← getEnv
  let mut count := 0
  for (name, _) in env.constants.toList do
    if (`FractalSync).isPrefixOf name then
      count := count + 1
      let axioms ← Lean.collectAxioms name
      for axiomName in axioms do
        unless allowed.contains axiomName do
          throwError "Unapproved axiom {axiomName} in {name}"
      logInfo m!"{name}: {axioms}"
  if count == 0 then
    throwError "No FractalSync declarations were imported; audit would be empty"
  logInfo m!"Axiom audit passed for {count} FractalSync declarations"
