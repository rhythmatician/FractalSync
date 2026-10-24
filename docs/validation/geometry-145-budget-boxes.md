# Budget-aware exterior proof experiment

This is an unshipped follow-up to the
[coherent-field enclosure candidate](geometry-145-ranged-field.md). It changes
no quality threshold, membership cap or original crossing test.

Fine tiles begin ordinary membership testing at `dyadic_n0(k) = 512 + 64*k`,
capped at 32768. The previous whole-box shortcut permits exterior acceptance
only for escape before update 512. The experiment supplies the tile's actual
minimum budget and accepts only a proved escape strictly before that budget.
Such a certificate applies at that budget and every subsequent doubling. The
default entry point retains its original 512 semantics.

The exact-box FIFO key includes the minimum budget. Uncertainty at budget 512
cannot be replaced by a cached exterior verdict obtained at budget 513.
Interior acceptance still requires the complete prefix bounded and a strict
periodic trap. The proof-work cap remains 4096; exhausting it returns uncertainty.

All 58 targeted tests and twelve derivative checks pass. Every serialized
derivative row matches the preceding candidate. Mutating the cache key to omit
the budget makes the partition regression fail. Changing the strict escape
cutoff to include the budget endpoint makes the original cutoff regression fail.
Independent ordinary-node comparisons exercise four starting budgets.

Three review axes found no concrete acceptance defect. Proof comments were
corrected to state that exterior certificates apply at or above the supplied
minimum. Larger unsuccessful prefixes can cost more, so this is not adopted as
an optimization on semantic correctness alone.

An exact-source coverage probe checks 1296 boxes around the recorded retained
node, at four levels and four spacings. It finds **zero additional exterior
certificates** in that sample. Existing certificates remain unchanged. This does
not establish a useful cost improvement. The separate recorded-point query's
terminal status is preserved in [the JSON report](geometry-145-budget-boxes.json).

The budget-aware isolated recorded-point query also timed out after 1200.28 seconds.

The preceding FIFO candidate's full original native crossing test times out at
1200 seconds at the same recorded difficult point. Its cache-only predecessor's
isolated point query also times out at 1200 seconds. These are failures to
complete validation, not passing crossings.

To reproduce, reconstruct the ranged-field candidate, apply
`experiments/geometry-145-box-fifo.patch`, then
`experiments/geometry-145-budget-boxes.patch`. The last patch was applied to the
exact reconstructed FIFO sources and compared with both updated source files.
Compile `experiments/geometry-145-budget-box-coverage.rs` with Rust 2021 and
optimization to repeat the coverage probe. The standalone harness uses
`std::sync::LazyLock` instead of `once_cell::Lazy`; proof arithmetic is unchanged.
The earlier FIFO-only benchmark belongs to the preceding three-coordinate-key
variant and should be run before the budget patch.
