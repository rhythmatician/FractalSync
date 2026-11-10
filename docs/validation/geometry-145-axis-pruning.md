# Geometry #145: conservative axis pruning

The isolated proof experiment now avoids a full capsule-distance calculation when an outward-rounded axis bound already proves the required separation. Production geometry is unchanged.

For a segment capsule with radius `r` and requested clearance `m`, a downward-rounded coordinate gap strictly greater than upward-rounded `r + m` proves clearance greater than `m`. Invalid inputs fall back to the existing calculation. During cover reuse, the shortcut can retain an already established lower bound; every edge and connector still participates in winding and positivity checks.

Validation passed all 78 experiment tests and caught all 21 cover mutations, including new mutations that omit radius, omit requested clearance, accept tangency, or skip winding after pruning. Original crossing tests and acceptance thresholds were not changed.

Frozen before/after runs both completed at the two difficult fixture locations. All 81 certified candidate checks, six reconstructed field components, uncertainty bounds, and derivative-check rows matched exactly.

| Fixture | Previous post-construction validation | Pruned validation | Previous cold total | Pruned cold total |
| --- | ---: | ---: | ---: | ---: |
| Native, period 85 | 6.252 s | 1.267 s | 478.807 s | 438.873 s |
| Python, period 154 | 0.555 s | 0.141 s | 100.142 s | 107.510 s |

These runs had different concurrent loads. The validation phase includes final bracket checks and diagnostic output; the totals do not establish a controlled cold speedup. Cold proof construction remains unsuitable for runtime use.

The coherent polynomial still bounds uncertainty relative to its ideal nodal reconstruction only. It does not establish between-node distance accuracy, derivative accuracy against the distance function, or regular/cut-locus classification. Runtime integration, shared membership policy, crossing and training validation, and practical construction cost remain open.

Exact outputs, source and binary hashes, mutation outcomes, and terminal statuses are recorded in [the machine-readable evidence](geometry-145-axis-pruning.json). The preceding field construction is documented in [the full-stencil report](geometry-145-full-stencil.md).
