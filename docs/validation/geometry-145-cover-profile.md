# Geometry #145: cold cover profile

Opt-in `GEOMETRY145_COVER_PROFILE=1` diagnostics now separate root solving, phase selection, certification, edge construction, joins, geometry checks, and final validation. Timers do not supply any certificate inputs or affect acceptance. Default and enabled builds each pass all 78 experiment tests.

The terminal Python fixture run reproduced all 81 certified samples and every field, uncertainty, and derivative-check output exactly. Its cover took 27.482 seconds:

| Phase | Calls | Seconds |
| --- | ---: | ---: |
| Certification | 236 | 22.548 |
| Root solving | 654 | 3.428 |
| Phase selection | 236 | 0.909 |
| Edge construction | 192 | 0.004 |
| Joins | 191 | 0.010 |
| Target geometry | 192 | 0.048 |
| Final cover/target validation | 2 | 0.073 |

Certification dominates cover construction. However, the complete cold run took 133.833 seconds, so these measurements do not attribute most of the whole run. Independent node-witness search and other work outside cover construction need separate profiling. Reducing certification retries is a candidate optimization, not a measured improvement yet.

The native fixture also completed and reproduced every certified sample, field component, uncertainty bound, and derivative output. Its cover took 421.064 seconds of a 498.902-second cold run. Certification used 375.156 seconds across 5,364 candidate patches, root solving 31.173 seconds across 13,474 calls, and phase selection 11.285 seconds. Edge construction, joins, target geometry, and final validation together used 2.927 seconds. These measurements support investigating certification retries before further geometry pruning.

No production behavior, threshold, or crossing test changed. These observations do not establish readiness to merge.

Source and executable hashes, exact-output comparisons, phase counts, and terminal status are in [the profile evidence](geometry-145-cover-profile.json).
