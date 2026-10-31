# Geometry #145: neighboring domain-size hints

The archived candidate tries a domain size suggested by the preceding patch's measured enclosure. It scales the enclosure by twice the squared ratio of multiplier radii, with the original small-domain floors. This is a numerical proposal only. A fresh strict inclusion/contraction proof issues every new certificate. A failed proposal falls back to the original bounded automatic search. The candidate is not adopted because it does not improve the native case.

The first experiment reused requested domain sizes instead of measured enclosures. That compounded slack across patches. It increased the Python cover from 182 to 202 edges and the native fixture refused at its unchanged depth limit after 106 edges. Those terminal failures are retained in the evidence; that proposal is not the implementation.

The measured-enclosure implementation passes all 79 experiment tests. All 12 branch mutations are caught, including bypassing the fresh proof and removing the fallback. The original contraction, strict inclusion, rounding, branch identity, and attracting-segment mutations remain caught.

The Python full-stencil comparison completes and preserves all 81 certified samples, six field components, error bounds, and derivative-check outputs exactly. It retains 182 accepted edges. Cover construction took 22.690 seconds versus the baseline's 27.482 seconds; certification took 17.674 versus 22.548 seconds. The complete cold run took 105.919 seconds versus 133.833 seconds. Different concurrent loads prevent a controlled speedup claim, especially for work outside the measured cover phases.

The native measured-enclosure comparison also completes and preserves all 81 certified samples and every field/error/derivative output exactly. However, it uses 2,773 edges versus 2,746. Cover construction takes 439.858 seconds versus 421.064 seconds, certification 393.853 versus 375.156 seconds, and the complete cold run 501.032 versus 498.902 seconds. A gain in one fixture does not establish a general cost fix.

The default experiment implementation is restored. To reproduce the rejected candidate, apply [the archived patch](experiments/geometry-145-nearby-domains.patch) at `1d2686f`; `git apply --check` passes against the restored source. Its 79-test and 12-mutation results belong to the applied candidate, while the default implementation retains 78 tests. The full native baseline phase profile is now recorded as well.

Production thresholds and crossing tests are unchanged. Runtime integration, original crossings, training, and practical query cost remain open. An isolated checkout at `tmp/geometry145-runtime-integration` reconstructs the preceding coherent field from the ten patches listed in the ranged-field report, ready for the next runtime investigation.

See [exact outcomes, hashes, and mutations](geometry-145-nearby-domains.json) and [the baseline phase profile](geometry-145-cover-profile.md).
