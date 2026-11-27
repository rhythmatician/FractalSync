# Finite critical-orbit return and trap selection

This offline fixture tests whether simple recurrence and configurable trap readouts have the same interpretation. It complements the [analytic navigation comparison](185-navigation-field-comparison.md) and [matched Controls trajectories](185-contour-controls-trial.md). It supplies validity controls for [#185](https://github.com/rhythmatician/FractalSync/issues/185).

The checker reuses the existing interval orbit recurrence \(P_0=0,\ P_{n+1}=P_n^2+c\) with 60-digit interval precision. It examines at most \(P_1,\ldots,P_{12}\) and stops before scoring the iterate that meets the proved escape guard \(|P_n|\ge4,\ |c|/|P_n|^2\le1/4\). A small return means a verified squared magnitude at most \(10^{-8}\) in this finite prefix. The check aborts if an interval makes the discrete threshold outcome ambiguous. An index remains a candidate unless a competitor's upper score bound lies strictly below its lower score bound. [Interval orbit evaluator and guard](../../formal/checks/check_exterior_ruler_jets.py), [exact selection lemmas](185-trap-selection-validity.md).

The exact centers are \(c=0\), \(c=-1\), and the real root near \(-1.75487766624669276005\) of \(c^3+2c^2+c+1=0\). The checker independently brackets the last root with exact rational endpoint tests. The polynomial is strictly increasing on the bracket, and \(P_3=c(c^3+2c^2+c+1)\). Thus \(P_3=0\) with \(P_1\ne0\), which gives exact least period three. Exact returns in the recurrent rows are known from the fixtures. The threshold test itself only detects a small finite return.

| Parameter | Scored prefix | First small return | Escape guard step | Closest to origin | Closest to \(-1\) |
| --- | ---: | ---: | ---: | --- | --- |
| Period-one center | 12 | 1 | | 1–12 | 1–12 |
| Period-two center | 12 | 2 | | 2, 4, 6, 8, 10, 12 | 1, 3, 5, 7, 9, 11 |
| Period-three center | 12 | 3 | | 3, 6, 9, 12 | 1, 4, 7, 10 |
| Nearby \(c=-0.99999\) | 12 | 2 | | 2 | 1 |
| Near-cusp \(c=0.2501\) | 12 | | | 1 | 1 |
| Exterior \(c=0.5+0.5i\) | 5 | | 6 | 1 | 4 |

The candidate sets report unresolved iteration identity when repeats or interval overlaps survive. Moving the point trap can change the selected phase of the same orbit. A third point configuration at \(1/4+i/8\) is included in the executable. A real-axis line trap ties at every iterate of all five real fixtures, even for the near-cusp transient. At \(c=-0.99999\), \(P_2=c(c+1)\) is nonzero yet below the return threshold. At \(c=0.2501\), the real orbit is known to escape independently of the 12-iterate guard. Its real increments satisfy \(P_{n+1}-P_n=(P_n-1/2)^2+0.0001\), so finite nonescape does not support an interior classification.

```powershell
.venv/Scripts/python.exe formal/checks/check_185_finite_recurrence.py
```

Keep these readouts experimental. The fixture distinguishes exact return, finite recurrence evidence, escape, and score-based selection. It exhibits configuration and parity effects, including cases where an index cannot be chosen uniquely. It measures neither predictive gain beyond existing Map features nor transfer to new trajectories and components. That question needs held-out comparisons for a declared navigation or visual target. The full research decision remains open.
