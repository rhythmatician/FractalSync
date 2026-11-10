# #145 closed attracting covers — 2026-10-26

PR #148 is not ready to merge. This experiment adds certified interior distance
lower bounds at both recorded high-period queries. It does not replace the
production provider, establish a distance jet, or clear the original crossings.
Thresholds and crossing tests remain unchanged.

## Construction and authority

The isolated Rust precision probe now assembles periodic-root certificates into
closed parameter curves. Every multiplier edge is strictly attracting, lies
inside a certified branch domain, and joins its neighbors on the same physical
cycle (allowing a certified phase advance). The final multiplier endpoint must
equal the first exactly. Automatic error-domain proposals still pass the
original strict inclusion and contraction gates.

Each parameter edge is enclosed by a capsule around its affine predictor.
Connector capsules enclose the homotopies between adjacent predictor endpoints.
Directed bounds must separate the query from every edge and connector. Only then
does an outward-checked ray count establish nonzero winding. Ambiguity, exhausted
work, incomplete closure, incompatible branches, and zero winding all refuse.
Numerical Newton solves and loop proposals never issue proof objects directly.

The API retains its exact query and directed lower bound privately. Decimal
values in diagnostics are approximate renderings, not proof comparisons.

## Why this bounds Mandelbrot interior distance

The mathematical premises are that the Mandelbrot set is full and that a
quadratic polynomial with an attracting finite cycle belongs to it. See
[Jung, *Homeomorphisms of the Mandelbrot Set*, section 2.3](https://web.ma.utexas.edu/mp_arc/c/03/03-529.pdf).
The latter implication uses attraction of the unique finite critical point.
A nonminimal certified return period is sufficient: its multiplier is a power
of the primitive multiplier, so modulus below one implies primitive attraction.

The following is the argument applied here, independently reviewed twice.
Let the certified closed attracting curve be gamma, and let its separation from
q be at least L > 0. Winding is zero near infinity and constant on the connected
complement of the full Mandelbrot set. It is therefore zero everywhere outside
the set. On the open disk B(q,L), which avoids gamma, winding is constant and
nonzero. Thus that entire open disk lies in the interior of the Mandelbrot set,
and distance from q to its boundary is at least L. The curve need not be simple.

This supplies neither an upper bound nor a nearest-boundary certificate.
It concerns the mathematical quadratic family at the represented parameter;
equivalence to the canonical finite floating orbit has not been established.

## Loop proposals and cost

Circular multiplier loops and a root-dented alternative share all proof gates.
The latter lowers the multiplier radius near angle zero, using a Gaussian
proposal with a configurable width, to avoid expensive subdivision near multiplier
one. All resulting straight edges still require strict attraction and the full
closure/winding check. These are alternative certified inner curves, not
substitutes for a completed near-boundary circle or global boundary identity.

Terminal results, commands, binary/source hashes and logs are recorded in the
accompanying JSON. The first circular inner runs at radius 0.9 completed with:

| Query | Period | Edges | Approximate lower bound | Cold wall time |
| --- | ---: | ---: | ---: | ---: |
| Native | 85 | 166 | 2.8772785041210325e-6 | 90.58 s |
| Python | 154 | 142 | 9.329120433167717e-8 | 111.07 s |

These costs are unsuitable for live queries. Shared machine load affects wall
times; the experiments ran concurrently. The root-dented native run completed
with 175 edges and lower bound approximately 4.791160347921405e-6 in 117 seconds.
Its Python counterpart completed with 146 edges and lower bound approximately
1.7240936517760065e-7 in 141.64 seconds. Both radius-0.9 circular results were
reproduced with the subsequently frozen executable.

Narrowing the root dent to 0.05 radians improves the native bound while retaining
completed closed proofs:

| Query | Edges | Approximate lower bound | Cold wall time |
| --- | ---: | ---: | ---: |
| Native | 234 | 6.95629739922484e-6 | 170.02 s |
| Python | 172 | 1.7240936517760065e-7 | 175.60 s |

The unchanged near-unit circular constructions at radius 0.999999 and 512 initial
edges exhaust their 1200-second process limits. Their accepted prefixes are not
closed-cover certificates. The dented alternatives establish neither completion
of those circles nor a live-query budget.

## Verification and remaining acceptance work

The isolated crate passes 39 tests across both binaries. All 26 retained and
new mutations are caught by assertion failures, rather than compilation errors.
New coverage includes successful end-to-end circular and dented cardioid
construction, distinct-component join refusal, independent connector separation,
edge intersection, uncertain orientation, closure, winding and budget exhaustion
after an accepted prefix. Mutation scripts restore exact original source bytes.

The unchanged original native crossing was rerun and still fails at
(0.21474166530729011, 0.12774018997553147), `CutLocus`, provider 6.
The Python fixture initially failed while trying to overwrite its own loaded
extension; a separate-process rebuild and verified extension precede the actual
crossing rerun recorded in the JSON.
That rerun reaches the original assertion and still refuses at
(0.35659381055433242, 0.06404656757776669), `Unresolved`, provider 6.

Merge readiness still requires a coherent production distance field with
consistent derivatives, reliable nearest/cut-locus classification, acceptable
query cost, and successful original native/Python crossings and live training.
Any adopted canonical change also requires version/golden updates and the
project's parity, architecture, Rust, Python, browser/WASM and CI gates. This
experiment alone changes none of those acceptance outcomes.
