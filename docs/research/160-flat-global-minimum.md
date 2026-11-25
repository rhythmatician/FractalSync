# #160: the flat Shore minimum, threshold, and equality case

The actual pairwise AC path infimum now has a global lower bound below the
flat shortcut threshold. Together with the earlier strict-shortcut construction,
this proves the exact threshold. The equality argument also classifies every
minimizing AC path at and below the threshold, including the endpoint
`L = pi * epsilon`.

The new sources are
[FlatShoreCalibration.lean](../../formal/FractalSync/FlatShoreCalibration.lean),
[FlatShoreMinimizers.lean](../../formal/FractalSync/FlatShoreMinimizers.lean),
[FlatShoreIsometries.lean](../../formal/FractalSync/FlatShoreIsometries.lean),
[FlatShoreThreshold.lean](../../formal/FractalSync/FlatShoreThreshold.lean), and
[ACScalarVariation.lean](../../formal/FractalSync/ACScalarVariation.lean).
They use the existing [flat Euclidean bridge](160-flat-intrinsic.md) and
[actual strict-shortcut theorem](160-flat-second-variation.md).

## 1. The actual path problem

Write `(r,y)` for normal and tangential coordinates in the actual Euclidean
plane, let Shore be `r=0`, and assume `epsilon>0`. The coordinate expression
already identified with the general AC graph-length functional is

\[
g_{\epsilon,a}
=\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2}\,dr^2
+\frac{1}{\epsilon^2+r^2}\,dy^2.
\]

The parameter `a` may be any real number in these theorems. Let
`D_epsilon,a` denote the actual infimum of graph lengths over all ordinary
Euclidean AC curves on `[0,1]` joining the specified endpoints.

For a nonnegative Shore separation `L`, the results give

\[
\boxed{
D_{\epsilon,a}((0,0),(0,L))=\frac L\epsilon
\quad\text{if }0\le L\le\pi\epsilon,
}
\tag{1}
\]

and the exact strict-shortcut criterion

\[
\boxed{
D_{\epsilon,a}((0,0),(0,L))<\frac L\epsilon
\quad\Longleftrightarrow\quad L>\pi\epsilon.
}
\tag{2}
\]

The forward implication in (2) uses the new unrestricted lower bound. The
reverse implication uses the earlier actual sine competitor and its checked
second derivative. Neither proof assumes that a minimizing path exists.
The previously proved general attainment theorem remains available separately.

## 2. A global calibration with a regularization parameter

Normalize `epsilon=1`, and write `R=1+r²`. Define the comparison speed

\[
\sigma(r;v_r,v_y)
=\sqrt{\frac{v_r^2}{R^2}+\frac{v_y^2}{R}}.
\tag{3}
\]

This is the round-sphere metric in latitude `arctan r` and longitude `y`.
The formal argument uses the explicit scalar function (3); it does not need
a bundled sphere, an external comparison theorem, or an assumed spherical
distance formula.

For `0<c<1`, use the globally regular scalar calibration

\[
\phi_c(r,y)=\arccos\!\left(\frac{c\cos y}{\sqrt{1+r^2}}\right).
\tag{4}
\]

Its arccos argument lies strictly between `-1` and `1` everywhere. Its actual
Fréchet derivative is checked:

\[
D\phi_c(r,y)[v]
=\frac{c\bigl(r\cos y\,v_r+R\sin y\,v_y\bigr)}
 {R\sqrt{R-c^2\cos^2 y}}.
\tag{5}
\]

The denominator is strictly positive. Set

\[
\Delta=R-c^2\cos^2y,\quad
V=v_r^2+Rv_y^2,\quad
N=r\cos y\,v_r+R\sin y\,v_y.
\]

The pointwise derivative bound follows from the checked sum-of-squares identity

\[
\boxed{
\Delta V-c^2N^2
=R(1-c^2)V
+c^2R\bigl(r\cos y\,v_y-\sin y\,v_r\bigr)^2\ge0.
}
\tag{6}
\]

Consequently

\[
|D\phi_c[v]|\le\sigma(r;v)
\le\sqrt{g_{1,a}(v,v)}.
\tag{7}
\]

The first inequality gives a global Lipschitz bound for the calibration in
the coordinate product norm. Composition with an ordinary AC path is therefore
AC. The existing a.e. chain rule and AC fundamental theorem apply to the actual
composite. The comparison speed is integrable along every admitted path.
For any AC path from `(0,0)` to `(0,ell)`, integration yields

\[
\phi_c(0,\ell)-\phi_c(0,0)
\le\int_0^1\sigma(\gamma(t);\dot\gamma(t))\,dt
\le L_{1,a}(\gamma).
\tag{8}
\]

For `0<=ell<=pi`, continuity as `c` tends to `1` makes the left side tend to
`acos(cos ell)-acos 1 = ell`. Lean obtains the endpoint inequality by closure
of the relevant scalar inequality set. The proof never differentiates arccos
at either singular endpoint. Thus

\[
\ell\le L_{1,a}(\gamma)
\quad\text{for every admitted AC path.}
\tag{9}
\]

Passing (9) through the actual candidate-length infimum and comparing with
the straight Shore segment proves (1) in normalized coordinates. This is a
global result: the path can backtrack or make an arbitrarily large excursion.

## 3. Equality forces the whole path onto Shore

The difference between the squared flat speed and (3) is

\[
g_{1,a}(v,v)-\sigma(r;v)^2
=\frac{(1+a^2)r^2v_r^2}{(1+r^2)^2}.
\tag{10}
\]

Suppose the actual graph length equals `ell`. After taking the calibration
limit, the chain `ell <= sphericalLength <= acGraphLength` must consist of
equalities. The regularized endpoint inequality for a fixed `c<1` need not
be an equality. Nonnegativity and integrability force
the two speeds to agree almost everywhere. Since `1+a²>0`, (10) gives

\[
r(t)\dot r(t)=0\quad\text{almost everywhere.}
\]

The square of an AC scalar trace is AC on this compact interval, and its
actual derivative is `2 r dot(r)` almost everywhere. The AC fundamental theorem
therefore makes `r²` constant. Its initial value is zero, so

\[
r(t)=0\qquad(0\le t\le1).
\tag{11}
\]

This argument includes `a=0`, the zero span, and `ell=pi`. It does not assume
smoothness of the minimizing path, nonvanishing speed, an Euler--Lagrange
equation, or a geodesic classification.

On Shore, the actual graph length reduces to the scalar total variation

\[
L_{1,a}(\gamma)=\int_0^1|\dot y(t)|\,dt.
\]

The new generic AC theorem proves

\[
\int_\alpha^\beta|f'(t)|\,dt=f(\beta)-f(\alpha)
\quad\Longleftrightarrow\quad
f\text{ is nondecreasing on }[\alpha,\beta]
\tag{12}
\]

for every AC real function and `alpha<=beta`, including the degenerate
interval. It uses the actual derivative and the AC fundamental theorem on
subintervals, with endpoint derivatives excluded only on a null set.

The complete path statement is therefore

\[
\boxed{
L_{1,a}(\gamma)=\ell
\quad\Longleftrightarrow\quad
r\equiv0\text{ on }[0,1]
\ \text{and}\ y\text{ is nondecreasing there},
}
\tag{13}
\]

under the prescribed endpoints and `0<=ell<=pi`. Positive dilation gives
the same classification with length `L/epsilon` for `0<=L<=pi*epsilon`.
Continuity and the endpoint values then identify the minimizing image as
the Shore segment. Parameterizations need not be unique: monotone AC
traversals can change speed and pause. At `L=0`, the classification forces
the constant path.

## 4. Dilation and translation apply to the full infimum

The isometry module proves equality of the entire candidate-length sets,
using forward and inverse transformations of actual AC paths. It obtains

\[
D_{\epsilon,a}(\epsilon x,\epsilon y)=D_{1,a}(x,y)
\quad(\epsilon>0)
\tag{14}
\]

and, for every real tangential shift `b`,

\[
D_{\epsilon,a}(x+(0,b),y+(0,b))=D_{\epsilon,a}(x,y).
\tag{15}
\]

The translation also preserves the actual coordinate and AC graph integrals.
The dimensional threshold statement is thus about the same unrestricted
Euclidean path metric used throughout the package. No minimizer is selected
to justify a change of variables.

The physical theorems are packaged for arbitrary real tangential source `b`,
with endpoints `(0,b)` and `(0,b+L)`. `intrinsicDistance_eq_shore_length` and
`minimizing_path_iff` give (1) and (13) in those coordinates.
`intrinsicDistance_lt_shore_length_iff` gives (2), and
`exists_shorter_path_iff` states the equivalent existence of an actual shorter
AC path. The latter passes from a strict infimum inequality to a candidate
using strict infimum approximation, without an attainment premise.

## 5. Formal boundary

The global minimum and equality classification at and below the threshold,
and the strict shortcut above it, are now kernel checked. The existing normal
Jacobi-field calculation explains the occurrence of `pi`, but the calibration
proof does not rely on a general conjugate-point or index-form theorem.

The following remain separate:

* Classifying all minimizers above the threshold, including the conventional
  claim of exactly two reflected excursions.
* Proving the global flat cut locus from the actual path metric. Its prior-art
  derivation remains conventional in [the flat-Shore note](160-flat-shore.md).
* Establishing smoothness and the geodesic ODE for arbitrary attained AC
  minimizers, and formalizing general Jacobi, second-variation, and conjugacy
  criteria.
* Formalizing the curved-Shore functional-analytic branch and local minimum.

The full module inventory and transitive axiom audit are recorded in
[formal/README.md](../../formal/README.md) and [verification.md](verification.md).

| New module | Explicit proved theorem/lemma declarations |
| --- | ---: |
| `ACScalarVariation` | 3 |
| `FlatShoreCalibration` | 13 |
| `FlatShoreMinimizers` | 6 |
| `FlatShoreIsometries` | 9 |
| `FlatShoreThreshold` | 5 |
| Total | 36 |

The targeted `lake build FractalSync.FlatShoreThreshold --wfail` passed with
2,602 jobs and no warnings. Counts include supporting lemmas.

The runtime, provider contracts, training, canonical parameters, and dependency
pins are unchanged. The research issues remain open and PR #164 remains a draft.
