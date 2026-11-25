# #160: metric curve length, attained pairwise minima, and Shore minimizers

The AC graph-length functional is now identified with the actual metric
curve length induced by its path infimum. The positive-floor distance is a
proper length metric, and every pair of endpoints has a minimizing AC path
with constant intrinsic speed. These statements are Lean checked.

The main setting is a finite-dimensional real normed space \(E\), a set
\(S\subseteq E\), \(\epsilon>0\), and \(a\in\mathbb R\). The positive-floor
results do not require closedness, nonemptiness, convexity, smoothness, or
uniqueness of nearest points. For the physical Shore interpretation one
usually takes \(S\) nonempty and closed. The all-minimizer segment
classification below additionally uses a strictly convex ambient norm,
which includes finite-dimensional real inner-product spaces.

Write
\[
r(x)=\operatorname{dist}(x,S),\qquad
\rho(x)=\sqrt{\epsilon^2+r(x)^2}.
\]
Here \(r\) is Mathlib's real-valued `Metric.infDist`. For empty \(S\) its
value is zero, not the extended-distance convention \(+\infty\). The
arbitrary-set statements use that exact convention. All pathwise AC,
monotonicity, constancy, and geodesic assertions below concern the parameter
interval \([0,1]\).
The existing definitions are retained:
\[
L(\gamma)=\int_0^1\ell_\gamma(t)\,dt,
\qquad
\ell_\gamma(t)=
\sqrt{\frac{\|\gamma'(t)\|^2}{\rho(\gamma(t))^2}
+\frac{a^2r(\gamma(t))^2(r\circ\gamma)'(t)^2}
       {\rho(\gamma(t))^4}},
\]
\[
D(x,y)=\inf\{L(\gamma):\gamma\text{ is AC},\;
\gamma(0)=x,\ \gamma(1)=y\}.
\]
Here AC is ordinary absolute continuity in the original norm. Its actual
finite-dimensional a.e. derivatives and the integrability of the displayed
density were established in the [earlier AC package](160-absolute-continuous-paths.md).
The [pairwise metric note](160-pairwise-metric-limit.md) supplies the metric
axioms, topology, properness, completeness, and fixed-pair zero-floor limits.

## 1. Subpaths and cumulative length

For \(0\le u\le v\le1\), define an actual affine subpath
\[
\gamma_{u,v}(t)=\gamma((v-u)t+u).
\]
Lean proves AC preservation and the exact integral identity
\[
L(\gamma_{u,v})=\int_u^v\ell_\gamma(t)\,dt.
\tag{1}
\]
The formula includes \(u=v\). It uses the actual affine change of variables
and homogeneous density, rather than a stipulated reparameterization law.

Define the cumulative integral
\[
A(t)=\int_0^t\ell_\gamma(s)\,ds.
\]
It is AC, nondecreasing on \([0,1]\), and satisfies
\[
A(0)=0,\qquad A(1)=L(\gamma),\qquad
D(\gamma(s),\gamma(t))\le |A(s)-A(t)|.
\tag{2}
\]
This last inequality follows by using (1) as an admissible path. Consequently
every ordinary AC path is also AC with respect to the intrinsic metric.
The formal theorem supplies that metric instance explicitly; it never
replaces the original normed-space structure globally.

Sources: [ACSubpaths.lean](../../formal/FractalSync/ACSubpaths.lean),
[IntrinsicLength.lean](../../formal/FractalSync/IntrinsicLength.lean), and
[IntrinsicVariation.lean](../../formal/FractalSync/IntrinsicVariation.lean).

## 2. The lifted chord recovers the full density

An endpoint displacement estimate alone cannot recover the ruler-change
term in \(\ell_\gamma\). Use instead the ordinary \(\ell^2\) product lift
\[
\Phi(x)=(x,a\rho(x))\in E\times_2\mathbb R.
\]
The AC chain rule proves
\[
\| (\Phi\circ\gamma)'(t)\|
=\rho(\gamma(t))\ell_\gamma(t)
\quad\text{a.e.}
\tag{3}
\]
No ambient derivative of \(r\) is required. Its scalar trace along an AC
curve has the derivatives needed for (3).

The already checked logarithmic ruler estimate gives
\[
\rho(\gamma(t))\le \rho(\gamma(0))e^{L(\gamma)}.
\]
Vector-valued AC FTC, (3), and integration therefore yield
\[
\|\Phi(\gamma(1))-\Phi(\gamma(0))\|
\le\rho(\gamma(0))e^{L(\gamma)}L(\gamma).
\]
Passing through the infimum gives the endpoint estimate
\[
\boxed{\|\Phi(y)-\Phi(x)\|\le\rho(x)e^{D(x,y)}D(x,y).}
\tag{4}
\]
The formal proof uses the closedness of the inequality as a condition on
the real path length and membership of the infimum in the closure of the
nonempty bounded-below length set. It assumes no minimizing curve.

More generally, suppose an AC real function \(B\) controls all path
increments by \(D(\gamma(s),\gamma(t))\le|B(s)-B(t)|\). Apply (4) at nearby
parameter values and divide by their separation. At points where both AC
derivatives exist, the exponential factor tends to one, giving
\[
\|(\Phi\circ\gamma)'(t)\|\le\rho(\gamma(t))|B'(t)|.
\]
Combining with (3) and cancelling the positive ruler proves
\[
\ell_\gamma(t)\le|B'(t)|\quad\text{a.e.}
\tag{5}
\]
This is `ac_graphSpeed_le_control_deriv`. The derivative comparison is
proved using actual punctured-neighborhood slope limits; it is not an
assumed metric-derivative identity.

Sources: [GraphChord.lean](../../formal/FractalSync/GraphChord.lean),
[ACVectorFTC.lean](../../formal/FractalSync/Analysis/ACVectorFTC.lean), and
[DerivativeControl.lean](../../formal/FractalSync/Analysis/DerivativeControl.lean).

## 3. Equality with metric curve length

Let \(\operatorname{Len}_D(\gamma)\) be the supremum of finite partition
sums of \(D\)-distances. In Lean this is Mathlib's actual `eVariationOn`,
with the explicit intrinsic metric instance. Summing (2) over a partition
gives
\[
\operatorname{Len}_D(\gamma)\le L(\gamma).
\tag{6}
\]
In particular the extended-real variation is finite.

Let \(V(t)\) be the real intrinsic variation from parameter zero to \(t\).
The checked partition and variation identities give, for \(s,t\in[0,1]\),
\[
D(\gamma(s),\gamma(t))\le |V(s)-V(t)|
\le |A(s)-A(t)|.
\tag{7}
\]
Thus \(V\) is nondecreasing and AC. Apply (5) with \(B=V\). Its derivative
is nonnegative a.e., so
\[
L(\gamma)\le\int_0^1V'(t)\,dt
=V(1)-V(0)=\operatorname{Len}_D(\gamma).
\]
Together with (6), this proves
\[
\boxed{\operatorname{Len}_D(\gamma)=L(\gamma).}
\tag{8}
\]
The real conversion is made only after finite variation is established.
The extended-real theorem is `eVariationOn_intrinsic_eq_acGraphLength`;
`acGraphLength_eq_intrinsicVariation` gives the terminal real variation
form. Both are in
[IntrinsicCurveLength.lean](../../formal/FractalSync/IntrinsicCurveLength.lean).

For every \(\delta>0\), the original infimum supplies an actual AC path
from \(x\) to \(y\) with metric curve length below \(D(x,y)+\delta\).
`exists_metric_length_lt_intrinsicDistance_add` packages this length-space
property as an explicit witness theorem. No separate Mathlib `LengthSpace`
instance is asserted.

## 4. Actual minimizing paths between arbitrary endpoints

The existence proof does not assume lower semicontinuity of the original
derivative-dependent functional. It uses the intrinsic metric already
proved proper.

First, (2) reparameterizes every candidate into an endpoint path with
intrinsic Lipschitz constant at most its graph length. The construction
chooses a parameter on each cumulative-length level. Flat portions of
\(A\) cause no ambiguity in the resulting point: (2) forces the path to be
constant on each level set. No inverse derivative or strict monotonicity
of \(A\) is required.

For each positive \(\delta\), consider endpoint paths with speed at most
\(D(x,y)+\delta\). They form a closed subset of the pointwise product of
compact intrinsic balls. Properness makes this family compact; the families
are nested as the speed bound decreases. The nonempty compact intersection
therefore contains a path with speed at most \(D(x,y)\). The triangle
inequality between its two endpoints forces equality on every subinterval:
\[
\boxed{D(\gamma(s),\gamma(t))=D(x,y)|s-t|.}
\tag{9}
\]
This compactness argument is checked in the general metric support module
[ProperGeodesics.lean](../../formal/FractalSync/Analysis/ProperGeodesics.lean).

The intrinsic displacement and ruler bounds give this path an ordinary
Lipschitz constant at most
\[
\rho(x)e^{2D(x,y)}D(x,y).
\]
It is therefore admissible in the original AC path class. Equation (5)
with the linear control \(B(t)=D(x,y)t\) gives \(L(\gamma)\le D(x,y)\),
while the definition of the infimum gives the opposite inequality. Hence
\[
\boxed{L(\gamma)=D(x,y).}
\tag{10}
\]
The existence theorem includes \(x=y\); then the metric geodesic is constant.

The principal declarations are `exists_intrinsic_ac_metricGeodesic` in
[IntrinsicGeodesics.lean](../../formal/FractalSync/IntrinsicGeodesics.lean),
`exists_acGraphLength_minimizer`, and
`intrinsicDistance_isLeast_pairwisePathLengths` in
[IntrinsicCurveLength.lean](../../formal/FractalSync/IntrinsicCurveLength.lean).
The conclusion is existence and constant intrinsic speed. It does not
assert uniqueness, ambient smoothness, or satisfaction of a smooth geodesic
differential equation at nonsmooth distance-field points.

## 5. All paths attaining the nearest-Shore distance

Suppose the ambient norm is strictly convex, \(\gamma(0)\in S\), and
\(\gamma\) is AC. The theorem `minimizing_shore_path_iff` in
[ShoreMinimizers.lean](../../formal/FractalSync/ShoreMinimizers.lean) proves
that
\[
L(\gamma)=F_{\epsilon,a}(r(\gamma(1)))
\]
holds exactly in the following cases:

- If \(r(\gamma(1))=0\), the whole path is constant.
- If \(r(\gamma(1))>0\), its starting point \(p=\gamma(0)\) is an actual
  nearest point in \(S\), and
  \[
  \boxed{\gamma(t)=p+\theta(t)(\gamma(1)-p),}
  \tag{11}
  \]
  where \(\theta\) is AC, nondecreasing, \(\theta(0)=0\), and
  \(\theta(1)=1\).

Both directions are checked, including traversals with pauses. The canonical
parameter in the necessity proof is
\[
\theta(t)=\frac{r(\gamma(t))}{r(\gamma(1))}.
\]
For paths toward Shore, a separate checked reversal theorem gives a
nonincreasing parameter from one to zero.

The proof extracts equality almost everywhere from the integrated signed
radial calibration. The nonnegative density defect has zero integral, so
\[
\|\gamma'(t)\|=(r\circ\gamma)'(t)\quad\text{a.e.}
\]
Integrating on subintervals, using vector AC FTC, and applying the
one-Lipschitz distance field proves exact displacement equality on each
ordered subinterval. The starting point is consequently nearest. Strict
convexity turns equality in the triangle inequality into membership in the
ordinary endpoint segment; the nearest-segment distance identity determines
the parameter. The converse differentiates the actual monotone AC segment
traversal and integrates its calibrated density.

Closedness is not needed for this equality-case theorem because an actual
starting point in \(S\) is already given. For a nonempty closed Shore, the
earlier identity between `intrinsicShoreDistance` and the radial profile
identifies this equality with minimizing the actual Shore-distance
infimum. The zero-distance and nearest-endpoint parts need no strict
convexity; that hypothesis is used for straight-segment classification.

The theorem does not assert uniqueness of the nearest point. It describes
the entire family: one geometric segment for each nearest point, with all
monotone AC traversals. In particular, a unique nearest point determines a
unique minimizing image but still permits different parameterizations.
The image is independent of \(\epsilon\) and \(a\). These last observations
are immediate mathematical consequences of the classification, not
additional named Lean uniqueness theorems.

## 6. Relationship to the remaining geometry

The metric curve-length bridge, positive-floor length-space property,
existence of arbitrary-endpoint AC minimizers, and complete nearest-Shore
segment classification are now checked. They are no
longer assumptions deferred by the pairwise package. The componentwise
zero-floor limit remains the earlier pointwise fixed-pair theorem; this
increment does not add uniform convergence over endpoints or convergence
of selected minimizers.

The flat cut locus and curved-Shore bifurcation still require their own
geometric arguments. Properness and attainment do not establish a cut
locus, smooth geodesic classification, conjugate points, or the
Lyapunov--Schmidt reduction. These distinctions are maintained in
[the flat-Shore note](160-flat-shore.md).
