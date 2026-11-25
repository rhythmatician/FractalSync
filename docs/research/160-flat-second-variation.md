# #160: actual sine-graph second variation and strict shortening

The flat-Shore package now proves the actual second amplitude derivative of
a literal curve-length integral. For every normalized Shore span `ell > pi`,
it also constructs a positive interval of sine amplitudes whose curves are
absolutely continuous, have the same two endpoints, and are strictly shorter
than the Shore segment.

The second-variation module is
[FlatShoreSecondVariation.lean](../../formal/FractalSync/FlatShoreSecondVariation.lean)
(24 explicit theorem declarations). The continuation adds
[FlatShoreShortcuts.lean](../../formal/FractalSync/FlatShoreShortcuts.lean)
(11 declarations), connecting the competitor to the actual unit-interval AC
path infimum and proving the physical positive-floor shortcut. One generic integration theorem is in
[Analysis/CompactParametricIntegral.lean](../../formal/FractalSync/Analysis/CompactParametricIntegral.lean)
and is counted as analysis support. These are exact-real statements under the
existing Lean and Mathlib pins.

## 1. The literal metric and fixed-endpoint sine family

The calculation uses the existing normalized flat coordinate metric with
`epsilon = 1`:

\[
g=
\frac{1+(1+a^2)q^2}{(1+q^2)^2}\,dq^2
+\frac{1}{1+q^2}\,dy^2.
\tag{1}
\]

Fix `ell > 0`, set `k = pi/ell`, and consider the actual curve

\[
\gamma_A(t)=\left(A\sin t,\frac{\ell}{\pi}t\right),
\qquad 0\le t\le\pi.
\tag{2}
\]

Lean proves its actual derivative

\[
\dot\gamma_A(t)=\left(A\cos t,\frac{\ell}{\pi}\right),
\]

its fixed endpoints `gamma_A(0)=(0,0)` and `gamma_A(pi)=(0,ell)`, and the
graph identity

\[
q_A(y)=A\sin\left(\frac{\pi y}{\ell}\right).
\]

The curve is proved absolutely continuous on every finite parameter
interval. Its actual derivative has product-norm bound
`max(|A|, |ell/pi|)`, so the mean value theorem gives a global Lipschitz bound
and hence absolute continuity. The competitors are therefore explicit
admissible coordinate curves, not formal variation fields alone.

The use of `t` in `[0,pi]` avoids a trigonometric change of variables in the
coefficient calculation. The graph-coordinate identity explicitly relates
this parameterization to the requested sine graph on `y` in `[0,ell]`.

## 2. The defined functional equals the actual coordinate length

With `s = sin(t)` and `c = cos(t)`, define

\[
N_{a,k}(A,t)=1+A^2(s^2+k^2c^2)
 +(1+a^2)A^4s^2k^2c^2,
\qquad D(A,t)=1+A^2s^2,
\]

\[
f_{a,k}(A,t)=\frac{\sqrt{N_{a,k}(A,t)}}{D(A,t)}.
\]

Both the radicand and denominator are strictly positive for every real
`a,k,A,t`. Expanding the actual existing bilinear metric at the actual curve
and tangent gives the checked identity

\[
g_{\gamma_A(t)}(\dot\gamma_A(t),\dot\gamma_A(t))
=\left(\frac\ell\pi\right)^2
 \frac{N_{a,\pi/\ell}(A,t)}{D(A,t)^2}.
\tag{3}
\]

The positive-span assumption removes the absolute value when taking the
square root. Consequently the defined integral

\[
\boxed{
\mathcal L_{\ell,a}(A)
=\frac\ell\pi\int_0^\pi f_{a,\pi/\ell}(A,t)\,dt
=\int_0^\pi
 \sqrt{g_{\gamma_A(t)}(\dot\gamma_A(t),\dot\gamma_A(t))}\,dt
}
\tag{4}
\]

is exactly the actual coordinate length. The rightmost Lean expression uses
`deriv (modeCurve ell A) t`, replaced by its proved actual derivative.
It does not take a proposed speed as an unverified input.

The source defines `oneModeLength` for all real `ell` because its algebraic
derivative statements remain valid there. For `ell <= 0` it is a signed
algebraic functional; its geometric-length interpretation in (4) explicitly
requires `ell > 0`.

## 3. Actual differentiation under the integral

The support theorem proves differentiation of a parameter integral from
joint continuity of the integrand and its actual parameter derivative.
It derives a uniform local derivative bound by compactness of a rectangle
`closedBall(A,1) × Icc(lo,hi)`, then applies Mathlib's dominated parametric
integral theorem. Its hypotheses do not include a derivative of the
integrated functional.

For (4), the source differentiates the square root and quotient explicitly.
The first density derivative has the form `A * modeRateFactor`. Joint `C^1`
regularity of this rate follows from the strictly positive radicand and
denominator. In particular the checked density jets are

\[
\partial_A f_{a,k}(0,t)=0,
\qquad
\partial_A^2 f_{a,k}(0,t)=k^2\cos^2t-\sin^2t.
\tag{5}
\]

The second differentiation under the integral uses the actual Fréchet
derivative of the jointly `C^1` first-density derivative, evaluated on the
amplitude direction `(1,0)`. The chain rule supplies the actual pointwise
parameter derivatives. Uniqueness of derivatives identifies their values at
zero with (5).

The exact trigonometric integrals then yield

\[
\boxed{
\mathcal L_{\ell,a}(0)=\ell,
\qquad \mathcal L'_{\ell,a}(0)=0,
\qquad
\mathcal L''_{\ell,a}(0)
=\frac\ell2\left[\left(\frac\pi\ell\right)^2-1\right].
}
\tag{6}
\]

The strongest second-derivative declaration is
`hasDerivAt_deriv_oneModeLength_zero`: it proves a `HasDerivAt` statement
about `deriv` of the actual integral functional. This is stronger than merely
evaluating an independently defined second-variation quadratic form.
The second derivative is independent of `a`; this increment does not prove
the fourth derivative of the actual functional or its quartic remainder.

## 4. Strictly shorter AC curves beyond the threshold

For `ell > pi`, the last expression in (6) is strictly negative. The source
proves

\[
\boxed{
\exists\delta>0\quad
\forall A\in(0,\delta),\qquad
\mathcal L_{\ell,a}(A)<\ell.
}
\tag{7}
\]

The argument uses the actual second derivative to show that the actual first
derivative is strictly negative immediately to the right of zero. The mean
value theorem then gives strict antitonicity on a sufficiently short amplitude
interval. The result establishes strict shortening, not merely a weak local
maximum of the amplitude functional.

The amplitude radius `delta` is existential and may depend on the span and
`a`; no numerical lower bound for it is provided. Equations (2), (4), and
(7), together with the proved absolute continuity, give actual shorter
fixed-endpoint coordinate competitors whenever `ell > pi`.

The second-variation module states its amplitude calculation for the
`epsilon = 1` coordinate metric. The following checked extension supplies the
Euclidean path-infimum and physical dilation steps.

## 5. The actual pairwise infimum and physical shortcut

`FlatShoreShortcuts` lifts (2) through the existing continuous linear chart
into the actual Euclidean plane `WithLp 2 (R x R)`, then reparameterizes it
from `[0,pi]` to `[0,1]` using `affineSubpath`. Lean proves that the resulting
`modePath` is an ordinary Euclidean AC path and has exactly the two designated
Shore endpoints. The chart is not assumed to be an isometry for the default
coordinate product norm.

The previously checked AC density identity and exact affine-subpath length
law give

\[
L_{1,a}(\operatorname{modePath}_{\ell,A})
=\mathcal L_{\ell,a}(A).
\tag{8}
\]

Here the left side is the actual `acGraphLength` from the general closed-set
path construction, including the actual distance trace and its a.e.
derivative. The proof permits the unsigned-distance cusp at Shore endpoints.

Every such path is therefore an admissible competitor for the existing
pairwise path infimum, giving

\[
D_{1,a}((0,0),(0,\ell))\le\mathcal L_{\ell,a}(A).
\tag{9}
\]

Choosing any amplitude from (7) proves a strict inequality for the actual
distance. This passage uses the bounded-below candidate-length set and an
explicit competitor. No minimizing curve or geodesic regularity is assumed.

The new module also checks the elementary metric dilation

\[
g_{\epsilon,a}(\epsilon r)(\epsilon v,\epsilon w)
=g_{1,a}(r)(v,w),\qquad \epsilon\ne0,
\]

and, for every actual Euclidean AC curve and positive floor,

\[
L_{\epsilon,a}(\epsilon\gamma)=L_{1,a}(\gamma).
\tag{10}
\]

This is proved through actual coordinate derivatives and the existing
coordinate/graph-length equality. It is not an assumed scaling convention.
With \(\ell=L/\epsilon\), the resulting physical theorem is

\[
\boxed{
\epsilon>0,\quad L>\pi\epsilon
\quad\Longrightarrow\quad
D_{\epsilon,a}((0,0),(0,L))<\frac L\epsilon.
}
\tag{11}
\]

The quantity \(L/\epsilon\) is the intrinsic length of the straight Shore
segment. All real \(a\) are allowed. The new result proves the strict
shortcut above the threshold for these endpoints. The separate
[global calibration continuation](160-flat-global-minimum.md) now proves the
converse, the exact threshold, and all AC equality cases at and below it.

## 6. Verification and remaining scope

The targeted command

```text
lake build --wfail FractalSync.FlatShoreShortcuts
```

completed successfully with 2,596 Lake jobs and no warnings or errors. It
includes the second-variation and compact parametric integral dependencies. Package-wide
verification and the transitive axiom audit are recorded in
[verification.md](verification.md).

| Checked component | Principal declarations |
| --- | --- |
| Actual density calculus | `hasDerivAt_modeDensity`, `hasDerivAt_modeRate_zero`, and joint regularity/positivity lemmas. |
| Actual integral derivatives | `hasDerivAt_modeIntegral`, `hasDerivAt_deriv_modeIntegral_zero`, `hasDerivAt_oneModeLength_zero`, `hasDerivAt_deriv_oneModeLength_zero`. |
| Actual competitor | `hasDerivAt_modeCurve`, `modeCurve_endpoints`, `modeCurve_graph`, `absolutelyContinuousOnInterval_modeCurve`. |
| Existing metric correspondence | `metric_modeCurve`, `metric_speed_modeCurve`, `oneModeLength_eq_coordinateIntegral`. |
| Strict shortening | `deriv_deriv_oneModeLength_neg`, `exists_shortening_amplitude_interval`. |
| Actual unit-interval Euclidean competitor and graph length | `hasDerivAt_euclideanMode`, `absolutelyContinuous_modePath`, `modePath_endpoints`, `acGraphLength_modePath`. |
| Actual path-infimum comparison and physical scaling | `intrinsicDistance_le_oneModeLength`, `intrinsicDistance_lt_shore_span`, `metric_dilation`, `coordinateLength_dilation`, `acGraphLength_dilation`, `intrinsicDistance_le_oneModeLength_scaled`, `intrinsicDistance_lt_shore_length`. |

This is a second-variation and shortening result for one explicit sine
family. It does not prove the general second-variation theorem, identify
arbitrary AC minimizers with smooth geodesics, classify all minimizers, or
establish the global cut locus. At `ell = pi` the proved second derivative
vanishes; equality-case minimality and uniqueness require the separate global
calibration argument, which is now checked in the continuation linked above.
The unit-interval path-infimum wrapper and physical
positive-floor shortcut are now checked by the continuation above.

The earlier [Jacobi calculations](160-flat-geodesics.md) and the conventional
global arguments in [the flat-Shore note](160-flat-shore.md) remain distinct.
The actual length functional, explicit Euclidean AC competitors, and the
general pairwise infimum are now connected through the checked physical
strict-shortcut theorem.
