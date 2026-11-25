# Normalized distance fields: exact charts, error propagation, and sampling limits

Research progress for [FractalSync #162](https://github.com/rhythmatician/FractalSync/issues/162), updated 2026-11-25.

This note derives mathematical contracts; it does not change `PlayerObservation`, the GeometryProvider, runtime authority, or the training schedule. Section 8 distinguishes the Lean-checked algebra, local differential, quadratic remainder, and combined provider/ruler bounds from the conventional proofs in this note. None of the results below is claimed to be novel. The main useful contribution at this stage is an explicit error budget for the actual normalized query, together with limitations that rule out overly strong interpretation of a finite Map tensor.

## 1. Definitions and assumptions

Let `E` be a finite-dimensional Euclidean space and let `D:E→R` be 1-Lipschitz. For a closed region `A`, the sign convention is positive outside and negative inside:

$$
D_A(x)=\operatorname{dist}(x,A)-\operatorname{dist}(x,A^c).
$$

For a nonempty proper region this is the signed distance to its boundary. The results concerning normalized differences only require that `D` be 1-Lipschitz; they need neither a smooth boundary nor a unique nearest point. Define

$$
M_{D,x,h}(u)=\frac{D(x+hu)-D(x)}h,\qquad h>0.
$$

Where `D` is differentiable at `x`, define

$$
R_{D,x,h}(u)=M_{D,x,h}(u)-\nabla D(x)\cdot u.
$$

The exact field obeys `M(0)=0`, `|M(u)|≤|u|`, and
`|M(u)-M(v)|≤|u-v|`. These follow immediately by cancellation and the Lipschitz assumption. None of those statements implies that an arbitrary numerical provider is 1-Lipschitz.

FractalSync specializes to

$$
\rho=\sqrt{D(x)^2+\epsilon^2},\qquad h=q\rho,
\qquad q=2^\zeta>0,\quad\epsilon>0.
$$

The following arguments keep the general physical scale `h` visible before specializing.

### Checked local differential and quadratic residual

The new calculus statements in
[`NormalizedMapDifferential.lean`](../../formal/FractalSync/NormalizedMapDifferential.lean)
hold in any real normed space. They do not require `D` to be an ideal signed
distance or globally 1-Lipschitz. With the anchor `x` and scale `h≠0` held
fixed, differentiability of `D` at the queried point gives

$$
\boxed{d_uM_{D,x,h}(u)=dD(x+hu).}
$$

At `u=0` this is exactly `dD(x)`. The affine query's factor `h` cancels the
normalizing denominator. This is a derivative in the window coordinate,
including when the fixed scale was chosen as `h=qρ(x)`; it does not
differentiate the anchor-dependent ruler. The theorem
`fderiv_normalizedMap` identifies the actual Fréchet differential fields.

Let `K=[x,x+hu]` be the closed physical query segment. Suppose `D` is
Fréchet differentiable at every point of `K`, the actual differential field
`fderiv ℝ D` is Fréchet differentiable there, and its derivative `H(z)` has
operator norm at most `L≥0` throughout `K`, including both endpoints. Then
the checked theorem `normalizedMap_quadratic_remainder_of_hessian_bound`
gives

$$
\boxed{|M_{D,x,h}(u)-dD(x)[u]|
\le\frac{L|h|}{2}\|u\|^2.}
$$

These are ambient Fréchet derivatives at the segment points. In Euclidean
space, `dD(x)[u]=∇D(x)·u`, and for the physical convention `h=qρ(x)>0` the
bound is `Lqρ(x)‖u‖²/2`. The Lipschitz-differential version accepts
`LipschitzOnWith L (fderiv ℝ D) K` instead of a Hessian field; the anchored
variation version needs only
`‖dD(x+thu)-dD(x)‖≤L‖thu‖`, together with differentiability on the segment.
No global `C²` hypothesis is imposed. A uniform residual bound over a window
requires these hypotheses with a common `L` on all its physical query
segments.

Separately, `normalizedMap_value_bound_on_segment` proves
`|M(u)|≤L0‖u‖` when `D` is only `L0`-Lipschitz on `K`, without any
differentiability assumption. The distinction matters at a cut locus: a
normalized value can remain well-defined and controlled while the
differential or Hessian hypotheses fail. None of these results certifies a
numerical provider's gradient or Hessian. The
[differential mechanics note](159-differential-mechanics.md) records the
proof structure and its correspondence to #159 Phase A.

## 2. Exact recentering and rescaling are one identity

**Theorem 1 (chart transition).** For any function `D`, any `h>0`, `α>0`, and `y=x+hv`,

$$
\boxed{
M_{D,y,\alpha h}(u)
=\frac{M_{D,x,h}(v+\alpha u)-M_{D,x,h}(v)}{\alpha}.}
\tag{1}
$$

**Proof.** The two terms in the numerator share the subtraction `D(x)`. Their difference is
`[D(x+hv+αhu)-D(x+hv)]/h`; division by `α` gives the result. No continuity or distance-field assumption is needed.

At differentiable `x,y`, the corresponding residual transition is

$$
R_{D,y,\alpha h}(u)=
\frac{R_{D,x,h}(v+\alpha u)-R_{D,x,h}(v)}{\alpha}
+[\nabla D(x)-\nabla D(y)]\cdot u.
\tag{2}
$$

Taking `v=0` recovers the cross-scale relation from the issue. Taking `α=1` gives pure recentering. This describes redundancy among continuous overlapping windows. A finite sampled window generally lacks the values at `v+αu` needed to execute the transition exactly.

For the particular ruler `h_x=qρ(x)` at fixed `q`, let `s=D(x)/ρ(x)` and let `v=(y-x)/h_x`. Since `D(y)=D(x)+h_x M_{D,x,h_x}(v)`, the next chart's scale ratio is

$$
\boxed{
\left(\frac{h_y}{h_x}\right)^2
=1+2qsM_{D,x,h_x}(v)+q^2M_{D,x,h_x}(v)^2.}
\tag{3}
$$

The normalized center distance `s` is needed; `M` alone deliberately removes the additive field value. Equation (3) is an exact consistency identity, not a request for another state variable. The existing geometry state can supply its terms.

## 3. Sharp stability bounds

**Theorem 2.** For 1-Lipschitz `D`,

$$
|M_{D,x,h}(u)-M_{D,y,h}(u)|\le\frac{2|x-y|}{h},
\tag{4}
$$

and

$$
\boxed{
|M_{D,x,h}(u)-M_{D,x,k}(u)|
\le 2|u|\frac{|h-k|}{\max(h,k)}.}
\tag{5}
$$

The pointwise ceiling `2|u|` also holds, so (4) can be replaced by its minimum with that ceiling.

The upper bounds (4) and (5) are checked by
`normalizedMap_center_change_bound` and `normalizedMap_scale_change_bound`
in [`NormalizedProviderError.lean`](../../formal/FractalSync/NormalizedProviderError.lean).
The latter allows arbitrary nonzero signed scales, with denominator
`max(abs(h),abs(k))` and numerator `2‖u‖ abs(h-k)`. Equation (5) is its
positive-scale specialization. These statements need only a real normed
space and `LipschitzWith 1 D`.

**Proof of (4).** Apply the triangle inequality separately to the two translated queries and the two center values; each difference is bounded by `|x-y|`.

**Proof of (5).** Suppose `k≥h`. Write `A_h=D(x+hu)-D(x)`. Then

$$
\frac{A_k}{k}-\frac{A_h}{h}
=\frac{A_k-A_h}{k}+A_h\left(\frac1k-\frac1h\right).
$$

Use `|A_k-A_h|≤(k-h)|u|` and `|A_h|≤h|u|`. Both resulting terms are `(k-h)|u|/k`. Exchange `h,k` for the other case.

Both factors of two are necessary even for genuine signed-distance fields. In one dimension take the closed interval `[0,2a]`, whose signed distance is `D(t)=|t-a|-a`. For (5), use `x=0,u=1,h=a,k>a`. The difference is exactly `2(k-h)/k`. For (4), take `x=0,y=δ`, with `0<δ<a`, and `h>a`. The difference is exactly `2δ/h`. These sharpness examples remain conventional calculations; the new Lean statements prove the upper bounds.

For a smooth local patch with `|H_D|op≤L` on all rays from `x` to the queried points, there is a different bound:

$$
\boxed{
|M_{D,x,h}(u)-M_{D,x,k}(u)|
\le\frac L2 |h-k|\,|u|^2.}
\tag{6}
$$

Indeed `M_{D,x,h}(u)=∫_0^1∇D(x+thu)·u\,dt`, so differentiation in `h` gives an integral bounded by `L|u|²∫_0^1t\,dt`. The region and Hessian hypothesis must hold throughout the swept segments. Equation (6) cannot be used across a cut locus. Where both bounds apply, take their minimum.

Equation (6) compares two physical scales and remains a conventional proof
here. The checked quadratic residual in Section 1 compares a fixed-scale
query with its anchor linearization; it does not by itself prove (6).

On a patch where `∇D` is `L`-Lipschitz, the same integral representation also gives the sharper center bound

$$
|M_{D,x,h}(u)-M_{D,y,h}(u)|\le L|u|\,|x-y|,
$$

provided the compared rays stay in that patch. Unlike (4), this bound does not diverge as `h→0`. It makes the smoothness assumption, rather than normalization alone, responsible for scale-independent center stability.

## 4. A combined provider-and-ruler error theorem

Let `Dhat=D+e`, but do not assume that `Dhat` is a distance field or Lipschitz. At the same center `x`, set

$$
\rho=\sqrt{D(x)^2+\epsilon^2},\quad
\widehat\rho=\sqrt{\widehat D(x)^2+\epsilon^2},\quad
h=q\rho,\quad\widehat h=q\widehat\rho.
$$

The computed Map query is

$$
\widehat M(u)=
\frac{\widehat D(x+\widehat h u)-\widehat D(x)}{\widehat h}.
$$

Define the **anchored error increment**

$$
\eta(u)=|e(x+\widehat h u)-e(x)|.
$$

**Theorem 3.**

$$
\boxed{
|\widehat M(u)-M_{D,x,h}(u)|
\le \frac{\eta(u)}{q\widehat\rho}
+2|u|\frac{|\widehat\rho-\rho|}{\max(\rho,\widehat\rho)}.}
\tag{7}
$$

**Proof.** Insert the exact query at the computed scale:

$$
\widehat M-M_{D,x,h}
=\frac{e(x+\widehat h u)-e(x)}{\widehat h}
+M_{D,x,\widehat h}-M_{D,x,h}.
$$

Apply (5) to the second term and cancel `q`. This also explains why the error must be evaluated at the *computed* query point `x+hhat u`, not the nominal one.

The exact decomposition is checked by
`normalizedMap_provider_scale_error_identity`. The theorem
`normalizedMap_recomputed_ruler_error_bound` proves (7) using the actual
definition `ruler epsilon s = sqrt(s²+epsilon²)`, with `ε>0`, `q>0`, and an
assumption bounding `abs(e(x+q rhoHat u)-e(x))` by a supplied `η`. Taking
that bound to be the exact increment gives the displayed formula. The
provider `D+e` needs no Lipschitz or differentiability assumption.

If the available guarantees are merely `|e(x)|≤e0` and `|e(x+hhat u)|≤eu`, replace `η` by `e0+eu`; this is `normalizedMap_recomputed_ruler_endpoint_bound`. A certified increment bound can be sharper because it retains error correlation. Consecutive-level estimates supplied by #145 are not automatically such certificates.

The map `s↦sqrt(s²+ε²)` is 1-Lipschitz, hence

$$
|\widehat\rho-\rho|\le |e(x)|.
\tag{8}
$$

This is the reverse triangle inequality for the two-dimensional vectors `(s,ε)` and `(s+e,ε)`. A more informative exact formula, writing `s=D(x)` and `b=e(x)`, is

$$
|\widehat\rho-\rho|
=\frac{|b(2s+b)|}{\widehat\rho+\rho}.
\tag{9}
$$

Equations (8) and (9) are checked by `ruler_perturbation_le` and
`ruler_perturbation_identity`. They hold for every real `s,b` when `ε>0`.

If `e0≤κρ` with `0≤κ<1`, then `ρhat≥(1-κ)ρ`. Thus (7) implies the simple uniform budget

$$
|\widehat M-M|
\le \frac{e0+eu}{q(1-\kappa)\rho}+2\kappa|u|.
\tag{10}
$$

This is `normalizedMap_recomputed_ruler_uniform_bound`. Its assumptions
include both endpoint error bounds and `e0≤κρ`, with `0≤κ<1`.
The correlated version `normalizedMap_recomputed_ruler_relative_bound`
uses `η` in place of `e0+eu` and only requires `abs(e(x))≤κρ` at the
center. The lower bound on `rhoHat` and the relative ruler-error estimate
are separately checked lemmas, rather than assumptions on the conclusion.

The first term improves as the window scale `q` increases. The rough ruler-error term has no such `1/q` improvement. With a certified Hessian bound the latter can instead be bounded by `Lq|ρhat-ρ||u|²/2` using (6).

### Correlated error and the Shore crest

If `e` is a constant `b` across the relevant queries, the anchored error increment is exactly zero. At fixed `h`, adding a constant to `D` leaves `M` unchanged. If the center ruler is recomputed, the scale term remains.

At an exact Shore center `D(x)=0`, however,

$$
\frac{\widehat\rho-\rho}{\widehat\rho}
=\frac{b^2}{\sqrt{\epsilon^2+b^2}
\,[\sqrt{\epsilon^2+b^2}+\epsilon]}
\le\frac{b^2}{2\epsilon^2}.
$$

Therefore a constant bias obeys

$$
\boxed{|\widehat M-M|\le (b/\epsilon)^2|u|.}
\tag{11}
$$

The scalar inequality is `ruler_crest_relative_bound`, and (11) is
`normalizedMap_crest_bias_bound`, for `ε,q>0`, `D(x)=0`, and any real
constant bias `b`. The proof uses the zero anchored error increment in
the combined provider theorem.

The effect is second order in `b/ε` at the crest, not the first-order `2|b|/h` suggested by a correlation-blind field-error bound. This is a bound, not a claim that every field attains it; affine fields are exactly independent of `h`.

For the residual, if a supplied gradient satisfies `|ghat-∇D(x)|≤eg`, add `eg|u|` to (7). This requires a valid derivative at `x`. An invalid gradient cannot be repaired by the fact that `M` is well-defined.

## 5. What finite samples can certify

There is a clean, established answer for function values. Suppose a finite nonempty set of queries `u_i` gives certified intervals `[l_i,r_i]` for an unknown 1-Lipschitz function `f`, including the exact anchor `(0,[0,0])`. Define

$$
L(u)=\max_i\{l_i-|u-u_i|\},\qquad
U(u)=\min_i\{r_i+|u-u_i|\}.
\tag{12}
$$

**Theorem 4 (finite interval envelopes).** A 1-Lipschitz function consistent with all intervals exists if and only if

$$
l_i\le r_j+|u_i-u_j|\quad\text{for every }i,j.
\tag{13}
$$

When consistent, `L` and `U` are both feasible 1-Lipschitz functions, and every feasible `f` obeys `L≤f≤U`. Hence the midpoint `(L+U)/2` has pointwise worst-case error exactly `(U-L)/2` over this feasible class. If the sample set is a `δ`-net of a window and every interval has radius at most `η`, this error is at most `η+δ`.

**Proof.** Necessity follows from `l_i≤f(u_i)≤f(u_j)+|u_i-u_j|≤r_j+|u_i-u_j|`. Maxima/minima of this finite family of cones are 1-Lipschitz. Condition (13) implies that `L(u_i)` and `U(u_i)` lie in the prescribed interval. The cone inequalities bound any feasible `f` from below by `L` and above by `U`. At each query `u`, both endpoints are achieved by feasible functions `L,U`, so no point estimate can have worst-case error below half the interval width. A nearest sample within `δ` gives `U-L≤2(η+δ)`.

This is an interval-data version of the classical McShane–Whitney extremal extension construction, not a new reconstruction theorem. The optimality statement is over all 1-Lipschitz functions; the smaller class of true signed-distance fields may admit tighter bounds. The result could guide a future sampling/error decision, but it does not propose another runtime Map representation.

### Exact samples do not certify connectivity

**Theorem 5 (finite signed-distance ambiguity).** Given any finite set `Q⊂R²`, there are two closed regions with smooth boundaries, one connected and one disconnected, whose exact signed distances agree at every point in `Q`.

**Proof.** Start with the halfplane `A0={(s,t):t≤0}`, whose signed distance is `D0(s,t)=t`. Pick a point `p=(p1,0)` on its boundary such that `p1` differs from every first coordinate of the finitely many samples. The point can be chosen in any prescribed nondegenerate open segment of that line. If that segment lies in the window's interior, the hidden component can be placed inside the sampled window. For each sample `z=(z1,z2)`,

$$
|p-z|>|z2|.
$$

The strict inequalities leave a small open neighborhood of `p` lying outside every closed ball `B(z,|z2|)`. Place a small closed disk `B` in this neighborhood, strictly above and disjoint from the halfplane. Set `A1=A0∪B`.

For every exterior sample, its distance to `B` exceeds its distance to the line, so its signed distance is unchanged. Every interior or boundary sample remains in the same halfplane, whose entire boundary line remains part of `∂A1`; its distance to the disk boundary cannot shorten the distance to the complement through the line. Thus all signed-distance samples agree. `A0` is connected, while `A1` has two components. Both boundaries are disjoint unions of smooth curves.

Include all Map centers and all finite multiscale query locations in `Q`. Their exact values, rulers computed from the centers, and normalized Map samples then agree as well. This is a limitation on guarantees derived only from those samples. It does not assert that the known Mandelbrot set is an arbitrary region, or rule out guarantees using its additional dynamics. Uniformly small function errors likewise do not imply topological equivalence without additional feature-size conditions. Existing reconstruction theory supplies such additional conditions [3].

## 6. Blow-ups: useful limits, with two essential qualifications

For any family of anchored 1-Lipschitz fields `f_k(0)=0`, the bounds `|f_k(u)|≤|u|` imply equiboundedness and equicontinuity on each compact ball. Arzelà–Ascoli and a diagonal subsequence yield a locally uniform limit, itself anchored and 1-Lipschitz.

For an unsigned distance `d_S` to a nonempty closed set `S`, at a center `x∈S`,

$$
M_{d_S,x,h}(u)=\frac{d_S(x+hu)}h
=\operatorname{dist}\left(u,\frac{S-x}{h}\right).
\tag{14}
$$

A locally uniform limit `f` of these functions is itself the distance to the closed set `T={f=0}`. To see this, `f` being 1-Lipschitz gives `f(u)≤dist(u,T)`. For fixed `u`, choose nearest points `z_k` in the scaled sets. They satisfy `|z_k|≤2|u|`, because those sets contain zero. A subsequence converges to `z`; local uniform convergence and the Lipschitz estimate imply `f(z)=0`, and `|u-z|=f(u)`. This gives the reverse inequality. Distance-function convergence is the appropriate Attouch–Wets perspective [2].

This argument does not apply verbatim to a centered distance difference at `x∉S`: subtracting `d_S(x)/h` need not leave a nonnegative function. Signed distances require retaining the two sides, for example by following both closed-region and closed-complement distance functions. Boundary convergence alone does not determine the signs.

### A tangent set need not be a cone or unique

Let

$$
S=\{0\}\cup\{2^{-n}:n=0,1,2,\ldots\}\subset\mathbb R.
$$

At `x=0`, the scales `h_k=2^{-k}` yield locally convergent distance functions for

$$
T=\{0\}\cup\{2^j:j\in\mathbb Z\}.
$$

This set is invariant under dilation by two but is not a cone. Scales `h'_k=(3/2)2^{-k}` yield `(2/3)T`, a different limit. At `u=1`, the first limiting distance is zero and the second is `1/3`. Thus uniqueness, conical structure, and a single asymptotic chart cannot be inferred from Lipschitz compactness alone.

### Fixed regularization does not reach infinitesimal scale

For the actual ruler with fixed `ε>0` and fixed `ζ`,

$$
h=2^\zeta\sqrt{D(x)^2+\epsilon^2}\ge2^\zeta\epsilon.
$$

Moving the center toward the Shore does not itself produce `h→0`. Any application of infinitesimal blow-up results must explicitly vary `ζ→-∞`, `ε→0`, or another scale parameter. Such a mathematical limit is not automatically an executable high-resolution promise from the current provider.

## 7. Decision and next useful work

The current results justify an engineering theorem package. They do not yet justify a standalone originality claim. The immediate useful contracts are the chart transition (1), the computed-location error budget (7), and the distinction between function-value and topology guarantees. The bias cancellation and crest estimate are useful specializations of elementary inequalities, rather than evidence of a new field of theory.

The Lean subset now includes the combined computed-location provider/ruler
budget (7), its relative and endpoint versions (10), and the crest bias
bound (11), supported by the actual scale-change and ruler-perturbation
lemmas. These are conditional certificates: a numerical provider must still
justify its error at the center and at the query its computed scale selects.
Its supplied gradient and Hessian must also be distinguished from the actual
`fderiv` fields used in Section 1.

The bridge in
[`geometry_provider.rs`](../../runtime-core/src/geometry_provider.rs)
uses a derivative stencil whose step is clamped between configured minimum
and maximum values. The specialized recomputed-ruler theorem applies when
the scale is exactly `q rhoHat`. For an actual clamped scale `k`, the more
general `normalizedMap_provider_scale_error_bound` accepts arbitrary
nonzero nominal and computed scales `h,k`; its certificate must bound
`abs(e(x+k u)-e(x))` at that actual query, and its scale term uses the
actual `abs(k-h)/max(abs(h),abs(k))`. Clamping does not inherit the
proportional-scale specialization automatically. Floating-point query
locations and arithmetic require their own correspondence estimates.

The smooth cross-scale bound (6), smooth center bound, residual transition
(2), ruler chart identity (3), and propagation of supplied gradient error
remain separate formal targets. The finite-sample envelopes, topology
obstruction, and Arzelà–Ascoli/Attouch–Wets arguments in Sections 5–6 retain
conventional proofs. They are not implied by the checked value-error
contracts and should not block training.

## 8. Checked Lean subset

The three modules below contain thirty-four checked theorems in namespace
`FractalSync.NormalizedDistanceFields`.

### Algebra, fixed-scale errors, and ideal-field bounds

[`NormalizedDistanceFields.lean`](../../formal/FractalSync/NormalizedDistanceFields.lean)
contains the original eight checked theorems:

| Theorem identifier | Exact scope |
| --- | --- |
| `normalizedMap_chart_transition` | Equation (1), for any real-valued function on a real module and nonzero scales. |
| `normalizedDifference_error_identity` | The exact anchored provider-error decomposition at a fixed scale. |
| `normalizedDifference_anchored_error_bound` | Error at most `η/h` given a certified anchored error increment and `h>0`. |
| `normalizedDifference_endpoint_error_bound` | Error at most `(eu+e0)/h` given separate endpoint bounds and `h>0`. |
| `normalizedMap_add_constant` | Constant field offsets cancel at fixed scale. |
| `normalizedMap_lipschitz_bound` | `abs(M(u)-M(v))≤‖u-v‖` for `LipschitzWith 1 D` on a real normed space and `h>0`. |
| `normalizedMap_zero` | Exact anchor `M(0)=0`. |
| `normalizedMap_value_bound` | `abs(M(u))≤‖u‖` under the same ideal-field assumptions. |

### Actual differentials and segment quadratic remainders

[`NormalizedMapDifferential.lean`](../../formal/FractalSync/NormalizedMapDifferential.lean)
adds eight checked theorems. All hold for a real normed space; finite
dimension and an inner product are unnecessary.

| Theorem identifier | Exact scope |
| --- | --- |
| `hasFDerivAt_normalizedMap` | At fixed anchor and `h≠0`, differentiability of `D` at `x+hu` gives the window derivative `dD(x+hu)`. |
| `hasFDerivAt_normalizedMap_zero` | At the window origin the differential is exactly `dD(x)`. |
| `fderiv_normalizedMap` | Equality of the actual window and physical `fderiv` values at a differentiable query. |
| `normalizedMap_value_bound_on_segment` | `abs(M(u))≤L0‖u‖` from an `L0`-Lipschitz bound on just `[x,x+hu]`, for `h≠0`. |
| `field_quadratic_remainder` | `abs(D(x+v)-D(x)-dD(x)[v])≤(L/2)‖v‖²` under differentiability and anchored differential variation on `[x,x+v]`. |
| `normalizedMap_quadratic_remainder` | The corresponding normalized residual is at most `(L abs(h)/2)‖u‖²`. |
| `normalizedMap_quadratic_remainder_of_lipschitz_fderiv` | The residual bound from differentiability of `D` and an `L`-Lipschitz actual differential on `[x,x+hu]`. |
| `normalizedMap_quadratic_remainder_of_hessian_bound` | The residual bound from differentiability of `D` and an actual derivative of `fderiv ℝ D` with operator norm at most `L` on the entire closed segment. |

### Combined provider errors, recomputed rulers, and scale changes

[`NormalizedProviderError.lean`](../../formal/FractalSync/NormalizedProviderError.lean)
adds eighteen checked theorems, again for arbitrary real normed spaces.
The ideal field is 1-Lipschitz where a Map bound requires it; the provider
error field is unrestricted except for the stated value certificates.

| Theorem identifier | Exact scope |
| --- | --- |
| `normalizedMap_value_bound_signed` | `abs(M(u))≤‖u‖` for every nonzero signed scale. |
| `normalizedMap_scale_difference_identity` | Exact two-scale difference decomposition for nonzero `h,k`. |
| `normalizedMap_scale_change_bound_right` | Scale error at most `2‖u‖ abs(h-k)/abs(k)`. |
| `normalizedMap_scale_change_bound` | Symmetric scale bound (5), with `max(abs(h),abs(k))` for signed scales. |
| `normalizedMap_provider_scale_error_identity` | Exact provider increment at the computed query, plus the ideal-field scale difference. |
| `normalizedMap_provider_scale_error_bound` | Combined increment and scale bound for arbitrary nonzero `h,k`. |
| `normalizedMap_provider_scaled_error_bound` | Specialization to `h=q rho`, `k=q rhoHat` for positive supplied scales. |
| `normalizedMap_recomputed_ruler_error_bound` | Equation (7) with both actual ruler definitions and a certified anchored error increment. |
| `normalizedMap_recomputed_ruler_endpoint_bound` | Equation (7) with separate center and computed-query error bounds. |
| `ruler_perturbation_le` | Equation (8): absolute ruler change at most the center field error. |
| `ruler_perturbation_identity` | Exact rationalized ruler change (9). |
| `ruler_perturbation_relative_lower` | `rhoHat≥(1-κ)rho` from `abs(b)≤κrho`. |
| `ruler_perturbation_relative_bound` | `abs(rhoHat-rho)/max(rho,rhoHat)≤κ` when `κ≥0` and `abs(b)≤κrho`. |
| `normalizedMap_recomputed_ruler_relative_bound` | Correlated error budget `η/[q(1-κ)rho]+2κ‖u‖`, for `0≤κ<1`. |
| `normalizedMap_recomputed_ruler_uniform_bound` | Endpoint budget (10), for `0≤κ<1`. |
| `ruler_crest_relative_bound` | Relative ruler error at zero center field at most `b²/(2ε²)`. |
| `normalizedMap_crest_bias_bound` | Constant-bias crest estimate (11), including cancellation of the anchored provider error. |
| `normalizedMap_center_change_bound` | Center bound (4); the signed-scale statement uses `abs(h)`. |

The physical statements use positive scales. The center-change theorem
also holds at `h=0` under Lean's totalized division, where both Maps vanish;
this algebraic endpoint does not define a physical zero-size query.

All three modules compile with Lean 4.24.0 and Mathlib commit
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`, with warnings treated as errors.
The original module's per-theorem axiom inspection reported only `propext`,
`Classical.choice`, and `Quot.sound`; the package-wide verification record is
maintained in [verification.md](verification.md). None of these modules contains
admitted proofs or custom axioms.

The smooth cross-scale bound (6), smooth center-stability bound, finite-data
envelopes, topology obstruction, and blow-up arguments still have
conventional proofs in this note and are not yet Lean-verified. The checked
differential, quadratic residual, and combined provider bounds do not remove
those separate obligations. No theorem equates a real-valued ideal field or
its actual derivatives with the floating-point GeometryProvider.

## References and provenance

1. E. J. McShane, *Extension of range of functions*, Bulletin of the American Mathematical Society 40 (1934), 837–842. [DOI](https://doi.org/10.1090/S0002-9904-1934-05978-0). Classical Lipschitz extension provenance. A current primary research treatment explicitly presents the extremal-envelope construction: [*On real-valued functions of Lipschitz type*](https://doi.org/10.1016/j.exmath.2025.125701). Bibliographic metadata and the latter paper's theorem text were inspected; the original AMS PDF was not successfully retrieved in this session.
2. R. Lowen and M. Sioen, *The Wijsman and Attouch-Wets topologies on hyperspaces revisited*, Topology and its Applications 70 (1996), 179–197. [DOI](https://doi.org/10.1016/0166-8641(95)00096-8). Publisher abstract inspected for distance-functional/uniform-convergence formulation; no claim is made that this paper originates every elementary distance-limit fact proved above.
3. F. Chazal, D. Cohen-Steiner, and A. Lieutier, *A Sampling Theory for Compact Sets in Euclidean Space*, Discrete & Computational Geometry 41 (2009), 461–479. [Author-hosted full paper](https://geometrica.saclay.inria.fr/team/Fred.Chazal/papers/DCG-ccsl-09.pdf), [DOI](https://doi.org/10.1007/s00454-009-9144-8). Full paper inspected, especially the distance-function background, feature-size hypotheses, and reconstruction statements. Cited to establish that topology guarantees require additional geometric hypotheses, not to transfer its theorems wholesale to the Mandelbrot boundary.

All equations (1)–(14) were derived explicitly in this note; references locate their established mathematical context. The companion synthetic checks exercise sharp examples and counterexamples and are not substitutes for the proofs or certificates for the real GeometryProvider.
