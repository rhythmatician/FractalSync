# Radial profile: checked calculus, sharp error and singular asymptotics

This increment for [Issue 160](https://github.com/rhythmatician/FractalSync/issues/160)
formalizes the scalar analysis underlying the
[distance-geometry note](160-distance-geometry.md).
[RadialProfile.lean](../../formal/FractalSync/RadialProfile.lean) contains
44 proved theorem declarations and six definitions. Its pinned Lean package
builds without warnings. This is a mathematical proof artifact, not a
floating-point implementation or a change to the runtime metric.

## 1. Definitions and scope

For epsilon > 0, define

$$
w_{\epsilon,a}(r)=
\frac{\sqrt{\epsilon^2+(1+a^2)r^2}}{\epsilon^2+r^2},
\qquad
F_{\epsilon,a}(r)=\int_0^r w_{\epsilon,a}(t)\,dt.
$$

The Lean definitions are the actual square-root density and interval integral,
`radialDensity epsilon a r` and `radialProfile epsilon a r`. They permit signed
radii using the oriented interval integral. Except for the stated slope cases,
the theorems permit every real a: the model depends on a squared, so a >= 0 is
not needed for these scalar results. Geometric distance statements still use
r >= 0.

Lean checks continuity and positivity of the density, the fundamental theorem
of calculus identity F'=w, strict monotonicity of F, F(0)=0, and positivity for
positive radii. It also checks the squared-density identity connecting this
analysis to the earlier `radialSpeedSq` definition.

Positive dilation is exact:

$$
F_{c\epsilon,a}(cr)=F_{\epsilon,a}(r),\qquad c>0,
\qquad
F_{\epsilon,a}(r)=F_{1,a}(r/\epsilon).
$$

This is scalar covariance. The corresponding dilation of the full distance
metric also involves scaling the closed set and its admissible paths.

## 2. The closed primitive is now checked

Put b=sqrt(1+a²), q=r/epsilon and s=sqrt(1+b²q²). The checked formula is

$$
\boxed{
F_{\epsilon,a}(r)
=b\operatorname{arsinh}(bq)
-a\log(s+aq)
+\frac a2\log(1+q^2).}
$$

The logarithm argument s+aq is strictly positive for all real a,q, because
s²-a²q²=1+q²>0. The formal proof differentiates the displayed expression,
checks its value at zero, and applies the fundamental theorem of calculus to
identify it with the integral. It does not rely on numerical quadrature or an
assumed antiderivative.

The original note writes the second part as
`-a atanh(aq/s)`. The elementary identity
`atanh(aq/s)=log(s+aq)-(1/2)log(1+q²)` relates the presentations.
The Lean result directly establishes the logarithmic expression; it does not
introduce an additional formal definition of atanh.

At a=0, the checked expression reduces to

$$
F_{\epsilon,0}(r)=\operatorname{arsinh}(r/\epsilon).
$$

The integrated density comparison also gives, for r >= 0,

$$
\operatorname{arsinh}(r/\epsilon)
\le F_{\epsilon,a}(r)
\le b\operatorname{arsinh}(r/\epsilon).
$$

## 3. Sharp additive error and the exact asymptotic constant

Define

$$
H_a(q)=b\operatorname{arsinh}(q)-F_{1,a}(q),
\qquad
C(a)=a\operatorname{arsinh}(a)-b\log b.
$$

Lean checks that H is nondecreasing, H(0)=0, and H(q) tends to C(a) as
q tends to positive infinity. Consequently

$$
\boxed{0\le b\operatorname{arsinh}(r/\epsilon)-F_{\epsilon,a}(r)
\le C(a),\qquad r\ge0.}
$$

The constant is optimal. The formal theorem
`radialDefectLimit_le_of_uniform_bound` states that any constant bounding H(q)
for every q >= 0 must be at least C(a). Nonnegativity of C(a) is also checked.

The limit proof removes divergent logarithms before taking a limit. For q>0
set u=1/q. Exact algebra rewrites H(q) as

$$
\begin{aligned}
&b\log(1+\sqrt{1+u^2})
-b\log(b+\sqrt{1+a^2+u^2})\\
&\quad +a\log(a+\sqrt{1+a^2+u^2})
-\frac a2\log(1+u^2).
\end{aligned}
$$

This expression is continuous at u=0 and evaluates there to C(a). The
formalization checks the exact rewrite, continuity, evaluation and limit.

Combining this result with the separately checked limit
`arsinh(q)-log(q) -> log(2)` yields

$$
F_{\epsilon,a}(r)-b\log(r/\epsilon)
\longrightarrow b\log 2-C(a)
=b\log(2b)-a\operatorname{arsinh}(a).
$$

There are checked theorems for both limits: fixed epsilon>0 with r increasing
to infinity, and fixed r>0 with epsilon decreasing to zero through positive
values. In particular, the scalar F diverges to infinity in both cases.
These scalar results do not themselves prove convergence of an infimum over
all admissible curves.

## 4. Sharp maximal radial density

The analysis also gives an exact global slope bound. For a>=0,

$$
\sup_{q\in\mathbb R}w_{1,a}(q)
=M(a):=
\begin{cases}
1,&0\le a\le1,\\
\dfrac{1+a^2}{2a},&a\ge1.
\end{cases}
$$

The two formulas agree at a=1. The formal module proves the upper bounds and
their attainment, rather than introducing a supremum wrapper. For a<=1 the
maximum is attained at q=0. For a>=1 it is attained at

$$
q=\sqrt{\frac{a^2-1}{1+a^2}}.
$$

The large-a bound follows from the checked certificate

$$
(1+a^2)^2(1+q^2)^2-4a^2[1+(1+a^2)q^2]
=\big[(1+a^2)q^2-(a^2-1)\big]^2\ge0.
$$

By the checked density dilation, the dimensional maximum is M(a)/epsilon.
The mean value theorem then gives the sharp global Lipschitz constant
M(a)/epsilon for F. This final Lipschitz corollary is conventional here;
the explicit Lean results are the derivative, the pointwise bounds, their
attainment, and dilation. No novelty claim is made for this elementary bound.

## 5. Exact boundary of formal verification

| Checked result family | Principal Lean declarations |
| --- | --- |
| Integral calculus and positivity | `continuous_radialDensity`, `hasDerivAt_radialProfile`, `radialProfile_strictMono` |
| Connection to squared graph speed | `radialDensity_sq` |
| Dilation | `radialDensity_scale`, `radialProfile_scale`, `radialProfile_normalize` |
| Full closed primitive | `hasDerivAt_normalizedRadialPrimitive`, `radialProfile_closed_form`, `radialProfile_scale_only` |
| Integrated comparison | `radialProfile_asinh_bounds` |
| Sharp density bounds and attainment | `radialDensity_le_one`, `radialDensity_le_peak`, `radialDensity_peak_attained` |
| Scalar divergence | `radialProfile_tendsto_atTop`, `radialProfile_tendsto_zero_floor` |
| Sharp additive constant and optimality | `radialDefect_tendsto`, `radialDefect_monotone`, `radialProfile_sharp_bounds`, `radialDefectLimit_le_of_uniform_bound` |
| Exact logarithmic asymptotics | `radialProfile_log_asymptotic`, `radialProfile_zero_floor_log_asymptotic` |

[PathCalibration.lean](../../formal/FractalSync/PathCalibration.lean) uses this
scalar package to prove integrated endpoint bounds for explicitly differentiable
traces with integrable densities, and the exact integral for a nearest-point
segment. Its hypotheses are visible in the theorem statements. The distinction
matters: the distance trace of a general absolutely continuous curve need not
be differentiable at every interior point.

The full unrestricted length-metric theorem for arbitrary closed sets still
needs the almost-everywhere chain rule and its integration for general
absolutely continuous traces, followed by the definition and infimum properties
of the intrinsic metric. Properness, minimizing-curve existence and the complete
componentwise metric limit remain separate global obligations. The scalar
proofs do not assert smoothness at cut loci, constant intrinsic curvature,
controllability, or validity of a nearest-point segment in a restricted domain.
