# A conformal transport angle budget

The endpoint-chord diagnostic in the experimental Physics rollout uses
normalized covariant Euler. Here is an error budget for the conformal case,
`lambda_sq = 0`, with a positive ruler, the same held signed turn on both
paths, and exact real arithmetic. Its first input is a bound on path error
throughout each substep. The committed endpoint trace supplies the chord.

[Endpoint-chord diagnostic](179-endpoint-chord-transport.md)

## Angle in the oriented coordinate frame

Write the metric as \(g=\rho^{-2}I\), the positive quarter turn as
\(J(v_x,v_y)=(-v_y,v_x)\), and the rotation one-form as
\(\alpha_z(v)=D\log\rho(z)[-Jv]\).
Thus \(\alpha=p\,dy-q\,dx\) for
\((p,q)=\nabla\log\rho\). The oriented orthonormal frame is
\((\rho e_x,\rho e_y)\), so a metric-unit direction is
\(d=\rho(\cos\theta,\sin\theta)\). The conformal connection gives
\(-\Gamma(v,d)=\beta(v)d+\alpha(v)Jd\), where
\(\beta=D\log\rho\). The derivative of \(\rho\) accounts for the radial
term \(\beta(v)d\). Consequently the real lifted angle has derivative
\(\theta'=\omega+\alpha_x(x')\). Integration uses the real lift through
branch crossings.

[Planar conformal connection](../../formal/FractalSync/ConformalRulerContracts.lean)

## Replacing the path by a chord

Let \(x\) be a continuously differentiable reference path and \(y\) its
affine reconstruction. Let \(e\) bound \(\|x(t)-y(t)\|\) for the full
interval. Put \(L=\sup\|D\alpha\|\) on a region containing the paths and
the straight segments between their paired positions. A common choice is
a convex enclosure for the whole substep.

The difference of the rotation integrals is

\[
\int (\alpha(x)-\alpha(y))[x']\,dt
-\int (\alpha\circ y)'[x-y]\,dt
+[\alpha(y)[x-y]]_a^b.
\]

This follows from the product rule for the evaluation of a one-form on
\(x-y\). The first two terms together have absolute value at most
\(L e(\operatorname{len}(x)+\operatorname{len}(y))\).
The signed endpoint term vanishes for shared endpoints. When summing
substeps, the terms at interior mesh points cancel for continuous
reference and reconstructed paths evaluated with one common form.
The outer terms can be bounded by \(M\) times the sum of the start and
end position errors when \(M\) bounds \(\|\alpha\|\) there.

[Lean path estimate](../../formal/FractalSync/ConformalTransportError.lean)

If a reference-trajectory certificate supplies \(\|x''\|\le A\) throughout
a substep of duration \(h\), and both committed endpoints have position
error at most \(\delta\), one can take \(e=\delta+Ah^2/8\).
Expand \(x(a)\) and \(x(b)\) to first order about \(x(t)\). The two
remainder norms are at most \(A(t-a)^2/2\) and \(A(b-t)^2/2\).
The chord weights \((b-t)/h\) and \((t-a)/h\) cancel the linear terms and
give a combined remainder at most \(A(t-a)(b-t)/2\le Ah^2/8\).
The same weights bound the endpoint error contribution by \(\delta\).
This argument already needs an acceleration bound on the reference
path. An acceleration bound only for queried endpoints leaves that
premise open.

The pointwise position premise is essential even if endpoint velocities
agree. For the flat fixture \(\rho=\sqrt{\epsilon^2+x_1^2}\),
consider the rectangle with corners \((0,0),(s,0),(s,H),(0,H)\),
traversed in that order and closed at the origin. Parameterize each side
by a cubic smoothstep so the velocity vanishes at every corner. The
stationary chord and the resulting continuously differentiable loop
agree at both sampled endpoints in position and velocity. The loop
accumulates \(Hs/(\epsilon^2+s^2)\) radians of transport and the chord
accumulates zero.

## Quadrature and normalized Euler

Write \(d_c=y(b)-y(a)\). The start sample \(\alpha(y(a))[d_c]\)
differs from exact chord transport by at most \(L\|d_c\|^2/2\).
On the unit chord parameter \(s\), the distance to the start is
\(s\|d_c\|\); the factor \(1/2\) is the integral of \(s\) from zero
to one.

For one real normalized Euler update, let
\(\beta_0=D\log\rho(y(a))[d_c]\) and
\(k=h\omega+\alpha(y(a))[d_c]\).
In the oriented frame the trial has coefficients \((1+\beta_0,k)\)
relative to the old ray and its positive quarter turn. Conformal
normalization scales the whole trial by a positive scalar. For
\(1+\beta_0>0\), the update turns by
\(\arctan(k/(1+\beta_0))\), and

\[
|\arctan(k/(1+\beta_0))-k|
\le \frac{|k|\,|\beta_0|}{1+\beta_0}+\frac{|k|^3}{3}.
\]

The arctangent is 1-Lipschitz. The cubic bound integrates
\(0\le 1-(\arctan)'(s)=s^2/(1+s^2)\le s^2\) on
\([0,|k|]\) and uses oddness for a negative turn.
For a sequence of substeps with the same initial frame angle, let
\(\delta_{\mathrm{ends}}\) be the sum of the position errors at the two
outer endpoints. The accumulated angle error is at most

\[
M\delta_{\mathrm{ends}} +
\sum_i \left[
L_i e_i(\operatorname{len}(x_i)+\|d_{c,i}\|)
+\frac{L_i\|d_{c,i}\|^2}{2}
+\frac{|k_i|\,|\beta_{0,i}|}{1+\beta_{0,i}}
+\frac{|k_i|^3}{3}
\right].
\]

A common held turn contributes equally to the two exact integrals.
Lean checks the path-replacement bound and the Cartesian connection
split. The chord quadrature, Euler defect, accumulation, and acceleration
interpolation bounds here are conventional derivations. Numerical
margin and roundoff, metric validity throughout the substep, a verified
all-time reference-path enclosure, and the nonconformal metric still
need separate evidence.

## Bounds from the signed field

For \(\rho=\sqrt{\epsilon^2+F^2}\), let \(G\) and \(H\)
bound \(\|DF\|\) and \(\|D^2 F\|\) as operator norms.
The scalar derivative of \(\log\sqrt{\epsilon^2+r^2}\) is bounded by
\(1/(2\epsilon)\) using \(2\epsilon|r|\le r^2+\epsilon^2\).
Its second derivative is bounded in absolute value by
\(1/\epsilon^2\) using
\(|\epsilon^2-r^2|\le\epsilon^2+r^2\).
The chain rule therefore permits
\(M=G/(2\epsilon)\) and
\(L=H/(2\epsilon)+G^2/\epsilon^2\).
The Cartesian quarter turn preserves both the Euclidean and the
coordinate maximum norm. Choose the same norm for the lengths,
position errors and operator bounds. For example the flat fixture with
\(\epsilon=0.08\) admits \(M=6.25\) and \(L=156.25\).

[Actual log-floor derivatives](../../formal/FractalSync/ConformalFloorCurvature.lean) ·
[Global operational Hessian bounds](162-global-operational-field.md)
