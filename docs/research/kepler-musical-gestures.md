# Metric-native musical gestures and Kepler clocks

Status: research proposal with a companion Lean development for scalar clocks and tangent-plane kinematics.
No runtime changes. Preserve free navigation and compare gesture tracking against it.

## Geometry

For a smooth positive definite Map metric `G`, choose a focus `p` and a
positively oriented `G(p)`-orthonormal frame `B`. On a certified normal disk
define `w(E)=B(a(cos E-e), b sin E)`, where `b=a√(1-e²)`, `0≤e<1` and
`a>0`. The target is `exp_p(w(E))`. The disk must contain all radii up to
`a(1+e)`.

The ellipse is the exponential image of a tangent-plane ellipse. The
two-geodesic-focus level set `d(p,x)+d(q,x)=2a` is a separate construction.
Both definitions allow variable Gaussian curvature. A frozen-metric affine
chart gives a cheaper approximation.

## Classical clock

The mean anomaly is `M(E)=E-e sin E`, with `M'(E)=1-e cos E≥1-e>0`.
Since `|M(E)-E|≤e`, the clock maps the real line onto itself. Its unique
unwrapped inverse advances by `2π` for every `2π` advance in mean anomaly.
In the oriented frame, `det(w,w_E)=ab(1-e cos E)`. The signed area from
periapsis is `ab M(E)/2`.

For fixed positive mean motion `n`, the tangent-plane speeds at periapsis
and apoapsis are `nb/(1-e)` and `nb/(1+e)`. Their ratio is
`(1+e)/(1-e)`. For `e=0.6`, the minor-to-major axis ratio is `b/a=0.8`
and periapsis speed is four times apoapsis speed. Evaluating target speed
on the Map uses the differential of `exp_p` and the metric at the target.

The tangent-plane radius is `a(1-e cos E)` and the acceleration is
`-n²a³ w/|w|³`. Kepler's third law follows across orbits with common
`μ=n²a³`. In a musical control mode with independent tempo and size,
`μ` is an effective parameter for each orbit. A fixed-`μ` mode would
couple these controls. The inverse-clock sensitivity is at most
`1/(1-e)`, and peak speed and acceleration diverge as `e→1` at fixed
positive `a,n`.

## Intrinsic equal-area clock

Let `j(w)>0` denote the pullback Riemannian area density in the oriented
normal frame. The radial fan `F(s,E)=exp_p(s w(E))`, `0≤s≤1`, has
tangent-plane Jacobian `det(w,s w_E)=s ab(1-e cos E)` in parameter order
`(s,E)`. The focus lies inside the ellipse and this determinant is
positive for `s>0`, so increasing anomaly sweeps wedges in the positive
orientation. Over a revolution each interior point away from the focus is
covered once, up to the shared boundary ray. The signed area per unit
anomaly is

```text
h(E) = ab(1-e cos E) ∫₀¹ s j(s w(E)) ds.
```

For fixed geometry and orbit, `h` is continuous, positive and periodic.
Its minimum and maximum over one period therefore bound it on the real
line. Set `H(E)=∫₀ᴱ h(u)du`, `A=H(2π)` and `M_G(E)=2π H(E)/A`.
This clock has a unique unwrapped inverse, and
`M_G(E+2π)=M_G(E)+2π`. Equal phase increments sweep equal signed
intrinsic areas, counting repeated revolutions with multiplicity. In
flat geometry `j=1`, giving `M_G(E)=E-e sin E`.

If `j` lies in `[1-δ,1+δ]` throughout the fan with `0≤δ<1`, set
`κ=(1-δ)/(1+δ)`. Applying the density bound to `h` and its period
integral `A` gives `κ≤M'_G(E)/(1-e cos E)≤1/κ`. In particular,
`M'_G≥κ(1-e)`, so a phase error of size `ε` moves the exact inverse by
at most `ε/[κ(1-e)]` radians of eccentric anomaly. The same bound applies
to a numerical estimate whose residual in the exact clock is at most
`ε`. These bounds control the clock. Tracking error also depends on the
geometry and Physics. Numerical quadrature and inverse tables need
certified errors.

## Dynamics boundary
In geodesic polar coordinates ds²=dr²+J(r,θ)²dθ², area density is J.
For a radial potential, Euler-Lagrange gives
d/dt(J² θdot)=J ∂θJ θdot². Thus angular momentum need not be conserved
without rotational symmetry. Even with symmetry, swept area rate is
θdot ∫₀ʳ J(s)ds, not generally proportional to J² θdot.
The existing Shore and wall potential is not generally radial about p.
Do not assert that a prescribed area-clock orbit is a free physical orbit.
For target γ(t), exact feedforward force covector is
G(γ) Dγdot/dt + dU(γ) + B γdot under friction -Bv.
Feedback, saturation, energy ledger, and lag require separate analysis.

## Musical control
Player chooses focus, orientation, intrinsic size, eccentricity, source,
gain, phase offset and tracking strength. CycleBank unwrapped phase drives
M or M_G; normalized causal RMS may drive size, eccentricity, or a
bounded nonnegative clock rate. Direct RMS-to-phase scrubbing is reversible
and is not a Kepler orbit in physical time. Equal areas per musical phase
are not equal areas per second unless phase rate is constant.
Two independently selected audio components can drive frame axes to
generate Lissajous gestures. Keep source confidence, phase continuity,
rate limits and explicit transition state. When parameters move, the
frozen-orbit area law does not imply equal physical swept area: the
derivatives of the moving focus/frame/axes must be accounted for.
Transport frames along focus motion; holonomy can change orientation.

## Experiments
Compare yoyo, free navigation, affine ellipse, normal-coordinate ellipse
with uniform anomaly, classical mean anomaly, and intrinsic area clock.
Compare kinematic reference vs bounded Physics tracking. Measure phase lag,
intrinsic path error, energy, Shore crossings, saturation, action bandwidth,
held-out musical coherence and blinded preference. Test circles, eccentric
orbits, deep scales, metric changes, cut loci, silence, phase resets and
polymeter. Rust remains runtime authority per ADR 0001.

## Verification boundary

The companion Lean modules `KeplerKinematics`, `KeplerAreaClock`,
`KeplerAreaDensity` and `KeplerEllipseClock` cover tangent-plane kinematics,
a scalar clock integrated from a continuous positively bounded rate, its
inverse and stability, and ellipse and flat-density specializations. The
geometric interpretation assumes a certified normal disk and a pullback
area density with the stated bounds. Establishing those hypotheses for
particular Map regions, and certifying moving-parameter dynamics, tracking
and numerical integration, remain separate research tasks.
