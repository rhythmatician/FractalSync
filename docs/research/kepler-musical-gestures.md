# Metric-native musical gestures and Kepler clocks

Status: research proposal, conventional derivations; Lean candidate not yet checked.
No runtime changes. Preserve free navigation and compare gesture tracking against it.

## Geometry
For a smooth positive definite Map metric G, choose a focus p and oriented
G(p)-orthonormal frame B. On a certified normal disk define
w(E)=B(a(cos E-e), b sin E), b=a sqrt(1-e²), 0≤e<1, a>0.
The target is exp_p(w(E)). The disk must contain radius a(1+e).
This is a normal-coordinate ellipse, not necessarily a two-geodesic-focus
ellipse. An alternative is the level set d(p,x)+d(q,x)=2a; compare rather
than identify the definitions. Neither assumes constant Gaussian curvature.
A cheap frozen-metric affine chart is a separate approximation.

## Classical clock
M=E-e sin E, M'=1-e cos E≥1-e>0. Hence a unique unwrapped inverse
exists (bounded difference from identity), with E(M+2π)=E(M)+2π.
The focus-relative determinant det(w,w_E)=ab(1-e cos E) in the
orthonormal frame. Swept area is ab M/2, up to an additive constant.
At fixed mean motion n, periapsis/apoapsis speed ratio is (1+e)/(1-e).
Radius is a(1-e cos E). The inverse-square acceleration is
-n² a³ w/|w|³, so Kepler III follows only when μ=n²a³ is held fixed.
For musical control choose n from tempo and a independently; μ is then
an effective parameter, not a universal constant. Compare optional fixed-μ
mode. Cap eccentricity: inverse-clock sensitivity ≤1/(1-e), peak speed
and acceleration grow without bound as e→1 at fixed a,n.

## Intrinsic equal-area clock
Let j(w)>0 be the pullback Riemannian area density in the normal disk.
The radial fan F(s,E)=exp_p(s w(E)), 0≤s≤1, has area rate
h(E)=det(w,w_E) ∫₀¹ s j(s w(E)) ds, using frame coordinates.
For fixed geometry and orbit h>0. Define H(E)=∫₀ᴱ h(u)du,
A=H(2π), and M_G(E)=2π H(E)/A. Invert this increasing clock.
Equal increments of M_G sweep equal intrinsic areas. In flat geometry
j=1 and M_G=E-e sin E. If j lies in [1-δ,1+δ], δ<1, the normalized
clock derivative relative to the classical derivative lies between
(1-δ)/(1+δ) and its reciprocal. This bounds clock distortion, not
trajectory tracking error. The area fan requires a valid normal disk.
Use quadrature and monotone inverse tables with certified errors later.

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
The companion Lean candidate in formal/research isolates algebraic
identities and positivity/ratio bounds. It is intentionally not imported
into the accepted proof inventory until the pinned build and axiom audit
pass. Remaining formal work: actual derivative and inverse theorem,
normal disk/area pullback and integration, moving-parameter derivatives,
tracking and numerical certificates. No claim of checked proofs yet.
