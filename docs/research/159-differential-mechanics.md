# Field differentials, normalized Map remainders, and continuous work-energy

This note records the checked differential extension of #157/#159. It uses the
existing positive local-ruler metric and inverse rather than creating a second
metric definition. The source modules are
[`RulerDifferential.lean`](../../formal/FractalSync/RulerDifferential.lean) and
[`ContinuousMechanics.lean`](../../formal/FractalSync/ContinuousMechanics.lean),
with the normalized Map extension in
[`NormalizedMapDifferential.lean`](../../formal/FractalSync/NormalizedMapDifferential.lean).

The Hessian results differentiate actual Fréchet differentials. The mechanics
results differentiate actual time traces and derive work-energy from the
equation of motion. They do not assume the identities they are intended to
prove.

## 1. The actual ruler and scale Hessians

Let D:E->R be a scalar field on a real normed space, epsilon>0, and
dref>0. Define

\[
\rho=\sqrt{D^2+\epsilon^2},\qquad
\sigma=\frac{\log(d_{\mathrm{ref}}/\rho)}{\log2}.
\]

At a point x, assume that D is differentiable on a neighborhood of x and
that its actual differential field `fderiv R D` has the stated Fréchet
derivative at x. This is sufficient for the following formulas; global C2
regularity is not required by the formal statements.

Using dD for the differential and H_D for the second differential,

\[
\boxed{d\rho=\frac{D}{\rho}\,dD,\qquad
H_\rho=\frac{D}{\rho}H_D+
\frac{\epsilon^2}{\rho^3}\,dD\otimes dD.}
\tag{1}
\]

The second term is a continuous bilinear map. The theorem
`hasFDerivAt_fderiv_ruler_comp` proves that the expression on the right is the
derivative of the actual `fderiv` of the composed ruler field.

For any locally differentiable positive ruler with a differentiable
differential field, the scale formulas are

\[
d\sigma=-\frac{1}{\rho\log2}\,d\rho,
\qquad
\boxed{H_\sigma=-\frac{1}{\rho\log2}H_\rho
+\frac{1}{\rho^2\log2}\,d\rho\otimes d\rho.}
\tag{2}
\]

For the regularized field, they can also be expressed directly in D:

\[
d\sigma=-\frac{D}{\rho^2\log2}\,dD,
\qquad
\boxed{H_\sigma=-\frac{D}{\rho^2\log2}H_D
+\frac{D^2-\epsilon^2}{\rho^4\log2}\,dD\otimes dD.}
\tag{3}
\]

Both actual second-derivative formulations are checked by
`hasFDerivAt_fderiv_scaleCoordinate_comp` and
`hasFDerivAt_fderiv_scaleCoordinate_ruler_comp`.

### A checked simplification at a smooth zero level

At a point where D=0 and the stated differentiability hypotheses hold,

\[
\boxed{H_\rho=\frac1\epsilon dD\otimes dD,\qquad
H_\sigma=-\frac{1}{\epsilon^2\log2}dD\otimes dD.}
\tag{4}
\]

The underlying field Hessian drops out entirely. Directions annihilated by
dD have zero second derivative in both expressions. If D is a smooth ideal
signed distance, dD is the unit normal covector, so (4) isolates the normal
crest curvature without using the Euclidean curvature of the Shore. The
formal theorem does not assume the ideal unit-gradient property, and (4)
does not create smoothness at a fractal boundary or a cut locus.

### Scale origin and summit

The module also checks

\[
\sigma_{\alpha d_{\mathrm{ref}}}(\rho)
=\sigma_{d_{\mathrm{ref}}}(\rho)+\frac{\log\alpha}{\log2}
\quad(\alpha>0),
\]

and

\[
\sigma\le\log_2(d_{\mathrm{ref}}/\epsilon),
\qquad
\sigma=\log_2(d_{\mathrm{ref}}/\epsilon)\iff D=0.
\]

These are `scaleCoordinate_reference_shift`,
`scaleCoordinate_ruler_le_summit`, and
`scaleCoordinate_ruler_eq_summit_iff`. The derivative formulas contain no
reference-scale parameter after differentiation.

## 2. Deriving the compact connection in local coordinates

Now let E be a real inner-product space. At a configuration write
u=gradient(rho), H for the derivative of that gradient field, and

\[
W=1+a^2\|u\|^2,\qquad
g(v,w)=\frac{\langle v,w\rangle+
a^2\langle u,v\rangle\langle u,w\rangle}{\rho^2}.
\]

The checked metric differential is

\[
(Dg[v])(w,z)
=-\frac{2\langle u,v\rangle}{\rho}g(w,z)
+\frac{a^2}{\rho^2}
\bigl(\langle Hv,w\rangle\langle u,z\rangle
+\langle u,w\rangle\langle Hv,z\rangle\bigr).
\tag{5}
\]

`hasDerivAt_metric_pairing` establishes its correspondence with actual
differentiation, also allowing the paired vectors to vary with time.

The compact connection expression is

\[
\boxed{
\begin{aligned}
\Gamma(v,w)={}&\frac{a^2\langle Hv,w\rangle}{W}u
-\frac{\langle u,w\rangle}{\rho}v
-\frac{\langle u,v\rangle}{\rho}w\\
&+\frac{\langle v,w\rangle+
a^2\langle u,v\rangle\langle u,w\rangle}{\rho W}u.
\end{aligned}}
\tag{6}
\]

With symmetric H, Lean checks Gamma(v,w)=Gamma(w,v) and the local Koszul
identity

\[
2g(\Gamma(v,w),z)
=(Dg[v])(w,z)+(Dg[w])(v,z)-(Dg[z])(v,w).
\tag{7}
\]

For rho>0, positive definiteness proves that Gamma(v,w) is the unique vector
satisfying (7). The corresponding metric-compatibility identity is

\[
(Dg[v])(w,z)=g(\Gamma(v,w),z)+g(w,\Gamma(v,z)).
\tag{8}
\]

Symmetry of H is needed for the torsion-free/Koszul assertions. It is not
needed for (8) once the metric differential is supplied by the local jets.
The principal identifiers are `connection_symm`, `metric_connection`,
`connection_metric_compatible`, `connection_koszul`, and
`connection_eq_of_koszul`.

This is a checked local-coordinate identification and uniqueness result.
The package does not construct a bundled Riemannian manifold connection or
prove an existence theorem for its geodesic differential equation.

## 3. Actual kinetic and total-energy derivatives

Let gamma be a differentiable position curve, v its velocity, and acc its
velocity derivative. Along the curve, the actual field chain rules give

\[
\dot\rho=\langle u,v\rangle,\qquad \dot u=Hv.
\]

For the existing kinetic energy K=g(v,v)/2, Lean proves

\[
\boxed{\frac{dK}{dt}=g(v,\mathrm{acc}+\Gamma(v,v)).}
\tag{9}
\]

The equation of motion with a time-independent potential U is

\[
\mathrm{acc}+\Gamma(v,v)=G^{-1}(Q-\nabla U),
\tag{10}
\]

where force and potential-gradient covectors are represented using the
ambient inner product. Pairing with v uses the already checked inverse:
g(v,G^-1 J)=inner(v,J). The actual chain rule for U therefore gives

\[
\boxed{\frac{d}{dt}(K+U)=\langle v,Q\rangle.}
\tag{11}
\]

`hasDerivAt_kinetic` and `hasDerivAt_totalEnergy` prove the time-trace forms.
`hasDerivAt_totalEnergy_of_fields` starts with the actual position derivative
and actual Fréchet derivatives of the ruler, its gradient, and the potential;
it discharges the trace identities by the Fréchet chain rule. Work-energy is
the conclusion, not a premise.

There are explicit checked zero-derivative corollaries for free geodesic
motion and conservative motion. These statements concern continuous
trajectories satisfying (10); they do not imply a discrete integrator
conserves energy.

## 4. Dissipation and the signed work ledger

For friction Q_f=-Bv and a positive-semidefinite quadratic form,

\[
\langle w,Bw\rangle\ge0\quad\text{for all }w
\quad\Longrightarrow\quad
\langle v,Q_f\rangle\le0.
\]

For metric-proportional friction the exact identity is

\[
Q_f=-\beta Gv\quad\Longrightarrow\quad
\langle v,Q_f\rangle=-2\beta K.
\tag{12}
\]

Nonpositive power follows when beta>=0. Symmetry of a general B is not
needed once its quadratic-form nonnegativity is stated. These are the
checked `friction_power_nonpos` and `metric_friction_power` results.

With continuous endpoint energy, the differential hypotheses on the open
time interval, and integrable power, Lean integrates (11):

\[
\boxed{\mathcal W:=\int_{t_0}^{t_1}\langle v,Q\rangle\,dt
=(K+U)(t_1)-(K+U)(t_0).}
\tag{13}
\]

The theorem is `integral_work_eq_energy_change`. It retains the signed
work of all included nonconservative forces, including friction. Since
final kinetic energy is nonnegative,

\[
\boxed{U(t_1)-U(t_0)\le K(t_0)+\mathcal W.}
\tag{14}
\]

`potential_increase_le_initial_kinetic_add_work` checks this algebraic
consequence of the ledger. The potential difference may contain both the
Shore contribution and an outer-wall term; neither is discarded. This is a
necessary energy condition, not a sufficient crossing or reachability theorem.
The separate AC crossing-action result in
[the #160 continuation](160-absolute-continuous-paths.md) bounds an integral
of kinetic energy rather than the work in (13).

## 5. The normalized Map differential and quadratic residual

Let D be a real-valued field on a real normed space. Hold the anchor c and
physical window scale h fixed, with h nonzero, and write

\[
M_{D,c,h}(u)=\frac{D(c+hu)-D(c)}h.
\]

Where D is Fréchet differentiable at the queried point c+hu, Lean proves the
actual window differential

\[
\boxed{d_uM_{D,c,h}(u)=dD(c+hu),\qquad
d_uM_{D,c,h}(0)=dD(c).}
\tag{15}
\]

The first identity requires differentiability at c+hu; the second requires
it at c. The differential of the affine query contributes a factor h, which
cancels the denominator. These are `hasFDerivAt_normalizedMap`,
`hasFDerivAt_normalizedMap_zero`, and `fderiv_normalizedMap`. In particular,
the last theorem identifies the actual `fderiv` fields, rather than a
separately supplied gradient formula. The scale remains fixed in this
derivative even when its chosen value is h=q rho(c). Differentiation with
respect to the anchor c, or a varying ruler, is a different calculation.

For the quadratic residual, put

\[
K_{c,h,u}=\{c+t(hu):0\le t\le1\},\qquad
R_{D,c,h}(u)=M_{D,c,h}(u)-dD(c)[u].
\]

The checked Hessian-bound theorem assumes the following on this entire
closed physical segment, including both endpoints:

- D is Fréchet differentiable at each z in K.
- The actual differential field `fderiv ℝ D` has Fréchet derivative H(z)
  at each such z, as a map into the space of continuous linear functionals.
- Its operator norm satisfies ||H(z)|| <= L, with L >= 0.

Under those hypotheses,

\[
\boxed{|R_{D,c,h}(u)|\le\frac{L|h|}{2}\|u\|^2.}
\tag{16}
\]

`normalizedMap_quadratic_remainder_of_hessian_bound` proves (16) for the
actual normalized query. The operator norm is that of
\(H(z)\in\mathcal L(E,\mathcal L(E,\mathbb R))\); the hypotheses concern ambient derivatives at
the segment points. They are not merely differentiability of a restricted
one-dimensional trace. No finite-dimensionality, inner product, global C2
regularity, or ideal signed-distance assumption is needed.

The proof first obtains an L-Lipschitz bound for the actual differential
on K, using the convex mean-value theorem. The theorem
`normalizedMap_quadratic_remainder_of_lipschitz_fderiv` also accepts that
Lipschitz hypothesis directly, without a Hessian field. The more general
`field_quadratic_remainder` and `normalizedMap_quadratic_remainder` need
only the anchored variation bound

\[
\|dD(c+t hu)-dD(c)\|\le L\|t hu\|\quad(0\le t\le1).
\]

A mean-value comparison with a quadratic scalar function checks the factor
1/2. For the physical convention h=q rho(c)>0, (16) becomes
|R(u)| <= L q rho(c) ||u||^2 / 2. The constant L may be specific to this
segment; a uniform statement over a whole window requires the same bound
on every queried segment.

There is also a weaker, nonsmooth value contract:
`normalizedMap_value_bound_on_segment` proves |M(u)| <= L0 ||u|| when D is
L0-Lipschitz only on K. Taking L0=1 recovers the ideal-field value bound
without a global Lipschitz hypothesis. This value theorem requires no
derivative. Normalization remains defined at a cut locus, but neither
(15) nor (16) applies there without its stated differentiability hypotheses.

The companion [normalized-field note](162-normalized-distance-fields.md)
records the chart identities and provider-error boundaries alongside these
new differential contracts.

## 6. Explicit potential/control continuation and remaining Phase A boundary

This increment supplies the full ruler/scale Hessians, scale-origin and
summit identities, local connection identification, and continuous work-energy
contracts. The normalized Map differential and its segment Hessian-bound
quadratic remainder are now checked as well. The subsequent
[potential and control extension](159-potential-controls.md) checks the
explicit Shore and secant-wall potential/force catalog, the full signed
crest-energy budget, momentum differentiation, intrinsic drive and drag,
candidate board friction and carve normalization, and fixed-metric steering.
The [discrete mechanics note](159-discrete-mechanics.md) adds exact
frozen-metric step identities and a sharp explicit-drag timestep criterion.

The local hypotheses do not certify a numerical GeometryProvider's sampled
derivatives. Rust remains the runtime authority; these exact-real proofs do
not establish floating-point implementation correspondence, numerical
stability, solution existence, or global smoothness of a regularized signed
distance field. In particular, an arbitrary sampled gradient or Hessian is
not identified with `fderiv` by these statements, and no error certificate is
derived for that approximation. The combined recomputed-ruler provider-error
bound in the [#162 note](162-normalized-distance-fields.md) is now checked
under explicit query-error certificates. It does not manufacture those
certificates for the numerical provider. Covariant steering along a varying
configuration and full changing-metric numerical-step bounds remain separate.
No runtime or training
contract changes follow from this development-time extension.
