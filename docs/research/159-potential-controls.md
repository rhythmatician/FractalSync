# #159: explicit potentials, forces, and metric control contracts

The current Shore potential and outer secant wall now have checked actual
Fréchet derivatives, force covectors, and metric-raised accelerations. Their
sum enters a checked continuous work-energy theorem and a necessary
Shore-crest energy budget. Additional control contracts cover drive, viscous
drag, momentum impulses, metric normalization, candidate board friction,
carve bias, and steering at a fixed metric configuration.

The source modules are
[PotentialForces.lean](../../formal/FractalSync/PotentialForces.lean) and
[ControlMechanics.lean](../../formal/FractalSync/ControlMechanics.lean).
They reuse the definitions in
[the shared metric package](157-scale-metric.md) and the actual local calculus
in [the differential mechanics note](159-differential-mechanics.md).
The [discrete step note](159-discrete-mechanics.md) treats a separate,
explicitly frozen-metric calculation. The subsequent
[moving-metric note](159-moving-metric-dynamics.md) checks conditional local
energy defects for the complete position update and continuous covariant
steering along a moving configuration.

## 1. Shared metric and the covector convention

Let \(E\) be a real inner-product space. At a fixed configuration write
\[
\rho>0,\qquad u=\nabla\rho,\qquad
W=1+a^2\|u\|^2,\qquad
G=\rho^{-2}(I+a^2u\otimes u).
\]
The existing inverse is
\[
G^{-1}Q=\rho^2\left(Q-\frac{a^2\langle u,Q\rangle}{W}u\right).
\tag{1}
\]
The displayed force vectors represent covectors by the ambient Riesz
identification. Their work pairing is \(\langle v,Q\rangle\); their
coordinate acceleration is \(G^{-1}Q\). These are different uses of the
same ambient representation. No unit-gradient assumption is needed.

## 2. The Shore ridge and reference-origin invariance

For \(d_{\rm ref}>0\), the canonical potential is
\[
U_\sigma(x)=\kappa\frac{\log(d_{\rm ref}/\rho(x))}{\log2}.
\]
Assuming the actual derivative \(d\rho_x=\langle u,\cdot\rangle\), Lean
proves
\[
dU_{\sigma,x}=-\frac{\kappa}{\rho\log2}\langle u,\cdot\rangle,
\qquad
Q_\sigma=\frac{\kappa}{\rho\log2}u.
\tag{2}
\]
Applying (1) yields
\[
\boxed{G^{-1}Q_\sigma=\frac{\kappa\rho}{W\log2}u.}
\tag{3}
\]
The apparent inverse-ruler factor in the covector is cancelled by the
inverse metric. Equation (3) includes \(u=0\). It does not require
\(\|u\|=1\), and it is independent of the reference scale.

For \(\alpha>0\), replacing \(d_{\rm ref}\) by
\(\alpha d_{\rm ref}\) adds the constant \(\kappa\log\alpha/\log2\).
The actual conservative differential is unchanged. For any two positive
ruler values,
\[
U_\sigma(\rho_1)-U_\sigma(\rho_0)
=\kappa\frac{\log(\rho_0/\rho_1)}{\log2}.
\tag{4}
\]
The principal declarations are `hasFDerivAt_shorePotential`,
`shore_acceleration`, `shorePotential_reference_shift`,
`shorePotential_difference`, and `totalPotential_fderiv_reference_invariant`.

## 3. The actual secant wall

For the current \(p=8\) outer wall, use
\[
s(x)=\frac{\|x\|^2}{4},\qquad
\varphi(x)=\frac\pi2s(x)^4,\qquad
U_{\rm wall}(x)=\mu(\sec\varphi(x)-1).
\]
The physical domain is \(\|x\|<2\). Lean proves
\(0\le\varphi<\pi/2\), hence \(\cos\varphi>0\), throughout that domain.
It differentiates the actual norm-square polynomial and trigonometric
composition to obtain
\[
\boxed{Q_{\rm wall}=-c(x)x,\qquad
c(x)=\mu\pi s(x)^3\sec\varphi(x)\tan\varphi(x).}
\tag{5}
\]
This is `hasFDerivAt_wallPotential`, with the negative differential
represented by `wallForce`. The default \(\mu=1/\pi\) specialization is
checked separately and cancels the factor \(\pi\).

For \(\mu\ge0\), the wall potential and coefficient are nonnegative, and
\[
\langle x,Q_{\rm wall}\rangle\le0.
\]
This is an inward radial covector statement. Its acceleration is
\[
\boxed{G^{-1}Q_{\rm wall}
=-c(x)\rho^2\left(x-\frac{a^2\langle u,x\rangle}{W}u\right).}
\tag{6}
\]
The metric projection term generally changes its Euclidean direction.
`wall_acceleration` retains that term rather than identifying the raised
force with a radial acceleration.

For \(\mu>0\), `wallPotential_tendsto_atTop` also checks
\[
U_{\rm wall}(x)\longrightarrow+\infty
\quad\text{whenever }\|x\|\to2\text{ from inside the domain}.
\tag{7}
\]
This is an actual filter limit. A separate checked condition in the
[moving-metric continuation](159-moving-metric-dynamics.md),
`norm(x)+abs(dt)*norm(v)+dt²*norm(A)<2`, keeps the proposed affine position
segment inside the disk. That conditional geometric result does not prove
unconditional step acceptance or floating-point safety. The totalized Lean
trigonometric expression outside the stated domain is not given a physical
interpretation.

## 4. Full continuous energy and the crest budget

Let \(U=U_\sigma+U_{\rm wall}\) and
\(Q_U=Q_\sigma+Q_{\rm wall}\). The checked differential of the actual sum is
\(dU=-\langle Q_U,\cdot\rangle\). For a trajectory satisfying
\[
\dot\gamma=v,\qquad
\dot v+\Gamma(v,v)=G^{-1}(Q_c-Bv+Q_U),
\]
the actual field chain rules and metric connection identity give
\[
\frac{d}{dt}(K+U)=\langle v,Q_c\rangle-\langle v,Bv\rangle,
\qquad K=\tfrac12g(v,v).
\tag{8}
\]
`canonical_work_energy` integrates (8):
\[
\boxed{(K+U)(t_1)-(K+U)(t_0)
=\int_{t_0}^{t_1}\langle v,Q_c\rangle\,dt
-\int_{t_0}^{t_1}\langle v,Bv\rangle\,dt.}
\tag{9}
\]
Its hypotheses include \(t_0\le t_1\), positive ruler and
\(\|\gamma(t)\|<2\) on the open time interval, actual position and velocity derivatives, actual
ruler and gradient-field Fréchet derivatives, the equation of motion on
the open time interval, continuity of energy on the closed interval, and
integrability of the work and friction terms. These are sufficient
pointwise calculus hypotheses. No solution-existence theorem is inferred,
and positive regularization alone does not establish field smoothness.

Now take \(\rho=\sqrt{\epsilon^2+D^2}\) with \(\epsilon>0\), and suppose
\(D(\gamma(t_1))=0\). The exact full potential difference is
\[
\Delta U=\kappa\frac{\log(\rho_0/\epsilon)}{\log2}
+U_{\rm wall}(\gamma(t_1))-U_{\rm wall}(\gamma(t_0)).
\]
Since final kinetic energy is nonnegative, (9) yields the checked necessary
condition
\[
\boxed{
\kappa\frac{\log(\rho_0/\epsilon)}{\log2}+\Delta U_{\rm wall}
\le K_0+W_c-\mathcal D_B,
}
\tag{10}
\]
where \(W_c\) is the signed control-work integral and
\(\mathcal D_B=\int\langle v,Bv\rangle\). For a positive-semidefinite
friction field, the latter is nonnegative. `canonical_crest_budget`
derives (10) directly from the stated trajectory calculus and dynamics;
its premises do not assume an energy ledger. Both wall change and negative
control work remain visible. Equation (10) is not a sufficient crossing
criterion.

## 5. Checked control primitives

The velocity differential of the actual kinetic energy and of
\(\mathcal L(x,v)=K(x,v)-U(x)\) is checked to be
\[
\frac{\partial K}{\partial v}
=\frac{\partial\mathcal L}{\partial v}=Gv.
\]
Thus the familiar momentum covector is obtained by actual differentiation.

| Primitive | Checked contract and conditions |
| --- | --- |
| Metric normalization | \(d=v/\sqrt{g(v,v)}\) has \(g(d,d)=1\) for \(\rho>0\) and \(v\ne0\). |
| Intrinsic drive | \(Q_d=FGd\) raises to \(Fd\). For a metric-unit heading, \(\|Q_d\|_{G^{-1}}=\lvert F\rvert\) and \(\lvert\langle v,Q_d\rangle\rvert\le\sqrt{2K}\lvert F\rvert\). Signed \(F\) is allowed. |
| Viscous drag | \(Q_b=-\beta Gv\) raises to \(-\beta v\), with power \(-2\beta K\le0\) when \(\beta\ge0\). |
| Momentum impulse | \(v_+=v+G^{-1}J\) implies \(Gv_+=Gv+J\). The existing exact impulse energy law is retained. |
| Heading projection | For metric-unit \(e\), \(P_ev=g(e,v)e\) is idempotent, and \(v-P_ev\) is metric-orthogonal to \(e\). If \(\langle u,e\rangle=0\), the projected vector is tangent to the ruler level. |
| Planar fixed-metric steering | For a symmetric coordinate form with positive determinant, the actual angle derivative of \(R_\theta d\) is its metric quarter-turn; a differentiable angle trace gives \(\dot\theta J_GR_\theta d\). The finite rotation preserves metric-unit length. |
| Redirection followed by impulse | Any metric-energy-preserving redirection changes the impulse cross term to \(\langle v_{\rm redirected},J\rangle\), with the same quadratic dual energy. |

The board-friction candidate is the bundled continuous linear map
\[
B=\beta_\parallel e^\flat\otimes e^\flat
+\beta_\perp n^\flat\otimes n^\flat.
\]
Lean proves
\[
\langle v,Bv\rangle
=\beta_\parallel g(e,v)^2+\beta_\perp g(n,v)^2\ge0
\tag{11}
\]
for nonnegative coefficients. This remains valid for a degenerate or
nonorthogonal pair of headings; interpreting them as an orthonormal board
frame is a separate premise. The existing inverse raises \(-Bv\) to the
negative weighted sum of those metric projections.

For the soft carve candidate
\[
b_\alpha(d)=(1-\alpha)d+\alpha P_ed,
\]
the theorem `carveBias_ne_zero` proves nonvanishing from \(d\ne0\),
\(\rho>0\), and \(0\le\alpha<1\). Its metric normalization is therefore
well-defined and unit. The strict bound matters: at \(\alpha=1\), a heading
orthogonal to \(e\) could be annihilated.

## 6. Correspondence and remaining boundary

The Shore and wall definitions match the algebraic expressions in
[`manifold.rs`](../../runtime-core/src/manifold.rs), including its default
wall strength and force-as-covector convention. The continuous equation
matches its force sum. The current `controls/2` path retains isotropic drag;
the board, carve, and steering primitives above remain candidate contracts
where the runtime has not adopted them. No new control policy, parameter
value, or runtime behavior is introduced by these Lean modules.

The moving-metric continuation now proves the actual norm derivative and
integrated preservation for the covariant steering equation, including its
connection term. Its metric-skew frame operator is the oriented quarter-turn
on an orthonormal frame plane; arbitrary-dimensional vectors outside that
plane are not assigned a quarter-turn by this construction. Existence of
solutions, a frame-construction algorithm, and discrete norm preservation
remain separate.

The same continuation proves conditional local energy and domain estimates
for the complete real-arithmetic step. A usable bound on the actual
phase-energy Hessian, potentially requiring higher ruler-field regularity,
must still be certified. Provider derivative accuracy, normalization
tolerance behavior, floating-point fidelity, unconditional multi-step safety,
and global rollout estimates are not established by the exact-real catalog.
