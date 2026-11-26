# Distance-based sensitivity metrics: a template theorem and two obstructions

Research note for [FractalSync #161](https://github.com/rhythmatician/FractalSync/issues/161), with application to [#150](https://github.com/rhythmatician/FractalSync/issues/150). Prepared 2026-11-24 against repository commit `55df760ff706210dcce3c88e79e476a45c5331f0`.

## Status and main result

The results below are conventional mathematical proofs. The accompanying numerical experiment checks formulas independently by differentiating rendered analytic templates; it is not a proof. The formalization inventory below identifies four norm/scalar lemmas checked by Lean 4.24.0 with Mathlib revision `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. Neither these proofs nor those lemmas establish a theorem about the current Julia renderer or its floating-point implementation.

The useful positive result is stronger than the Gaussian control example in #150: every nonzero Schwartz template, real or complex, produces a nondegenerate normalized location-scale family whose pullback metric is a constant-Gram upper-half-space metric. An explicit shear identifies that entire parameter space with scaled real hyperbolic space, without assuming a symmetric template, isotropic translation sensitivity, or vanishing mixed terms. Restricting this family to any positive differentiable ruler gives uniform comparison with the graph metric used by FractalSync. A quantitative derivative-perturbation criterion makes this comparison robust to model error.

The negative result is equally useful: distance to a singular set does not select a universal exponent. A scalar distance multiplier cannot remove unequal directional sensitivity exponents. Regularizing the metric's ruler also does not regularize a genuinely singular observable.

These are reusable known-structure results and elementary consequences, not an established novel mathematical contribution. The concrete open question is whether an independently chosen Julia observable meets a useful version of the transfer assumptions on resolved patches.

## 1. An arbitrary localized template gives hyperbolic location-scale geometry

Let $n\ge1$ and let $f\in\mathcal S(\mathbb R^n;\mathbb C)$ be nonzero, with $\|f\|_{L^2}=1$. Real-valued templates are included. Regard complex $L^2$ as a **real** Hilbert space, with

$$\langle u,v\rangle_{\mathbb R}=\operatorname{Re}\int\overline{u(y)}v(y)\,dy.$$

The Schwartz assumption is deliberately sufficient rather than minimal: it makes differentiation under the integral and the argument below immediate. Weak Sobolev extensions require an explicit generator-domain proof and are not silently included.

Define

$$F(\mu,s)(x)=s^{-n/2}f((x-\mu)/s),\qquad \mu\in\mathbb R^n,\ s>0.$$

Let $U_{\mu,s}g=s^{-n/2}g((x-\mu)/s)$ and define the translation and dilation generators

$$T_i f=\partial_i f,\qquad Af=\frac n2f+y\cdot\nabla f.$$

The maps $U_{\mu,s}$ are real-unitary, and direct differentiation in $L^2$ gives

$$DF(\mu,s)[u,v]=-\frac1s U_{\mu,s}\left(u\cdot\nabla f+vAf\right).$$

Let $K$ be the real Gram matrix of $(T_1f,\ldots,T_nf,Af)$, written

$$K=\begin{pmatrix}A_0&b\\b^\top&c_0\end{pmatrix}.$$

Then the full parameter-space metric is

$$g=\frac{d\mu^\top A_0d\mu+2b^\top d\mu\,ds+c_0ds^2}{s^2}. \tag{1}$$

### Theorem 1: the Gram matrix is positive definite

For every nonzero Schwartz template above, $K\succ0$.

**Proof.** A null real Gram vector $(u,v)$ would satisfy

$$u\cdot\nabla f+v\left(\frac n2f+y\cdot\nabla f\right)=0$$

in $L^2$, hence everywhere by smoothness. If $v=0$ and $u\ne0$, $f$ is constant along every line parallel to $u$. Schwartz decay along those lines forces $f=0$.

If $v\ne0$, set $y_0=-u/v$. The equation becomes

$$(y-y_0)\cdot\nabla f(y)=-\frac n2f(y).$$

Along the scaling flow, for any $y$,

$$f(y_0+e^t(y-y_0))=e^{-nt/2}f(y).$$

As $t\to-\infty$, the left side stays bounded by continuity near $y_0$. Since $n>0$, the right side can stay bounded only if $f(y)=0$. This again contradicts $f\ne0$. Thus no nonzero null vector exists. $\square$

This is a statement about infinitesimal identifiability of the **full image or signal in $L^2$**. An invariant or lossy descriptor applied afterward can destroy it.

### Theorem 2: an explicit hyperbolic coordinate change

Set

$$\delta=c_0-b^\top A_0^{-1}b>0,\qquad
\xi=A_0^{1/2}(\mu+A_0^{-1}bs),\qquad z=\sqrt\delta\,s.$$

These coordinates define a global diffeomorphism of the upper half-space. Completing the square in (1) gives

$$g=\delta\frac{|d\xi|^2+dz^2}{z^2}. \tag{2}$$

Thus the full location-scale manifold has constant sectional curvature $-1/\delta$. Nonzero translation-scale cross terms require a shear of the location coordinate, not a new geometry. Translation anisotropy requires a fixed linear change of coordinates. The graph of a position-dependent ruler inside this space generally has variable curvature.

**Proof.** Positive definiteness of $K$ implies $A_0\succ0$ and $\delta>0$, and

$$d\mu^\top A_0d\mu+2b^\top d\mu\,ds+c_0ds^2
=|A_0^{1/2}(d\mu+A_0^{-1}b\,ds)|^2+\delta ds^2.$$

Use $s^2=z^2/\delta$. The curvature statement follows by constant rescaling of the standard upper-half-space metric. $\square$

For a positive real template, $p=f^2$ is a density and the Fisher metric of its location-scale family is $4g$ under the usual differentiability and integrability hypotheses. For a general complex template, the phase carries additional $L^2$ information; identifying this metric with the Fisher metric of $|f|^2$ would discard it.

## 2. Pullback to a ruler graph, with no gradient bound

Let $\Omega\subset\mathbb R^n$ be open, $\rho:\Omega\to(0,\infty)$ be $C^1$, and define the template observable

$$T_\rho(c)=F(c,\rho(c)).$$

Fix $a>0$ and write $r=\nabla\rho(c)$,

$$G_a(c)=\rho^{-2}(I+a^2rr^\top),\quad
J_a=\begin{pmatrix}I\\ ar^\top\end{pmatrix},\quad
K_a=\operatorname{diag}(I,1/a)K\operatorname{diag}(I,1/a).$$

### Theorem 3: uniform comparison on every such graph

Let $\lambda_-$ and $\lambda_+$ be the smallest and largest eigenvalues of the **constant** SPD matrix $K_a$. Then

$$0<\lambda_-G_a\preceq P_T\preceq\lambda_+G_a,\qquad P_T=(DT_\rho)^*DT_\rho. \tag{3}$$

**Proof.** The chain rule and (1) yield the identities

$$P_T=\rho^{-2}J_a^\top K_aJ_a,
\qquad G_a=\rho^{-2}J_a^\top J_a.$$

Evaluate the eigenvalue inequalities for $K_a$ on $J_av$. $\square$

No Lipschitz bound on $\rho$ is needed. This matters outside ideal-distance geometry: the graph correction can absorb arbitrarily steep scale change for this model class. In FractalSync's ideal signed-distance setting, $|\nabla\rho|\le1$, so the scale-only metric is already uniformly comparable to the full graph metric. The theorem does not override the need for that ablation.

If $A_0=b_0I$, $b=0$, and $c_0>0$, choosing $a^2=c_0/b_0$ gives exact equality $P_T=b_0G_a$. For the centered normalized Gaussian $f(y)=\pi^{-n/4}e^{-|y|^2/2}$, $A_0=I/2$, $b=0$, and $c_0=n/2$. Hence $a^2=n$ gives $P_T=G_a/2$. The two-dimensional example in #150 is the case $n=2$.

## 3. A quantitative transfer condition for a non-template observable

Let $V:\Omega\to\mathcal H$ be differentiable into the same real Hilbert space as $T_\rho$. Suppose, on a stated patch,

$$\|(DV-DT_\rho)v\|\le\eta\,\|v\|_{G_a}\quad\text{for every tangent }v. \tag{4}$$

### Theorem 4: robustness under derivative error

If $0\le\eta<\sqrt{\lambda_-}$, then

$$\bigl(\sqrt{\lambda_-}-\eta\bigr)^2G_a
\preceq (DV)^*DV
\preceq\bigl(\sqrt{\lambda_+}+\eta\bigr)^2G_a. \tag{5}$$

**Proof.** Apply the triangle and reverse-triangle inequalities to $DVv=DT_\rho v+(DV-DT_\rho)v$, then use (3) and square the resulting nonnegative inequalities. $\square$

The hypothesis measures derivative error in units of intrinsic displacement. Small image reconstruction error by itself does not imply (4): the scalar error $E_\varepsilon(t)=\varepsilon\sin(t/\varepsilon^2)$ has uniformly small amplitude but derivatives of order $1/\varepsilon$.

If both observations instead arise from a non-isometric descriptor or projection, its action on the relevant derivative subspaces also needs a lower bound. An $L^2$ immersion alone does not establish descriptor identifiability.

Equation (5) provides a concrete mathematical bridge worth testing in #150/#151. It does not assert that any Julia image is a translated and dilated template, or prescribe a descriptor to make that hypothesis true. Establishing (4) for independently defined observables is the remaining application-specific work.

For any linearized controls response $B$, a comparison $\ell G_a\preceq P\preceq uG_a$ immediately gives

$$\ell B^\top G_aB\preceq B^\top PB\preceq uB^\top G_aB.$$

This transfers useful bounds to the action space; it does not remove null directions of $B$ or of the observed response. Stacking multiple times can change the identifiable subspace and must be analyzed as a separate operator.

## 4. Scalar distance factors cannot repair unequal exponents

The following asymptotic statements use the **unregularized** distance $d\to0$, or a resolved joint limit with $\epsilon/d$ bounded. They do not take $\rho_\epsilon\to0$ while keeping a fixed positive $\epsilon$.

Let $J(d)$ be an injective observable Jacobian, with singular values $s_1(d)\ge\cdots\ge s_n(d)>0$. Let

$$G_p(d)=d^{-2p}h(d),\qquad mI\preceq h(d)\preceq MI$$

for fixed $0<m\le M<\infty$. The generalized sensitivity eigenvalues are those of

$$C_p=d^{2p}h^{-1/2}J^*Jh^{-1/2}.$$

The generalized min-max characterization gives, for every ordered eigenvalue,

$$\frac{d^{2p}s_i(d)^2}{M}\le\lambda_i(C_p)\le
\frac{d^{2p}s_i(d)^2}{m}. \tag{6}$$

### Theorem 5: a necessary and sufficient exponent condition

Suppose for fixed exponents $q_i$ and positive constants $a_i,b_i$,

$$a_i d^{-q_i}\le s_i(d)\le b_i d^{-q_i}$$

for all sufficiently small positive $d$. Then all eigenvalues of $C_p$ remain bounded above and bounded away from zero **if and only if** $q_i=p$ for every $i$.

**Proof.** In (6), each eigenvalue is bounded above and below by positive constants times $d^{2(p-q_i)}$. This stays bounded above and away from zero precisely when the exponent vanishes. $\square$

More generally,

$$\kappa(C_p)\ge\frac mM\left(\frac{s_1(d)}{s_n(d)}\right)^2. \tag{7}$$

The scalar factor cancels from the condition number. A uniformly bounded graph correction can alter constants, but cannot cure an asymptotically diverging directional ratio. Rank failure is an even stronger obstruction: an invertible metric change preserves the Jacobian rank.

### Explicit fold model

The equilibrium branch of $\dot x=t-x^2$ is $x_*(t)=\sqrt t$ for $t>0$. Its derivative is $1/(2\sqrt t)$, so for the scalar observable $V(t)=\sqrt t$,

$$P(t)=\frac1{4t},\qquad G_{1/2}=t^{-1},\qquad C_{1/2}=\frac14.$$

The correct distance exponent for this observable is $p=1/2$, whereas $p=1$ yields $C_1=t/4\to0$. In one dimension every positive scalar Hessian has condition number one; the distinction here is a uniform nonzero sensitivity scale across parameters, not directional conditioning.

For the two-parameter observable $V(t,y)=(\sqrt t,y)$, with singular set $\{t=0\}$ and distance $d=t$, one has

$$P=\operatorname{diag}(1/(4t),1).$$

For the unregularized graph correction $h=\operatorname{diag}(1+a^2,1)$,

$$C_p=\operatorname{diag}\left(\frac{t^{2p-1}}{4(1+a^2)},t^{2p}\right).$$

For sufficiently small $t$, its condition number is exactly $1/[4(1+a^2)t]$, for **every** exponent $p$. This is an analytic nontrivial model where no scalar distance preconditioner with a bounded correction can work in all directions.

The same singular set can support a different exponent for a different observable: $V(t)=\log t$ has $P=t^{-2}$ and exact $p=1$ scaling. Therefore the set and the dynamical bifurcation alone do not determine the observable metric.

## 5. The regularization floor has a precise limit

In the fold example, take $\rho_\epsilon=\sqrt{t^2+\epsilon^2}$ and $G_{1/2}=\rho_\epsilon^{-1}$. Then

$$C_{1/2}=\frac{\rho_\epsilon}{4t}.$$

For fixed $\epsilon>0$, this diverges as $t\downarrow0$: a finite metric floor cannot bound an actually singular derivative. On the explicit resolved domain $t\ge k\epsilon$, with $k>0$,

$$\frac14\le C_{1/2}\le\frac14\sqrt{1+k^{-2}}. \tag{8}$$

Conversely, if an observable has bounded derivatives $\|DV\|\le L$ while $G_p\succeq m\rho^{-2p}I$, then

$$\lambda_{\max}(G_p^{-1/2}P G_p^{-1/2})\le\frac{L^2}{m}\rho^{2p}.$$

For $p>0$ this tends to zero in a limit $\rho\to0$. A fixed-resolution smooth renderer cannot be assumed to keep a uniform lower sensitivity bound under arbitrarily finer geometric scaling. The resolved scale range and the actual renderer derivative must be measured together.

## 6. Prior art and originality verdict

The following primary sources were checked during this investigation:

1. Laurent Jacques and Christophe De Vleeschouwer, *A Geometrical Study of Matching Pursuit Parametrization*, IEEE Transactions on Signal Processing 56 (2008), 2835–2848. [arXiv:0801.3372](https://arxiv.org/abs/0801.3372), [DOI](https://doi.org/10.1109/TSP.2008.917379). Their framework treats parameterized dictionaries as submanifolds of signal space. Section 4.2 specifically takes a **symmetric real one-dimensional** mother function and obtains a constant diagonal Gram matrix times inverse scale squared. This is direct prior art for the location-scale calculation, though it is not the full arbitrary complex-template statement proved above.
2. Srdjan Vukmirović and Tijana Šukilović, *Geodesic completeness of the left-invariant metrics on $\mathbb RH^n$* (2020). [Publisher](https://umj.imath.kiev.ua/index.php/umj/article/view/645), [DOI](https://doi.org/10.37863/umzh.v72i5.645). Theorems 2.1–2.3 classify left-invariant metrics on the hyperbolic Lie group using the same block shear and Schur complement and establish constant curvature. They credit the Riemannian constant-curvature result to Milnor's 1976 work. This is direct structural prior art for the general affine shear result, including absence of a symmetry requirement in the Riemannian case.
3. Carlos Beltrán, Jean-Pierre Dedieu, Gregorio Malajovich, and Mike Shub, *Convexity Properties of the Condition Number*, SIAM J. Matrix Anal. Appl. 31 (2010), 1491–1506. [Author manuscript](https://personales.unican.es/beltranc/archivos/LogConvexCN.pdf), [DOI](https://doi.org/10.1137/080718681). Theorem 2 treats the inverse-distance-squared conformal metric for a complete $C^2$ submanifold without boundary, on the open domain of unique nearest-point projection, and proves self-convexity there. It does not automatically apply to a fractal singular set, a regularized floor, or the additional graph term.
4. Yi-Chiuan Chen and Tomoki Kawahira, *Simple Proofs for the Derivative Estimates of the Holomorphic Motion near Two Boundary Points of the Mandelbrot Set*. [Author manuscript](https://www1.econ.hit-u.ac.jp/kawahira/works/ChenKawahira_DerivativeEstimates.pdf), [arXiv:1808.08335](https://arxiv.org/abs/1808.08335). The derivative upper bounds concern moving Julia points on particular parameter approaches near $1/4$ and $-2$. They do not establish a two-sided renderer sensitivity estimate in distance to the full Mandelbrot boundary.

The explicit formulas and counterexamples in this note are derivations made for this project. The literature already contains the central geometric structures, and the comparison and perturbation arguments are standard linear algebra and norm inequalities. **No standalone novelty claim is justified by the current evidence.** A useful independent contribution could emerge from a new theorem verifying a derivative-transfer hypothesis for a nontrivial dynamical observable, a sharp regime classification near particular bifurcations, or a new robust numerical certification method. None is proved here.

## 7. Concrete follow-up and formalization inventory

The [residual curvature and response bounds](161-response-bounds.md) complete
the comparison package with the actual least-squares second derivative and
the finite-horizon action bound. The renderer and Controls measurements
remain in #150 and #151.

- Preserve the arbitrary-template theorem as a mathematical control family for #150/#151, alongside the Gaussian special case.
- In measurements, report both normalized singular values; fitting a single scale exponent can conceal a direction that is becoming null.
- Separate variation across scale from pointwise directional condition number. Scalar rescaling changes the former but cancels from the latter.
- Use (4) only if derivative error is actually assessed. Small template reconstruction residual is insufficient.
- Report the ratio $d/\epsilon$ along with sensitivity near a regularization floor.
- Small suitable Lean targets are completion of the square and Schur positivity in one location dimension, pullback of quadratic-form inequalities, and the norm-perturbation lower/upper comparison. General differentiability of the $L^2$ affine action and the global curvature theorem require substantially more library work; they are conventional proofs here.

The module `formal/FractalSync/SensitivityBounds.lean` now verifies:

| Lean declaration | Conventional content | Scope |
| --- | --- | --- |
| `FractalSync.sensitivity_norm_transfer` | Model-response norm bounds transfer with additive derivative-error constants. | Arbitrary seminormed additive commutative group; no differentiability claim. |
| `FractalSync.sensitivity_squared_transfer` | Squaring gives the constants $(\ell-\eta)^2$ and $(u+\eta)^2$. | Explicit nonnegative input radius and lower margin; the norm/algebra core of Theorem 4. |
| `FractalSync.location_scale_complete_square` | $Au^2+2buv+cv^2=A(u+(b/A)v)^2+(c-b^2/A)v^2$. | Real scalar location coordinate; assumes $A\ne0$. |
| `FractalSync.location_scale_schur_pos` | $A>0$ and $Ac-b^2>0$ imply $c-b^2/A>0$. | Scalar positive-definiteness prerequisite. |

`lake build FractalSync.SensitivityBounds` succeeds. Transitive axiom reports contain only Lean's standard `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx`, project axioms, or unfinished proofs occur. The analytic theorems, exponent asymptotics, and curvature statement remain conventional proofs.

No runtime code, observation contract, reward, training prerequisite, or canonical value of $a$ or $p$ changes as a consequence of this note.
