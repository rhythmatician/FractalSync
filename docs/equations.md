# FractalSync equations

**Last consolidated:** 2026-09-04

This is the current consolidated equation set after the scale-relative Mandelbrot geometry, intrinsic visual-gesture, audiovisual-entrainment, and offline-teacher pivots. It deliberately separates current mechanical authority from research hypotheses and training-only supervision.

## Status legend

| Status | Meaning |
| --- | --- |
| **Canonical** | Current mathematical authority or accepted destination contract. |
| **Implemented** | Present in the authoritative runtime path. Implementation status can lag a newer canonical equation. |
| **Candidate** | Concrete formulation selected for prototype/ablation, but not yet frozen as permanent authority. |
| **Research** | Falsifiable hypothesis. Do not silently promote it into runtime or training contracts. |
| **Diagnostic** | Derived quantity for inspection/research; not automatically a `PlayerObservation` channel. |
| **Training-only** | Offline target, auxiliary head, loss, or evaluator absent from the live runtime input path. |

## Parameter, time, and sign conventions

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 1 | Positive parameters | \(\displaystyle \boxed{\epsilon,d_{\rm ref},\lambda,\kappa,\mu,G_0,N>0,\qquad \beta\ge 0,\qquad B\succeq0}\) | **Canonical.** \(\epsilon\) regularizes the Shore scale; \(B\) is a positive-semidefinite friction tensor. |
| 2 | Canonical analysis/Physics step | \(\displaystyle \boxed{\Delta t_{\rm canonical}=\frac{H_{\rm hop}}{f_s}=\frac{1024}{48000}\ \mathrm{s}}\) | **Canonical/implemented.** Musical and audio-driven physical time follows sample time, not render cadence. |
| 3 | Generalized-force types | \(\displaystyle \boxed{v,\hat d,a\in T_cM,\qquad p,Q,J\in T_c^*M}\) | **Canonical.** Velocity, directions, and coordinate acceleration are tangent vectors; momentum, force, and impulse are covectors. |
| 4 | Musical flat/sharp maps | \(\displaystyle \boxed{v^\flat=Gv,\qquad Q^\sharp=G^{-1}Q}\) | **Canonical.** The metric converts tangent vectors to covectors and the inverse metric converts covectors to tangent vectors. |
| 5 | Mandelbrot scale-origin invariance | \(\displaystyle d_{\rm ref}\mapsto \alpha d_{\rm ref}\implies \sigma\mapsto\sigma+\log_2\alpha\) | **Canonical.** This adds a constant to \(U_\sigma\) but leaves \(\nabla\sigma\), \(G\), forces, and trajectories unchanged. |
| 6 | Log-scale reference invariance | \(\displaystyle \rho_{\rm ref}\mapsto\alpha\rho_{\rm ref}\implies\eta_M\mapsto\eta_M+\ln\alpha\) | **Canonical.** Only an additive coordinate origin changes; \(\dot\eta_M\) is unaffected. The same applies to the audio reference scale \(s_{\rm ref}\). |
| 7 | Visual phase gauge | \(\displaystyle \Theta\mapsto\Theta+\chi_0\) | **Canonical convention.** Parallel-transported visual phase is defined up to one constant per gesture epoch. Phase differences or fitted offsets absorb \(\chi_0\). |

## Fractal geometry and Mandelbrot Map

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 8 | Julia iteration | \(\displaystyle \boxed{z_{n+1}=z_n^2+c}\) | **Canonical.** Fundamental Julia dynamics. |
| 9 | Mandelbrot iteration | \(\displaystyle \boxed{z_0=0,\qquad z_{n+1}=z_n^2+c}\) | **Canonical.** Defines Mandelbrot membership of \(c\). |
| 10 | Complex parameter | \(\displaystyle \boxed{c=x+iy}\) | **Canonical.** The two independent Mandelbrot configuration coordinates are \((x,y)\). |
| 11 | Smooth escape iteration | \(\displaystyle \boxed{\nu=n+1-\log_2\!\bigl(\log\lvert z_n\rvert\bigr)}\) | **Implemented Map quantity.** Used to construct a smooth escape field. |
| 12 | Normalized escape field | \(\displaystyle \boxed{F(c)=\operatorname{clamp}\!\left(\frac{\nu(c)}{N},0,1\right)}\) | **Implemented Map quantity.** |
| 13 | Escape-field sensitivity | \(\displaystyle \boxed{G_F(c)=\lVert\nabla F(c)\rVert}\) | **Implemented Map quantity.** Local sensitivity of the escape field. |
| 14 | Shore sensitivity/proximity | \(\displaystyle \boxed{S(c)=\frac{G_F(c)}{G_F(c)+G_0}}\) | **Implemented Map quantity.** Not geometric distance. |
| 15 | Signed Shore distance | \(\displaystyle \boxed{D(c)=\begin{cases}+\operatorname{dist}(c,\partial M),&c\notin M\\-\operatorname{dist}(c,\partial M),&c\in M\end{cases}}\) | **Canonical geometry.** Sign gives realm; magnitude gives geometric distance. The practical field is finite-resolution. |
| 16 | Unsigned Shore distance | \(\displaystyle \boxed{d(c)=\lvert D(c)\rvert=\operatorname{dist}(c,\partial M)}\) | **Canonical geometry.** |

## Mandelbrot scale manifold

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 17 | Regularized local ruler | \(\displaystyle \boxed{\rho(c)=\sqrt{D(c)^2+\epsilon^2}}\) | **Canonical.** Removes the \(D=0\) scale singularity, but does not globally smooth signed-distance cut loci. |
| 18 | Mandelbrot scale | \(\displaystyle \boxed{\sigma(c)=\log_2\frac{d_{\rm ref}}{\rho(c)}}\) | **Canonical.** Logarithmic Mandelbrot scale induced by position. |
| 19 | Finite Shore summit | \(\displaystyle \boxed{\rho_{\min}=\epsilon,\qquad\sigma_{\max}=\log_2\frac{d_{\rm ref}}{\epsilon}}\) | **Canonical.** \(\epsilon\) sets the current smallest local ruler and finite Shore summit. |
| 20 | Exact inverse scale law | \(\displaystyle \boxed{\rho(c)=d_{\rm ref}2^{-\sigma(c)}}\) | **Canonical.** Each \(+1\) in \(\sigma\) halves the local spatial ruler. |
| 21 | Asymptotic geometric-distance law | \(\displaystyle \boxed{d(c)\approx d_{\rm ref}2^{-\sigma(c)}}\) | **Approximation.** Valid away from the regularized crest where \(\rho\approx d\). |
| 22 | Scale differential | \(\displaystyle \boxed{d\sigma=-\frac{1}{\ln2}\frac{d\rho}{\rho}}\) | **Canonical.** |
| 23 | Regularized-distance gradient | \(\displaystyle \boxed{\nabla\rho=\frac{D}{\rho}\nabla D}\) | **Canonical where differentiable.** |
| 24 | Scale gradient | \(\displaystyle \boxed{\nabla\sigma=-\frac{1}{\rho\ln2}\nabla\rho=-\frac{D}{\rho^2\ln2}\nabla D}\) | **Canonical where differentiable.** |
| 25 | Scale Hessian | \(\displaystyle \boxed{H_\sigma=\nabla^2\sigma}\) | **Diagnostic/internal.** Numerically sensitive near cut loci and finite-resolution artifacts. |
| 26 | Ruler Hessian | \(\displaystyle \boxed{H_\rho=\nabla^2\rho}\) | **Canonical internal input** to the compact connection where a valid twice-differentiable or explicitly smoothed field exists. |
| 27 | Independent generalized coordinate | \(\displaystyle \boxed{r=(x,y)}\) | **Canonical.** The physical configuration space is two-dimensional. |
| 28 | Position-scale embedding | \(\displaystyle \boxed{q(c)=\bigl(x,y,\sigma(c)\bigr)}\) | **Canonical.** An embedding, not three independent degrees of freedom. |
| 29 | Embedding Jacobian | \(\displaystyle \boxed{J_q(c)=\frac{\partial q}{\partial(x,y)}=\begin{bmatrix}1&0\\0&1\\\sigma_x&\sigma_y\end{bmatrix}}\) | **Canonical.** |
| 30 | Planar velocity | \(\displaystyle \boxed{v=\dot r=(\dot x,\dot y)}\) | **Canonical.** Persistent Mandelbrot velocity. |
| 31 | Embedded velocity | \(\displaystyle \boxed{\dot q=J_qv}\) | **Canonical.** |
| 32 | Induced scale velocity | \(\displaystyle \boxed{\dot\sigma=\nabla\sigma\cdot v}\) | **Canonical.** There is no independent \(v_\sigma\). |
| 33 | Local-ruler velocity | \(\displaystyle \boxed{\dot\rho=\nabla\rho\cdot v}\) | **Canonical where differentiable.** |

The \(\epsilon\)-regularization makes \(\rho>0\), but it does not make \(D\) globally \(C^2\). At medial-axis/cut-locus points the nearest Shore point is non-unique. Any equation using \(\nabla\rho\), \(H_\rho\), or \(\Gamma\) therefore applies either where those derivatives exist or to the separately defined smoothed/baked derivative authority used by the implementation.

## Scale-relative / hyperbolic metric and kinetic energy

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 34 | Ambient position-scale metric | \(\displaystyle \boxed{H(q)=\operatorname{diag}\!\left(\rho^{-2},\rho^{-2},\lambda^2\right)}\) | **Current destination geometry.** Horizontal distance is measured in units of the local Mandelbrot ruler. |
| 35 | Ambient metric in \(\sigma\) coordinates | \(\displaystyle \boxed{H(q)=\operatorname{diag}\!\left(\frac{2^{2\sigma}}{d_{\rm ref}^2},\frac{2^{2\sigma}}{d_{\rm ref}^2},\lambda^2\right)}\) | **Canonical equivalent.** |
| 36 | Induced configuration metric | \(\displaystyle \boxed{G(c)=J_q(c)^{\mathsf T}H(q)J_q(c)}\) | **Current destination geometry.** |
| 37 | Expanded induced metric | \(\displaystyle \boxed{G(c)=\rho^{-2}I+\lambda^2\nabla\sigma\nabla\sigma^{\mathsf T}}\) | **Current destination geometry.** Replaces the old fixed-planar metric. |
| 38 | Factorized metric | \(\displaystyle \boxed{G(c)=\rho^{-2}\!\left(I+a^2\nabla\rho\nabla\rho^{\mathsf T}\right),\qquad a=\frac{\lambda}{\ln2}}\) | **Canonical equivalent.** Isotropic scale factor plus one rank-one graph correction. |
| 39 | Graph-metric auxiliaries | \(\displaystyle \boxed{h=I+a^2\nabla\rho\nabla\rho^{\mathsf T},\qquad W=1+a^2\lVert\nabla\rho\rVert^2,\qquad G=\rho^{-2}h}\) | **Canonical.** |
| 40 | Exact inverse metric | \(\displaystyle \boxed{G^{-1}=\rho^2\!\left[I-\frac{a^2\nabla\rho\nabla\rho^{\mathsf T}}{W}\right]}\) | **Canonical.** Sherman-Morrison inverse. |
| 41 | Metric determinant | \(\displaystyle \boxed{\det G=\rho^{-4}W}\) | **Canonical.** Useful for orientation, area, and metric quarter-turns. |
| 42 | Riemannian area element | \(\displaystyle \boxed{dA_G=\sqrt{\det G}\,dx\,dy=\rho^{-2}\sqrt W\,dx\,dy}\) | **Canonical.** |
| 43 | Position-scale line element | \(\displaystyle \boxed{ds_M^2=\frac{dx^2+dy^2}{\rho^2}+\lambda^2d\sigma^2}\) | **Current destination geometry.** |
| 44 | Hyperbolic \(\rho\)-form | \(\displaystyle \boxed{ds_M^2=\frac{dx^2+dy^2+a^2d\rho^2}{\rho^2},\qquad a=\frac{\lambda}{\ln2}}\) | **Canonical equivalent.** |
| 45 | Standard upper-half-space coordinate | \(\displaystyle \boxed{z=a\rho}\) | **Canonical coordinate change.** |
| 46 | Ambient hyperbolic metric | \(\displaystyle \boxed{ds_M^2=a^2\frac{dx^2+dy^2+dz^2}{z^2}}\) | **Exact ambient statement.** The ambient \((x,y,z)\) space is upper-half-space \(H^3\) with curvature scale \(a\). |
| 47 | Kinetic energy | \(\displaystyle \boxed{K(c,v)=\frac12v^{\mathsf T}G(c)v}\) | **Canonical.** |
| 48 | Expanded kinetic energy | \(\displaystyle \boxed{K=\frac12\!\left(\frac{\lVert v\rVert^2}{\rho^2}+\lambda^2\dot\sigma^2\right)}\) | **Canonical.** Equal intrinsic motion automatically shrinks in Euclidean \(c\)-space at deep scale. |
| 49 | Ruler-form kinetic energy | \(\displaystyle \boxed{K=\frac{1}{2\rho^2}\!\left(\lVert v\rVert^2+a^2\dot\rho^2\right)}\) | **Canonical equivalent.** |
| 50 | Iso-scale metric eigenvalue | \(\displaystyle \boxed{g_t=\rho^{-2}}\) | **Canonical where \(\nabla\sigma\neq0\).** Cost along an iso-scale tangent. |
| 51 | Scale-normal metric eigenvalue | \(\displaystyle \boxed{g_n=\rho^{-2}+\lambda^2\lVert\nabla\sigma\rVert^2}\) | **Canonical where \(\nabla\sigma\neq0\).** Cost along the maximum-scale-change direction. |
| 52 | Anisotropy ratio | \(\displaystyle \boxed{\frac{g_n}{g_t}=1+\lambda^2\rho^2\lVert\nabla\sigma\rVert^2=1+a^2\lVert\nabla\rho\rVert^2}\) | **Canonical.** |
| 53 | Signed-distance anisotropy | \(\displaystyle \boxed{\frac{g_n}{g_t}=1+\frac{\lambda^2}{(\ln2)^2}\frac{D^2}{D^2+\epsilon^2}\lVert\nabla D\rVert^2}\) | **Canonical where differentiable.** |
| 54 | Ideal-SDF anisotropy bound | \(\displaystyle \boxed{1\lesssim\frac{g_n}{g_t}\lesssim1+\frac{\lambda^2}{(\ln2)^2}}\) | **Approximation.** For \(\lVert\nabla D\rVert\approx1\); with \(\lambda=1\), the upper value is about \(3.08\). |

The ambient space is exactly hyperbolic upper-half-space. The physical Mandelbrot world is the two-dimensional graph \(z=a\rho(x,y)\) embedded in that space. Its induced Gaussian curvature is generally not constant and depends on the first and second derivatives of \(\rho\).

## Local Shore, scale, and metric frames

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 55 | Coordinate-plane quarter-turn | \(\displaystyle \boxed{J_0=\begin{bmatrix}0&-1\\1&0\end{bmatrix}}\) | **Canonical orientation convention.** Positive rotation follows the orientation of \(c=x+iy\). |
| 56 | Geometric Shore normal | \(\displaystyle \boxed{\hat n_D=\frac{\nabla D}{\lVert\nabla D\rVert}}\) | **Diagnostic/canonical geometry** where \(\nabla D\neq0\). |
| 57 | Scale normal | \(\displaystyle \boxed{\hat n_\sigma=\frac{\nabla\sigma}{\lVert\nabla\sigma\rVert}}\) | **Diagnostic** where \(\nabla\sigma\neq0\). It becomes undefined at symmetric extrema such as the regularized summit. |
| 58 | Iso-scale tangent | \(\displaystyle \boxed{\hat t_\sigma=J_0\hat n_\sigma}\) | **Diagnostic.** Tangent to a local iso-\(\sigma\) contour. |
| 59 | Euclidean scale-frame components | \(\displaystyle \boxed{v_t=v\cdot\hat t_\sigma,\qquad v_n=v\cdot\hat n_\sigma}\) | **Diagnostic.** Coordinate components, not yet intrinsic amplitudes. |
| 60 | Metric-unit scale frame | \(\displaystyle \boxed{e_t=\frac{\hat t_\sigma}{\sqrt{g_t}}=\rho\hat t_\sigma,\qquad e_n=\frac{\hat n_\sigma}{\sqrt{g_n}}}\) | **Diagnostic.** \(\langle e_\alpha,e_\beta\rangle_G=\delta_{\alpha\beta}\). |
| 61 | Intrinsic scale-frame components | \(\displaystyle \boxed{\nu_t=\langle v,e_t\rangle_G=\sqrt{g_t}\,v_t,\qquad\nu_n=\langle v,e_n\rangle_G=\sqrt{g_n}\,v_n}\) | **Diagnostic.** Metric-normalized contour and cross-contour motion. |
| 62 | Terrain-frame motion phasor | \(\displaystyle \boxed{m_\sigma=\nu_t+i\nu_n}\) | **Diagnostic/research.** \(\lvert m_\sigma\rvert^2=v^{\mathsf T}Gv=2K\), but its angle inherits rotation/degeneracy of the local scale frame. |
| 63 | Metric quarter-turn | \(\displaystyle \boxed{J_G=\frac1{\sqrt{\det G}}\begin{bmatrix}-G_{12}&-G_{22}\\G_{11}&G_{12}\end{bmatrix}}\) | **Canonical orientation-compatible operator.** \(J_G^2=-I\) and \(J_G^{\mathsf T}GJ_G=G\). |
| 64 | Metric rotation | \(\displaystyle \boxed{R_G(\vartheta)=\cos\vartheta\,I+\sin\vartheta\,J_G}\) | **Canonical local construction.** \(R_G(\vartheta)^{\mathsf T}GR_G(\vartheta)=G\). |

## Connection and manifold dynamics

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 65 | Levi-Civita connection | \(\displaystyle \boxed{\Gamma^i_{jk}=\frac12G^{i\ell}\!\left(\partial_jG_{k\ell}+\partial_kG_{j\ell}-\partial_\ell G_{jk}\right)}\) | **Canonical.** |
| 66 | Compact connection | \(\displaystyle \boxed{\Gamma^i_{jk}=\frac{a^2\rho_i\rho_{jk}}{W}-\frac{\delta^i_j\rho_k+\delta^i_k\rho_j}{\rho}+\frac{h_{jk}\rho_i}{\rho W}}\) | **Canonical for the scale-relative metric** wherever \(\rho\) is twice differentiable. |
| 67 | Free geodesic motion | \(\displaystyle \boxed{\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k=0}\) | **Canonical.** Free motion follows the manifold rather than flat Euclidean integration. |
| 68 | Covariant acceleration | \(\displaystyle \boxed{a_{\rm cov}^i=\frac{Dv^i}{dt}=\dot v^i+\Gamma^i_{jk}v^jv^k}\) | **Canonical invariant acceleration vector.** |
| 69 | Coordinate connection term | \(\displaystyle \boxed{a_\Gamma^i=-\Gamma^i_{jk}v^jv^k}\) | **Diagnostic only.** Coordinate-dependent inertial term, not an independent physical force. |
| 70 | Metric gradient | \(\displaystyle \boxed{\operatorname{grad}_G U=G^{-1}\nabla U}\) | **Canonical.** |
| 71 | Lagrangian | \(\displaystyle \boxed{L(r,v)=K(r,v)-U(r)}\) | **Canonical.** |
| 72 | Forced Euler-Lagrange equation | \(\displaystyle \boxed{\frac{d}{dt}\frac{\partial L}{\partial v}-\frac{\partial L}{\partial r}=Q}\) | **Canonical.** \(Q\) contains only explicit nonconservative Controls and dissipation. |
| 73 | Full dynamics | \(\displaystyle \boxed{\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k=-G^{ij}\partial_jU+G^{ij}Q_j}\) | **Canonical destination equation.** Physics itself is musically ignorant. |
| 74 | Invariant acceleration balance | \(\displaystyle \boxed{a_{\rm cov}=-\operatorname{grad}_G U+G^{-1}Q}\) | **Canonical equivalent.** |

Here \(i,j,k\in\{x,y\}\), \(\rho_i=\partial_i\rho\), and \(\rho_{jk}=\partial_j\partial_k\rho\). The compact connection follows from the graph metric \(h\) and conformal factor \(\rho^{-2}\).

## Conservative potentials and energy

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 75 | Native Shore potential | \(\displaystyle \boxed{U_\sigma(c)=\kappa\sigma(c)}\) | **Canonical candidate/implemented baseline.** Makes the Shore a finite high-potential ridge. |
| 76 | Shore force covector | \(\displaystyle \boxed{Q_\sigma=-\nabla U_\sigma=-\kappa\nabla\sigma}\) | **Canonical.** |
| 77 | Metric-converted Shore acceleration | \(\displaystyle \boxed{-G^{-1}\nabla U_\sigma=\frac{\kappa\rho}{\ln2}\frac{\nabla\rho}{W}}\) | **Canonical.** The inverse metric cancels the apparent deep-scale divergence of \(\nabla\sigma\). |
| 78 | Shore barrier to the crest | \(\displaystyle \boxed{\Delta U_{\rm Shore}(\rho_0\to\epsilon)=\kappa\log_2\frac{\rho_0}{\epsilon}}\) | **Canonical.** \(d_{\rm ref}\) cancels from the energy difference. |
| 78a | Native crossing energy condition | \(\displaystyle \boxed{K_0+W_{\rm control}-D_{\rm friction}\gtrsim\Delta U_{\rm Shore}}\) | **Necessary energetic criterion, not sufficient.** Direction, local geometry, control timing, and numerical validity still determine whether the trajectory actually crests the Shore. |
| 79 | Outer-wall radial coordinate | \(\displaystyle \boxed{s_c=\frac{\lvert c\rvert^2}{4}}\) | **Canonical.** Maps \(\lvert c\rvert<2\) to \(0\le s_c<1\). |
| 80 | Outer-wall phase | \(\displaystyle \boxed{\varphi_c=\frac\pi2s_c^4}\) | **Canonical.** |
| 81 | General secant-wall potential | \(\displaystyle \boxed{U_{\rm wall}=\mu\!\left[\sec\!\left(\frac\pi2s_c^4\right)-1\right]}\) | **Canonical candidate/implemented baseline.** Nearly flat inside and divergent as \(\lvert c\rvert\to2^-\). |
| 82 | General wall force covector | \(\displaystyle \boxed{Q_{\rm wall}=-\mu\pi s_c^3\sec\varphi_c\tan\varphi_c\,(x,y)}\) | **Canonical.** Radially inward. |
| 83 | Chosen wall strength | \(\displaystyle \boxed{\mu=\frac1\pi}\) | **Implemented default.** |
| 84 | Chosen wall potential | \(\displaystyle \boxed{U_{\rm wall}=\frac1\pi\!\left[\sec\!\left(\frac\pi2s_c^4\right)-1\right]}\) | **Implemented default.** |
| 85 | Chosen wall force | \(\displaystyle \boxed{Q_{\rm wall}=-s_c^3\sec\varphi_c\tan\varphi_c\,(x,y)}\) | **Implemented default.** |
| 86 | Total conservative potential | \(\displaystyle \boxed{U(c)=U_\sigma(c)+U_{\rm wall}(c)}\) | **Canonical current landscape.** Shore ridge plus outer-domain bowl. |
| 87 | Total conservative force covector | \(\displaystyle \boxed{Q_U=-\nabla U=Q_\sigma+Q_{\rm wall}}\) | **Canonical.** |
| 88 | Mechanical energy | \(\displaystyle \boxed{E=K+U}\) | **Canonical.** |
| 89 | Expanded mechanical energy | \(\displaystyle \boxed{E=\frac12v^{\mathsf T}Gv+\kappa\sigma+U_{\rm wall}}\) | **Canonical.** |
| 90 | Valid-domain invariant | \(\displaystyle \boxed{\lvert c\rvert<2}\) | **Canonical.** The secant potential is physical; hard rejection remains a numerical safety guard. |

Because \(\rho_{\min}=\epsilon\), the regularizer is also a physical model parameter: it fixes the minimum ruler, maximum \(\sigma\), and finite Shore barrier. It should be versioned with Physics rather than treated as a numerically irrelevant epsilon.

## Generalized Controls, friction, impulses, and steering

The symbol \(u\) is reserved below for intrinsic speed. Metric-unit control directions use \(\hat d\).

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 91 | Metric-unit drive direction | \(\displaystyle \boxed{\hat d^{\mathsf T}G\hat d=1}\) | **Canonical.** A unit control direction has the same intrinsic magnitude at every Mandelbrot scale. |
| 92 | Steering update | \(\displaystyle \boxed{\hat d^+=R_G(\Delta\vartheta)\hat d^-}\) | **Candidate/clean interpretation.** Left/right steering rotates intent in the local metric tangent plane. |
| 93 | Drive-force covector | \(\displaystyle \boxed{Q_{\rm drive}=F_{\rm drive}\,G\hat d}\) | **Canonical.** \(G^{-1}Q_{\rm drive}=F_{\rm drive}\hat d\). |
| 94 | Continuous control power | \(\displaystyle \boxed{P_{\rm control}=v^{\mathsf T}Q_{\rm control}}\) | **Canonical.** Signed rate at which continuous Controls change mechanical energy. |
| 95 | Metric-consistent drag | \(\displaystyle \boxed{Q_{\rm drag}=-\beta Gv}\) | **Canonical baseline.** |
| 96 | General friction | \(\displaystyle \boxed{Q_{\rm friction}=-Bv,\qquad B\succeq0}\) | **Canonical.** Grip/drift may alter magnitude and anisotropy without allowing friction to inject energy. |
| 97 | Friction power | \(\displaystyle \boxed{P_{\rm friction}=v^{\mathsf T}Q_{\rm friction}=-v^{\mathsf T}Bv\le0}\) | **Canonical.** |
| 98 | Simple drag power | \(\displaystyle \boxed{P_{\rm drag}=-\beta v^{\mathsf T}Gv=-2\beta K\le0}\) | **Canonical.** |
| 99 | Generalized momentum | \(\displaystyle \boxed{p=\frac{\partial L}{\partial v}=Gv}\) | **Canonical.** Momentum is a covector. |
| 100 | Generalized impulse | \(\displaystyle \boxed{p^+=p^-+J}\) | **Canonical.** Applied at fixed \(c\). |
| 101 | Dual-metric impulse magnitude | \(\displaystyle \boxed{\lVert J\rVert_{G^{-1}}^2=J^{\mathsf T}G^{-1}J}\) | **Canonical.** Natural scale-independent impulse bound. |
| 102 | Impulse velocity jump | \(\displaystyle \boxed{v^+=v^-+G^{-1}J}\) | **Canonical.** |
| 103 | Continuous energy ledger | \(\displaystyle \boxed{\frac{dE}{dt}=P_{\rm control}+P_{\rm friction}}\) | **Canonical between impulses**, up to integration error. |
| 104 | Impulse energy jump | \(\displaystyle \boxed{\Delta E_J=\frac12(p^-+J)^{\mathsf T}G^{-1}(p^-+J)-\frac12(p^-)^{\mathsf T}G^{-1}p^-}\) | **Canonical.** Instantaneous kinetic-energy change at fixed \(c\). |

## Metric-native special moves

These remain candidate game/debug actions. Every move must reduce to ordinary metric/Map primitives and preserve the distinction between Mandelbrot scale and Julia zoom.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 105 | Kick / jump | \(\displaystyle \boxed{p^+=p^-+J_{\rm kick},\qquad J_{\rm kick}^{\mathsf T}G^{-1}J_{\rm kick}\le J_{\max}^2}\) | **Candidate.** Bounded intrinsic impulse; no discontinuous change in \(c\) or \(\sigma\). |
| 106 | Intrinsic Grow footprint | \(\displaystyle \boxed{f_R(c)=\frac{\int_M K\!\left(d_G(c,c')/R\right)f(c')\,dA_G(c')}{\int_M K\!\left(d_G(c,c')/R\right)\,dA_G(c')}}\) | **Research candidate.** Temporarily increases the spatial footprint used to sense/interact with selected Map fields; it does not directly alter \(\sigma(c)\). Exact affected fields remain unresolved. |
| 107 | Iso-scale tangent projector | \(\displaystyle \boxed{P_t=e_t e_t^\flat=e_t e_t^{\mathsf T}G}\) | **Canonical local operator** where the iso-scale frame is valid. |
| 108 | Carve-lock direction | \(\displaystyle \boxed{\hat d_{\rm lock}=\operatorname{normalize}_G\!\left((1-\alpha)\hat d+\alpha P_t\hat d\right),\qquad0\le\alpha<1}\) | **Candidate.** Soft bias toward iso-scale carving, not a hard path constraint. |
| 109 | Board-frame drift tensor | \(\displaystyle \boxed{B_{\rm drift}=\beta_\parallel e_b^\flat\!\otimes e_b^\flat+\beta_\perp n_b^\flat\!\otimes n_b^\flat,\qquad\beta_\parallel,\beta_\perp\ge0}\) | **Candidate.** Drift changes longitudinal/lateral grip while preserving \(B\succeq0\). |
| 110 | Energy-preserving redirection | \(\displaystyle \boxed{v'=R_G(\vartheta)v,\qquad K(c,v')=K(c,v)}\) | **Candidate primitive.** Useful for a mechanically explicit slingshot release. |
| 111 | Slingshot release | \(\displaystyle \boxed{v^+=R_G(\vartheta)v^-+G^{-1}J_{\rm release}}\) | **Research candidate.** Rotation redirects existing energy; only the bounded impulse contributes unaccounted kinetic change, which must appear in \(\Delta E_J\). |

## Intrinsic visual gesture geometry

This is the canonical visual-side compression to derive and expose first as diagnostic state. It does not automatically become `PlayerObservation`.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 112 | Intrinsic speed | \(\displaystyle \boxed{u=\lVert v\rVert_G=\sqrt{v^{\mathsf T}Gv}=\sqrt{2K}}\) | **Mathematically derived / diagnostic authority.** Motion amplitude in intrinsic units. |
| 113 | Unit trajectory tangent | \(\displaystyle \boxed{T=\frac vu}\) | **Canonical where \(u>0\).** |
| 114 | Unit trajectory normal | \(\displaystyle \boxed{N=J_G T}\) | **Canonical where \(u>0\).** Positive metric-normal quarter-turn. |
| 115 | Covariant Frenet decomposition | \(\displaystyle \boxed{a_{\rm cov}=\dot u\,T+u\Omega\,N=\dot u\,T+u^2\kappa_gN}\) | **Canonical derived identity** during smooth motion. |
| 116 | Longitudinal gesture | \(\displaystyle \boxed{a_\parallel=\langle a_{\rm cov},T\rangle_G=\dot u}\) | **Diagnostic authority.** Intrinsic surge/braking. |
| 117 | Lateral gesture | \(\displaystyle \boxed{a_\perp=\langle a_{\rm cov},N\rangle_G=u\Omega=u^2\kappa_g}\) | **Diagnostic authority.** Intrinsic carve/lateral acceleration. |
| 118 | Intrinsic turn rate | \(\displaystyle \boxed{\Omega=\left\langle\frac{DT}{dt},N\right\rangle_G=\frac{a_\perp}{u}}\) | **Diagnostic authority** where \(u\) exceeds the validity floor. A free geodesic has \(\Omega\approx0\). |
| 119 | Geodesic curvature | \(\displaystyle \boxed{\kappa_g=\left\langle\frac{DT}{ds},N\right\rangle_G=\frac{\Omega}{u}=\frac{a_\perp}{u^2}}\) | **Diagnostic authority** where valid. |
| 120 | Parallel-transported frame | \(\displaystyle \boxed{\frac{DE_1}{dt}=0,\qquad\frac{DE_2}{dt}=0,\qquad E_2=J_G E_1}\) | **Canonical construction along a trajectory.** Initial orientation is arbitrary. |
| 121 | Transported heading phase | \(\displaystyle \boxed{T=\cos\Theta\,E_1+\sin\Theta\,E_2,\qquad\dot\Theta=\Omega}\) | **Diagnostic authority.** \(\Theta\) is defined up to an epoch constant. |
| 122 | Parallel-transported motion phasor | \(\displaystyle \boxed{m_\parallel=\langle v,E_1\rangle_G+i\langle v,E_2\rangle_G=u e^{i\Theta}}\) | **Research/diagnostic.** Preferred visual complex phase carrier. |
| 123 | Visual logarithmic derivative | \(\displaystyle \boxed{\frac{\dot m_\parallel}{m_\parallel}=\frac{\dot u}{u}+i\Omega=\frac{a_\parallel}{u}+i\Omega}\) | **Derived identity** during smooth, nonzero-speed motion. |
| 124 | Mandelbrot inverse log-scale | \(\displaystyle \boxed{\eta_M=-\ln\frac{\rho}{\rho_{\rm ref}}=(\ln2)(\sigma-\sigma_{\rm ref})}\) | **Canonical derived coordinate.** Reference shift is additive only. |
| 125 | Mandelbrot scale rate | \(\displaystyle \boxed{\dot\eta_M=-\frac{\dot\rho}{\rho}=(\ln2)\dot\sigma}\) | **Canonical derived gesture.** Rate of travel through fractal scale. |
| 126 | Fractional speed-change rate | \(\displaystyle \boxed{\chi_u=\frac{d}{dt}\ln u=\frac{a_\parallel}{u}}\) | **Diagnostic** only above the speed floor. |
| 127 | Phase-validity gate | \(\displaystyle \boxed{q_V=C_{\rm geom}\frac{u}{u+u_0}}\) | **Candidate diagnostic confidence.** Phase/log-rate matter only when motion amplitude and geometry validity support them. |
| 128 | Impulse phase jump | \(\displaystyle \boxed{\Delta\Theta_J=\operatorname{atan2}\!\left(\langle J_G v^-,v^+\rangle_G,\langle v^-,v^+\rangle_G\right)}\) | **Diagnostic candidate.** Records a discrete heading jump rather than smearing it into continuous \(\Omega\). |
| 129 | Force-source decomposition | \(\displaystyle \boxed{a_{\rm cov}=a_U+a_{\rm control}+a_{\rm friction},\qquad a_U=-\operatorname{grad}_G U}\) | **Diagnostic.** Between impulses; the Christoffel coordinate term is not listed as a physical force. |

At \(u\approx0\), heading, \(\Theta\), \(\Omega/u\), \(\kappa_g\), and \(\dot{\ln u}\) become undefined or ill-conditioned. The correct response is explicit validity/confidence and deterministic gesture epochs, not fabricated zero phase or silent clipping.

## Intrinsic curvature and holonomy extensions

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 130 | Intrinsic Gaussian curvature | \(\displaystyle \boxed{K_M=\frac{R_{1212}}{\det G}}\) | **Research/diagnostic.** Curvature of the physical two-dimensional Mandelbrot manifold. |
| 131 | Gauss equation in ambient \(H^3\) | \(\displaystyle \boxed{K_M=-\frac1{a^2}+k_1k_2}\) | **Exact geometric relation** where principal extrinsic curvatures \(k_1,k_2\) are defined. |
| 132 | Geodesic deviation | \(\displaystyle \boxed{\frac{D^2\xi}{ds^2}+K_M\xi=0}\) | **Research interpretation.** Scalar transverse form in two dimensions; negative curvature tends to separate nearby geodesics. |
| 133 | Geometric holonomy | \(\displaystyle \boxed{\Delta\Theta_{\rm geo}=\oint_\gamma\omega\equiv\iint_\Sigma K_M\,dA_G\pmod{2\pi}}\) | **Research extension.** Sign depends on orientation/connection convention. A closed loop can return to position with changed transported orientation. |

Curvature and holonomy are promising long-form gesture diagnostics, not yet foundations of the musical reward.

## Continuous multi-resolution Map infrastructure

The MIP pyramid remains useful numerical infrastructure. It is no longer the primary semantic definition of Mandelbrot scale and should not force the Player to rediscover \(\rho\), \(\sigma\), or \(G\) from raw resolution stacks.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 134 | Continuous MIP coordinate | \(\displaystyle \boxed{\ell=\ell(\sigma),\qquad k=\lfloor\ell\rfloor,\qquad\alpha=\ell-k}\) | **Map-provider infrastructure.** Maps continuous scale to stored resolutions. |
| 135 | Adjacent-MIP interpolation | \(\displaystyle \boxed{f(c,\sigma)=(1-\alpha)f_k(c)+\alpha f_{k+1}(c)}\) | **Map-provider infrastructure.** Representative interpolation for any stored field \(f\). |
| 136 | Cross-scale derivative | \(\displaystyle \boxed{\partial_\sigma f\approx\frac{f_{k+1}-f_k}{\Delta\sigma}}\) | **Candidate Map diagnostic.** Not an independent scale authority. |
| 137 | Player-facing raw MIP evidence | \(\displaystyle \boxed{A_{\rm MIP}\subset O_t\quad\text{only if an ablation shows independent value}}\) | **Current design rule.** Prefer compact intrinsic geometry first. |

## Audio, CycleBank, and complex time-scale geometry

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 138 | CycleBank mode | \(\displaystyle \boxed{z_i^A=A_i e^{i\phi_i},\qquad[\cos\phi_i,\sin\phi_i,\log_2f_i,A_i,C_i]}\) | **Canonical conceptual audio mode.** Phase, continuous frequency, strength, and confidence. |
| 139 | Rational audio relation | \(\displaystyle \boxed{\psi_{ij}^{(n:m)}=\operatorname{wrap}(n\phi_i-m\phi_j)}\) | **Canonical diagnostic.** Generic harmonic/polymetric phase relation. |
| 140 | Latent temporal undertone | \(\displaystyle \boxed{f_i\approx n_i f_0}\) | **Deferred research.** An inferred slow organizer, not required for v1. |
| 141 | Complex Morlet transform | \(\displaystyle \boxed{W_x(s,\tau)=\frac1{\sqrt s}\int x(t)\,\psi^*\!\left(\frac{t-\tau}{s}\right)dt}\) | **Research/common analysis notation.** Runtime may use a causal one-sided wavelet/filter-bank equivalent rather than a centered textbook CWT. |
| 142 | Morlet atom | \(\displaystyle \boxed{\psi(u)\approx C_\psi e^{i\omega_0u}e^{-u^2/2}}\) | **Schematic.** Exact admissibility correction and causal realization are implementation choices. |
| 143 | Complex coefficient decomposition | \(\displaystyle \boxed{W_x(s,t)=A_x(s,t)e^{i\phi_x(s,t)}}\) | **Canonical representation.** Amplitude says whether a mode exists; phase says where it is in its cycle. |
| 144 | Instantaneous angular frequency | \(\displaystyle \boxed{\omega_x(s,t)=\operatorname{Im}\!\left(\frac{\partial_tW_x}{W_x}\right)}\) | **Canonical analytic-field diagnostic** where amplitude is sufficient. |
| 145 | Audio logarithmic derivative | \(\displaystyle \boxed{\frac{\dot z_A}{z_A}=\frac{\dot A}{A}+i\dot\phi}\) | **Derived identity** for a nonzero complex audio mode. |
| 146 | Audio inverse log-scale | \(\displaystyle \boxed{\eta_A=-\ln\frac{s}{s_{\rm ref}}}\) | **Research organizing coordinate.** Fine temporal scales have larger \(\eta_A\). |
| 147 | Affine time-scale metric | \(\displaystyle \boxed{ds_A^2=\frac{d\tau^2+\alpha_A^2ds^2}{s^2}}\) | **Research/common geometry.** Representative scale-relative metric on the wavelet time-scale half-plane. |
| 148 | Log-scale time-scale metric | \(\displaystyle \boxed{ds_A^2=\frac{e^{2\eta_A}}{s_{\rm ref}^2}d\tau^2+\alpha_A^2d\eta_A^2}\) | **Equivalent form.** |
| 149 | Morlet-ridge phase/displacement relation | \(\displaystyle \boxed{d\phi\approx\omega_0\frac{d\tau}{s}}\) | **Approximation along a coherent ridge.** Phase counts time displacement in units of the current temporal ruler. |
| 150 | Audio phase-validity gate | \(\displaystyle \boxed{q_A=C_A\frac{A}{A+A_0}}\) | **Candidate confidence.** Phase is ignored when coefficient amplitude/confidence is insufficient. |
| 151 | AudioActivation | \(\displaystyle \boxed{A_{\rm audio}(t)=\operatorname{concat}(A_{\rm cycles},A_{\rm intensity},A_{\rm spectral},A_{\rm texture})}\) | **Destination contract.** Compact causal music perception; exact channels remain subject to ablation/versioning. |

The important shared principle is not that audio time-scale space and Mandelbrot position-scale space are globally identical. It is that both normalize coordinate displacement by a local ruler before comparing dynamics.

## Common intrinsic gesture state

For nonzero, valid modes, the current compression is a three-coordinate gesture state plus its tangent vector.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 152 | Audio gesture coordinate | \(\displaystyle \boxed{\xi_A=(\ln A,\phi,\eta_A)}\) | **Research organizing representation.** |
| 153 | Audio gesture tangent | \(\displaystyle \boxed{\dot\xi_A=\left(\frac{\dot A}{A},\dot\phi,\dot\eta_A\right)}\) | **Research organizing representation.** |
| 154 | Visual gesture coordinate | \(\displaystyle \boxed{\xi_V=(\ln u,\Theta,\eta_M)}\) | **Research organizing representation.** |
| 155 | Visual gesture tangent | \(\displaystyle \boxed{\dot\xi_V=\left(\frac{a_\parallel}{u},\Omega,\dot\eta_M\right)}\) | **Research organizing representation.** |
| 156 | Gesture-state manifolds | \(\displaystyle \boxed{\mathcal G_A\cong\mathbb R_{\ln A}\times S^1_\phi\times\mathbb R_{\eta_A},\qquad\mathcal G_V\cong\mathbb R_{\ln u}\times S^1_\Theta\times\mathbb R_{\eta_M}}\) | **Research hypothesis.** Both modalities use amplitude, phase, and log scale as natural coordinates. |
| 157 | Six-dimensional phase-space state | \(\displaystyle \boxed{X_A=(\xi_A,\dot\xi_A)\in T\mathcal G_A,\qquad X_V=(\xi_V,\dot\xi_V)\in T\mathcal G_V}\) | **Research compression.** “Six features” are really a 3D gesture state plus its tangent vector. |
| 158 | Coordinate correspondence | \(\displaystyle \boxed{(\ln A,\phi,\eta_A;\dot{\ln A},\dot\phi,\dot\eta_A)\longleftrightarrow(\ln u,\Theta,\eta_M;\dot{\ln u},\Omega,\dot\eta_M)}\) | **Research common currency.** A correspondence to learn, not an asserted equality. |

This representation removes absolute tempo, Euclidean Mandelbrot depth, raw screen heading, and many arbitrary feature choices from the core coupling problem. Validity/confidence must travel with every phase-bearing coordinate.

## Visual complex modes for audiovisual comparison

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 159 | Direct motion mode | \(\displaystyle \boxed{b_0(t)=m_\parallel(t)=u(t)e^{i\Theta(t)}}\) | **Research candidate.** Native geometric visual phasor. |
| 160 | Scalar intrinsic gesture bank | \(\displaystyle \boxed{g_j(t)\in\{u,\Omega,a_\parallel,a_\perp,\eta_M,\dot\eta_M,\ldots\}}\) | **Research candidate.** \(\kappa_g\) is allowed only with low-speed validity gating. |
| 161 | Wavelet-derived visual mode | \(\displaystyle \boxed{b_j(r,t)=W_{g_j}(r,t)=B_j(r,t)e^{i\psi_j(r,t)}}\) | **Research candidate.** Applies the same complex multiscale analysis to intrinsic visual gestures. |
| 162 | Direct-vs-wavelet ablation | \(\displaystyle \boxed{\{m_\parallel\}\quad\text{vs}\quad\{W_{g_j}\}\quad\text{vs}\quad\{m_\parallel,W_{g_j}\}}\) | **Required research comparison.** Do not assume one representation is canonical before held-out evidence. |

## Phase-harmonic audiovisual coupling

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 163 | Phase-harmonic operator | \(\displaystyle \boxed{[z]^k=\begin{cases}\lvert z\rvert e^{ik\arg z},&z\neq0\\0,&z=0\end{cases}}\) | **Research/DSP primitive.** Multiplies phase while preserving amplitude. |
| 164 | Cross-modal relative phase | \(\displaystyle \boxed{\chi_{ij}^{(n:m)}=n\phi_i-m\psi_j}\) | **Research.** Supports rational timing relations. |
| 165 | Windowed phase-harmonic cross-moment | \(\displaystyle \boxed{C_{ij}^{n,m}(s,r,\tau;t)=\mathcal S_t\!\left([a_i(s,t)]^n\,\overline{[b_j(r,t-\tau)]^m}\right)}\) | **Research.** \(\mathcal S_t\) is a documented causal/windowed smoother; \(\tau\) is signed lead/lag. |
| 166 | Normalized phase-harmonic coherence | \(\displaystyle \boxed{\gamma_{ij}^{n,m}=\frac{C_{ij}^{n,m}}{\sqrt{\mathcal S_t(\lvert a_i\rvert^2)\,\mathcal S_t(\lvert b_j\rvert^2)+\varepsilon_{\rm num}}}}\) | **Research candidate.** Must carry amplitude/confidence masks and fold-safe normalization. |
| 167 | Weighted phase order parameter | \(\displaystyle \boxed{Z_{ij}^{n:m}=\frac{\sum_{\tau'\in T_t}q_{ij}(\tau')e^{i\chi_{ij}^{(n:m)}(\tau')}}{\sum_{\tau'\in T_t}q_{ij}(\tau')+\varepsilon_{\rm num}}}\) | **Research candidate.** Measures persistence of any relative phase. |
| 168 | Preferred signed phase fit | \(\displaystyle \boxed{R_{ij}^{n:m}=\operatorname{Re}\!\left(e^{-i\delta_{ij}^{n:m}}Z_{ij}^{n:m}\right)=\left\langle\cos(\chi_{ij}^{(n:m)}-\delta_{ij}^{n:m})\right\rangle_q}\) | **Research candidate.** Distinguishes a contextually preferred lead/lag from mere concentration. |
| 169 | Multichannel spectral matrices | \(\displaystyle \boxed{S_{AA}=\mathbb E[aa^\dagger],\qquad S_{VV}=\mathbb E[bb^\dagger],\qquad S_{VA}=\mathbb E[ba^\dagger]}\) | **Research.** Complex vectors may contain selected phase-harmonic modes. |
| 170 | Whitened cross-modal operator | \(\displaystyle \boxed{C_{VA}=S_{VV}^{-1/2}S_{VA}S_{AA}^{-1/2}}\) | **Research candidate.** Requires regularization/shrinkage in finite data. |
| 171 | Canonical coherence modes | \(\displaystyle \boxed{C_{VA}=U\Sigma V^\dagger}\) | **Research.** Stable dominant singular modes would support a low-dimensional choreography space. |
| 172 | Audio relation matrix | \(\displaystyle \boxed{H_A=\mathbb E[\Phi_A\Phi_A^\dagger]}\) | **Stronger research hypothesis.** \(\Phi_A\) is the retained phase-harmonic audio feature vector. |
| 173 | Visual relation matrix | \(\displaystyle \boxed{H_V=\mathbb E[\Phi_V\Phi_V^\dagger]}\) | **Stronger research hypothesis.** |
| 174 | Relational-organization hypothesis | \(\displaystyle \boxed{H_V(t)\approx\mathcal T_{z_{\rm slow}(t)}\!\bigl(H_A(t)\bigr)}\) | **Research.** The visual system may preserve the evolving organization of musical relations rather than copying individual coordinates. |

## Context-conditioned entrainment reward

Maximal coherence everywhere is not the objective. Music forms, weakens, slips, reorganizes, and restores relationships. The target is context-appropriate evolution of coupling.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 175 | Slow causal context | \(\displaystyle \boxed{z_{\rm slow}(t)=f_{\rm context}(O_{\le t})}\) | **Research.** A slowly evolving causal state, not a semantic mood label supplied at runtime. |
| 176 | Context-conditioned coupling | \(\displaystyle \boxed{w_\ell=w_\ell(z_{\rm slow}),\qquad\delta_\ell=\delta_\ell(z_{\rm slow}),\qquad\pi_k=\pi_k(z_{\rm slow})}\) | **Research.** Strengths, preferred offsets, and possibly a small mixture of coupling regimes. |
| 177 | Fast entrainment score | \(\displaystyle \boxed{R_{\rm entrain}(t)=\sum_\ell w_\ell(z_{\rm slow})\operatorname{Re}\!\left(e^{-i\delta_\ell(z_{\rm slow})}Z_\ell(t)\right)}\) | **Research candidate.** \(\ell\) indexes only pairs/scales/lags/ratios supported by the falsification probe. |
| 178 | Audio-support gate | \(\displaystyle \boxed{g_A(s,t)=\frac{P_A(s,t)}{P_A(s,t)+P_0}}\) | **Candidate safeguard.** Bounded evidence that audio oscillation exists at a scale. |
| 179 | Unsupported visual activity | \(\displaystyle \boxed{L_{\rm unsupported}=\sum_s P_V(s,t)\bigl(1-g_A(s,t)\bigr)}\) | **Candidate safeguard.** Penalizes visual oscillation where corresponding audio support is absent. |
| 180 | Contrastive calibration | \(\displaystyle \boxed{R_{\rm contrast}=S(a_t,b_t)-\mathbb E_{\tilde b\sim\mathcal N_t}S(a_t,\tilde b)}\) | **Research/training candidate.** Subtracts plausible mismatch/null score. Live use requires explicit causal negative semantics. |
| 181 | Coupling budget | \(\displaystyle \boxed{\sum_\ell\lvert w_\ell\rvert\le W_{\max}}\) | **Candidate safeguard.** Prevents all-to-all activation; low-rank or sparse alternatives may replace it. |
| 182 | Total reward ledger | \(\displaystyle \boxed{R_{\rm total}=R_{\rm competence}+\lambda_E R_{\rm entrain}-\lambda_U L_{\rm unsupported}-\lambda_C L_{\rm chatter}-\lambda_A L_{\rm action}+R_{\rm retained\ visual}}\) | **Research schematic.** Every term remains separately logged. No generic penalty should suppress legitimate groove/repetition by definition. |

A reward is not accepted because it can be optimized. It must preserve matched-vs-shifted ranking, resist constant circles/jitter/impulse spam, generalize to held-out songs and starts, and improve blinded human judgments.

## Player observation, policy, and action space

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 183 | Compact local geometry activation | \(\displaystyle \boxed{A_{\rm geom}=\operatorname{pack}(D,\rho,\sigma,\text{first-order frame/metric summaries},\text{validity})}\) | **Destination direction; exact contract unresolved.** Prefer canonical intrinsic geometry over a large raw MIP stack. |
| 184 | Physics activation | \(\displaystyle \boxed{A_{\rm phys}=\operatorname{pack}(p,K,U,E,\text{grip/friction},\text{other justified state})}\) | **Destination direction; exact contract unresolved.** Intrinsic gesture diagnostics enter only after explicit ablation. |
| 184a | Intrinsic gesture activation candidate | \(\displaystyle \boxed{A_{\rm gesture}=\operatorname{pack}(u,\cos\Theta,\sin\Theta,\Omega,a_\parallel,a_\perp,\eta_M,\dot\eta_M,q_V)}\) | **Diagnostic/ablation candidate.** Compact and geometrically natural, but not automatically part of `PlayerObservation`. |
| 185 | PlayerObservation | \(\displaystyle \boxed{O_t=\operatorname{concat}(A_{\rm audio},A_{\rm geom},A_{\rm phys},A_{\rm JuliaView})}\) | **Destination contract.** Entirely causal. |
| 186 | Causal PlayerPolicy | \(\displaystyle \boxed{a_t=\pi_\theta(O_{\le t},h_{t-1})}\) | **Destination.** Small persistent temporal policy; exact architecture remains an empirical choice. |
| 187 | Controls v2 | \(\displaystyle \boxed{a_t=(\Delta\vartheta,\text{throttle},\text{brake},\text{grip/drift},J,\Delta Z_J,\Delta\theta_J,\Delta\mathcal P,\ldots)}\) | **Destination.** Simple motion actions plus independent Julia presentation actions. No direct Mandelbrot \(\sigma\) control. |
| 188 | Optional special-action extension | \(\displaystyle \boxed{a_t^{\rm special}\in\{\text{kick},\text{grow},\text{carve lock},\text{drift},\text{slingshot}\}}\) | **Candidate/debug game extension.** Each action must reduce to the metric-native primitives above. |

The model does not need to relearn the meaning of Mandelbrot scale from raw MIPs if \(\rho\), \(\sigma\), \(G\), and compact intrinsic state are already available. Raw Map windows remain possible ablations for residual local structure that compact geometry cannot capture.

## Julia presentation and ColorIntent

Mandelbrot scale \(\sigma(c)\) and Julia zoom \(Z_J\) are separate domains. The Player directly controls Julia zoom, rotation, and palette evolution through persistent presentation state.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 189 | Julia visual | \(\displaystyle \boxed{J_t=J(c_t;Z_{J,t},\theta_{J,t},\mathcal P_t)}\) | **Canonical.** Mandelbrot parameter plus independent Julia presentation state. |
| 190 | Julia zoom update | \(\displaystyle \boxed{Z_{J,t+\Delta t}=\Phi_Z(Z_{J,t},\Delta Z_{J,t})}\) | **Canonical schematic.** Player-controlled, bounded state transition; exact integration/range is Rust-owned. |
| 191 | Julia rotation update | \(\displaystyle \boxed{\theta_{J,t+\Delta t}=\operatorname{wrap}\!\bigl(\theta_{J,t}+\Delta\theta_{J,t}\bigr)}\) | **Canonical schematic.** |
| 192 | OKLCH ColorIntent state | \(\displaystyle \boxed{\mathcal P=(L,C,h,\mathcal H,w_a)}\) | **Destination.** Lightness, chroma, circular anchor hue, small harmony mode, and accent weight. |
| 193 | Circular hue observation | \(\displaystyle \boxed{h\mapsto(\cos h,\sin h)}\) | **Destination representation.** Avoids the \(2\pi\to0\) discontinuity. |
| 194 | Harmony offset | \(\displaystyle \boxed{\Delta h_{\mathcal H}\in\left\{0,\frac\pi6,\pi\right\}}\) | **Current restrained candidate.** Monochrome, analogous, or opponent. |
| 195 | Second semantic color | \(\displaystyle \boxed{h_2=\operatorname{wrap}(h+\Delta h_{\mathcal H})}\) | **Destination.** At most two intentional semantic colors; renderer interpolation does not create extra independent intent colors. |
| 196 | Palette-state evolution | \(\displaystyle \boxed{\mathcal P_{t+\Delta t}=\Phi_{\mathcal P}(\mathcal P_t,\Delta\mathcal P_t)}\) | **Canonical schematic.** Persistent bounded change with rate limits/hysteresis. |
| 197 | Circular hue loss | \(\displaystyle \boxed{d_{S^1}(h_1,h_2)=1-\cos(h_1-h_2)}\) | **Candidate training primitive.** Appropriate when a soft hue target exists. |
| 198 | ColorIntent weak target | \(\displaystyle \boxed{\mathcal P^T(t)=\mathcal R_{\rm human}\!\left(V^T(t),A^T(t),p_F^T(t),p_B^T(t),\text{causal descriptors}\right)}\) | **Training-only research.** Human-authored editable rules transform offline affect/structure annotations into soft ColorIntent preferences. No runtime rule controller. |
| 199 | Confidence-weighted color loss | \(\displaystyle \boxed{L_{\rm color}=w_{\mathcal P}(t)\,d_{\mathcal P}\!\left(\mathcal P_t,\mathcal P^T(t)\right)}\) | **Training-only candidate.** \(d_{\mathcal P}\) combines circular hue, continuous OKLCH, and harmony terms. |

The math constrains a small, stable ColorIntent state. Human-authored offline weak supervision may steer taste, but the live Player remains the only musical color policy.

## Offline structural teachers and SongFormer distillation

SongFormer is an offline, potentially noncausal teacher. Its dense outputs belong only on the target/evaluation side.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 200 | Teacher boundary probability | \(\displaystyle \boxed{p_B^T(t)=\operatorname{sigmoid}(z_B^T(t))}\) | **Training-only.** Preserve the dense timeline before peak-picking. |
| 201 | Teacher section distribution | \(\displaystyle \boxed{p_F^T(t;\tau_{\rm KD})=\operatorname{softmax}\!\left(\frac{z_F^T(t)}{\tau_{\rm KD}}\right)}\) | **Training-only.** Soft structural-role knowledge; \(\tau_{\rm KD}\) is the distillation temperature. |
| 202 | Causal shared state | \(\displaystyle \boxed{h_t=f_\theta(O_{\le t})}\) | **Runtime-capable Student state.** Contains no SongFormer output or future audio. |
| 203 | Boundary-now Student | \(\displaystyle \boxed{p_B^S(t)=\operatorname{sigmoid}(g_B(h_t))}\) | **Training-only auxiliary head.** May be removed at export. |
| 204 | Boundary distillation loss | \(\displaystyle \boxed{L_B=\operatorname{BCE}\!\left(p_B^S(t),p_B^T(t)\right)}\) | **Training-only candidate.** Soft-label boundary-now supervision. |
| 205 | Horizon-readiness target | \(\displaystyle \boxed{y_H^T(t)=\mathcal K_H\!\left[p_B^T\right](t)}\) | **Training-only research.** \(\mathcal K_H\) is a documented max, integrated-mass, hazard, or smooth time-to-boundary operator over \((t,t+H]\); it is not yet frozen. |
| 206 | Readiness Student | \(\displaystyle \boxed{y_H^S(t)=g_H(h_t)}\) | **Training-only auxiliary head.** Student predicts future structural evidence from causal history only. |
| 207 | Readiness loss | \(\displaystyle \boxed{L_R=\sum_{H\in\mathcal H}\operatorname{BCE}\!\left(y_H^S(t),y_H^T(t)\right)}\) | **Training-only primary bet.** Initial candidate horizons \(\mathcal H=\{0.5,1,2,4\}\,\mathrm{s}\). |
| 208 | Section Student | \(\displaystyle \boxed{p_F^S(t;\tau_{\rm KD})=\operatorname{softmax}\!\left(\frac{g_F(h_t)}{\tau_{\rm KD}}\right)}\) | **Training-only auxiliary head.** Named sections do not become runtime ontology. |
| 209 | Teacher-entropy confidence | \(\displaystyle \boxed{w_F(t)=1-\frac{H(p_F^T(t))}{\log K_F}}\) | **Candidate.** Downweights ambiguous section labels; other calibrated confidence rules may replace it. |
| 210 | Section distillation loss | \(\displaystyle \boxed{L_F=w_F(t)\tau_{\rm KD}^2D_{\rm KL}\!\left(p_F^T(t;\tau_{\rm KD})\,\Vert\,p_F^S(t;\tau_{\rm KD})\right)}\) | **Training-only candidate.** Prefer distribution distillation over uncalibrated raw-logit MSE. |

The acceptance question is not whether the Student predicts SongFormer. It is whether the shared causal trunk produces better-timed, less frantic, more musically coherent motion and Julia presentation on held-out songs.

## Offline valence/arousal teachers

Keep causal acoustic energy distinct from perceived arousal.

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 211 | Causal acoustic energy | \(\displaystyle \boxed{\mathcal E_{\rm aud}(t)=\operatorname{RMS/envelope}(\text{audio}_{\le t})}\) | **Runtime causal descriptor.** Not definitionally equal to arousal. |
| 212 | Teacher affect outputs | \(\displaystyle \boxed{(V_k^T(t),A_k^T(t))=\mathcal T_k(\text{offline audio window})}\) | **Training-only.** \(k\) indexes Music2Emo, Essentia VA heads, or another screened teacher. |
| 213 | Per-teacher calibration | \(\displaystyle \boxed{\tilde V_k=f_{V,k}(V_k^T),\qquad\tilde A_k=f_{A,k}(A_k^T)}\) | **Training-only candidate.** Maps native teacher scales into one documented semantic range. |
| 214 | Robust ensemble affect | \(\displaystyle \boxed{\hat V^T=\operatorname{median}_k\tilde V_k,\qquad\hat A^T=\operatorname{median}_k\tilde A_k}\) | **Training-only candidate.** Only after teacher scales are calibrated. |
| 215 | Disagreement confidence | \(\displaystyle \boxed{C_V=\exp\!\left(-\frac{\operatorname{MAD}_k(\tilde V_k)}{\tau_V}\right),\qquad C_A=\exp\!\left(-\frac{\operatorname{MAD}_k(\tilde A_k)}{\tau_A}\right)}\) | **Candidate.** Teacher disagreement becomes uncertainty rather than being hidden by averaging. |
| 216 | Causal affect auxiliary heads | \(\displaystyle \boxed{(\hat V_t^S,\hat A_t^S)=g_{VA}(h_t)}\) | **Training-only.** Student sees only causal Player input. |
| 217 | Affect distillation loss | \(\displaystyle \boxed{L_{VA}=C_V\,\ell_V(\hat V_t^S,\hat V^T)+C_A\,\ell_A(\hat A_t^S,\hat A^T)}\) | **Training-only candidate.** Valence-only, arousal-only, joint, energy-only, and energy-plus-arousal variants require ablation. |
| 218 | Offline affect timeline | \(\displaystyle \boxed{\mathcal A^T(t)=\bigl(\hat V^T,\hat A^T,C_V,C_A,\{\tilde V_k,\tilde A_k\}_k,\text{provenance}\bigr)}\) | **Training-only artifact.** Versioned window, alignment, calibration, model, checkpoint, and source-audio metadata. |

Valence and arousal are hypotheses about useful latent supervision, especially for color. They are not mandatory runtime observation fields and do not define deterministic hue or temperature laws.

## Training-only objective composition

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 219 | Auxiliary objective composition | \(\displaystyle \boxed{L_{\rm train}=L_{\rm primary}+\lambda_B L_B+\lambda_R L_R+\lambda_F L_F+\lambda_{VA}L_{VA}+\lambda_{\rm color}L_{\rm color}+\cdots}\) | **Training-only schematic.** Every head/loss must receive a keep/drop decision from downstream behavioral ablation. |
| 220 | Runtime export | \(\displaystyle \boxed{h_t\longrightarrow\text{Controls v2};\qquad\{g_B,g_H,g_F,g_{VA}\}\ \text{may be discarded}}\) | **Destination invariant.** Rich offline supervision can shape the trunk without creating live dependencies or product outputs. |
| 221 | Teacher/runtime causality separation | \(\displaystyle \boxed{T^T(t)=\mathcal T(\text{audio}_{\rm whole})\quad\text{is allowed only as target},\qquad O_t=\mathcal O(\text{audio}_{\le t},\text{state}_t)}\) | **Canonical training rule.** Future-aware teacher state never enters the Student input. |

## Scale- and energy-stratified resets

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 222 | Scale-stratified target | \(\displaystyle \boxed{\sigma_{\rm target}\sim U(\sigma_{\min},\sigma_{\max})}\) | **Research/reset contract.** Exposes deep scale bands deliberately. |
| 223 | Scale-conditioned location | \(\displaystyle \boxed{c_0\sim p\!\left(c\mid\sigma(c)\approx\sigma_{\rm target}\right)}\) | **Research/reset contract.** |
| 224 | Mixed reset distribution | \(\displaystyle \boxed{p(c_0)=w_{\rm area}p_{\rm area}+w_{\rm scale}p_\sigma+w_{\rm replay}p_{\rm replay}}\) | **Research candidate.** Preserve ordinary starts while adding deep coverage. |
| 225 | Approximate depth-scaled perturbation | \(\displaystyle \boxed{\lVert\delta c\rVert\propto2^{-\sigma}}\) | **Cheap approximation.** |
| 226 | Fixed intrinsic perturbation | \(\displaystyle \boxed{\delta r^{\mathsf T}G(c_0)\delta r=\delta s^2}\) | **Preferred geometric reset criterion.** |
| 227 | Metric-consistent initial kinetic energy | \(\displaystyle \boxed{K_0=\frac12v_0^{\mathsf T}G(c_0)v_0}\) | **Canonical reset quantity.** |
| 228 | Total-energy reset | \(\displaystyle \boxed{E_0=K_0+U(c_0)}\) | **Canonical reset quantity.** |
| 229 | Independent Julia reset | \(\displaystyle \boxed{\mathcal P_0,Z_{J,0},\theta_{J,0}\sim p_{\rm Julia}\quad\text{independently of }\sigma(c_0)}\) | **Destination invariant.** Mandelbrot scale does not determine Julia zoom or palette. |

## End-to-end system and compact spines

| # | Concept | Equation | Status / meaning |
| -: | --- | --- | --- |
| 230 | Runtime causal chain | \(\displaystyle \boxed{\text{audio}_{\le t}\to A_{\rm audio}\to O_t\to\pi_\theta\to\text{Controls v2}\to\{Q,J,\text{Julia deltas}\}\to\text{manifold Physics}\to c_t\to J_t}\) | **Destination runtime.** Offline teachers and auxiliary heads are absent. |
| 231 | Geometry spine | \(\displaystyle \boxed{D\to\rho\to\sigma\to q\to J_q\to H\to G\to G^{-1}\to\Gamma}\) | **Canonical geometry.** |
| 232 | Mechanical spine | \(\displaystyle \boxed{G\to K\to p,\qquad U\to Q_U,\qquad a_{\rm cov}=-\operatorname{grad}_G U+G^{-1}Q,\qquad E=K+U}\) | **Canonical mechanics.** |
| 233 | Visual gesture spine | \(\displaystyle \boxed{(c,v,G,\Gamma,Q)\to u,T,N,a_\parallel,a_\perp,\Omega,\kappa_g,\Theta,\eta_M}\) | **Derived diagnostic geometry.** |
| 234 | Audio gesture spine | \(\displaystyle \boxed{x(t)\to W_x(s,t)=Ae^{i\phi}\to(\ln A,\phi,\eta_A;\dot{\ln A},\dot\phi,\dot\eta_A)}\) | **Research/common representation.** |
| 235 | Shared gesture hypothesis | \(\displaystyle \boxed{X_A\in T\mathcal G_A\quad\longleftrightarrow\quad X_V\in T\mathcal G_V}\) | **Research.** Compare two multiscale dynamical geometries rather than arbitrary audio features and shader parameters. |
| 236 | Entrainment hypothesis | \(\displaystyle \boxed{\text{musical coherence}\ \approx\ \text{context-conditioned multiscale phase-harmonic entrainment}}\) | **Umbrella research hypothesis.** |
| 237 | Offline teacher spine | \(\displaystyle \boxed{\text{whole-song teachers}\to\text{versioned soft timelines}\to\text{auxiliary targets / weak labels / evaluation}\to\text{causal Student}}\) | **Training-only architecture.** |

The compact mechanical destination remains

$$
\boxed{
\begin{aligned}
q(c)&=(x,y,\sigma(c)),\\
a&=\frac{\lambda}{\ln2},\\
H(q)&=\operatorname{diag}(\rho^{-2},\rho^{-2},\lambda^2),\\
G(c)&=\rho^{-2}\!\left(I+a^2\nabla\rho\nabla\rho^{\mathsf T}\right),\\
G^{-1}(c)&=\rho^2\!\left[I-\frac{a^2\nabla\rho\nabla\rho^{\mathsf T}}{1+a^2\lVert\nabla\rho\rVert^2}\right],\\
K&=\tfrac12v^{\mathsf T}Gv,\\
p&=Gv,\\
U&=\kappa\sigma+U_{\rm wall},\\
E&=K+U,\\
\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k&=-G^{ij}\partial_jU+G^{ij}Q_j.
\end{aligned}
}
$$

The compact visual-gesture destination is

$$
\boxed{
\begin{aligned}
u&=\sqrt{2K},\\
T&=v/u,\\
N&=J_G T,\\
a_{\rm cov}&=a_\parallel T+a_\perp N,\\
a_\parallel&=\dot u,\\
a_\perp&=u\Omega=u^2\kappa_g,\\
\dot\Theta&=\Omega,\\
m_\parallel&=ue^{i\Theta},\\
\eta_M&=-\ln(\rho/\rho_{\rm ref}),\\
\dot\eta_M&=-\dot\rho/\rho.
\end{aligned}
}
$$

And the current audiovisual research object is

$$
\boxed{
C_{ij}^{n,m}(s,r,\tau;t)
=
\mathcal S_t\!\left(
[a_i(s,t)]^n
\overline{[b_j(r,t-\tau)]^m}
\right),
\qquad
[z]^k=\lvert z\rvert e^{ik\arg z},
}
$$

with a slowly context-conditioned signed fit rather than indiscriminate maximum coherence.
