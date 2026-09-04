Here is the **current equation set after the latest Mandelbrot-native Physics pivot**. I’ve marked things that are still candidate/research rather than pretending they are frozen.

### Fractal geometry and Mandelbrot Map

|  # | Concept                         | Equation                                                                                                                                 | Status / meaning                                        |                                     |                                                |
| -: | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------- | ---------------------------------------------- |
|  1 | Julia iteration                 | \(\displaystyle z_{n+1}=z_n^2+c\)                                                                                                        | Fundamental Julia dynamics.                             |                                     |                                                |
|  2 | Mandelbrot iteration            | \(\displaystyle z_0=0,\qquad z_{n+1}=z_n^2+c\)                                                                                           | Defines Mandelbrot membership of \(c\).                 |                                     |                                                |
|  3 | Complex parameter               | \(\displaystyle c=x+iy\)                                                                                                                 | Independent Mandelbrot configuration coordinate.        |                                     |                                                |
|  4 | Smooth escape iteration         | (\displaystyle \nu=n+1-\log_2(\log                                                                                                       | z_n                                                     | ))                                  | Used to construct the smooth escape field.     |
|  5 | Normalized escape field         | \(\displaystyle F(c)=\operatorname{clamp}\!\left(\frac{\nu(c)}{N},0,1\right)\)                                                           | Existing baked \(F\) field.                             |                                     |                                                |
|  6 | Escape-field gradient magnitude | \(\displaystyle G_F(c)=\|\nabla F(c)\|\)                                                                                                 | Local escape-field sensitivity.                         |                                     |                                                |
|  7 | Shore sensitivity/proximity     | \(\displaystyle S(c)=\frac{G_F(c)}{G_F(c)+G_0}\)                                                                                         | Existing baked \(S\) field. **Not geometric distance.** |                                     |                                                |
|  8 | Signed Shore distance           | \(\displaystyle D(c)=\begin{cases}+\operatorname{dist}(c,\partial M),&c\notin M\\-\operatorname{dist}(c,\partial M),&c\in M\end{cases}\) | Gives both realm and geometric distance.                |                                     |                                                |
|  9 | Unsigned Shore distance         | (\displaystyle d(c)=                                                                                                                     | D(c)                                                    | =\operatorname{dist}(c,\partial M)) | Geometric distance to the Mandelbrot boundary. |

### Mandelbrot scale manifold

|  # | Concept                           | Equation                                                                                                                  | Status / meaning                                                                |            |                           |
| -: | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- | ------------------------- |
| 10 | Smooth finite-resolution distance | \(\displaystyle \boxed{\rho(c)=\sqrt{D(c)^2+\epsilon^2}}\)                                                                | **Leading candidate** replacing the old hard `max(d,d_min)` clamp.              |            |                           |
| 11 | Mandelbrot scale                  | \(\displaystyle \boxed{\sigma(c)=\log_2\frac{d_{\rm ref}}{\rho(c)}}\)                                                     | Current leading scale formulation. Scale belongs only to the Mandelbrot domain. |            |                           |
| 12 | Asymptotic scale law              | \(\displaystyle d(c)\approx d_{\rm ref}2^{-\sigma(c)}\)                                                                   | Valid away from the regularized Shore crest.                                    |            |                           |
| 13 | Embedded manifold                 | \(\displaystyle \boxed{q(c)=\bigl(x,y,\sigma(c)\bigr)}\)                                                                  | A 2D configuration manifold embedded in 3D position-scale space.                |            |                           |
| 14 | Independent planar coordinate     | \(\displaystyle r=(x,y)\)                                                                                                 | The actual two generalized coordinates.                                         |            |                           |
| 15 | Embedding Jacobian                | \(\displaystyle \boxed{J_q(c)=\frac{\partial q}{\partial(x,y)}=\begin{bmatrix}1&0\\0&1\\\sigma_x&\sigma_y\end{bmatrix}}\) | Converts tangent-space motion into embedded position-scale motion.              |            |                           |
| 16 | Planar velocity                   | \(\displaystyle v=\dot r=(\dot x,\dot y)\)                                                                                | Persistent Mandelbrot Momentum/velocity.                                        |            |                           |
| 17 | Embedded velocity                 | \(\displaystyle \boxed{\dot q=J_q(c)\,v}\)                                                                                | Core kinematic relationship.                                                    |            |                           |
| 18 | Induced scale velocity            | \(\displaystyle \boxed{\dot\sigma=\nabla\sigma(c)\cdot v}\)                                                               | No independent \(v_\sigma\).                                                    |            |                           |
| 19 | Iso-scale direction               | \(\displaystyle \nabla\sigma(c)\cdot u=0\)                                                                                | A tangent direction that locally preserves Mandelbrot scale.                    |            |                           |
| 20 | Maximum scale-change direction    | \(\displaystyle u\parallel\pm\nabla\sigma(c)\)                                                                            | Locally maximizes (                                                             | \dot\sigma | ) for fixed planar speed. |

### Local Mandelbrot frame

|  # | Concept             | Equation                                                                                                   | Status / meaning                                                                               |
| -: | ------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 21 | Local normal        | \(\displaystyle \hat n=\frac{\nabla S}{\|\nabla S\|}\)                                                     | Useful local Shore/sensitivity frame where defined.                                            |
| 22 | Local tangent       | \(\displaystyle \hat t=(-n_y,n_x)\)                                                                        | Tangent to the local \(S\)-contour.                                                            |
| 23 | Normal velocity     | \(\displaystyle v_n=v\cdot\hat n\)                                                                         | Local normal component.                                                                        |
| 24 | Tangential velocity | \(\displaystyle v_t=v\cdot\hat t\)                                                                         | Local carving component.                                                                       |
| 25 | Scale Hessian       | \(\displaystyle H_\sigma(c)=\begin{bmatrix}\sigma_{xx}&\sigma_{xy}\\\sigma_{xy}&\sigma_{yy}\end{bmatrix}\) | Second-order geometry. Likely required internally for Physics; optional as Player observation. |

### Metric and kinetic energy

|  # | Concept                                   | Equation                                                                             | Status / meaning                                                                           |
| -: | ----------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 26 | Ambient position-scale metric             | \(\displaystyle H=\operatorname{diag}(1,1,\lambda^2)\)                               | Sets relative mechanical weight of scale displacement.                                     |
| 27 | Induced Mandelbrot metric                 | \(\displaystyle \boxed{G(c)=J_q(c)^{\mathsf T}H\,J_q(c)}\)                           | Core local metric.                                                                         |
| 28 | Expanded induced metric                   | \(\displaystyle \boxed{G(c)=I+\lambda^2\nabla\sigma(c)\nabla\sigma(c)^{\mathsf T}}\) | Simplest current metric candidate.                                                         |
| 29 | Line element                              | \(\displaystyle ds^2=dx^2+dy^2+\lambda^2d\sigma^2\)                                  | Ambient geometric interpretation.                                                          |
| 30 | Kinetic energy                            | \(\displaystyle \boxed{K(c,v)=\frac12v^{\mathsf T}G(c)v}\)                           | Current canonical KE formulation.                                                          |
| 31 | Expanded KE                               | \(\displaystyle K=\frac12\|v\|^2+\frac12\lambda^2\dot\sigma^2\)                      | Shows planar + induced-scale contributions.                                                |
| 32 | Local tangent/normal/scale interpretation | \(\displaystyle K=\frac12m_t v_t^2+\frac12m_n v_n^2+\frac12m_\sigma\dot\sigma^2\)    | Interpretive/generalized form; coefficients should derive from one authoritative geometry. |

### Connection and manifold dynamics

|  # | Concept                         | Equation                                                                                                                          | Status / meaning                                                                             |
| -: | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 33 | Christoffel symbols             | \(\displaystyle \boxed{\Gamma^i_{jk}=\frac12G^{i\ell}\left(\partial_jG_{k\ell}+\partial_kG_{j\ell}-\partial_\ell G_{jk}\right)}\) | The missing piece that makes curved geometry actually redirect motion.                       |
| 34 | Free geodesic motion            | \(\displaystyle \boxed{\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k=0}\)                                                               | Free rider moving on the Mandelbrot manifold.                                                |
| 35 | Native potential                | \(\displaystyle \boxed{U(c)=\kappa\,\sigma(c)}\)                                                                                  | **Leading candidate.** Makes The Shore a high-potential ridge.                               |
| 36 | Potential force                 | \(\displaystyle F_U=-\operatorname{grad}_G U=-G^{-1}\nabla U\)                                                                    | Conservative “gravity” supplied by Mandelbrot geometry rather than attraction to the origin. |
| 37 | Lagrangian                      | \(\displaystyle L=K-U\)                                                                                                           | Compact mechanics formulation.                                                               |
| 38 | Forced Euler–Lagrange equation  | \(\displaystyle \frac{d}{dt}\frac{\partial L}{\partial\dot r}-\frac{\partial L}{\partial r}=Q\)                                   | Adds generalized non-conservative Controls/friction.                                         |
| 39 | Full Mandelbrot-native dynamics | \(\displaystyle \boxed{\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k=-G^{ij}\partial_jU+G^{ij}Q_j}\)                                    | Current destination equation of motion.                                                      |
| 40 | Mechanical energy               | \(\displaystyle \boxed{E=K+U}\)                                                                                                   | Derived mechanical-energy diagnostic/invariant.                                              |

### Generalized Controls, friction, and energy accounting

|  # | Concept                | Equation                                                                                 | Status / meaning                                                                                          |
| -: | ---------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 41 | Drive force            | \(\displaystyle Q_{\rm drive}=a_{\rm drive}\,\eta\)                                      | Schematic: throttle supplies generalized work along a selected tangent-space covector/direction \(\eta\). |
| 42 | Metric-consistent drag | \(\displaystyle \boxed{Q_{\rm drag}=-\beta G(c)v}\)                                      | Simple isotropic dissipation candidate.                                                                   |
| 43 | General friction       | \(\displaystyle \boxed{Q_{\rm friction}=-B(c,\mathrm{grip},\ldots)v}\)                   | \(B\succeq0\); drift/grip can control its anisotropy.                                                     |
| 44 | Friction power         | \(\displaystyle P_{\rm friction}=-v^{\mathsf T}Bv\le0\)                                  | Guarantees friction cannot inject energy.                                                                 |
| 45 | Simple drag power      | \(\displaystyle P_{\rm drag}=v^{\mathsf T}Q_{\rm drag}=-\beta v^{\mathsf T}Gv\le0\)      | Explicit dissipation proof.                                                                               |
| 46 | Energy ledger          | \(\displaystyle \boxed{\frac{dE}{dt}=P_{\rm control}+P_{\rm impulse}-P_{\rm friction}}\) | Up to numerical integration error.                                                                        |
| 47 | Impulse update         | \(\displaystyle p^+=p^-+J_{\rm impulse}\)                                                | Schematic generalized-momentum interpretation of taps/impulses.                                           |

### Continuous multi-resolution Map

|  # | Concept                    | Equation                                                                            | Status / meaning                                                   |
| -: | -------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 48 | Continuous MIP coordinate  | \(\displaystyle \ell=\ell(\sigma),\qquad k=\lfloor\ell\rfloor,\qquad\alpha=\ell-k\) | Maps continuous Mandelbrot scale onto stored discrete resolutions. |
| 49 | Adjacent-MIP interpolation | \(\displaystyle \boxed{S(c,\sigma)=(1-\alpha)S_k(c)+\alpha S_{k+1}(c)}\)            | Representative continuous cross-MIP interpolation.                 |
| 50 | Cross-scale derivative     | \(\displaystyle \partial_\sigma S\approx\frac{S_{k+1}-S_k}{\Delta\sigma}\)          | Candidate observation/geometry evidence.                           |

### Audio / CycleBank

|  # | Concept                   | Equation                                                                                                                            | Status / meaning                                      |
| -: | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 51 | CycleBank mode            | \(\displaystyle [\cos\phi_i,\sin\phi_i,\log_2 f_i,A_i,C_i]\)                                                                        | Current conceptual Player-facing mode representation. |
| 52 | Rational phase relation   | \(\displaystyle \boxed{\psi_{ij}=\operatorname{wrap}(n\phi_i-m\phi_j)}\)                                                            | Detects stable rational timing relationships.         |
| 53 | Latent temporal undertone | \(\displaystyle f_i\approx n_i f_0\)                                                                                                | Deferred #97 hypothesis.                              |
| 54 | AudioActivation           | \(\displaystyle \boxed{A_{\rm audio}(t)=\operatorname{concat}(A_{\rm cycles},A_{\rm intensity},A_{\rm spectral},A_{\rm texture})}\) | Composite causal music tensor.                        |

### Player observation and policy

|  # | Concept             | Equation                                                                                                                                                                     | Status / meaning                                         |
| -: | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 55 | PlayerObservation   | \(\displaystyle \boxed{O_t=\operatorname{concat}(A_{\rm audio},A_{\rm Map/Geometry},A_{\rm Physics},A_{\rm JuliaView})}\)                                                    | Complete ambient feedback.                               |
| 56 | Causal PlayerPolicy | \(\displaystyle \boxed{a_t=\pi_\theta(O_{\le t})}\)                                                                                                                          | Learned causal action policy.                            |
| 57 | Controls v2         | \(\displaystyle a_t=(\text{throttle},\text{steer},\text{brake},\text{grip/drift},\text{taps},\text{Julia zoom }\Delta,\text{rotation }\Delta,\text{palette }\Delta,\ldots)\) | Motion actions + independent Julia presentation actions. |

### Julia presentation

|  # | Concept                 | Equation                                                                           | Status / meaning                                                                                            |
| -: | ----------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 58 | Julia visual            | \(\displaystyle J_t=J(c_t;\,Z_t,\theta_t,\mathcal P_t)\)                           | Julia parameter \(c_t\) plus independent zoom \(Z_t\), rotation \(\theta_t\), and palette \(\mathcal P_t\). |
| 59 | Zoom integration        | \(\displaystyle Z_{t+\Delta t}=Z_t+\Delta Z_t\)                                    | Schematic; exact bounded/log-space representation still to be frozen.                                       |
| 60 | Rotation integration    | \(\displaystyle \theta_{t+\Delta t}=\operatorname{wrap}(\theta_t+\Delta\theta_t)\) | Persistent cinematic rotation.                                                                              |
| 61 | Palette-state evolution | \(\displaystyle \mathcal P_{t+\Delta t}=\Phi(\mathcal P_t,\Delta\mathcal P_t)\)    | Schematic Rust-owned bounded `ColorIntent` evolution.                                                       |

### Scale- and energy-stratified training resets

|  # | Concept                       | Equation                                                                                                     | Status / meaning                                                             |
| -: | ----------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 62 | Scale-stratified reset        | \(\displaystyle \boxed{\sigma_{\rm target}\sim U(\sigma_{\min},\sigma_{\max})}\)                             | Exposes the Player to deep scale bands during training.                      |
| 63 | Reset conditioned on scale    | \(\displaystyle c_0\sim p(c\mid\sigma(c)\approx\sigma_{\rm target})\)                                        | Samples a location in the requested scale band.                              |
| 64 | Mixed reset distribution      | \(\displaystyle \boxed{p(c_0)=w_{\rm area}p_{\rm area}+w_{\rm scale}p_\sigma+w_{\rm replay}p_{\rm replay}}\) | Current research hypothesis for #110.                                        |
| 65 | Depth-scaled perturbation     | \(\displaystyle \|\delta c\|\propto2^{-\sigma}\)                                                             | Rough Euclidean approximation to scale-aware neighborhood sampling.          |
| 66 | Better perturbation criterion | \(\displaystyle \delta r^{\mathsf T}G(c)\delta r=\delta s^2\)                                                | Fixed **manifold-distance** perturbation rather than fixed Euclidean jitter. |
| 67 | Metric-consistent initial KE  | \(\displaystyle \boxed{K_0=\frac12v_0^{\mathsf T}G(c_0)v_0}\)                                                | Lets resets normalize Momentum by actual mechanical energy.                  |
| 68 | Total-energy reset            | \(\displaystyle E_0=K_0+U(c_0)\)                                                                             | Allows training to stratify not only scale but mechanical regime.            |

### End-to-end system

|  # | Concept          | Equation                                                                                                                                                                                                                                            |
| -: | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 69 | Full system      | \(\displaystyle \boxed{\text{audio}\rightarrow A_{\rm audio}\rightarrow O_t\rightarrow\pi_\theta\rightarrow a_t\rightarrow\{Q,\text{Julia view deltas}\}\rightarrow\text{manifold Physics}\rightarrow c_t\rightarrow J_t\rightarrow\text{reward}}\) |
| 70 | Mechanical spine | \(\displaystyle \boxed{\begin{aligned}q(c)&=(x,y,\sigma(c))\\G(c)&=J_q^{\mathsf T}HJ_q\\K&=\tfrac12v^{\mathsf T}Gv\\U&=\kappa\sigma(c)\\E&=K+U\\\ddot r^i+\Gamma^i_{jk}\dot r^j\dot r^k&=-G^{ij}\partial_jU+G^{ij}Q_j\end{aligned}}\)               |

The biggest change from our earlier equation lists is that **\(J_q\) and \(G\) are no longer the end of the Physics story**. The current destination is the full line from

$$
\boxed{D(c)\rightarrow\sigma(c)\rightarrow q(c)\rightarrow J_q(c)\rightarrow G(c)\rightarrow\Gamma(c)}
$$

to

$$
\boxed{
K,\ U,\ E,\ Q
\rightarrow
\text{manifold equations of motion}.
}
$$

That is now the mathematical core of the Mandelbrot-native engine.
