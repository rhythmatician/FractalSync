# Covariant phase harmonics: equivalence, obstructions, and stability

Research progress for [FractalSync #163](https://github.com/rhythmatician/FractalSync/issues/163), with consequences for [#124](https://github.com/rhythmatician/FractalSync/issues/124) and [#126](https://github.com/rhythmatician/FractalSync/issues/126).

**Status, 2026-11-24:** The statements below have conventional proofs. The accompanying numerical script checks selected consequences and counterexamples; numerical checks are not proofs. A separate Lean module records the exact subset that has been machine checked. No claim of mathematical originality is made. These results neither establish audiovisual fit nor change the preregistered primary experiment.

## 1. Main conclusions

1. Along one specified trajectory and one valid gesture epoch, a transport-aware phase-harmonic statistic is exactly the ordinary statistic of the existing parallel-transported phasor. It is not a distinct ablation.
2. Replacing actual-trajectory transport by arbitrary pairwise geodesic transport can destroy positive semidefiniteness. A supposed squared coherence can become negative. The precise condition is trivial cycle holonomy in the selected harmonic representation.
3. Arbitrary phases at independent epoch starts make a globally pooled moment unidentifiable. The exact range of possible pooled magnitudes is determined by the polygon inequality. Epoch-invariant summaries necessarily discard cross-epoch phase information.
4. The phase-harmonic Lipschitz bound requested in the issue is already exactly Lemma 2.1, equation (2.19), of Mallat, Zhang, and Rochette. It should be reused with attribution.
5. Finite-weight coherence admits explicit perturbation bounds separating mode error, transport-angle error, and normalization support. Positive numerical regularization makes the expression defined at zero; it does not supply missing physical phase or a uniform accuracy guarantee on unbounded inputs.

## 2. Conventions and finite data

Let an oriented Riemannian surface have a metric-compatible connection. Fix a trajectory, its actual time ordering, and one valid epoch. At sample times indexed by a finite set I, let b_i be the complex coordinates of a tangent signal in chosen oriented orthonormal frames. Let a_i be a scalar complex signal, already evaluated at the declared lagged sample time. Nonnegative weights w_i use the same supports for numerator and both energies. Their sum need not equal one, although the units and numerical regularizer must match that convention.

For integer k, define

$$
H_k(z)=\begin{cases}|z|(z/|z|)^k,&z\ne0,\\0,&z=0.\end{cases}
$$

An integer power of a unit complex number is unambiguous, including negative k. In particular H_0(z)=|z|, including z=0. Write L_k=max(1,|k|).

Let p_i be the unit complex number transporting the coordinates at sample i back along the specified trajectory to a fixed reference fiber. Thus the transported harmonic is

$$
y_i=p_i^m H_m(b_i),\qquad x_i=H_n(a_i),\qquad
C=\sum_iw_i x_i\overline{y_i}.
$$

The weighted energies and regularized coherence are

$$
A=\left(\sum_iw_i|a_i|^2\right)^{1/2},\quad
B=\left(\sum_iw_i|b_i|^2\right)^{1/2},\quad
\gamma_\varepsilon=\frac{C}{\sqrt{A^2B^2+\varepsilon}}.
$$

Here epsilon is nonnegative; if epsilon=0 require A B>0. This epsilon has units of the product of the two energy terms. It is unrelated to the geometric Shore regularization parameter. Multiplying all weights by a positive factor alpha preserves the regularized statistic only when epsilon is also multiplied by alpha squared. Fixed epsilon breaks exact amplitude/weight-scale invariance near its floor, so the weight normalization belongs in the statistic's specification.

## 3. Exact covariance and the no-duplicate result

Represent a coordinate gauge change by unit complex numbers g_i: b_i'=g_i b_i. This is a convention for coordinate components; if a frame itself is rotated by chi_i, its components rotate by g_i=exp(-i chi_i). Transport changes as

$$
p_i'=g_0p_i\overline{g_i}.
$$

**Proposition 1 (harmonic covariance).** For every integer m,

$$
H_m(g_i b_i)=g_i^mH_m(b_i),\qquad
(p_i')^mH_m(b_i')=g_0^m p_i^mH_m(b_i).
$$

Consequently C'=overline(g_0)^m C, while A, B, |C|, and |gamma_epsilon| are invariant. If the scalar audio phase also has a constant change a_i'=h a_i, then C'=h^n overline(g_0)^m C.

**Proof.** Unit multiplication preserves modulus. For nonzero b the normalized unit direction is multiplied by g_i, and integer powers multiply. At zero both sides vanish. The second identity cancels overline(g_i)^m g_i^m=1. The constant reference factor then leaves the finite sum. All norm claims follow because that factor has modulus one.

A signed score requires an offset transforming with the same factor. For a unit reference r, the score Re(overline(r) C) is invariant under C'=q C and r'=q r. Holding r fixed while changing C's phase reference does not define an invariant signed score. This is a type requirement, not a property that fitting can evade.

**Corollary 1 (existing transported-frame statistics).** Set $\widetilde b_i=p_i b_i$. Then

$$
C=\sum_iw_iH_n(a_i)\overline{H_m(\widetilde b_i)}.
$$

Thus the proposed covariant construction and #124's transported phasor produce exactly the same numbers when the path, frame initialization, weights, supports, lag, and numerical transport are the same. Additional correction by a connection phase would apply transport twice. A comparison of those mathematically identical constructions would measure implementation discrepancies, not a new representation.

## 4. A pairwise formula and its precise obstruction

Transport from sample j to sample i **along the same time-parameterized path** is

$$
U_{ij}=\overline{p_i}p_j.
$$

Expansion of the finite sum gives the reference-free identity

$$
|C|^2=
\sum_{i,j}w_iw_j x_i\overline{x_j}\,
\overline{H_m(b_i)}U_{ij}^{m}H_m(b_j).
\tag{1}
$$

Indeed U_ij^m=overline(p_i^m)p_j^m, and (1) is C overline(C). It is nonnegative despite complex-looking summands. The matrix K_ij=U_ij^m is positive semidefinite and rank one, because for any complex coefficients v_i,

$$
\sum_{i,j}\overline{v_i}K_{ij}v_j
=\left|\sum_jp_j^mv_j\right|^2.
\tag{2}
$$

**Proposition 2 (unit transport kernel criterion).** Suppose a nonempty finite matrix K is Hermitian, K_ii=1, and |K_ij|=1. The following are equivalent:

1. K is positive semidefinite.
2. K has the form K_ij=overline(q_i)q_j for unit q_i.
3. Every triangle product K_ij K_jk K_ki equals 1.
4. Every cycle product equals 1.

Under these conditions K has rank one.

**Proof.** A PSD matrix is a Gram matrix K_ij=<v_i,v_j>. Unit diagonal gives unit vectors. Equality |<v_i,v_j>|=1 in Cauchy-Schwarz forces all the vectors to be collinear, proving 2. A factorization gives (2), proving PSD and rank one. It also telescopes around every cycle. Conversely fix a reference 0 and set q_i=K_0i. The triangle product K_0i K_ij K_j0=1 implies K_ij=overline(q_i)q_j. Cycles include triangles, and the factorization gives all cycles. The one-point case is immediate.

Applied to K_ij=U_ij^m, the required condition is

$$
\bigl(U_{ij}U_{jk}U_{ki}\bigr)^m=1.
$$

For m=0 it always holds. For nonzero m, raw holonomy need not be trivial if it is invisible to that harmonic, for example a half-turn at even m. Therefore the condition must be stated in the selected representation.

**Exact counterexample.** Consider

$$
K=\begin{pmatrix}1&1&-1\\1&1&1\\-1&1&1\end{pmatrix},\qquad
v=(1,-1,1)^T.
$$

Its cycle product is -1, K v=-v, and v^T K v=-3. Choose w_i=1/3, x=(1,-1,1), and H_m(b_i)=1. Replacing actual-path transport in (1) by this inconsistent K gives -1/3. Both energies equal one. Thus even the alleged normalized squared coherence is negative. This is a failure of the squared-coherence interpretation. An ordinary signed cross-modal score is allowed to be negative and is not contradicted by this example.

The example has a geometric realization. Three vertices of a regular tetrahedron on the unit sphere have mutual dot products -1/3. Their unique shortest great-circle arcs form a spherical triangle with three angles 2pi/3, hence area pi. Levi-Civita holonomy around the boundary is a half-turn. Gauge choices can make two edge transports 1 and the third -1. This yields the displayed matrix for odd m. For a general geodesic triangle with harmonic holonomy exp(i phi), its unit-transport matrix has determinant 2 cos(phi)-2, which is strictly negative when the harmonic holonomy is nontrivial.

**Consequence for implementation.** Retain the actual gesture trajectory's transport. Do not replace it with pairwise shortest-path transport and continue using Gram or squared-coherence guarantees. A connection-Laplacian construction with its own positive kernel can be legitimate, but it is a different statistic requiring a separate derivation and experiment.

## 5. Epoch phases are an identifiable obstruction

Suppose E>=1 independent gesture epochs have moments C_1,...,C_E. For a nonzero visual harmonic m, unknown initial frames allow independent phase changes C_e -> exp(i alpha_e) C_e. The factor m does not restrict the possible phase, since multiplication by a nonzero integer maps the circle onto itself.

**Proposition 3 (sharp pooling range).** Let r_e=|C_e|, R=sum_e r_e, and r_max=max_e r_e. As the unknown phases vary, the pooled modulus ranges over exactly

$$
\left[\max(0,2r_{\max}-R),\ R\right].
\tag{3}
$$

**Proof.** Triangle and reverse triangle inequalities give both endpoints as bounds. For exact attainability, a sum of vectors with lengths r_1,...,r_E has a rotationally symmetric set of possible values. Inductively suppose its possible radii form [L,U]. Adding a vector of length r gives radii in the union over s in [L,U] of [|s-r|,s+r], which is [max(0,L-r,r-U),U+r]. Beginning with [r_1,r_1] gives (3). Every intermediate radius is attained by continuity of the angles.

Two identical perfect-coherence epochs with C_1=C_2=1/2 can therefore have a pooled modulus anywhere from 0 to 1, without changing the within-epoch signals at all. Normalization energies are unchanged. Global pooled magnitude across unknown epochs is consequently not a well-defined physical feature.

Independent uniform phase averaging gives

$$
\mathbb E_\alpha\left|\sum_e e^{i\alpha_e}C_e\right|^2
=\sum_e|C_e|^2.
$$

For unregularized within-epoch energies A_e,B_e, the optional block summary

$$
\Gamma_{\rm block}^2=
\frac{\sum_e|C_e|^2}{\sum_e A_e^2B_e^2}\in[0,1]
$$

is a weighted mean of squared within-epoch coherences, if its denominator is positive. It deliberately discards cross-epoch phase alignment. Changing the epoch partition changes the question: if every epoch has one nonzero sample, each within-epoch coherence magnitude is one. This summary must not be used to repair a failed experiment by fragmenting epochs or optimizing offsets on held-out data.

Known inter-epoch phase bridges can identify a longer epoch. Unknown bridges cannot be supplied by arbitrary phase stitching. This statement concerns comparison across the break, not the existence of an amplitude-preserving zero H_k(0)=0 at an isolated zero-valued sample.

## 6. Exact Lipschitz constant, with prior-art attribution

**Proposition 4 (established result).** For every integer k and z,w in C,

$$
|H_k(z)-H_k(w)|\le L_k|z-w|,\qquad L_k=\max(1,|k|),
\tag{4}
$$

and the constant is sharp.

This is precisely Lemma 2.1, equation (2.19), in Mallat, Zhang, and Rochette [R1], not a proposed new theorem. A short independent proof is useful for formalization. Write z=r xi, w=s eta with r,s nonnegative and xi,eta unit. For k nonnegative, a telescoping geometric sum gives |xi^k-eta^k|<=k|xi-eta|. Negative k reduces to positive k by complex conjugation. Then

$$
|r\xi^k-s\eta^k|^2
=(r-s)^2+rs|\xi^k-\eta^k|^2
\le L_k^2\bigl((r-s)^2+rs|\xi-\eta|^2\bigr).
$$

Zero input cases follow immediately from modulus preservation. For k=0 use the reverse triangle inequality. Radial variation proves that a constant below 1 is impossible. For |k|>=1, unit-circle arguments with arbitrarily small angular separation have ratio tending to |k|.

## 7. A complete finite-weight perturbation bound

Put X_i=sqrt(w_i) H_n(a_i) and Y_i=sqrt(w_i) p_i^m H_m(b_i). Write ||X||=A and ||Y||=B. For another data set on the same finite support write X',Y',A',B', and

$$
d_X=\|X'-X\|_2,\qquad d_Y=\|Y'-Y\|_2,\qquad
D=\sqrt{A^2B^2+\varepsilon},\quad D'=\sqrt{(A')^2(B')^2+\varepsilon}.
$$

**Proposition 5 (normalized correlation perturbation).** Assume D,D'>0 and the same epsilon. Then

$$
|\gamma_\varepsilon'-\gamma_\varepsilon|
\le \min\!\left(2,
\frac{2\min\{A d_Y+B'd_X,\ B d_X+A'd_Y\}}{\max(D,D')}
\right).
\tag{5}
$$

**Proof.** Cauchy-Schwarz gives |C|<=A B<=D, so |gamma|<=1. Expanding the difference in either order gives

$$
|C'-C|\le Q:=\min\{A d_Y+B'd_X,\ B d_X+A'd_Y\}.
$$

The reverse triangle inequality for norms gives |A'B'-AB|<=Q as well. The map s -> sqrt(s^2+epsilon) is 1-Lipschitz for s>=0, hence |D'-D|<=Q. If D>=D', write

$$
\gamma'-\gamma=(C'-C)/D+C'(1/D'-1/D).
$$

Using |C'|<=D' gives an upper bound (|C'-C|+|D'-D|)/D<=2Q/D. Exchange primed and unprimed quantities when D'>D. Finally both coherences have modulus at most one, giving the cap 2.

For mode perturbations e_a=||a'-a||_w, e_b=||b'-b||_w, suppose p_i'=p_i exp(i delta_i). Proposition 4 and the triangle inequality imply

$$
d_X\le L_n e_a,
\quad
d_Y\le L_m e_b+
\left(\sum_iw_i|b_i'|^2\,4\sin^2(m\delta_i/2)\right)^{1/2}.
\tag{6}
$$

The second term is bounded by B' min(2,|m| max_i|delta_i|). For m=0 it vanishes. Transport error is therefore explicitly weighted by signal amplitude, and it grows at most linearly with harmonic order for small angular errors. A bound on connection integration error can feed delta_i; changed trajectories or changed connections need their own bounds.

At epsilon=0 and positive A,A',B,B', one can additionally use the direction-normalization inequality

$$
\left\|X/A-X'/A'\right\|
\le\frac{2d_X}{A+A'}
$$

to obtain

$$
|\gamma_0'-\gamma_0|
\le\min\left(2,\frac{2d_X}{A+A'}+\frac{2d_Y}{B+B'}\right).
\tag{7}
$$

For the normalization inequality, set u=X/A, u'=X'/A'. The identity

$$
\|X-X'\|^2-\frac{(A+A')^2}{4}\|u-u'\|^2
=(A-A')^2\left(1-\frac{\|u-u'\|^2}{4}\right)\ge0
$$

proves it since unit vectors differ by at most 2. Equation (7) follows by adding and subtracting one normalized inner product.

Changing masks or weights is covered only if their contribution to d_X,d_Y is included. For example a weight change contributes terms involving |sqrt(w_i')-sqrt(w_i)| times mode amplitude. The theorem does not assume a hard validity threshold changes continuously.

### Why nonzero support is a real condition

With epsilon=0, take Y=(1,0), X_t=(t,0), and X_t'=(0,t). Coherences are 1 and 0 while ||X_t-X_t'||=sqrt(2)t tends to zero. No continuous extension at X=0 preserves normalized coherence.

Even epsilon>0 does not give a uniform global Lipschitz constant without amplitude control. In one sample, take X=0, X'=1/B, Y=Y'=B. Then the input difference tends to zero as B grows, while the coherence difference stays 1/sqrt(1+epsilon). With a declared amplitude bound A,A',B,B'<=M, (5) does yield the global bound 2M(d_X+d_Y)/sqrt(epsilon).

Rest, low confidence, unknown epoch resets, and physically undefined heading remain validity decisions in #124. A numerical denominator does not establish them.

## 8. Holonomy and intrinsic turning use different signs and roles

In a local coordinate frame, choose the convention that the covariant derivative of a complex vector is db/dt+i omega(t)b. Then transport back to time zero is p(t)=exp(i integral_0^t omega(s) ds). If b=u exp(i theta), its intrinsic turning is Omega=theta_dot+omega. Therefore the transported heading Theta=theta+integral omega satisfies Theta_dot=Omega, matching #124.

Forward holonomy around a closed path has angle h=-integral omega modulo 2pi. If a regular tangent velocity returns to itself and the local coordinate frame is periodic, theta(T)-theta(0)=2pi j for an integer j, so

$$
\Theta(T)-\Theta(0)+h=2\pi j.
$$

Intrinsic turning and connection rotation must not both be added as independent corrections. On a flat simply connected patch h=0. On an oriented constant-curvature surface, holonomy is, up to the stated orientation convention, curvature times enclosed area modulo 2pi. Actual-path transport is still coherent on the time interval even when the path closes in the base surface: the two endpoint times are distinct until one imposes an additional periodic identification.

## 9. Originality assessment and next work

The phase-harmonic bound is explicitly prior art. Gauge covariance, common-reference trivialization, rank-one unit Gram matrices, the polygon inequality, and Cauchy-Schwarz perturbation arguments are established mathematical ingredients. The useful contribution here is their exact combination into testable FractalSync contracts and concrete failure cases. This investigation has **not established an independently publishable new mathematical result**.

The strongest practical consequences are already actionable: no duplicate covariant ablation, no pairwise shortcut that destroys PSD, no silent cross-epoch pooling, and explicit harmonic-order/transport/normalization error budgets. A research paper would need a genuinely sharper theorem, a new identifiable statistic with information unavailable in the current transported stream, or substantial empirical/general-domain findings. Those are further research questions, not prerequisites for training.

### Primary references reviewed

- [R1] Stéphane Mallat, Sixin Zhang, and Gaspar Rochette, *Phase Harmonic Correlations and Convolutional Neural Networks*, arXiv:1810.12136v2, Lemma 2.1 and equation (2.19); published in *Information and Inference* 9(3), 2020. https://arxiv.org/abs/1810.12136 ; https://doi.org/10.1093/imaiai/iaz019 . The exact single-harmonic Lipschitz theorem is already present; the paper also develops stability of unnormalized covariance operators.
- [R2] Amit Singer and Hau-tieng Wu, *Vector Diffusion Maps and the Connection Laplacian*, arXiv:1102.0075. https://arxiv.org/abs/1102.0075 . Relevant prior work for comparing vector-valued data through connection information. Its diffusion construction is not the raw unit-entry kernel rejected above.
- [R3] Amit Singer, *Angular Synchronization by Eigenvectors and Semidefinite Programming*, arXiv:0905.3174. https://arxiv.org/abs/0905.3174 . Relevant prior work on recovering phases from pairwise offsets up to a common phase, and the effect of incompatible/noisy pairwise measurements.
- [R4] Taco Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling, *Gauge Equivariant Convolutional Networks and the Icosahedral CNN*, ICML 2019. https://proceedings.mlr.press/v97/cohen19d.html . Relevant prior work for gauge transformation rules of features and transport, not a claim that local covariance is novel here.

The search was a targeted first pass, not a comprehensive novelty clearance.

## Rust comparison over accepted gesture samples

The prepared-field rollout can now emit a motion phasor from the velocity
in its transported frame. The harmonic comparison consumes this phasor
once and requires every gesture sample to share the trajectory, epoch,
and reference identifiers. The caller supplies aligned sample arrays. A
positive lag pairs gesture index `i` with audio index `i - lag`. Weights
index gesture samples. Missing motion phases and zero weights remove the
same pair from the cross-moment and both energies. The comparison divides
all three sums by their common weight and reports no result on insufficient
support or zero energy. Coherence is the magnitude of the normalized
cross-moment divided by the geometric mean of the normalized energies.

Changing the epoch reference rotates the complex moment by the expected
harmonic phase factor. Magnitude and coherence are invariant under a
constant reference rotation. A signed fit must rotate its preferred
reference by the same factor. The current result carries the complex
moment so a probe can make that comparison. It supplies no angle-error
estimate when any contributing frame lacks one. The experimental
transport currently supplies no certified angle bound. The held-out
audiovisual fit experiment and production gesture source remain open.
