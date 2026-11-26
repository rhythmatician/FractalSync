# Acceleration budgets on orbit-trap contours

For a prescribed exact-trap path, intrinsic arc length separates its shape
from the speed at which the Player follows it. A high-curvature pass near
the saddle forces a drop in speed, but the pass itself shrinks. On the
critical level, a change to a perpendicular branch requires a full stop.
[Turning near the orbit-trap saddle](185-trap-saddle-curvature.md)

## Acceleration along a prescribed contour

Let $\gamma(s)$ be a regular contour arc parameterized by positively
oriented intrinsic arc length in $G=\rho^{-2}I$. Let $s=s(t)$ increase,
$v=ds/dt>0$, and $w(s)=v(s)^2$. Write $T=\gamma'(s)$ for its $G$-unit
tangent, $N$ for the $G$-unit left normal, and $\kappa_G$ for signed
geodesic curvature. The derivative $w'$ below is with respect to $s$.
Since $\nabla_T T=\kappa_G N$ and $dv/dt=v\,dv/ds=w'/2$,

$$
\nabla_{\dot{\gamma}}\dot{\gamma}
  = \frac{w'}2 T + \kappa_G w N,
\qquad
\big\|\nabla_{\dot{\gamma}}\dot{\gamma}\big\|_G^2
  = \left(\frac{w'}2\right)^2 + \kappa_G^2 w^2.
$$

For an acceleration cap $a_{\mathrm{cap}}>0$, the formula gives a
pointwise test of a proposed schedule. For a conservative schedule, choose
nonnegative tangential and normal allocations $a_t,a_n$ with
$a_t^2+a_n^2\le a_{\mathrm{cap}}^2$. If $K\ge|\kappa_G|$ throughout the
arc, impose $|w'|\le 2a_t$ and $K w\le a_n$. When $K=0$, the normal
constraint is automatic. The full cap is available to a constant-speed
pass. For a variable speed the residual tangential allowance at an
already feasible $w$ is $\sqrt{a_{\mathrm{cap}}^2-\kappa_G^2 w^2}$,
which may exceed the chosen fixed allocation.

The tangential constraint gives the endpoint condition
$|w(s_1)-w(s_0)|\le2a_t(s_1-s_0)$ by the mean-value theorem, with
continuity at the endpoints. Braking from entry speed $v_0$ to rest
therefore requires distance at least $v_0^2/(2a_t)$ for $a_t>0$.
With stops at both ends, the pointwise envelope is

$$
w(s)\le \min\{2a_t(s-s_0),\,2a_t(s_1-s),\,a_n/K\},
$$

with the last bound used for $K>0$. For the full cap itself, replace
$a_t$ by $a_{\mathrm{cap}}$ in the endpoint condition to get a necessary
bound regardless of how the acceleration is split.

## A stop-to-stop benchmark

For $L=s_1-s_0>0$, choose a positive squared peak speed $W$ with
$2W/L\le a_t$ and $KW\le a_n$. Set $u=(s-s_0)/L$ and

$$
w(s)=4W u(1-u),\qquad
s(t)=s_0+\frac{L}{2}\left(1-\cos\frac{2\sqrt{W}\,t}{L}\right),
\qquad 0\le t\le\frac{\pi L}{2\sqrt W}.
$$

The schedule satisfies $\dot s=\sqrt{w(s)}$ on this interval and both
ends are at rest. The maximum of $w$ is $W$, and $|w'|$ is at most
$4W/L$. The resulting travel time is exactly $\pi L/(2\sqrt W)$.
Its endpoint tangential acceleration has magnitude $2W/L$, which matters
if the arc is joined to another schedule or a hold.

## Regular levels near the saddle

Write $P(c)=c^2+c$, $c_s=-1/2$, $r=|P(c)|$,
$\delta=|1-4r|$, and $\sigma=\operatorname{sgn}(1-4r)$. On the phase
window $\theta=\pi+\delta\xi$ with $|\xi|\le T_0$, $T_0>0$ fixed,
the earlier curvature expansion is uniform on either regular side.
In particular, $\sqrt{\delta}\,\kappa_G$ tends to
$2\sigma\rho(c_s)/(1+\xi^2)^{3/4}$. The limit is bounded away from zero
throughout that fixed window. A positive, smooth ruler near $c_s$ makes
the window's intrinsic length proportional to $\sqrt{\delta}$ in order.

There is an explicit curvature bound for schedules that need a finite
$K$. Take $0<\delta\le1/2$, and let $\rho\le R$ and
$\|\nabla_E\rho\|\le B$ throughout the window. Its discriminant
$D=1-4r e^{i\delta\xi}$ satisfies
$\delta\le |D|\le \delta(1+3T_0/2)$. The lower bound is the reverse
triangle inequality. The upper bound uses $4r\le3/2$ and
$|e^{ix}-1|\le |x|$. Insert these bounds and $r\ge1/8$ into
$\kappa_G=\rho\kappa_E+\partial_{N_E}\rho$, where $N_E$ is the
Euclidean unit normal, to get

$$
|\kappa_G|\le
\frac{4R(1+\delta\sqrt{1+3T_0/2})}{\sqrt{\delta}}+B.
$$

The bounds on $D$ also give the length order directly, since
$ds/d\theta=r/(\rho\sqrt{|D|})$ and $\rho$ is bounded above and away
from zero near $c_s$. With $K$ chosen as the displayed bound, a
constant-speed pass at $v=\sqrt{a_{\mathrm{cap}}/K}$ has speed of order
$\delta^{1/4}$ and elapsed time of order $\delta^{1/4}$. The same travel
time order is achievable with stops at both ends using
$W=\min\{a_t L/2,a_n/K\}$ for fixed positive allocations. The uniform
curvature lower bound forces $v=O(\delta^{1/4})$ everywhere in the window
under the full cap, so any such traversal takes at least order
$\delta^{1/4}$. These estimates concern only the shrinking window.
Braking to enter it from a fixed speed can cost finite distance and time
on the rest of the contour.

## Critical contour handoff

On $r=1/4$, set the handoff time to $t=0$ and write
$\phi(t)=\theta(t)-\pi$. For increasing phase the incoming and outgoing
tangent lines at the saddle are perpendicular. A differentiable handoff
with matching one-sided velocities must have zero velocity, since the
common velocity lies in both lines. The ruler stays positive here, so
zero coordinate velocity also means zero intrinsic speed.

An explicit phase schedule is $\phi(t)=\beta\,\operatorname{sgn}(t)
|t|^{2m}$ for $\beta>0$. The exact contour has

$$
(2c+1)^2=1-e^{i\phi}=-i\phi H(\phi),\qquad
H(x)=\int_0^1 e^{iux}\,du.
$$

The function $H$ is analytic, with $H(0)=1$, so near zero it has an
analytic square root with value $1$ there. On the two sides use

$$
c(t)-c_s=\tfrac12 |t|^m B_\pm\sqrt{H(\phi(t))},\qquad
B_-^2=i\beta,\qquad B_+^2=-i\beta.
$$

Choose the signs of $B_-$ and $B_+$ to match the prescribed incident
branches and set $c(0)=c_s$. On either side the analytic factor is
$1+O(|t|^{2m})$ with derivatives obtained by termwise differentiation
of its convergent local series. For $m=2$ the velocity tends to zero and
the coordinate acceleration tends to $B_\pm$ on each side. Thus the
joined path is $C^{1,1}$ with bounded one-sided acceleration. The
connection term is quadratic in velocity and tends to zero, so the
one-sided intrinsic acceleration magnitudes tend to
$\sqrt{\beta}/\rho(c_s)$. Taking this number strictly below the cap gives
a sufficiently short local handoff within the cap on both sides.
For $m=3$ both one-sided second derivatives tend to zero and the path
is $C^2$ through the handoff. Its covariant acceleration also tends to
zero there.

## Checked scope

Lean checks the time derivative of speed from the squared-speed
derivative, the squared norm of the orthonormal-frame components, the
split inequality, the endpoint ramp and two-stop envelope, and the
pointwise budget for the polynomial stop-to-stop profile. It also checks
the zero-velocity consequence of matching velocities in orthogonal
directions. The covariant derivative decomposition, curvature bound,
traversal-time formulas, and critical analytic lift regularity use
conventional calculus and differential geometry.
[TrapSteeringBudget.lean](../../formal/FractalSync/TrapSteeringBudget.lean)
