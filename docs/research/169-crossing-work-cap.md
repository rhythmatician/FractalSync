# Sustained work and Shore arrival

The duration and kinetic-action bounds in [#169](https://github.com/rhythmatician/FractalSync/issues/169) use an ideal geometric lower bound on travel. A separate necessary condition comes from how much signed mechanical work the applied controls can supply before the trajectory first reaches the Shore.

## Work available on the approach

On the impulse-free interval $[t_0,t_1]$, use the graph metric $G$ along the path. Write $s=\|v\|_G=\sqrt{2K}$, $L=\int s\,dt$, $T=t_1-t_0$ and $W_c=\int v^{\mathsf T}Q_c\,dt$. Assume the speed and signed power are integrable. A certified almost-everywhere bound $\|Q_c\|_{G^{-1}}\le F$ gives

$$
W_c\le FL.
$$

This follows by integrating the existing primal–dual Cauchy–Schwarz inequality. At any nonzero speed, the aligned covector $Q_c=(F/s)Gv$ has dual norm $F$ and power $Fs$, so the pointwise bound is attained. If a separate certified cap $s\le V$ holds almost everywhere, then $W_c\le FVT$. This second bound uses actual length bounded **above** by $VT$. The radial and displacement bounds supply lower bounds on length and enter the separate inequality $B\le VT$. Using $FB$ as an upper bound on control work would incorrectly exclude some paths.

## Required work at the Shore

For a first crest event at $c_1$, let $D(c_1)=0$, $\rho_0=\sqrt{D(c_0)^2+\varepsilon^2}$ and let $K_*$ be a required minimum arrival kinetic energy. Under the existing signed energy ledger, the necessary work deficit is

$$
R=\kappa\log_2(\rho_0/\varepsilon)
  +U_{\rm wall}(c_1)-U_{\rm wall}(c_0)
  +D_f+K_*-K_0.
$$

Here $D_f$ is integrated frictional dissipation, nonnegative for positive semidefinite friction. The sign of the wall difference follows the actual endpoints. When the friction integral is unknown but known nonnegative, the optimistic lower deficit sets $D_f=0$. A certified lower bound on the remaining terms can also be used when the arrival set or crest location is uncertain. The exact deficit satisfies $R\le W_c$ for an impulse-free approach meeting $K_*$, so $R>FVT$ excludes it by the deadline. If $F,V>0$, the conditions combine to give

$$
T\ge \max\left(\frac{B}{V},\frac{\max(0,R)}{FV}\right).
$$

Use a bound $B$ for the same interval and event as the work comparison. For a through-Shore deadline, evaluate the required crest budget on its first-arrival prefix. The full path's travel bound can separately use the sum of start and end radial costs when crossing is established. Meeting the crest budget establishes neither the approach direction nor later crossing. A positive $K_*$ constrains speed at the event; it does not require any outward normal velocity.

An explicitly accounted impulse at fixed position with pre-event speed at most $V_-$ and dual norm at most $J_{\max}$ can add at most $V_-J_{\max}+J_{\max}^2/2$ to kinetic energy. Use the actual signed impulse work in a replay ledger or its bound in a deadline test. This needs a cap at the pre-event state itself. An almost-everywhere continuous power cap does not include instantaneous impulses.

## An impulse-free rest bound without a speed cap

Suppose the approach starts with $K_0=0$, the total potential $U(t)$ never falls below $U(t_0)$ on the interval, and the continuous total energy is differentiable on the open interval. With force bounded by $F$ and nonnegative frictional dissipation rate, the excess $e(t)=K(t)+U(t)-U(t_0)$ is nonnegative, and

$$
e'(t)\le F\sqrt{2K(t)}\le F\sqrt{2e(t)}.
$$

The scalar comparison applies pointwise on the open interval. Differentiate $\sqrt{2e+\delta^2}$ for $\delta>0$, bound its derivative by $F$, and use the mean-value theorem. Sending $\delta$ to zero gives $e(t_1)\le F^2T^2/2$. A crest rise plus minimum arrival kinetic energy therefore requires $\Delta U+K_*\le F^2T^2/2$. This envelope is attained by constant aligned force from rest in a constant metric with constant potential and zero friction. An actual route whose total potential dips below the initial value lacks the hypothesis for this comparison. The signed work cap remains available there if its own hypotheses hold.

## Analytic example

Take the conformal restriction of the flat Shore model with $a=0$, $\varepsilon=0.01$, $|D(c_0)|=0.02$, $\kappa=4$, wall and friction disabled, and start at rest. The radial distance to the Shore is $\operatorname{asinh}(2)\approx1.44364$ in metric units. The potential rise is $4\log_2\sqrt5\approx4.64386$ in energy units. With force cap $F=2$ and speed cap $V=1$, the geometric duration bound is about $1.44364$ and the work duration bound is about $2.32193$. At a deadline of $2$, the geometric bound alone leaves the task open while the work cap is only $4$ and rules out the arrival. Without a speed cap, the impulse-free rest envelope still gives $T\ge\sqrt{2(4.64386)}/2\approx1.524$ under the no-potential-drop condition.

If the approach is required to follow the straight normal segment without reversals, its actual length is the radial distance. In that specific case $FL\approx2.88727<4.64386$, which rules out the route at any duration. At rest, the intrinsic potential slope opposing the inward normal initially equals $\kappa|D(c_0)|/(\rho_0\ln 2)\approx5.162$. A cap of $2$ cannot supply inward acceleration at that starting point. This local force comparison concerns the initial normal direction. It supplies no conclusion about a route with a different momentum or an intervening potential drop.

## Checked scope and remaining issue work

[ShoreCrossingWorkBound.lean](../../formal/FractalSync/ShoreCrossingWorkBound.lean) checks the sustained power and work inequalities, impulse jump cap, full crest deficit with minimum arrival kinetic energy, combined duration bound, and scalar rest envelope. The physical application of the rest envelope uses `hasDerivAt_canonicalEnergy` and the pointwise force pairing. The signed energy identity and exact Shore/wall potential difference come from the existing mechanics package. The crossing-specific geometry and integrated kinetic action remain separate bounds. [Potential forces and work–energy](159-potential-controls.md), [absolutely continuous crossing bounds](160-absolute-continuous-paths.md).

The [Rust feasibility check and replay trial](169-navigation-feasibility.md)
evaluate necessary conditions on the canonical sample clock. Hard curriculum
exclusions still need certified target and metric correspondence, an applicable
cap, and event accounting when impulses are present. The navigation and
training consumers remain open. [Navigation bounds](https://github.com/rhythmatician/FractalSync/issues/174),
[work-energy ledger](https://github.com/rhythmatician/FractalSync/issues/166).


## A sampled-action condition

For a finite accepted history with durations $h_i$, associate a length
$ℓ_i$ and the speed $v_i$ used in the sampled kinetic action. If a
certified arrival bound $B$ satisfies $B\le\sum_i ℓ_i$ and each interval
satisfies $ℓ_i\le h_i v_i$, then the sampled action obeys
$\sum_i h_i v_i^2/2\ge B^2/(2\sum_i h_i)$ for positive total duration.
The Lean theorem uses weighted finite Cauchy–Schwarz, so unequal durations
are allowed. To use it with the runtime ledger, a provider and update
certificate must establish the per-interval length coverage for the actual
sampled speeds. Endpoint speeds alone may miss motion between the samples.

If an interval certificate instead gives $ℓ_i\le h_i v_i+e_i$, where
$e_i$ bounds the length missed by the sampled speed, subtract the
total allowance before applying the action bound. For
$T=\sum_i h_i>0$ and nonnegative sampled speeds, the resulting floor is

$$
\sum_i h_i v_i^2/2\ge\frac{\max(0,B-\sum_i e_i)^2}{2T}.
$$

For example, let $w(t)$ be the actual intrinsic speed on
$[t_i,t_i+h_i]$. If $w(t)-v_i\le b_i+L_i(t-t_i)$ almost everywhere,
integrating gives $e_i=h_i b_i+L_i h_i^2/2$. Here $b_i$ can bound the
discrepancy between the true starting speed and the sampled speed, and
$L_i$ can bound the subsequent growth of intrinsic speed. An impulse or
any other event that changes the speed estimate's premises needs its
own interval boundary. A bound across an uninterrupted interval needs
evidence for the metric provider along the path and for the velocity
used to evaluate the intrinsic speed. The intervals must cover the same
history and metric as $B$. The Lean theorem
`sampledAction_lower_bound_of_excess` takes the resulting length
allowances as hypotheses. Rust now checks that the panels cover the
arrival prefix on the canonical sample clock, accumulates each speed
gap and growth allowance upward, and rounds the resulting action
floor downward. The unequal-panel test uses a speed jump at a panel
boundary and refuses a gap, an incomplete prefix, and an estimated
target length. Production use still needs independent bounds for
the initial speed gap and speed growth on each RK interval.

[Sampled-action proof](../../formal/FractalSync/SampledAction.lean)
