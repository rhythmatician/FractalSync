# Transport in the full graph metric

The endpoint-chord diagnostic measures transport along the affine reconstruction
of the committed substep positions. A varying oriented orthonormal frame gives
an angle one-form for the full graph metric. The exact-real error bound below
applies when the frame, connection, and angle one-form have bounds throughout
a chord. An affine signed field supplies a closed-form calibration of the
compiled diagnostic on moving, sheared routes.

[Endpoint-chord diagnostic](179-endpoint-chord-transport.md)

## Angle and Euler error in a varying frame

Let \(G=S^\top S\), with \(S\) an orientation-preserving matrix factor, for
example the runtime's upper Cholesky factor. Use Euclidean vector and induced
operator norms in this derivation. Write \(J(x,y)=(-y,x)\) and let
\(\Gamma_z(v)d=\Gamma_z(v,d)\). For a metric-unit direction \(d\) moving
with held turn rate \(\omega\), the frame coordinates \(w=S(z)d\) obey
\[
 w'=\bigl(D S_z[z']S^{-1}-S\Gamma_z(z')S^{-1}+\omega J\bigr)w .
\]
Metric compatibility gives \(DG_z[v]=\Gamma_z(v)^\top G+G\Gamma_z(v)\).
Differentiating \(S^\top S\) then makes
\(DS_z[v]S^{-1}-S\Gamma_z(v)S^{-1}\) skew symmetric. In two dimensions it
equals \(\alpha_z(v)J\), where \(\alpha_z(v)\) is its entry in row two and
column one. The lifted angle obeys \(\theta'=\omega+\alpha_z(z')\).

For a chord from \(z_0\) to \(z_1\) with displacement \(\Delta\), duration
\(h\), and \(\tau=h\omega\), put
\[
 B=S_1S_0^{-1}-I,\qquad D=DS_0[\Delta]S_0^{-1},\qquad
 C=S_0\Gamma_{z_0}(\Delta)S_0^{-1}.
\]
The Euler trial in endpoint frame coordinates is the action of
\((I+B)(I+\tau J-C)\) on the old unit coordinates. Endpoint normalization
divides these trial coordinates by their positive Euclidean norm. With
\(k=\tau+\alpha_{z_0}(\Delta)\), the trial matrix has the decomposition
\[
 (I+B)(I+\tau J-C)=I+kJ+R,\qquad R=(B-D)+B(\tau J-C).
\]
If \(\|DS\|\le K\), \(\|D^2S\|\le H\), and \(\|D\alpha\|\le L\)
along the chord, the integral and Taylor bounds give
\[
 \|R\|\le r=\|S_0^{-1}\|
 \left[\tfrac12H\|\Delta\|^2+K\|\Delta\|(|\tau|+\|C\|)\right].
\]
The trial with \(R=0\) has length \(\sqrt{1+k^2}\) and angle increment
\(\arctan k\). For \(r<1\), the perturbed trial has projection at least
\(1-r\) onto that ray and transverse component at most \(r\), so the angle
perturbation is at most \(r/(1-r)\). The chord integral of \(\alpha\) differs
from its start sample by at most \(L\|\Delta\|^2/2\). The resulting one-step
angle error bound is
\[
 \frac{L\|\Delta\|^2}{2}+\frac{|k|^3}{3}+\frac{r}{1-r}.
\]
The cubic term bounds \(|\arctan k-k|\). This estimate is uniform over the
initial unit direction, so it adds over successive substeps with a common
initial frame angle. For an independently enclosed continuous reference path,
the existing Lean line-integral theorem applies to the full-metric one-form
under its path-deviation and derivative premises. The above frame identity
and Euler estimate are conventional derivations. Applying them to a production
trace needs certified metric and derivative bounds and a roundoff budget.
The committed positions provide the chords. A reference-path certificate
must supply the deviation throughout each substep.

[Checked path-replacement theorem](../../formal/FractalSync/ConformalTransportError.lean) ·
[Conformal error budget](179-conformal-transport-error.md)

## Closed-form moving calibration

Take \(F(x,y)=x+2y\), with oriented normal and tangent coordinates
\(u=(x+2y)/\sqrt5\) and \(v=(-2x+y)/\sqrt5\). Put
\[
 m=1+\frac{5\lambda^2}{(\log 2)^2},\qquad
 \rho=\sqrt{\epsilon^2+F^2},\qquad s=\sqrt{\epsilon^2+m F^2}.
\]
The full graph metric is \(a^2\,du^2+b^2\,dv^2\), with \(a=s/\rho^2\)
and \(b=1/\rho\) depending only on \(u\). The positively oriented
orthonormal frame is \((a^{-1}\partial_u,b^{-1}\partial_v)\). In that frame
\(\nabla_{\partial_v}(a^{-1}\partial_u)=(b'/a)b^{-1}\partial_v\), and the
same derivative in the \(u\) direction is zero. The angle one-form is
\[
 \alpha=-\frac{b'}{a}\,dv=\frac{\sqrt5 F}{\rho s}\,dv .
\]
For a chord with \(\Delta F\ne 0\), its integral is
\[
 \frac{\sqrt5\,\Delta v}{\sqrt m\,\Delta F}
 \log\frac{\sqrt m\,\rho_1+s_1}{\sqrt m\,\rho_0+s_0}.
\]
This follows by substituting \(d\rho=F\,dF/\rho\). For constant \(F\),
the integral is \(\sqrt5\,\Delta v F/(\rho s)\).

The compiled check uses \(\epsilon=0.08\) and \(\lambda^2=0.7\). One route
stays on the positive side and another crosses \(F=0\). Each accumulates
the closed-form integral on all four committed substep chords per outer
step, plus the held turn over 0.32 mechanical seconds. The analytic
normal/tangent frame measures the runtime direction and its companion at
every outer endpoint. A conformal calculation uses the same chords so
the test can detect omission of the graph term independently of any change
in the mechanical trajectory. Refinement compares each run to its own
committed chords.

| Outer steps | Chords | Positive-side error (rad) | Positive-side bound (rad) | Crossing error (rad) | Crossing bound (rad) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 0.0200704 | 0.0532005 | 0.0647522 | 0.355654 |
| 16 | 64 | 0.0100064 | 0.0264831 | 0.0326058 | 0.174656 |
| 32 | 128 | 0.00499527 | 0.0132148 | 0.0163255 | 0.0867597 |

The conformal angle differs from the full graph angle by 1.37 radians on
the positive-side route and 1.49 radians on the crossing route at these
resolutions.

### Per-chord angle bound on the affine fixture

In the analytic normal/tangent coframe above, both \(B=S_1S_0^{-1}-I\)
and \(D=DS_0[\Delta]S_0^{-1}\) are diagonal since the oriented normal
and tangent directions are constant. Let their diagonal entries be
\(b_i,d_i\). The \(b_i\) are endpoint scale ratios minus one and
the \(d_i\) are the directional derivatives of the log scales at the
start. With \(k=h\omega+\alpha_0(\Delta)\), the diagonal entries of
\(R=B-(I+B)D+kBJ\) are \(b_i-(1+b_i)d_i\), and its off-diagonal
entries are \(-k b_1\) and \(k b_2\). Its Frobenius norm \(r\) bounds
its operator norm. The unperturbed trial has length \(\sqrt{1+k^2}\),
so when \(r<\sqrt{1+k^2}\) the bound for one chord is
\[
 \arctan\!\frac{r}{\sqrt{1+k^2}-r}
 +\frac{|k|^3}{3}
 +\left|\int_{\mathrm{chord}}\alpha-\alpha_0(\Delta)\right|.
\]
The last term uses the closed-form chord integral. This bound holds
uniformly over starting unit directions. Since the exact chord update
adds an angle independently of the starting direction, the per-chord
bounds add over the full route.

The compiled tests evaluate the bound on each committed chord for both
moving routes above. At each sampled resolution the computed sum covers
the measured frame error and falls by at least 30% under step halving.
The formulas use exact-real arithmetic. The test comparisons use
binary64 evaluations, and the bound covers the chord reconstruction.
Production use would require a roundoff enclosure and separate
continuous-path evidence.

### Perturbing the chord and its held turn

Let \(P=I+B\) and \(M=I-D+kJ\), so the Euler trial for an initial
unit vector \(w\) in coframe coordinates is \(v=PMw\). A perturbed
chord uses its own start and endpoint coframes and coefficients
\(\widehat P,\widehat D,\widehat k\). Compare trials with the same
unit coordinates at their respective starts. The trial discrepancy
satisfies \(\|\widehat v-v\|\le e\) for
\[
 e=\|\widehat P-P\|\big(\|I-D\|+|k|\big)
    +\|\widehat P\|\big(\|\widehat D-D\|+|\widehat k-k|\big).
\]
This follows by subtracting the trial products as
\((\widehat P-P)M+\widehat P(\widehat M-M)\). Thus the endpoint
scale ratio, directional derivative, and turn and connection increment
can supply separate input-error budgets.

Use a nominal lower bound \(\ell=\sqrt{1+k^2}-r>0\) for the trial
length. If \(e<\ell\), the perturbed length is at least \(\ell-e\).
The perturbed trial projects at least \(\ell-e\) along the nominal ray
and at most \(e\) transverse to it in absolute value. The wrapped angle
difference is therefore at most \(\arctan(e/(\ell-e))\). The distance
between the two normalized vectors is at most \(2e/\ell\). To see this,
divide both vectors by the nominal length, which costs at most
\(e/\ell\) in vector discrepancy. Renormalizing the perturbed vector
then costs at most another \(e/\ell\) by the reverse triangle inequality.
If the strict margin fails, the envelope gives no angle or nonvanishing
guarantee.

The Rust fixture compares two separately committed one-chord RK4 traces
at each perturbation size. It changes initial position, velocity and
held turn together and checks both bounds on two coordinate axes and
one oblique direction of the analytic coframe. These are single-chord
comparisons in the declared coframes. To compose them along a path,
one must also propagate the incoming axis error or compare each step
against exact isometric transport. The fixture uses binary64 comparisons
under an exact-real bound.

## Continuous free geodesic on the affine fixture

The affine metric also gives a continuous reference trajectory through
conservation of speed \(V\) and tangent momentum \(p\). The latter follows
because the metric coefficients are independent of \(v\). With the mechanical
potential, wall strength, drag and control set to zero, these constants are
\[
 V^2=a^2 \dot u^2+b^2\dot v^2,\qquad p=b^2\dot v.
\]
Before a normal turning point, let \(\eta=\operatorname{sgn}(\dot u)\). Then
\[
 \dot F=\eta\frac{\sqrt5\,\rho^2}{s}\sqrt{V^2-p^2\rho^2},
 \qquad \dot v=p\rho^2 .
\]
Inverting the flight-time integral of \(1/\dot F\) from \(F(0)\) gives
\(F(T)\). Integrating \(p\rho^2/\dot F\) over the same field interval gives
the tangent displacement. Dividing the angle rate \(\alpha(\dot z)\) by
\(\dot F\) gives \(\eta\,d\arcsin(p\rho/V)/dF\), so the total angle
increment for held turn rate \(\omega\) is
\[
 \eta\bigl[\arcsin(p\rho(T)/V)-\arcsin(p\rho(0)/V)\bigr]+\omega T.
\]

The compiled test uses the default \(\lambda^2=1\), the fixture floor
\(\epsilon=0.08\), initial position \((0.04,0.05)\), initial velocity
\((-0.12,0.4)\), and \(\omega=0.9\) over \(T=0.16\) mechanical seconds.
The normal branch is increasing. The flight-time bracket ends at \(F=0.32\),
where \(|p|\rho/V < 0.7\) bounds the ratio throughout the bracket. The
reference uses bisection and composite Simpson integration of the conserved
quantities. Doubling the integration panels from 512 to 1,024 changes the
reference endpoint and angle by less than \(10^{-11}\). The reference
quadrature uses the scalar metric formulas directly. At \(T=0.16\), it
gives \(F(T)\approx 0.271622591187\), endpoint
\((-0.010211568875,0.140917080031)\), and a connection angle increment of
\(0.251911410360\) radians before adding the held turn.

The compiled transport angle uses the analytic coframe at the compiled
endpoint, and the continuous angle uses the same coframe field at the
reference endpoint. The position error measures the two endpoints in
Euclidean coordinates. An additional angle comparison integrates the exact
one-form along the committed chords and measures the gap to the continuous
reference.

| Outer steps | Chords | Endpoint position error | Transport angle error (rad) | Chord-to-continuous angle gap (rad) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | \(2.05444\times10^{-8}\) | 0.00973903 | \(3.46158\times10^{-5}\) |
| 8 | 32 | \(1.30852\times10^{-9}\) | 0.00490345 | \(8.66566\times10^{-6}\) |
| 16 | 64 | \(8.25603\times10^{-11}\) | 0.00246011 | \(2.16716\times10^{-6}\) |

The final mechanical clock matched the reconstructed reference time of
0.16 seconds to binary64 precision at all three resolutions. These are
numerical observations for this free geodesic before its turn. A certified
error bound would require quadrature and floating-point enclosures as well
as the trajectory hypotheses above.

```powershell
rtk proxy cargo test --release -p runtime_core --features experimental-operational-field --lib frame_transport -- --nocapture
```

[Compiled calibration](../../runtime-core/src/operational_engine/frame_transport.rs) ·
[Transport issue](https://github.com/rhythmatician/FractalSync/issues/179) ·
[Gesture-epoch issue](https://github.com/rhythmatician/FractalSync/issues/170)

## Varying signed turn rates on accepted chords

The prepared-field diagnostic accepts one signed rate per chord in the
committed endpoint trace. It uses the profile duration divided by the
chord count, so even a large absolute timestamp that rounds away the clock
advance leaves the transport duration intact. In the stationary calibration,
a chord of duration `h` with rate `ω` rotates the normalized Euler frame by
`atan(h ω)`. The regression exercises positive, zero, and negative rates,
checks bit-for-bit equality with the held-rate entry point for a constant
schedule, and checks that malformed schedules leave the next valid result
unchanged. The pose and gesture epoch needed by the harmonic statistic
remain a separate runtime input.

## Retracing a moving path with reversed turn rates

In the same affine field, a free-geodesic replay runs for 0.16 mechanical
seconds with `λ² = 0.7`, starting at `(0.04, 0.05)` with velocity
`(-0.12, 0.4)`. Each outer step holds `[0.9, -0.4, 1.4, 0.3]` radians per
mechanical second over its four committed chords. The return replay starts
at the actual forward endpoint with reversed velocity and carries the
forward frame into it. It negates and reverses the rate schedule.

The comparison integrates the exact one-form on the chords of both
replays, since RK4 can return along a slightly different path. Intentional
turn integrates to zero across the two legs. The remaining reference angle
gap measures the chord mismatch. Both the outbound frame error and the
error after the two legs fall by roughly a factor of two with each
refinement.

| Outer steps per leg | Outbound frame error (rad) | Round-trip frame error (rad) | Round-trip reference angle gap (rad) | Return position error |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.00837865 | 0.0168392 | `5.69e-10` | `3.68e-10` |
| 8 | 0.00420407 | 0.00842878 | `1.77e-11` | `1.15e-11` |
| 16 | 0.00210557 | 0.00421632 | `5.54e-13` | `3.59e-13` |

Frame errors use the analytic coframe at each measured endpoint. The
round-trip error can exceed the outbound error even when the two
reconstructed paths nearly coincide. The comparison gives a numerical
calibration of the normalized Euler update on this fixture.
