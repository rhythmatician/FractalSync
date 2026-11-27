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

| Outer steps | Chords | Positive-side error (rad) | Crossing error (rad) |
| ---: | ---: | ---: | ---: |
| 8 | 32 | 0.0200704 | 0.0647522 |
| 16 | 64 | 0.0100064 | 0.0326058 |
| 32 | 128 | 0.00499527 | 0.0163255 |

The conformal angle differs from the full graph angle by 1.37 radians on
the positive-side route and 1.49 radians on the crossing route at these
resolutions.

```powershell
rtk proxy cargo test --release -p runtime_core --features experimental-operational-field --lib frame_transport -- --nocapture
```

[Compiled calibration](../../runtime-core/src/operational_engine/frame_transport.rs) ·
[Transport issue](https://github.com/rhythmatician/FractalSync/issues/179) ·
[Gesture-epoch issue](https://github.com/rhythmatician/FractalSync/issues/170)
