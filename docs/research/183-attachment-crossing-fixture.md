# Analytic attachment crossing through frozen Physics

This [#183](https://github.com/rhythmatician/FractalSync/issues/183) fixture extends the local ruler experiments with a damped real-axis crossing of the period-two bulb and main-cardioid attachment \(c_*=-3/4\). All three profiles use the existing frozen Physics engine. The third profile exercises the analytic two-component quartic floor. The two distance profiles use the same tangent-line comparison field. [Rust example](../../runtime-core/examples/attachment_ruler_crossing.rs), [period-two comparison](183-period-two-physics-ablation.md).

## Fields and scope

Write \(x=\operatorname{Re}(c-c_*)\) and \(s=\sqrt{1-4c}\) on the local branch with positive real part. The locally extended rulers are

\[
r_1=(1-|1-s|^2)|s|/4,\qquad
r_2=1/8-2|c+1|^2 .
\]

Their positive sets in the patch are the main cardioid and period-two bulb and are disjoint. With \(r=\max(0,r_1,r_2)\), the local analytic ruler is \(R=(\epsilon^4+r^4)^{1/4}\), including a constant-floor region wherever both rulers are nonpositive. This is the two-component fixture from the attachment analysis. It does not model the exterior or identify a realm elsewhere in that region. [Attachment gluing and axis formulas](184-period-two-attachment.md).

The existing engine accepts a scalar \(D\) and forms \(\rho=\sqrt{D^2+\epsilon^2}\). Preparing the effective scalar

\[
D_{\rm eff}=-\frac{r^2}{\sqrt{\sqrt{r^4+\epsilon^4}+\epsilon^2}}
\]

makes \(\rho=R\) in exact arithmetic before spline preparation. Squaring the expression gives \(D_{\rm eff}^2=\sqrt{r^4+\epsilon^4}-\epsilon^2\). The rationalized expression avoids that subtraction in the implementation. The sign of \(D_{\rm eff}\) is an encoding choice. The physical ruler is \(C^2\) according to the gluing result, while the effective scalar itself can lose second-derivative continuity at a join into the constant-floor region.

The distance comparison uses \(D=x\), signed distance to the tangent line. Its magnitude agrees with the ideal local distance on the short real-axis segment inside either component, where the attachment is the nearest Shore point. Off the axis, the tangent line is a synthetic extension. The distance graph uses \(\lambda^2=1\) and both conformal profiles use \(\lambda^2=0\). All profiles use floor \(\epsilon=2^{-7}\) and a \(641\times65\) coefficient patch with spacing \(2^{-13}\), or 64 grid intervals per floor unit. The provider requires the full stencil at every stage. [Intracomponent distance formulas](184-period-two-attachment.md), [cardioid negative-axis distance](184-main-cardioid-ruler.md#comparison-on-the-same-paths).

The fixture supplies an empty synthetic cut list. The engine's built-in refusals and disk guard still apply. An optional separate probe uses the retained field and its original policy. On this run it returned \(D=-3.8091\times10^{-4}\) at the attachment with reference-distance accuracy marked `unknown`. The retained sample supplies no verification of true-Shore distance or of an archived crossing. It does not change the fixture's fields or policy. [Archived field and policy provenance](../validation/bounded-operational-engine.md).

## Axis reference and measured crossing

All three profiles start at \(c_*-2\epsilon\) with coordinate velocity \((0.05,0)\), drag \(\beta=0.2\), and held covector zero. The wall and scale potentials are disabled for the reference calculation, with \(\mu=\kappa=0\). The hard disk constraint remains active. The two tested time steps are \(0.005\) s and \(0.0025\) s, with four RK4 substeps per attempted step and a total duration of 1 s.

Reflection fixes the real axis for each ideal metric. Along that axis let \(m(x)=\sqrt{G_{xx}(x)}\) and \(V_0=m(-2\epsilon)\cdot0.05\). The ideal forward intrinsic speed \(V\) obeys \(\dot V=-\beta V\). If \(A_*=\int_{-2\epsilon}^0m(x)\,dx\), the crossing time is \(-\log(1-\beta A_*/V_0)/\beta\) and the crossing coordinate speed is \(\epsilon(V_0-\beta A_*)\). The endpoint is found by inverting the traveled length \(V_0(1-\exp(-\beta t))/\beta\). The reference integrates each side separately. Increasing from 256 to 512 Simpson panels changed the reference crossing times by at most \(1.27\times10^{-12}\) s and the endpoints by at most \(3.89\times10^{-14}\). This is an independent numerical evaluation of the ideal formulas.

The compiled unit test checks the encoded physical ruler against the independent real-axis formulas at 257 grid points. A further comparison of the prepared ruler with its analytic axis value and derivatives uses 1,153 points, including spline joins, off-grid points, and the attachment. Over the interval \([-3\epsilon,3\epsilon]\), the maximum discrepancies in \(R\), \(R'\), and \(R''\) were \(1.37\times10^{-7}\), \(4.17\times10^{-5}\), and \(3.01\times10^{-2}\). These are sampled interpolation and arithmetic discrepancies, with derivatives measured in the unscaled coordinate \(x\).

| Profile | Ideal crossing time, s | Fine crossing time, s | Fine coordinate speed | Fine intrinsic speed | Fine endpoint \(c\) on axis |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tangent-line distance graph | 0.426074592 | 0.426074747 | 0.033521726 | 4.290781 | \(-0.729420894\) |
| Tangent-line distance conformal | 0.531677707 | 0.531677846 | 0.020105091 | 2.573452 | \(-0.738893671\) |
| Quartic dynamical conformal | 0.526622125 | 0.526617777 | 0.022832042 | 2.922501 | \(-0.738427914\) |

Crossing time and coordinate speed use linear interpolation between bracketing committed states. Intrinsic speed projects that velocity onto the prepared axis metric at the attachment. Both resolutions completed one crossing in every profile, with transverse position below \(2\times10^{-18}\) in the fine runs. Step halving changed crossing time by at most \(4.09\times10^{-7}\) s and matched-time states by at most \(1.28\times10^{-13}\) in position and \(5.28\times10^{-13}\) in velocity. The maximum coarse energy-work residual was \(2.98\times10^{-10}\). Each coarse run used 18,400 prepared geometry queries during its attempted steps.

For the quartic profile, the fine crossing was \(4.35\times10^{-6}\) s earlier than the ideal axis prediction, and the fine endpoint differed from that reference by \(3.29\times10^{-7}\). Its time-step comparison is substantially smaller than this ideal-reference difference. Its nonzero speed demonstrates a crossing of the prepared floor fixture over the tested schedule. Equal initial coordinate states give unequal intrinsic energies across metrics, so the numbers do not compare equal-energy maneuvers.

## Reproduce

```powershell
cargo test --offline -p runtime_core --features experimental-operational-field --example attachment_ruler_crossing
cargo run --offline -p runtime_core --features experimental-operational-field --example attachment_ruler_crossing
cargo run --offline -p runtime_core --features experimental-operational-field --example attachment_ruler_crossing -- --probe-retained tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json
```

Keep the quartic candidate opt-in. This local axis crossing supports a prepared research comparison, while offset attachment crossings, a certified full-domain provider with seam evidence, visual response, and a geometry adoption decision remain open. [#183](https://github.com/rhythmatician/FractalSync/issues/183), [#182](https://github.com/rhythmatician/FractalSync/issues/182).
