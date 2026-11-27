# Transport along committed endpoint chords

The experimental `FrozenRollout` can evaluate an oriented unit frame along
chords joining its committed mechanical endpoints. The order and endpoints
come from the integrator. Each chord is a declared affine reconstruction
between those samples. The rollout that produced the trace evaluates the
frame with its frozen field, profile, and reference policy. [Committed trace](../validation/bounded-operational-engine.md#committed-mechanical-endpoints)

For a chord with displacement \(\Delta c\) and duration \(h\), the update
starts with a metric-unit direction \(d\). It forms the trial direction
\(d+h\omega J_Gd-\Gamma(\Delta c,d)\) using the start jet and a held
signed turn rate \(\omega\). It normalizes the trial in the endpoint metric
and constructs its positive quarter-turn companion. The duration comes from
the integrator profile, even when rounded absolute times coincide. The
candidate refuses mismatched provenance, unavailable endpoint geometry, and
numerical cases without adequate factorization or nonvanishing margin.
[Endpoint-normalized update](159-accumulated-dynamics.md) ·
[Rust diagnostic](../../runtime-core/src/operational_engine/frame_transport.rs)

On the flat conformal fixture with signed field \(F(x,y)=x\) and floor
\(\epsilon=0.08\), the reference parallel-transport angle along a chord is
the exact integral of \(x\,dy/(\epsilon^2+x^2)\). The compiled diagnostic
uses the actual four-substep traces. From 8 to 16 to 32 outer steps, the
angular error falls by at least 35% at each refinement. Renormalizing the
same coordinate direction throughout preserves its unit norm. That control's
angle differs from the reference by over 0.1 radians.

The stationary full-metric fixture uses \(F(x,y)=x+2y\). Its independent
analytic metric confirms unit axes, orthogonality, positive orientation, and
the signed angle for a held turn. The successful trial reads one jet per
endpoint. Refusal checks confirm that an unchanged trace remains reusable.

The same fixture also has an exact moving-chord reference in an oriented
normal/tangent frame. Two moving routes exercise the shear with opposite held
turn signs. [Full-metric angle and calibration](179-anisotropic-transport.md)

```powershell
rtk proxy cargo test --release -p runtime_core --features experimental-operational-field --lib frame_transport
```

The refinement test concerns transport along the endpoint-chord
reconstruction. Validity and quantitative transport error along the
continuous Physics trajectory still need bounds between the sampled
endpoints before the gesture-epoch statistic can consume them.
[Transport issue](https://github.com/rhythmatician/FractalSync/issues/179) ·
[Gesture-epoch issue](https://github.com/rhythmatician/FractalSync/issues/170)
