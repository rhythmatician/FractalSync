# Main-cardioid and period-two contours under bounded Controls

This [#185](https://github.com/rhythmatician/FractalSync/issues/185) trial compares a natural contour and a prepared Map iso-scale contour through the same Rust Physics and Controls v2 drive. It measures how two guidance fields behave when their tangents become desired velocities and the existing Controls mapping supplies the force. [Rust fixture](../../runtime-core/examples/contour_controls_trial.rs), [earlier pointwise navigation comparison](185-navigation-field-comparison.md).

## Matched setup

Both runs use the retained prepared field `9721bb7647a0b8e6` and its unchanged reference-refusal policy. Both start at \(c=-0.45+0.1i\) with intrinsic speed \(0.3\) in their respective upward contour tangents. The initial coordinate velocity therefore differs. For the natural field, the level is \(|1-\sqrt{1-4c}|^2\), and its tangent uses the analytic derivative of the main-cardioid multiplier. For the Map alternative, the level is the prepared signed scalar \(D\). Within this patch \(D<0\), so its level sets are also the level sets of \(\sigma=\log_2(d_{\rm ref}/\sqrt{D^2+\epsilon^2})\). Its tangent comes from the stored spline jet. [Main-cardioid chart](184-main-cardioid-ruler.md), [retained asset and policy](../validation/bounded-operational-engine.md).

At each of 60 outer steps of \(0.01\) s the fixture computes a metric-unit contour tangent and sets the desired velocity to \(0.3\) times that tangent. A proportional velocity error chooses the direction and throttle, capped at one, with gain 8. The unmodified Rust `MotionControls` computes the drive covector, and `FrozenRollout` holds it across the four RK4 substeps. The profiles use the same graph metric, neutral grip and brake, passive drag \(0.05\), and zero impulse or conservative potentials. This example's feedback is only the caller that selects ordinary Controls inputs. It has no position correction toward the initial level set and no path constraint.

## Main-cardioid results

Level errors below are maximum absolute changes from the starting level along each committed trajectory. Compare within columns since the two level functions have different units.

| Guidance | Final \((\operatorname{Re}c,\operatorname{Im}c)\) | Peak throttle | Peak intrinsic speed error | Peak multiplier level error | Peak prepared Map scalar error |
| --- | --- | ---: | ---: | ---: | ---: |
| Multiplier tangent | \((-0.435317,0.148857)\) | 0.05070 | 0.00199 | 0.001282 | 0.002173 |
| Map iso-scale tangent | \((-0.433990,0.148634)\) | 0.05176 | 0.00202 | 0.000884 | 0.000868 |

Both commands stayed well below the throttle cap. The natural tangent did not produce the smaller multiplier-level error in this particular finite-response trial. Its exact tangent describes an instantaneous contour direction. The inertial rollout, discrete holds, and velocity-only feedback allow the path to drift from that contour.

## Period-two results

The second pair starts at \((-0.9,0.1)\), with the same intrinsic speed, time step, force mapping, and feedback parameters. The exact period-two multiplier is \(4(c+1)\), so the natural level is \(16|c+1|^2\) on the open bulb. Its contours also agree with the exact bulb-distance contours. The prepared Map contour is a distinct numerical estimate. Their initial coordinate velocities were \((-0.0230333,0.0230333)\) and \((-0.0229268,0.0231404)\). [Period-two multiplier and Shore distance](184-period-two-attachment.md#exact-bulb-and-shore-distance).

| Guidance | Final \((\operatorname{Re}c,\operatorname{Im}c)\) | Peak throttle | Peak intrinsic speed error | Peak multiplier level error | Peak prepared Map scalar error |
| --- | --- | ---: | ---: | ---: | ---: |
| Multiplier tangent | \((-0.914414,0.112986)\) | 0.05082 | 0.00204 | 0.001454 | 0.000317 |
| Map iso-scale tangent | \((-0.914397,0.112997)\) | 0.04910 | 0.00202 | 0.001539 | 0.000336 |

Both bulb paths finished close together. The natural tangent had slightly less multiplier-level drift in this run. The exact geometric coincidence of the bulb contours limits what this pair can say about guidance independence.

All four trials stayed inside their checked multiplier chart and covered negative-scalar Map patch at accepted samples. Candidate generation rejects points outside either multiplier chart and unresolved tangents. The run tests both cases in both charts and separately confirms refusal of the archived reference cut at \((-1,0)\).

Run from the repository root with the retained asset.

```powershell
cargo run --offline -p runtime_core --features experimental-operational-field --example contour_controls_trial -- tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json
```

The JSON includes all four trajectories with 61 samples each, with the bulb pair under `period_two`.

## Frozen Julia comparison

Save the replay JSON in UTF-8 as `tmp/navigation-185/controls-trial.json`, then
run the same frozen renderer used by the Julia sensitivity probe from the
repository root.

```sh
node frontend/scripts/check_julia_sensitivity.mjs --contours tmp/navigation-185/controls-trial.json
```

The diagnostic records 16 frames per path at 40 ms intervals. It uses the
tracked 500-iteration fragment shader with time, view, color and lighting
fixed, at 80×80 RGB8. The table reports normalized RGB RMS distances. The
paired difference compares the two guidance paths at the same mechanical time.

| Component | Median consecutive change, multiplier | Median consecutive change, Map | Largest paired difference |
| --- | ---: | ---: | ---: |
| Main cardioid | 0.02105 | 0.02098 | 0.01635 |
| Period-two bulb | 0.01775 | 0.01752 | 0.01163 |

Consecutive changes have similar medians for both guidance choices in each
component. The paired differences show the rendered paths can diverge even
with equal launch speed and close mechanical endpoints. These measurements
concern one display configuration and two starts. The [navigation maps and
animations](185-navigation-field-comparison.md) give additional kinematic
comparisons. The report retains the shader and replay hashes, presentation
metadata, registered times, and pair distances.

Keep natural contours as opt-in diagnostic guidance. The small, mixed
level-tracking effect and similar image-change medians give no reason to add
a Player observation or change default steering. Configurable recurrence,
route performance, and broader provider and perceptual evidence remain open
under [#185](https://github.com/rhythmatician/FractalSync/issues/185).
