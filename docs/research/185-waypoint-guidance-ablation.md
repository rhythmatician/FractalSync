# Waypoint guidance under retained-field Controls

A configurable point trap reduced the final waypoint miss on four of six
cardioid routes at the primary settings. The largest reduction was
0.005509 in parameter-plane distance, about 4.8% on that route. Neither trap
nor multiplier guidance improved the hit count. The gains failed to carry
over to the supported period-two routes.
[Issue #185](https://github.com/rhythmatician/FractalSync/issues/185),
[earlier contour Controls trial](185-contour-controls-trial.md).

## Task and ablation

Six route layouts were declared before running the experiment. The layouts
are distinct from the starts in the earlier contour trials. The two chart
maps place the same layouts in the cardioid and period-two bulb. They
specify a Cartesian start, waypoint and launch heading for each case.
The endpoint objective is Euclidean distance to the waypoint after 1.2 s.
The objective uses the waypoint coordinates alone. The best miss over
committed samples gives a second measure for crossing a goal radius.
[Route definitions and Rust fixture](../../runtime-core/examples/waypoint_guidance_ablation.rs).

All arms start from the same mechanical state with intrinsic speed 0.3.
They use 10 ms outer steps, four held RK4 substeps, passive drag 0.05, and
ordinary Controls v2 velocity feedback with gain 8. Desired intrinsic
speed is capped at 0.3 and tapers within the waypoint neighborhood using
the norm of the displacement in the current metric. Grip, brake,
impulse and conservative potentials are zero. Each arm uses the retained
prepared provider `9721bb7647a0b8e6` and its archived refusal policy.
[Prepared provider and reference policy](../validation/bounded-operational-engine.md).

Every 60 ms the same controller tries the waypoint heading, the current
velocity heading, and both signs of the Map normal and tangent. The Map
arm also tries the two signs of a 45-degree bisector in the metric. The
other arms replace those two slots with the two signs of either the
multiplier-magnitude tangent or the point-trap tangent. The trap is
\(F(c)=|P_2(c)-(1/4+i/8)|^2\), with no closest-index selection. It is one
configuration from the preceding finite recurrence study. The
multiplier candidate requires its component chart, and both candidates
require a resolved tangent. When the replacement is unavailable, the
controller uses the Map bisector and records the fallback.
[Trap configuration and finite recurrence limits](185-finite-recurrence-controls.md).

Candidate headings use copied mechanical state and the same Controls
and throttle ceiling as the committed path. The controller commits the
first 60 ms of the candidate with the smallest forecast miss, using a
stable candidate order to break ties. Each trial starts with the same
number of candidate slots. A provider refusal in a forecast disqualifies
that forecast. A launch or execution refusal retains the partial trace
and gets no final miss.

## Results

The primary setup uses a 120 ms forecast and a 0.25 throttle ceiling.
Changes are alternative minus Map in parameter-plane units, so negative
means improved. A difference of at most \(10^{-6}\) in either direction
counts as a tie. The table counts completed pairs.

| Chart | Replacement | Median change | Range | Wins / ties / losses |
| --- | --- | ---: | ---: | ---: |
| Cardioid | Multiplier | -0.000055 | -0.001067 to +0.000167 | 3 / 2 / 1 |
| Cardioid | Point trap | -0.000207 | -0.005509 to 0 | 4 / 2 / 0 |
| Period-two bulb | Multiplier | 0 | -0.000004 to +0.001603 | 1 / 1 / 1 |
| Period-two bulb | Point trap | 0 | 0 to +0.001615 | 0 / 2 / 1 |

The Map arm's closest samples ranged from 0.015649 to 0.195889 across the
six cardioid routes, and from 0.028460 to 0.085118 across the three
supported bulb routes. No sample in any arm reached within 0.005 or 0.010
of its waypoint under any setting. The Map arm selected its bisector at
12 of 20 replans on bulb route d in the primary setup. The controller
never selected the trap tangent across the supported bulb routes.
Replacing the bisector made route d worse by 0.001615.

The provider refused the starts of bulb routes a, c and e with
`GeometryUnavailable` in every arm and setting. These are three of the
six declared bulb cases. The comparison retains them as shared
launch refusals. There were no forecast refusals, subsequent committed
refusals, or unavailable tangent replacements on the supported routes.
The win and loss counts therefore describe only the covered patch.

| Change from primary | Cardioid multiplier | Cardioid trap | Bulb multiplier | Bulb trap |
| --- | ---: | ---: | ---: | ---: |
| Double forecast to 240 ms | 3 / 2 / 1 | 3 / 2 / 1 | 0 / 2 / 1 | 0 / 2 / 1 |
| Double throttle ceiling to 0.50 | 3 / 2 / 1 | 3 / 2 / 1 | 0 / 2 / 1 | 0 / 2 / 1 |

The largest trap gain, on cardioid route c, shrank from 0.005509 to
0.003526 with the longer forecast and to 0.000959 with the higher
throttle ceiling. The longer forecast improved the largest multiplier
gain from 0.001067 to 0.002455 on cardioid route a. The winner and effect
size depend on the route and the steering budget.

Keep the two fields as opt-in diagnostics. Drop both proposals for
changing the Player observation or default steering on this evidence.
The trial gives a direct measure of task value under Controls. The mixed
effects, shared provider refusals and small number of route layouts leave
transfer open. Prior matched-speed maps, animations and the frozen Julia
comparison cover separate trajectories. This trial scores mechanical
waypoint error. Perceptual and predictive gain remain open.
[Navigation maps and animations](185-navigation-field-comparison.md#reproduce-the-trial),
[frozen Julia comparison](185-contour-controls-trial.md#frozen-julia-comparison).

## Reproduce

Set `$waypointAsset` to the retained `geometry145-retained-fine.bin`.
From the repository root, write the replay and verify matched starting
states, throttle ceilings, sample counts and distance calculations.

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
rtk cargo build --offline --release -p runtime_core --features experimental-operational-field --example waypoint_guidance_ablation
rtk proxy .\target\release\examples\waypoint_guidance_ablation.exe $waypointAsset docs/validation/bounded-reference-policy.json |
    Set-Content -Encoding utf8 tmp/185-waypoint-guidance.json
rtk proxy .\.venv\Scripts\python.exe formal/checks/check_185_waypoint_guidance.py tmp/185-waypoint-guidance.json
```

The replay records every committed state and every refusal. The checker
prints per-route signed changes, selected replacement counts, goal hits,
and full refusal tallies. The replay was byte-identical on
`67b887d` and `891cc86` when captured with the same PowerShell command.
