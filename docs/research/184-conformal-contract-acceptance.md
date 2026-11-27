# Conformal ruler proof contracts

The conformal candidate in [#184](https://github.com/rhythmatician/FractalSync/issues/184) uses the curvature $-1$ Poincaré density $\lambda$ of each open component, with raw ruler $r=1/\lambda$. The quadratic floor is $R_\varepsilon(r)=\sqrt{r^2+\varepsilon^2}$ for $\varepsilon>0$. The mechanical metric under study is $R_\varepsilon^{-2}|dc|^2$ in the parameter plane. The distance and rank-one graph results in [#160](160-distance-geometry.md) keep their original hypotheses and remain the baseline.

## Conditional comparison and curvature

For a given positive density, flooring its reciprocal gives $R_\varepsilon(1/\lambda)^{-1}=\lambda/\sqrt{1+(\varepsilon\lambda)^2}$. Assume $d\ge0$ and $d/2\le r\le 2d$. With the same $\varepsilon$ for both rulers, squaring shows $R_\varepsilon(d)/2\le R_\varepsilon(r)\le2R_\varepsilon(d)$. Taking reciprocals and multiplying by the same Euclidean speed gives a factor-two conformal speed comparison. Integration gives the same factor for the length of any common path whose two speed integrands are integrable and satisfy the comparison on the parameter interval. Squaring gives a factor-four kinetic energy comparison at fixed velocity. The associated scale coordinates with the same positive reference length differ by at most one octave.

The bracket comes from Schwarz's lemma and Koebe's quarter theorem on inland components. A separate lift through the exterior coordinate gives the bracket in the collar specified in the gluing note. The Lean transfer takes the bracket as a hypothesis. In particular, the comparison supplies no bound on gradient or Hessian errors, and applies only where the bracket is available. [Inland and exterior collar hypotheses](184-shore-floor-gluing.md), [Douady and Hubbard's exterior coordinate](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf#page=64).

For a smooth positive ruler $R$, the metric pairing is $\langle v,w\rangle/R^2$, the inverse action is $R^2v$, and the connection term is

$$
\Gamma(v,w)=\langle v,w\rangle\nabla\log R
 -\langle\nabla\log R,w\rangle v-\langle\nabla\log R,v\rangle w.
$$

These are the $a=0$ specialization of the graph metric. In Cartesian coordinates, writing $(p,q)=\nabla\log R$, the actual curvature commutator on the coordinate plane, divided using the metric Gram determinant, yields $K=R^2(\partial_xp+\partial_yq)$. The positive ruler and actual first derivatives of $p,q$ specify the inputs to this checked identity. The quadratic- and quartic-floor proofs give the relevant log derivatives and weighted-trace identities from raw ruler jets. [Conformal comparison and curvature proofs](../../formal/FractalSync/ConformalRulerContracts.lean), [quadratic-floor curvature](184-conformal-floor-curvature.md), [quartic-floor curvature](184-quartic-ruler-floor.md).

## Flat normal fixture

In Euclidean normal and tangential coordinates $(n,y)$, take the exact raw model $r(n,y)=b|n|$ for a constant $b>0$. The squared floored ruler is $\varepsilon^2+b^2n^2$ on both sides, so the metric is smooth through $n=0$. The actual first and second derivatives of $\log R_\varepsilon$ in the normal coordinate give

$$
K(n)=b^2\frac{\varepsilon^2-b^2 n^2}{\varepsilon^2+b^2 n^2}.
$$

The metric is independent of $y$ and its first derivative vanishes on $n=0$. Along the straight Shore, $y$ advances by $\varepsilon s$ in intrinsic unit-speed arclength $s$. The normal Jacobi equation is $j''+b^2j=0$. Its nonzero solution with $j(0)=0$ is a multiple of $\sin(bs)$, so its first positive zero is at $s=\pi/b$. The tangent equation $j''=0$ contributes no nonzero field vanishing at both ends. Thus the ideal fixture's first conjugate separation is $\pi\varepsilon/b$ in Euclidean tangential distance. The sine derivatives and first-zero statements in both coordinates are checked in Lean. The Jacobi and conjugacy interpretation uses the standard geometric equation, with the actual $b=1$ flat-Shore coordinate residual already checked. [Flat-Shore geodesic and Jacobi proofs](../../formal/FractalSync/FlatShoreGeodesics.lean).

The exact model assumes a straight Shore and the same linear slope on either side. For distinct one-sided slopes $b_+,b_-$, the second derivative at the join of $(\varepsilon^2+b_\pm^2n^2)^{-1}$ approaches $-2b_\pm^2/\varepsilon^4$ from each side. A positive floor therefore does not imply a $C^2$ metric. Since $R_\varepsilon\ge\varepsilon$, a crossing with integrable speed has conformal length at most its Euclidean length divided by $\varepsilon$. This checked length bound requires no differentiability at the crossing. A mechanical trajectory through the Shore instead needs a regular metric and force field on a common certified region.

## Dependency audit

| Area | Checked in Lean | Classical or conditional input |
| --- | --- | --- |
| General positive ruler and comparisons | `ScaleMetric`, `ContinuousMechanics`, `ConformalRulerContracts` | Use the actual ruler gradient in the connection and actual log-gradient derivatives in the curvature. Assume the pointwise density bracket and path integrability. |
| Period one | `MainCardioidRuler` verifies parameter-map derivative, tangent speed, chord brackets, and exact cusp-axis distance and ruler | The multiplier chart identifies the component's Poincaré ruler. The smooth-boundary asymptotics and off-axis cusp calculations use the conventional derivations in the [main-cardioid study](184-main-cardioid-ruler.md). |
| Period two | `PeriodTwoAttachment` verifies the quadratic period factor, cycle multiplier, disk criterion, circle distance and ruler, and conditional quartic attachment gluing | The disk density fixes normalization. The identification of signed local extensions and interpretation as a smooth attachment metric are in the [period-two study](184-period-two-attachment.md). |
| Exterior | `ExteriorRulerBounds` verifies guarded orbit and logarithmic derivative increments, finite envelopes and ratio enclosure. `ExteriorRulerJets` verifies common escape disk bounds, holomorphic tail and center derivative estimates | The Böttcher coordinate identifies the exterior potential and density. The identification of the analytic limit jets with that coordinate and the interval arithmetic experiment have the scope stated in the [exterior ruler notes](184-exterior-ruler-jets.md). |
| Boundary extension and smoothing | `ConformalFloorCurvature`, `QuarticRulerFloor`, and `ShoreFloorGluing` verify actual log-floor derivatives, weighted-trace identities, and $C^3$ gluing from uniform distance-power bounds on derivative fields | The domain-specific collar estimates and their global application to the canonical ruler are conventional. The [gluing study](184-shore-floor-gluing.md) also explains the quartic cusp obstruction to $C^4$. |
| Mechanics and changing coordinates | `CoordinateCovariance`, `NonlinearCoordinates`, and `ContinuousMechanics` give covariance and work-energy; `OperationalFieldGluing` and `OperationalTrajectoryCertificate` cover selected provider and trajectory obligations | Apply only with their actual differentiability, positive-ruler, force regularity and compact-domain bounds. Operational interpolation and convergence require a specified asset and corresponding jet and error certificates. |

The acceptance scope is the conditional conformal layer, exact period-one and period-two fixtures, exterior guard and jet bounds, and regularity obligations above. General multiplier uniformization and Böttcher limit identification remain classical inputs. The ideal normal fixture makes no claim about a generic fractal Shore. Provider fidelity and representable arithmetic have their own scope in [#181](https://github.com/rhythmatician/FractalSync/issues/181). The candidate provider and Physics comparison belong to [#182](https://github.com/rhythmatician/FractalSync/issues/182) and [#183](https://github.com/rhythmatician/FractalSync/issues/183). Fractional conformal powers and further distance-specific global cut-locus and curved-Shore classification remain separate research.
