# Exterior rays and equipotential navigation

The exterior coordinate distinguishes two natural directions. An
external ray holds external angle fixed and changes the Green
potential. An equipotential holds the Green potential fixed and
changes the angle. Both can be traversed at unit speed in the exterior
conformal metric. [Exterior ruler](184-exterior-ruler-bounds.md)

## Exact directions and rates

Use the local holomorphic coordinate $F=\log\Phi=g+i\theta$,
where $g>0$ and $u=F'\ne0$ in the finite exterior. The unfloored
ruler is $\rho=g/|u|$. Choose the velocities
$$
v_{\mathrm{ray}}=g/u,\qquad v_{\mathrm{equip}}=ig/u .
$$
They are perpendicular, with Euclidean magnitude $\rho$.
Multiplying by $u$ gives $\dot F=g$ and $\dot F=ig$,
respectively. Thus $g(s)=g(0)e^s$ on the outward ray, and
$\theta(s)-\theta(0)=g(0)s$ on the positive equipotential.
The metric becomes $(dg^2+d\theta^2)/g^2$ on the cover.
External rays are geodesics. Equipotentials are horocycles with
curvature magnitude one, closing after length $2\pi/g$ in the
quotient. These geometric identifications follow from the right
half-plane hyperbolic metric. [Exterior coordinate and covering](184-exterior-ruler-bounds.md#exterior-coordinate)

A floored conformal metric uses the same direction lines. Its
unit-speed chart rates are multiplied by $\rho_{\varepsilon}/\rho$,
so the unfloored rates and geodesic conclusions have a specific
scope. The phase origin is local, and neither external angle
nor orbit index is musical time.

## Error in the computed direction

If a truncated computation returns $\hat u\ne0$ with
$|u-\hat u|\le E$, then using $g/\hat u$ or $ig/\hat u$
changes the predicted complex chart rate by at most $g E/|\hat u|$.
This follows from $u(g/\hat u)-g=g(u-\hat u)/\hat u$, and the
quarter turn preserves that bound. It limits unintended change in
$g$ on a computed equipotential and in $\theta$ on a computed ray.
A finite path also needs regional
error and time integration bounds. Lean checks the exact rates,
speed and inequality. [ExteriorNavigation.lean](../../formal/FractalSync/ExteriorNavigation.lean)

## Matched exterior paths

From $0.5+0.5i$, compare five paths of length $0.75$ in the
unfloored exterior conformal metric. The initially tangent geodesic
in chart coordinates has $g(s)=g(0)\operatorname{sech}(s)$ and
$\theta(s)=\theta(0)+g(0)\tanh(s)$. The Map path rotates the returned
distance gradient, and the trap path rotates the gradient of
$F_{\rm trap}(c)=|c^2+c|^2$. Those tangents are scaled to the same
conformal speed.

| Path | End $g/g(0)$ | Change in sampled $D$ | Change in trap $F_{\rm trap}$ |
| --- | ---: | ---: | ---: |
| Outward ray | 2.11700 | +0.046132 | +0.566424 |
| Equipotential | 1 | −0.074722 | +0.157478 |
| Initially tangent geodesic | 0.772390 | −0.088460 | +0.043742 |
| Map-gradient contour | 1.35627 | −0.002207 | +0.312169 |
| Fixed trap contour | 0.701708 | −0.092398 | below $10^{-14}$ in magnitude |

This is a numerical comparison using a large escape guard and
adaptive ODE solving. The Map data came from the installed
`geometry-provider/6` wheel. The checked-out runtime source uses
a different provider version. The drift in reported $D$ along the
Map-gradient path is an observation about that sample and evaluator.
The paths show that an equipotential and a fixed trap can preserve
different quantities. They do not establish predictive value for
the Player. Retain the directions as diagnostic references for
[#185](https://github.com/rhythmatician/FractalSync/issues/185).
[Interior comparison](185-navigation-field-comparison.md)

~~~powershell
.venv/Scripts/python.exe formal/checks/check_exterior_navigation.py
~~~
