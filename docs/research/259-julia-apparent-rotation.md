# Apparent rotation of a marked Julia eye

In the hands-on review of #251, Gain 2 in Distinct Directions appeared to
turn the arms around each large spiral eye at the default anchor. The
measurement proposed here follows an arm around its own eye. The angle of the
line joining the two eyes is a separate observable.

Fix the presentation. Let p(c) be a marked eye center and q(c) a corresponding
arm landmark. Use coordinates with positive imaginary axis up, so positive
angles turn counterclockwise. Set w(c) = q(c) - p(c). At a base c₀ with
w(c₀) nonzero, take the angle of w(c)/w(c₀). The ratio is 1 at c₀, so this
choice works even if the original landmark lies on the cut of the principal
argument. A common translation of the eye and arm cancels in w.

With a real Fréchet derivative Dw at c₀, the angular rate in input direction h
is Im(Dw[h]/w(c₀)). The proof differentiates the normalized complex logarithm
near 1 and takes its imaginary part. Its norm bound
|Im(Dw[h]/w(c₀))| ≤ |Dw[h]|/|w(c₀)| also bounds the contribution of a velocity
error when the base landmark vector is fixed. Center error also perturbs the
denominator. Marks too close to the eye center make the angular estimate
unstable.

Lean checks the local branch derivative, the angle-free application formula,
and the norm bound in
[ApparentRotation.lean](../../formal/FractalSync/ApparentRotation.lean).
The derivative is real-linear in the parameter increment. It requires a
differentiable landmark track, with no holomorphicity assumption.

## Synthetic calibration

For each landmark use w(t) = r exp(iφ) exp((s + iω)t). Set
p(t) = (0.4 + 0.1i) + (3 − 2i)t and q(t) = p(t) + w(t). Use r = 0.02 and
φ = π for the inner landmark, and r = 0.08 and φ = 0.8 for the outer one.
With s = 0.5 for both, recover angular rates by central differences of
arg(w(t)/w(0)). The inner mark is deliberately at the angle cut before
normalization. At h = 0.01, 0.001 and 0.0001 the double-precision results
agreed with the following values to nine decimal places.

| Imposed motion | Inner rate | Outer rate |
| --- | ---: | ---: |
| Translation and dilation | 0 | 0 |
| Translation, dilation and uniform turn | 0.4 | 0.4 |
| Translation, dilation and differential twist | 0.75 | −0.25 |

## Applying the differential to renders

Start from the frozen presentation in the #151 sensitivity probe. Compare
opposite parameter offsets along the two directions used by the gesture.
Follow several arm marks across inner and outer radii of each eye. Report
the signed responses per landmark, their agreement, and the steps at which
the estimates stabilize. Keep track losses and uncertain correspondence
visible. Compare solo source plays with the combined replay after the
landmark measurements are stable. A patch fit can suggest apparent turn,
while disagreement between marked arms can reveal deformation hidden by
one fitted angle.

At fixed c, the argument of a nonzero periodic multiplier describes
the rotation of the linearized iterate in z. The measured quantity here
describes the change of a relative image landmark as c varies. To compare
them, first identify a candidate dynamical organizing point in #258, then
measure whether the marked arm follows it and whether angle changes agree
under the same parameter perturbations. The synthetic numbers above do not
establish a rotation measurement or an orbit correspondence for the Julia
renderer.
