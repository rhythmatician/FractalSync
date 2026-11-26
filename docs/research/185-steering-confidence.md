# A sharp gradient-error gate for trap steering

This follows the [fixed second-iterate trap](185-orbit-trap-steering.md) for
[#185](https://github.com/rhythmatician/FractalSync/issues/185). The gate uses an
observed gradient \(\hat g\) and an upper bound \(e\) on the Euclidean error
\(|g-\hat g|\). It certifies the angle between the observed and true oriented
tangents at the query point. For a conformal ruler \(G=\rho^{-2}I\) with
\(\rho>0\), applying the same quarter turn to the two gradients and normalizing
them preserves their angle.

## The gate

For a maximum steering error \(\theta\in[0,\pi/2]\), choose
\(k=\cos^2\theta\) once. Set \(M=\hat g\cdot\hat g\) and \(E=e^2\). With
a valid nonnegative error budget, accept when both comparisons pass.

\[
E<M,\qquad E\le (1-k)M.
\]

The comparisons require only the squared norm and basic arithmetic at the
query point. The first comparison certifies regularity and a positive dot
product with every consistent gradient. The second certifies that each
consistent gradient makes an angle at most \(\theta\) with \(\hat g\).
For example, at \(30^\circ\) the error budget may be at most half the observed
gradient norm. At \(0^\circ\) the budget must be zero, and at \(90^\circ\)
the strict regularity check still applies. When \(E\ge M\) the error ball
contains the zero gradient, so the input cannot certify a regular direction.

## Proof and sharpness

Write \(P=\hat g\cdot g\), \(D=\det(\hat g,g)\), \(N=g\cdot g\) and
\(Q=|\hat g-g|^2\). If \(P\le0\), then
\(Q=M-2P+N\ge M\), contradicting \(Q\le E<M\). The planar identity
\(P^2+D^2=MN\) gives

\[
QN-D^2=(N-P)^2\ge0,\qquad D^2\le EN,\qquad P^2\ge(M-E)N.
\]

These inequalities give the exact cosine floor
\(\sqrt{1-E/M}\). For the estimated conformal unit tangent
\(\hat T=\rho J\hat g/\sqrt M\), they also give a pointwise trap drift bound
\(|dF(\hat T)|=\rho |D|/\sqrt M\le\rho e |g|/\sqrt M\).
A finite step needs its own drift estimate.

Let \(A=M-E\), \(B=\sqrt{EA}\), and take
\(g_*=(A\hat g+B J\hat g)/M\). Then
\(\hat g\cdot g_*=|g_*|^2=A\), \(\det(\hat g,g_*)=B\), and
\(|g_* - \hat g|^2=E\). The witness attains the cosine floor and the
determinant bound. When \(E<M\), the threshold for \(k\) is therefore
necessary and sufficient over the full error ball.

The [Lean module](../../formal/FractalSync/OrbitTrapConfidence.lean) checks
the dot/determinant identity, the positive-dot cone bound, its boundary
witness, the squared gate equivalence, and the zero-gradient obstruction.
The conformal angle and drift interpretations use the tangent formula in
the preceding note.

## Near the three critical points

For \(F(c)=|c^2+c|^2\),
\[
|g(c)|=4|c|\,|c+1|\,|c+1/2|.
\]
Write \(r=|c-c_*|\) for distance to one of the three critical points.
The triangle inequality in the two remaining factors shows
\(|g|/r\to2\) at either minimum and \(|g|/r\to1\) at the saddle, uniformly
over approach directions. The following bounds hold for \(r\le1/4\).

| Center \(c_*\) | Lower bound on \(|g|\) | Upper bound on \(|g|\) |
| --- | ---: | ---: |
| \(0,-1\) | \(4r(1-r)(1/2-r)\) | \(4r(1+r)(1/2+r)\) |
| \(-1/2\) | \(4r(1/4-r^2)\) | \(4r(1/4+r^2)\) |

At the saddle, combine the two remaining factors as
\(|(c+1/2)^2-1/4|\). This yields the quadratic correction inside the
parentheses in the table. All three lower bounds are at least \(3r/4\).
Consequently, for \(e>0\) and \(0<\theta<\pi/2\), the gate passes at every
consistent estimate whenever

\[
\frac{4e}{3}\left(1+\frac{1}{\sin\theta}\right)\le r\le\frac14 .
\]

Indeed, \(|\hat g|\ge|g|-e\) then implies
\(|\hat g|\ge e/\sin\theta>e\). In the opposite direction, if
\(|g|<e(1+1/\sin\theta)\), some consistent estimate fails the gate.
An estimate displaced by \(e\) against the true gradient attains the minimum
possible observed norm \(\max(|g|-e,0)\), with the zero estimate usable when
\(|g|\le e\). These bounds explain why a fixed absolute error budget can
obscure steering near every critical point.

## Obtaining an error budget for this fixture

If the requested point \(c\) and the point used by the gradient evaluator
\(\hat c\) both lie within radius \(1/4\) of the same center, a position
certificate \(|c-\hat c|\le\delta\) gives
\(e=L\delta+e_{\rm eval}\). Here \(e_{\rm eval}\) bounds the gradient
evaluation error at \(\hat c\). To obtain \(L\), differentiate
\(g(c)=2\overline{(2c+1)}(c^2+c)\) in the real direction \(h\).

\[
Dg(c)[h]=4\bar h(c^2+c)+2|2c+1|^2 h .
\]

In either minimum's ball, \(|c^2+c|\le5/16\) and
\(|2c+1|\le3/2\), so \(L=23/4\) suffices. In the saddle ball the same
bound on \(|c^2+c|\) holds and \(|2c+1|\le1/2\), giving \(L=7/4\).
Integrating along the straight segment gives the position contribution
because the ball is convex.

The critical-point scaling and position budget above are conventional
estimates. The gate and Lean proof use exact real arithmetic. A
finite-precision gate needs conservative comparisons with the angular
threshold. The calculation below supplies one evaluation budget for the
fixed trap.

## Binary64 evaluation example

For binary64 coordinates in `[-2, 2]²`, use the arithmetic order below.
Assume separately rounded operations, round-to-nearest ties-to-even and
gradual underflow. The budget is relative to the exact gradient at the
supplied coordinates. Use the position bound above if the coordinates also
carry error.

```text
h = 2*x + 1
k = 2*y
A = (x*x - y*y) + x
B = h*y
gx = 2*(h*A + k*B)
gy = 2*(-k*A + h*B)
```

Set `η = 2^-52`. A rounded operation whose exact result on machine
operands has magnitude at most `M ≥ 1` contributes at most `Mη`.
Doubling and negation are exact in the ranges here. For a product of
quantities with exact magnitudes bounded by `B₁, B₂` and errors bounded
by `E₁, E₂`, the pre-rounding error is at most
`E₁(B₂ + E₂) + B₁E₂`. Adding operand errors for sums gives these enclosures.

| Quantity | Bound on absolute error in units of `η` |
| --- | ---: |
| `h, k, A, B` | `5, 0, 28, 21` |
| `h*A, k*B, -k*A, h*B` | `242, 125, 153, 207` |
| `gx, gy` | `916, 902` |

The sum of the component bounds is below `2^-41`, so `e_eval = 2^-41`
also bounds Euclidean error. A spot check compared 4,420 Python binary64
evaluations, including neighbors of the critical coordinates, against
exact rational arithmetic in the same formula. The largest observed
error was `2.381e-14` rounded upward.

For a 5° angular tolerance with evaluation error alone, this budget
requires observed gradient norm at least about `5.22e-12`. Output
normalization and the gate comparison need their own finite-precision
allowances.
