# A stable form of the radial profile

The radial profile measures the unrestricted ideal graph distance to the Shore. For $\varepsilon>0$, $r\ge0$, and $a\ge0$, put $q=r/\varepsilon$, $b=\sqrt{1+a^2}$ and $k=1/(b+a)$. The existing integral and its logarithmic primitive are checked in Lean. The same exact value is

$$
F_{\varepsilon,a}(r)=
k\operatorname{asinh}(bq)
+a\operatorname{asinh}\left(kq\frac{\sqrt{1+(bq)^2}}{\sqrt{1+q^2}}\right).
$$

Both summands are nonnegative. The radius-zero case is exactly zero and zero anisotropy reduces to $\operatorname{asinh}(q)$. [Radial-profile integral and exact primitive](160-radial-profile.md), [stable-form Lean proofs](../../formal/FractalSync/StableRadialProfile.lean).

## Exact algebra

Let $d=\sqrt{1+q^2}$ and $s=\sqrt{1+(bq)^2}$. Because $s^2=d^2+(aq)^2$, the logarithmic part of the existing primitive is equivalent to $-a\operatorname{asinh}(aq/d)$. Thus the existing primitive is $b\operatorname{asinh}(bq)-a\operatorname{asinh}(aq/d)$.

The hyperbolic-sine subtraction formula and $\cosh(\operatorname{asinh}x)=\sqrt{1+x^2}$ give

$$
\operatorname{asinh}x-\operatorname{asinh}y
=\operatorname{asinh}\left(x\sqrt{1+y^2}-y\sqrt{1+x^2}\right).
$$

Write $X=\operatorname{asinh}(bq)$ and $Y=\operatorname{asinh}(aq/d)$. Then $bX-aY=(b-a)X+a(X-Y)$. The argument of the difference identity for $X-Y$ simplifies to $(b-a)qs/d$. Since $(b-a)(b+a)=1$, use the reciprocal $k$ for both occurrences of $b-a$. The algebraic equivalence holds for signed real $a,q$ as well. The same-sign evaluation applies on the physical domain $a,q\ge0$.

## Binary64 evaluation range

For ordinary finite intermediate values, use `hypot(1, a)` for $b$ and `hypot` for the square roots of sums of squares. A large $a$ makes $b-a$ susceptible to rounding to zero. Even its rationalized denominator $b+a$ can overflow. Compute $C=1/(1+a/b)$ and $k=C/b$ instead. The second $\operatorname{asinh}$ argument is exactly

$$
z=q C\frac{\operatorname{hypot}(1/b,q)}{\operatorname{hypot}(1,q)}.
$$

Evaluating the ratio before multiplying by $q$ avoids forming $bq$ in this term. The ratio lies in $[0,1]$ because $b\ge1$. The first term still needs $\operatorname{asinh}(bq)$, which can be finite when $bq$ overflows. Likewise $q=r/\varepsilon$ can overflow even for a finite final profile. For example $r=10^{300}$, $\varepsilon=10^{-100}$ and $a=1$ yield a finite profile of order $10^3$ in exact arithmetic even though $q=10^{400}$ cannot be materialized in binary64.

The following log-scaled identities show how to avoid those intermediate range limits for positive radius. Use an ordinary quotient when it is safely representable, or carry $\ell_q=\log r-\log\varepsilon$. Set $\ell_b=\log b$ and $\ell_C=-\log1p(a/b)$. For log arguments $u,v$ define

$$
H(u,v)=\max(u,v)+\tfrac12\log1p\left(e^{-2|u-v|}\right)
=\log\operatorname{hypot}(e^u,e^v).
$$

The first $\operatorname{asinh}$ argument has log $\ell_b+\ell_q$. The second has log $\ell_q+\ell_C+H(-\ell_b,\ell_q)-H(0,\ell_q)$. When the two $H$ terms share the same maximum, cancel those maxima before subtracting the corrections. For $\ell\ge0$ use

$$
\operatorname{asinh}(e^\ell)=\ell+\log1p\left(\sqrt{1+e^{-2\ell}}\right).
$$

For $\ell<0$ and representable $x=e^\ell$, rationalize the small square-root increment and evaluate $\log1p(x+x^2/(1+\operatorname{hypot}(1,x)))$. An underflowed $x$ can still contribute after multiplication by a large $a$. For $a=10^{300}$ and $q=10^{-300}$, the second argument is about $7.07\times10^{-601}$, while its weighted contribution is about $7.07\times10^{-301}$. Dropping it would lose more than half the finite profile. In that range a weighted evaluation can retain the product's exponent before exponentiating, using $\operatorname{asinh}x/x\to1$ and an explicit approximation budget for any small-argument shortcut. Log scaling can lose accuracy on nearby input ratios, especially where it subtracts large input logarithms. Ordinary arithmetic with safe intermediates and log fallback need their own crossover and error tests. An unrepresentable final value must be distinguished from overflow in intermediate calculations.

The Lean result establishes equivalence to the radial integral and the nonnegative terms. A binary64 error contract needs separate tests or enclosures for the supported parameter range and targets. Rust specifies no fixed precision for the standard `f64` implementations of `hypot` and `asinh`. The existing sharp distance sensitivity $M(a)\delta/\varepsilon$ propagates a certified radius error $\delta$ for fixed parameters. Rounding of the scalar evaluator and provider-to-geometry correspondence have separate error budgets. [Rust `f64` methods](https://doc.rust-lang.org/std/primitive.f64.html#method.hypot), [sharp radial Lipschitz bound](160-radial-profile.md).

## Reproducible scalar probe

Run `python formal/checks/check_stable_radial_profile.py` in the project
environment from the repository root. The probe uses a base-two exponent
split for the input ratio and sums the two positive terms in log space. It
compares 321 finite and 15 overflow cases against the subtractive primitive at
780 decimal digits. The largest relative discrepancy for reference values at
least $10^{-300}$ was $2.30\times10^{-13}$. At $q=10^{-300}$ and $a=10^{300}$,
direct binary64 subtraction returned zero while the scaled result was
$1.1477935747\times10^{-300}$. These are sampled scalar comparisons.

## Rust and cockpit diagnostics

`runtime_core::shore_distance::RadialProfile` evaluates the profile with explicit input and overflow errors. It uses the ordinary quotient for finite values and logarithmic asymptotics when the quotient or first argument overflows. The Rust tests check independent quadrature, sampled binary64 extremes, plane and circle paths, and the radius sensitivity bound. [Rust implementation](../../runtime-core/src/shore_distance.rs), [numerical tests](../../runtime-core/tests/test_shore_distance.rs).

Cockpit replay snapshots carry `diagnostics.geometry.shoreDistanceEstimate` alongside the signed distance, provider version, tile, resolution, and validity for the same query. The estimate records the active `epsilon` and `anisotropy`. It is available for a finite raster sample, including at unresolved or singular jets, and absent outside the provider domain or when its scalar has unknown Shore-distance correspondence. The current raster has no certified radius error. A future certified radius error can be propagated via `propagate_radius_error`, with a separate allowance for floating-point evaluation.

The analytic value is exact for the unrestricted ideal model and a true Euclidean radius. It gives a lower bound in a restricted admissible domain when the metrics correspond, even if the nearest segment is unavailable there. The cockpit displays a raster estimate, which carries no such route guarantee. [Issue #168](https://github.com/rhythmatician/FractalSync/issues/168).
