# Exterior conformal ruler derivatives from a common escape disk

The [pointwise exterior bounds](184-exterior-ruler-bounds.md) enclose the ruler itself. A disk on which every parameter escapes at a common stage also gives bounds for its gradient and Hessian. Let $z_0(c)=0$, $z_{j+1}(c)=z_j(c)^2+c$, and let $n\ge1$ be that stage. Use the Green-function normalization $g=\log|\Phi|=\lim_j 2^{1-j}\log|z_j|$ from the pointwise note.

## Certifying the disk

For a center $c_0$ and radius $h>0$, take $C=|c_0|+h$. Calculate $e_0=0$ and

$$
e_{j+1}=2|z_j(c_0)|e_j+e_j^2+h.
$$

For $|c-c_0|\le h$, induction gives $|z_j(c)-z_j(c_0)|\le e_j$. The quadratic difference is $2z_j(c_0)(z_j(c)-z_j(c_0))+(z_j(c)-z_j(c_0))^2$. Thus a lower bound $R\le |z_n(c_0)|-e_n$ gives $|z_n(c)|\ge R$ throughout the closed disk. Require $R\ge4$ and $C\le R^2/4$. These tests are sufficient, and a failed test leaves the disk unresolved.

## Holomorphic tail and Cauchy bounds

For each parameter in the disk, the [escape guard](184-exterior-ruler-bounds.md#exact-input-truncation-bounds) gives $|z_{n+j}(c)|\ge3^jR$. The principal complex logarithm is holomorphic at $1+a$ for $|a|\le1/4$, since $\operatorname{Re}(1+a)>0$. Integrating $a/(1+ta)$ along $0\le t\le1$ gives $|\operatorname{Log}(1+a)|\le(4/3)|a|$. Hence the complex tail

$$
T_n(c)=\sum_{j=0}^\infty 2^{-n-j}\operatorname{Log}\!\left(1+\frac{c}{z_{n+j}(c)^2}\right)
$$

converges uniformly on the closed disk, is holomorphic in its interior, and satisfies

$$
|T_n(c)|\le\frac43\,2^{-n}\frac{C}{R^2}\sum_{j=0}^\infty18^{-j}
=\frac{24}{17}\,2^{-n}\frac{C}{R^2}=:E.
$$

Choose a holomorphic logarithm of $z_n$ on the open disk and put $F_n=2^{1-n}\log z_n$. The real parts of the summands telescope the finite Green approximations, so $\operatorname{Re}(F_n+T_n)=g$. A local logarithm $F=\log\Phi$ differs from $F_n+T_n$ by an imaginary constant wherever both are defined. Cauchy's derivative estimate at the disk center, applied on circles of radius $s<h$ and then with $s\uparrow h$, gives

$$
|g(c_0)-\operatorname{Re}F_n(c_0)|\le E,\qquad
|F^{(k)}(c_0)-F_n^{(k)}(c_0)|\le \frac{k!E}{h^k}\quad (k=1,2,3).
$$

## From the finite orbit to the ruler Hessian

Write $d=z_j'$, $q=z_j''$, $t=z_j'''$. Start all three at zero and update from the same old iterate $z=z_j$ and its derivatives:

$$
(z,d,q,t)\gets(z^2+c,\ 2zd+1,\ 2zq+2d^2,\ 2zt+6dq).
$$

At stage $n$, the unweighted approximations to $g,U=F',V=F'',W=F'''$ are

$$
\left(\log|z|,\ \frac d z,\ \frac q z-\left(\frac d z\right)^2,\
\ \frac t z-\frac{3qd}{z^2}+2\left(\frac d z\right)^3\right).
$$

Multiply by $2^{1-n}$ to get the normalized quantities. The same positive factor cancels from the ruler $\rho=g/|U|$ and all the expressions below. An evaluator can use the unweighted centers directly with error radii $(12/17)(C/R^2) k!/h^k$ for $k=0,1,2,3$.

Identify real directions $v,w\in\mathbb R^2$ with complex numbers. Because $g=\operatorname{Re}F$ and $U$ is holomorphic and nonvanishing, the chain rule gives

$$
\begin{aligned}
D\log\rho[v]&=\frac{\operatorname{Re}(Uv)}{g}-\operatorname{Re}\!\left(\frac VU v\right),\\
D^2\log\rho[v,w]&=\operatorname{Re}\!\left[\left(\frac Vg-\frac WU+\left(\frac VU\right)^2\right)vw\right]
-\frac{\operatorname{Re}(Uv)\operatorname{Re}(Uw)}{g^2}.
\end{aligned}
$$

The terms with denominator $g$ or $g^2$ come from $\log g$. The remaining terms come from $-\operatorname{Re}\log U$. Evaluating on $1,i$ gives gradient and Hessian entries. For derivatives of $\rho$ itself, use $D\rho=\rho D\log\rho$ and $D^2\rho=\rho(D^2\log\rho+D\log\rho\otimes D\log\rho)$.

To enclose these expressions, form a real interval for $g$ and complex disks for $U,V,W$ from the finite values and the error radii. Require positive lower bounds for $g$ and $|U|$ before dividing. A disk for $U$ with center $u$ and radius $r<|u|$ gives the lower bound $|u|-r$ even if a containing rectangular interval is too wide to certify division. The $h^{-k}$ factor makes a small certified disk expensive for higher derivatives. Once a disk is certified at one stage, the guard persists on that disk, so taking additional iterates with the same $h$ can reduce the tail bound.

## Interval experiment

The common disk gives a derivative truncation error at its **center**.
The comparison below propagates those bounds to the unregularized
ruler, gradient, and Hessian using 60-digit interval arithmetic in
`mpmath` 1.3.0. The iteration uses the center perturbation test until
the disk passes. On later steps it updates the uniform lower bound by
$R\leftarrow R^2-C$, which follows directly from the orbit recurrence
across the disk. The complex truncation disks are enclosed by
rectangles for interval propagation.

| Center $c_0$ | Disk radius $h$ | Iteration $n$ | Value width | Widest gradient entry | Widest Hessian entry |
| ---: | ---: | ---: | ---: | ---: | ---: |
| $0.5+0.5i$ | $10^{-4}$ | 6 | unresolved | unresolved | unresolved |
| $0.5+0.5i$ | $10^{-4}$ | 8 | $1.18\times10^{-8}$ | $2.35\times10^{-4}$ | $7.05$ |
| $0.5+0.5i$ | $10^{-4}$ | 9 | $6.27\times10^{-18}$ | $1.25\times10^{-13}$ | $3.76\times10^{-9}$ |
| $0.2501$ | $10^{-8}$ | 313 | $5.46\times10^{-7}$ | $3.84\times10^2$ | $2.94\times10^{11}$ |
| $0.2501$ | $10^{-8}$ | 315 | $3.37\times10^{-12}$ | $6.81\times10^{-4}$ | $2.02\times10^5$ |
| $0.2501$ | $10^{-8}$ | 316 | $1.34\times10^{-18}$ | $2.71\times10^{-10}$ | $8.05\times10^{-2}$ |
| $0.2501$ | $10^{-8}$ | 317 | $4.26\times10^{-31}$ | $8.62\times10^{-23}$ | $2.56\times10^{-14}$ |

A disk passes the escape test at iteration 6 in the first case, but
the interval denominator test for the jets first passes at iteration 7.
Near the cusp, the ruler value is already tightly enclosed at
iteration 315 while its Hessian interval is still wide. A provider
can select a stopping criterion based on requested component widths.
The experiment concerns the stated disk and arbitrary-precision
interval evaluator. A Rust binary64 implementation would need its
own orbit arithmetic and range bounds.

~~~powershell
.venv/Scripts/python.exe formal/checks/check_exterior_ruler_jets.py
~~~

[Interval checker](../../formal/checks/check_exterior_ruler_jets.py)

## Checked scope

The [Lean module](../../formal/FractalSync/ExteriorRulerJets.lean) checks the orbit perturbation bound, the closed disk test, the geometric majorant for the complex logarithm series, and the tail's differentiability in the open disk and continuity on its closure. It proves the Cauchy bound for every natural derivative order of the actual infinite tail using only the center-orbit and radius hypotheses. The identification with the branch of $\log\Phi$, the finite orbit derivative recurrences, and the formulas for propagating the tail bounds through the ruler use the conventional arguments above.
