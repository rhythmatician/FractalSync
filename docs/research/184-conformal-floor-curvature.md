# Curvature of the floored main-cardioid ruler

The unregularized conformal metric has Gaussian curvature $-1$ in the open main cardioid. A fixed positive ruler floor raises curvature to $+1$ in the limit at every smooth boundary point. At the cusp the limiting curvature depends on the approach. The limit is $7$ on the real axis and $4$ on the multiplier path $u=1-t(1+i)$. This calculation concerns the candidate conformal metric in the [main-cardioid ruler study](184-main-cardioid-ruler.md).

## Curvature in parameter coordinates

Let $\rho>0$ be a $C^2$ ruler on a planar region. Write $g=\rho^{-2}I=e^{2\phi}I$, where $\phi=-\log\rho$. Use the Euclidean Laplacian $\Delta=\partial_x^2+\partial_y^2$ in parameter coordinates. The Christoffel symbols in these coordinates are

$$
\Gamma^k_{ij}
=\delta^k_i\partial_j\phi+\delta^k_j\partial_i\phi-\delta_{ij}\partial_k\phi .
$$

With the convention $K=\langle R(\partial_x,\partial_y)\partial_y,\partial_x\rangle_g/\det g$, the first component of $R(\partial_x,\partial_y)\partial_y$ is $-\partial_x^2\phi-\partial_y^2\phi$. The two quadratic connection terms cancel. The numerator is therefore $-e^{2\phi}\Delta\phi$ and the denominator is $e^{4\phi}$. This gives

$$
K=-e^{-2\phi}\Delta\phi
  =\rho^2\Delta\log\rho
  =\rho\,\Delta\rho-|\nabla\rho|^2.
$$

Now fix $\varepsilon>0$ and set $R_\varepsilon=\sqrt{\rho^2+\varepsilon^2}$. For $f(r)=\log\sqrt{r^2+\varepsilon^2}$, the scalar derivatives are

$$
f'(r)=\frac{r}{r^2+\varepsilon^2},
\qquad
f''(r)=\frac{\varepsilon^2-r^2}{(r^2+\varepsilon^2)^2}.
$$

Taking the Euclidean trace of the composite Hessian and multiplying by $R_\varepsilon^2$ gives the exact Gaussian curvature of $R_\varepsilon^{-2} I$:

$$
\begin{aligned}
K_\varepsilon
&=\rho\,\Delta\rho+
\frac{\varepsilon^2-\rho^2}{\rho^2+\varepsilon^2}|\nabla\rho|^2\\
&=K+\frac{2\varepsilon^2}{\rho^2+\varepsilon^2}|\nabla\rho|^2 .
\end{aligned}
$$

The correction is nonnegative, is at most $2|\nabla\rho|^2$, and vanishes at critical points of $\rho$. At a fixed point in the interior it is $O(\varepsilon^2)$ as $\varepsilon\to0$. Its fraction of $2|\nabla\rho|^2$ is close to one when $\rho\ll\varepsilon$.

## Main cardioid

Use the attracting multiplier $u$ with $|u|<1$. Set $c=F(u)=u(2-u)/4$, $a=1-|u|^2$, and $s=|1-u|$, so that $\rho(F(u))=as/4$ and $|F'(u)|=s/2$. The chain rule for the holomorphic coordinate map gives $\Delta_u(h\circ F)=|F'|^2(\Delta_c h)\circ F$. Here the component Hessian traces of $F$ vanish and $DF\,DF^{\mathsf T}=|F'|^2 I$. Since $\Delta_u\log s=0$ and $\Delta_u\log a=-4/a^2$ on the open disk, $K=\rho^2|F'|^{-2}(-4/a^2)=-1$.

The derivative $DF$ is a rotation times $s/2$, so the chain rule for the gradient norm gives

$$
|\nabla_c\rho(F(u))|
=\left|-u+\frac{a(u-1)}{2s^2}\right|.
$$

Expanding the square and using $u\cdot(u-1)=(s^2-a)/2$ yields

$$
|\nabla_c\rho(F(u))|^2=1-\frac{3a}{2}+\frac{3a^2}{4s^2}.
$$

The inequality $0<a\le 2s$ follows from $1-|u|\le|1-u|$. It gives $|\nabla_c\rho|^2\le4$ and hence $-1\le K_\varepsilon\le7$ throughout the open cardioid. Pointwise, the curvature is positive exactly where $2|\nabla_c\rho|^2>1+(\rho/\varepsilon)^2$.

### Smooth boundary

As $u\to b$ with $|b|=1$ and $b\ne1$, we have $a\to0$ and $s\to|1-b|>0$. The squared gradient tends to $1$ and $\rho\to0$, so $K_\varepsilon\to+1$ for each fixed $\varepsilon>0$. The local inverse of $F$ and the positive square root give a smooth metric across a neighborhood of $F(b)$.

In a joint limit with $\varepsilon\to0$ and $d_S/\varepsilon\to\lambda<\infty$, the [smooth Shore-distance ratio](184-main-cardioid-ruler.md#smooth-boundary-and-cusp) gives $\rho/\varepsilon\to\lambda$. Therefore

$$
K_\varepsilon\longrightarrow -1+\frac{2}{1+\lambda^2}.
$$

This limiting profile changes sign at $\lambda=1$.

### Cusp

On $u=1-t(1+i\alpha)$ with real $\alpha$ and $t\downarrow0$, one has $s=t\sqrt{1+\alpha^2}$ and $a=2t-(1+\alpha^2)t^2$. Thus

$$
|\nabla_c\rho|^2\longrightarrow 1+\frac{3}{1+\alpha^2},
\qquad
K_\varepsilon\longrightarrow 1+\frac{6}{1+\alpha^2}
\quad(\varepsilon>0\text{ fixed}).
$$

The bound $0\le a/s\le2$ puts all cusp accumulation values of $K_\varepsilon$ in $[1,7]$. All values in $(1,7]$ occur along the straight multiplier approaches above. The endpoint $1$ occurs along the tangential multiplier approach $u=(1-t^2)e^{it}$, where $s/t\to1$ and $a/s\to0$.

The ruler tends to $\varepsilon$ at the cusp. In fact, $\rho\le s^2/2=2|c-1/4|$, so $R_\varepsilon-\varepsilon=O(|c-1/4|^2)$ for fixed $\varepsilon$. The gradient of $R_\varepsilon$ tends to zero because $\nabla R_\varepsilon=(\rho/R_\varepsilon)\nabla\rho$ and $|\nabla\rho|\le2$. A positive $C^2$ metric extension across the cusp would give continuous Gaussian curvature there. The limits $7$ and $4$ along the axis and the $\alpha=1$ path preclude such an extension.

On the cusp axis, $\rho/d_S\to2$ and $|\nabla_c\rho|^2\to4$. In a joint limit along this axis with $d_S/\varepsilon\to\lambda<\infty$ and $\varepsilon\to0$,

$$
K_\varepsilon\longrightarrow -1+\frac{8}{1+4\lambda^2}.
$$

This limiting profile changes sign at $\lambda=\sqrt{7}/2$. The smooth profile and the cusp-axis profile describe how the sign change scales in units of physical Shore distance as the floor shrinks.

## Sample values at a fixed floor

At `ε = 10^-4`, evaluating the curvature at the following rational
multiplier coordinates gives these values.

| `t` | Smooth, `u = -1 + t` | Cusp axis, `u = 1 - t` | Oblique cusp, `u = 1 - t - it` |
| ---: | ---: | ---: | ---: |
| `10^-2` | `-0.99980205` | `5.31696234` | `2.27566189` |
| `10^-4` | `-0.00020000` | `6.99880003` | `3.99880007` |
| `10^-6` | `0.99979402` | `6.99998800` | `3.99998800` |

The calculation uses exact rational values of `ρ² = a²s²/16` and the
squared gradient formula above. As a cross-check, differentiating
`ρ ∘ F = as/4` by the product rule gives
$\rho\Delta_c\rho=a(-4s^2-4u\cdot(u-1)+a)/(4s^2)$, which yields the same
curvature at all nine points. The table rounds only the displayed values.

## Checked scope

[ConformalFloorCurvature.lean](../../formal/FractalSync/ConformalFloorCurvature.lean) checks both scalar derivatives, the second Fréchet derivative of the log floor, the weighted-trace correction and its bounds, and the scalar squared-gradient simplification and upper bound. The coordinate curvature identification, the main-cardioid Laplacian calculation and gradient formula, and the boundary limits are conventional derivations above. The conformal metric under study is defined solely by the candidate ruler.
