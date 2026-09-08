# 0004 — Scale-aware differential Mandelbrot geometry provider

Status: Accepted (2026-09-07) — implemented by #145
Related: #120, #145, #84, #111, #137, #138

This ADR records the destination architecture for the Mandelbrot geometry consumed by FractalSync Physics.

It does **not** reopen the scale-relative metric adopted by issue #120. It changes the representation that supplies the signed-distance geometry from which that metric is derived.

## Context

FractalSync now treats Mandelbrot motion as mechanics on a two-dimensional configuration manifold whose geometry is induced by a scale-relative ruler.

The canonical chain is

```text
D(c)
  -> rho(c) = sqrt(D(c)^2 + epsilon^2)
  -> sigma(c) = log2(d_ref / rho(c))
  -> G(c)
  -> Gamma(c)
```

with

```text
G(c) = rho(c)^-2 I
     + lambda^2 grad(sigma) grad(sigma)^T.
```

The ordinary rider configuration remains `c = x + i y`; `sigma(c)` is derived geometry, not an independent degree of freedom.

The current Physics geometry authority is a fixed-resolution global 1024² signed-distance raster. `runtime-core/src/manifold.rs` samples that field, finite-differences `sigma` to estimate its gradient, then finite-differences those gradients again to estimate the Hessian needed by the Levi-Civita connection.

Issue #120 exposed a numerical limitation in this arrangement. A native Shore-crossing trajectory continues to cross under timestep refinement, but conservative-energy error at first crossing does not converge. The sampled derivative field also shows large Hessian variation near difficult Shore geometry. The current evidence identifies sampled-field derivative quality as the leading source while leaving the integrator contribution to be isolated separately.

That observation reveals a broader architectural mismatch independent of the exact numerical attribution:

> **Scale-relative Physics requires geometry whose resolution and differential quality remain meaningful across Mandelbrot scale. A single fixed-world-resolution Cartesian raster cannot be the destination authority for that job.**

Moving from 1024² to 2048² would only double linear resolution, approximately one additional octave. It would not remove the fixed-resolution assumption from a system intended to traverse many scale octaves.

FractalSync also already has a separate 2048² mip pyramid. That pyramid carries

```text
F = fractional escape iteration
S = unsigned Shore-proximity / sensitivity
```

for Player minimap/cartographic use. It is a different representation with a different role and is not automatically a signed-distance/Hessian authority.

A Rust-owned deep-zoom evaluator also exists for resolution-unlimited on-demand Mandelbrot sampling. It is useful infrastructure, but its current unsigned output does not by itself satisfy the signed-distance, differential, continuity, or singularity semantics required by Physics.

ADR 0001 remains in force: deterministic runtime geometry shared by training and runtime is authoritative in `runtime-core`.

---

## Decision

FractalSync Physics will consume a **scale-aware differential Mandelbrot geometry provider** rather than treating a fixed-resolution raster plus repeated finite differences as the destination geometry authority.

Conceptually:

```text
Physics
   |
   v
GeometryProvider
   |
   v
GeometryJet(c)
   |
   +-- D
   +-- grad_D
   +-- H_D
   +-- resolved scale / error capability
   +-- validity
   +-- singularity classification where known
```

The exact type names and storage representation are implementation details. The architectural invariants are not.

### 1. One coherent differential sample

For a given local geometry query, the provider must return signed distance and the first- and second-order differential information required by the ordinary Physics path from **one coherent local representation**.

The provider must not define the destination geometry as

```text
sample scalar D
  -> finite-difference sigma
  -> finite-difference that gradient
  -> Hessian / Gamma.
```

Instead, the intended dependency is

```text
local Map representation
        -> D, grad D, H_D
        -> rho, grad rho, H_rho
        -> sigma, grad sigma, H_sigma
        -> G, Gamma.
```

This means Physics differentiates the geometry representation, not repeated samples of an already sampled representation.

### 2. Downstream derivatives are analytic consequences

Given a coherent signed-distance jet,

```text
rho = sqrt(D^2 + epsilon^2)
```

so

```text
grad rho = (D/rho) grad D
```

and

```text
H_rho = (D/rho) H_D
      + (epsilon^2/rho^3) grad D grad D^T.
```

Then `grad sigma` and `H_sigma` follow by the chain rule from

```text
sigma = log2(d_ref / rho).
```

The metric, inverse metric, connection, energy, and generalized-force semantics remain those already adopted by #120.

### 3. Resolution is scale-aware

The provider must not have one permanent world-space texel size as its destination resolution model.

It must be able to refine local geometry as Mandelbrot scale demands and report the actual resolution or error capability of the returned geometry.

A criterion of the general form

```text
local cell size <= alpha * max(rho, epsilon)
```

may be useful, but ADR 0004 does not standardize a specific `alpha`, tiling scheme, refinement law, or error estimator. Those are implementation and validation decisions owned by #145 and #120.

### 4. Genuine singularities remain explicit

Regularizing

```text
rho = sqrt(D^2 + epsilon^2)
```

removes the `rho = 0` singularity but does not make every signed-distance cut locus, medial-axis point, cusp, or non-unique nearest-Shore configuration differentiable.

The provider must distinguish, as far as practically possible, between:

```text
regular geometry
numerically unresolved geometry
provider-domain failure
known/suspected genuine nonsmooth geometry
```

Ordinary Levi-Civita Physics may require a valid smooth jet. At genuine nonsmooth locations, the system must use explicit deterministic validity/failure/special semantics rather than silently smoothing the geometry until a unique tangent or Hessian appears.

### 5. Map roles remain separate

The project will not require one artifact to simultaneously serve as:

```text
Player minimap
cartography
signed realm authority
local metric authority
Hessian / connection authority
arbitrary deep-zoom geometry.
```

The existing 2048² `F/S` mip pyramid remains a Player observation/cartography representation unless separate work changes that decision.

The Physics geometry provider owns signed differential geometry.

The deep-zoom evaluator may be reused as a source for local geometry generation, but it is not promoted to Physics authority merely because it is resolution-unlimited.

### 6. Rust remains authoritative

The GeometryProvider and the semantics of its GeometryJet live in `runtime-core` under ADR 0001.

Python and WASM consumers receive the same versioned geometry semantics. Training mirrors may remain differentiable approximations where necessary, but they must be parity-pinned to the Rust forward semantics at the owned seam.

---

## Representation is intentionally not fixed by this ADR

ADR 0004 chooses the **contract**, not the concrete storage method.

Plausible implementations include:

- adaptive/local `C^2` spline tiles;
- a quadtree or other multiresolution coefficient field;
- another deterministic local representation capable of coherent value/gradient/Hessian evaluation;
- on-demand local geometry generated from a deeper Mandelbrot evaluator and cached behind the same provider seam.

A tensor-product cubic B-spline is an attractive candidate because it is compact, local, `C^2` across regular cell boundaries, and analytically differentiable, but it is **not** adopted by this ADR.

The implementation should be selected by #145 using numerical quality, runtime cost, determinism, cache behavior, and #120 validation evidence.

---

## Consequences

### Positive

- Physics no longer bakes a fixed raster resolution into its mathematical authority.
- Value, gradient, and Hessian become mutually consistent by construction on regular regions.
- Raster-specific derivative tuning such as `pixel / 24` can leave destination Physics.
- Deep Mandelbrot travel can refine local geometry rather than eventually exhausting a global texture.
- Genuine cut loci can be represented as explicit validity/singularity semantics instead of being confused with interpolation artifacts.
- The same geometry-provider seam can support navigation, debug diagnostics, and future special mechanics without creating additional distance authorities.
- The 2048² Player mip pyramid can remain optimized for observation rather than being burdened with second-order Physics requirements.

### Negative / cost

- The geometry provider becomes a more substantial runtime subsystem.
- Adaptive/local representations require deterministic caching, versioning, and invalidation rules.
- Second-order differential quality must be tested explicitly across tile/refinement boundaries.
- Genuine signed-distance singularities still require explicit semantics; a better provider does not make them disappear.
- Python/WASM parity and runtime cost must be revalidated.

---

## Migration

The current 1024² signed-distance field becomes a **bridge/fallback**, not destination authority.

Migration is one-way:

```text
legacy 1024² signed raster
        -> temporary GeometryProvider adapter if useful
        -> destination scale-aware GeometryProvider
        -> Physics
```

Destination Physics must not depend on raster-specific concepts such as fixed pixel spacing or a tuned finite-difference fraction.

Issue #84 tracks bridge/deletion consequences.

The current `derivative_step()` and nested finite-difference derivative path should be deleted once no surviving destination consumer requires them.

The existing 2048² `F/S` mip pyramid is not part of this demolition. It remains valid for Player minimap/cartographic evidence unless separately superseded.

---

## Validation

Issue #120 remains the validation owner. Issue #145 implements the replacement provider.

After a candidate provider exists, validation should distinguish at least:

```text
1. smooth analytic/control geometry
2. regular Mandelbrot Shore geometry across multiple scales
3. genuine cusp / cut-locus / singular stress geometry
```

For regular regions, acceptance should include:

- coherent `D`, `grad D`, and `H_D`;
- stable metric eigenvalues/condition number;
- stable connection behavior;
- understood timestep and conservative-energy convergence;
- scale-relative Euclidean motion and control authority;
- deterministic behavior across refinement/tile boundaries;
- runtime cost appropriate for the live visualizer.

For genuine nonsmooth regions, acceptance is different:

- deterministic classification or validity semantics;
- bounded/fail-closed behavior;
- no fabricated unique normal/Hessian;
- no hidden local smoothing introduced solely to make conservation tests pass.

#111 should expose provider/version, requested/resolved scale, validity/singularity state, and bridge-vs-destination provenance for eyes-on diagnosis.

---

## Alternatives considered

### Keep the 1024² raster and widen tolerances

Rejected. This masks a representation mismatch and does not address scale depth or derivative consistency.

### Replace 1024² with a larger fixed raster

Rejected as the destination. A 2048² field buys only one additional octave of linear resolution. Larger global rasters postpone the same failure while increasing memory and bake cost.

### Reuse the 2048² F/S mip pyramid as Physics authority

Rejected. `F` and `S` are Player/cartographic fields with different semantics. `S` is unsigned proximity/sensitivity, not canonical signed distance, and the pyramid does not provide the coherent second-order signed geometry required by Physics.

### Promote `deep_zoom_field` directly to Physics authority

Rejected in its current form. It is useful resolution-unlimited infrastructure but currently lacks the signed differential and singularity semantics required by the Physics geometry contract.

### Locally smooth Hessians inside Physics

Rejected. That creates a second hidden geometry authority and can erase genuine singularities. Any regularization must belong to the canonical provider representation and be versioned/validated there.

### Change the canonical scale-relative metric

Rejected by this ADR. #120 already adopted the metric. ADR 0004 addresses the representation supplying its geometry.

---

## Follow-up

- #145 implements the scale-aware GeometryProvider and migration seam.
- #120 validates the new provider and remains open until the numerical/eyes-on acceptance program is satisfied.
- #84 records bridge/deletion triggers for the 1024² raster and derivative-step path.
- #111 exposes geometry-provider diagnostics.
- #137 consumes the provider through an explicit exterior/interior handoff rather than extending raster-edge extrapolation as permanent geometry.
- #138 should consume intrinsic geometry through the same authoritative provider rather than developing a separate navigation distance authority.

The durable rule is:

> **The Map must provide differential geometry at the scale the geometry itself demands.**
