# Aeneas and Charon feasibility for FractalSync

**Decision for [#158](https://github.com/rhythmatician/FractalSync/issues/158).** Defer integration. Controls offers a small discrete proof candidate. Both extraction attempts stopped before LLBC generation at a missing compiler driver, so the spike produced zero generated Lean theorems. For the inverse metric, the Lean backend's rejection of float literals is a separate blocker visible in upstream source. Keep the existing formal calculations and runtime parity checks in place. Revisit integration when a pinned runner can complete the Controls proof or when the float backend can model the operations in a selected Physics function.

## Pins and runtime constraints

| Set | Aeneas | Charon | Rust | Lean and Mathlib |
| --- | --- | --- | --- | --- |
| Matches `formal/` by declared dependency pins | `d6ff3182a6b0b9c5a0376ac0ff19bf69a8a74afc` | `b51c1fcb513d7c41a5c2a6d73b5cca4d3ad908a8` | `nightly-2025-07-20` with `rustc-dev`, `llvm-tools-preview`, and `rust-src` | Lean 4.24.0 and Mathlib `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` |
| Prebuilt diagnostic release, 2026-11-26 | `69280e07ed995241a4e23ef15937f3a2dea18fd2` | `5376a9cf0b58d006b5da7ec70b14f6e0cac68498` | `nightly-2026-11-25` | Lean and Mathlib 4.36.0 |

The first row comes from the Aeneas Lean package's manifest, its Charon pin, and Charon's compiler pin. The matching dependencies allow a future experiment in a temporary copy of `formal/` with its existing Lean installation and Mathlib cache. Building that older tool pair needs the pinned Rust nightly and the Aeneas build environment. This spike checked the declared pins and left the package untouched. [Aeneas Lean toolchain](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/backends/lean/lean-toolchain), [Lean package manifest](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/backends/lean/lake-manifest.json), [Charon pin](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/charon-pin), [Rust toolchain](https://github.com/AeneasVerif/charon/blob/b51c1fcb513d7c41a5c2a6d73b5cca4d3ad908a8/rust-toolchain), [build instructions](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/README.md).

The release used for the diagnostic ships Linux and macOS binaries. WSL2 Debian x86-64 ran the Linux launchers from `/dev/shm`. The Aeneas bundle reported the exact Charon commit in its pin. The standalone Charon nightly with the same date reported `c80ab163893fdb3b4825339e6f75458c592ccbd3`. Pair by commit, since matching release dates can select different builds. Charon's driver links against a particular nightly's compiler internals. Both host Windows and Debian initially listed stable Rust toolchains only. [Aeneas release assets](https://github.com/AeneasVerif/aeneas/releases/tag/nightly-2026.11.26-69280e0), [bundled Charon pin](https://github.com/AeneasVerif/aeneas/blob/69280e0/charon-pin), [release Lean toolchain](https://github.com/AeneasVerif/aeneas/blob/69280e0/backends/lean/lean-toolchain), [standalone Charon release](https://github.com/AeneasVerif/charon/releases/tag/nightly-2026.11.26).

## Translation attempts

The source revision was `34f5fd2bc619956620a207069b6e039f778bb767`. The Controls fixture copies the `Harmony` enum and the two indexing methods, omitting derives and unrelated impl members. The Physics fixture copies `inverse_2x2` verbatim. It is the inverse used by `apply_generalized_force` to turn a force covector into coordinate acceleration. [Controls](../../runtime-core/src/controls.rs#L182), [inverse metric](../../runtime-core/src/manifold.rs#L328), [generalized force](../../runtime-core/src/manifold.rs#L552).

The following commands run from a WSL scratch directory. Set `repo` to the absolute WSL path of the checkout. The stable compiler passed both fixture compilations, with an unused-function warning for the isolated inverse.

```bash
mkdir aeneas
curl -fLsS https://github.com/AeneasVerif/aeneas/releases/download/nightly-2026.11.26-69280e0/aeneas-linux-x86_64.tar.gz |
    tar -xz -C aeneas

rev=34f5fd2bc619956620a207069b6e039f778bb767
git -C "$repo" show "$rev:runtime-core/src/controls.rs" > controls.rs
git -C "$repo" show "$rev:runtime-core/src/manifold.rs" > manifold.rs
sed -n '182,188p;200,214p' controls.rs > harmony.rs
echo '}' >> harmony.rs
sed -n '329,339p' manifold.rs > inverse.rs

rustc --crate-type lib harmony.rs
rustc --crate-type lib inverse.rs
aeneas/aeneas -version
aeneas/charon version
aeneas/charon toolchain-version

CHARON_TOOLCHAIN_IS_IN_PATH=1 aeneas/charon rustc --preset=aeneas \
    --sysroot default --dest-file=harmony.llbc -- --crate-type=lib harmony.rs
CHARON_TOOLCHAIN_IS_IN_PATH=1 aeneas/charon rustc --preset=aeneas \
    --sysroot default --dest-file=inverse.llbc -- --crate-type=lib inverse.rs
aeneas/aeneas -backend lean harmony.llbc
aeneas/aeneas -backend lean inverse.llbc
```

`CHARON_TOOLCHAIN_IS_IN_PATH` selects the path that skips Charon's Rust installation code. Both Charon commands stopped while querying the driver compiler version. The dynamic loader reported `librustc_driver-a0f7070336ff7044.so` was missing, and `ldd aeneas/charon-driver` identified the same missing library. Aeneas confirmed both LLBC inputs were absent, so there was no generated Lean representation to import for either proof. A `rustup which --toolchain ...` availability probe itself triggered downloads for both nightlies. I stopped the install and used `rustup toolchain uninstall` for those two names. The resulting toolchain list again contained only the original stable toolchain. The releases and fixtures were extracted into RAM because the Windows host had about 1.1 GB free at the start. [Charon toolchain selection code](https://github.com/AeneasVerif/charon/blob/5376a9cf0b58d006b5da7ec70b14f6e0cac68498/charon/src/bin/charon/toolchain.rs).

### Controls

The proposed theorem is that for every unsigned machine index \(i\), the generated `from_index(i)` succeeds and the generated `index` of its result succeeds with \(i \bmod 3\). The remainder lies in \(\{0,1,2\}\), so the panic arm is unreachable. Each of those cases maps through the corresponding enum constructor back to the same remainder. This property is about the two method bodies and their success results. It needs no initial streaming state or environmental assumptions. The extraction failure prevented a proof against the generated definitions. [Upstream Lean tutorial and proof examples](https://github.com/AeneasVerif/aeneas/tree/d6ff3182a6b0/tests/lean/Tutorial).

### Physics

The inverse includes two product terms, subtraction, an absolute value call, comparison with a float literal, division, negation, more multiplications, fixed-array access, and construction of a `Result` with a string on failure. The small fixture has no Map provider or third-party crate. Charon's checked-in test fixtures show how simple `f64` inputs and sums appear in LLBC. Both Aeneas pins explicitly reject float literals during export, including the kind used for the determinant threshold and reciprocal numerator. Its ability to print a float type name gives no semantics for those operations. Extraction of this function's `f64::abs` and error-path string conversion remains untested. A handwritten external model for either call would require a fidelity argument. [Charon float fixture and golden output](https://github.com/AeneasVerif/charon/blob/b51c1fcb513d7c41a5c2a6d73b5cca4d3ad908a8/charon/tests/ui/float.out), [Lean 4.24 extractor](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/src/extract/ExtractTypes.ml), [diagnostic-release extractor](https://github.com/AeneasVerif/aeneas/blob/69280e0/src/extract/ExtractTypes.ml).

The real-valued reference is the inverse of \(G=\rho^{-2}I+a\,g g^T\) under the hypotheses in the existing determinant and inverse identities. The Rust guard compares the absolute value of a computed determinant with the binary64 representation of \(10^{-14}\). Two stable Rust tests on the fixture illustrate where a theorem about ideal positive definiteness stops applying.

- The symmetric matrix with rows \((1,1)\) and \((1,1+2^{-52})\) has positive determinant and returns `Err("singular metric")`.
- The symmetric matrix with rows \((10^{200},10^{200})\) and \((10^{200},2\cdot10^{200})\) has finite entries and is positive definite when its represented entries are interpreted as reals. Both products overflow to infinity. Their subtraction yields NaN, the comparison evaluates to false, and the implementation returns `Ok` with four NaNs.

Both assertions passed with `rustc 1.98.0` under WSL. These are tests of the isolated inverse function. They leave open whether a valid provider and Physics configuration can reach those matrices. [Lean scale metric specification](../../formal/FractalSync/ScaleMetric.lean).

A candidate correctness statement is that under an entry bound, finite-intermediate assumptions, and a determinant margin, the inverse returns `Ok` and its entries approximate the real inverse of the represented matrix with a stated error. Write \(D=m_{00}m_{11}-m_{01}m_{10}\). The absolute determinant error is bounded by the sum of the product and subtraction rounding errors. To certify the success branch, the lower bound on \(|D|\) minus that error budget must exceed the binary64 threshold. The reciprocal is sensitive to this margin because the exact difference between reciprocals grows as \(|\widehat D-D|/(|D||\widehat D|)\), before rounding the division and the final four products. A proof must handle underflow or constrain it, exclude NaN and infinities on the promised domain, and establish the provider bounds separately. FloatSpec has relevant arithmetic and error-bound definitions. Its inspected revision uses Lean 4.34.0-rc2, and connecting a float model to generated Rust would still be a separate obligation. [FloatSpec library and scope](https://github.com/Beneficial-AI-Foundation/FloatSpec/tree/158263e983ec3925e02b10f5b312498bf414e0f1), [FloatSpec toolchain](https://github.com/Beneficial-AI-Foundation/FloatSpec/blob/158263e983ec3925e02b10f5b312498bf414e0f1/lean-toolchain).

## Verification boundary and cost

- A generated-code proof would depend on the pinned rustc MIR, Charon's LLBC translation, Aeneas's translation and standard-library models, and Lean's kernel. Audit the produced theorem with `#print axioms` through the package's audit workflow. Also review external models and the extraction options. An axiom audit cannot establish that a handwritten definition implements Rust or that an extracted Linux result covers a WASM build. In particular, `Harmony` uses target-sized `usize`. A future proof must record the target width or quantify over supported widths. [Aeneas translation and proof scope](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/README.md), [Lean audit instructions](../../formal/README.md).
- Restart with the Lean 4.24 pair on a Linux runner with the pinned compiler already available. Generate each LLBC with the paired Charon, run `aeneas -backend lean harmony.llbc`, then import the generated code into a temporary copy of the existing package. Trace the source symbols and generated result branches, prove the modulo property, and run the package build and axiom audit. Upstream's Lean tutorial uses the generated monadic code and its specification lemmas. CI would eventually need to regenerate from the actual source and check the reviewed proof. Each toolchain update would require checking translated output and its theorem again. Time for a successful extraction and proof remains unmeasured. [Pinned Aeneas commands and Lean package setup](https://github.com/AeneasVerif/aeneas/blob/d6ff3182a6b0/README.md), [Aeneas agent tactic guidance for newer releases](https://github.com/AeneasVerif/aeneas/blob/69280e0/.claude/skills/aeneas-tactics-quickref/SKILL.md).
- The most promising future seams are small, integer-only Controls or index transitions with few external models. For Physics, keep using Lean for exact mechanics and separately track Map/provider error, integration truncation error, and per-operation rounding and conditioning. Retain numerical and parity tests for operational inputs and error bounds that lack a checked implementation-level theorem. A module-level floating-point extraction pilot would first need literal, arithmetic, comparison, and library-call models covering the selected function.
