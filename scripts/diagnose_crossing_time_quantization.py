"""Check the original sampled crossing criterion against an exact solution.

This is a diagnostic, not a replacement for the original crossing regression.
"""

from fractions import Fraction


def main() -> None:
    # Exact solution x(t) = crossing_time - t, x(0) > 0.  The first sampled
    # negative state is deliberately recorded exactly as the regression does.
    crossing_time = Fraction(3863, 1000) + Fraction(270565, 1_000_000_000)
    steps = (Fraction(2, 1000), Fraction(1, 1000), Fraction(1, 2000))
    samples = []
    for dt in steps:
        index = crossing_time // dt + 1
        samples.append(index * dt)
    coarse, middle, fine = samples
    coarse_gap = abs(coarse - middle)
    fine_gap = abs(middle - fine)
    assert coarse_gap == 0 and fine_gap == Fraction(1, 2000)
    print(f"exact crossing: {float(crossing_time):.9f} s")
    print("first sampled negative state:", *(float(t) for t in samples), "s")
    print(f"coarse gap: {float(coarse_gap):.4f} s; fine gap: {float(fine_gap):.4f} s")
    print("original strict gap-refinement criterion:", fine_gap < coarse_gap)


if __name__ == "__main__":
    main()
