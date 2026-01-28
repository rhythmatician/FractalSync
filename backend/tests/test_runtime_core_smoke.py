from types import ModuleType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime_core import FeatureExtractor, OrbitState, ResidualParams, Complex  # noqa: F401


def test_runtime_core_smoke(runtime_core_module: ModuleType) -> None:
    rc = runtime_core_module

    # Basic shape: public constants and types
    assert hasattr(rc, "SAMPLE_RATE")

    # FeatureExtractor: construct and call `test_simple`
    assert hasattr(rc, "FeatureExtractor"), "FeatureExtractor class missing"
    fe: FeatureExtractor = rc.FeatureExtractor()
    assert hasattr(fe, "test_simple")
    res: list[float] = fe.test_simple()
    # Some builds return `True`; others return a non-empty numeric sequence.
    print(f"LOOK============================={type(res)}")
    assert isinstance(res, list), "test_simple() must return a list of floats"
    assert len(res) > 0
    assert all(
        isinstance(x, float) for x in res
    ), "test_simple() list must contain floats"

    # For this test we require the OrbitState API to be present and test it strictly
    assert hasattr(rc, "OrbitState"), "OrbitState must be present in this build"

    # Ensure there is a known constructor available
    orbit_state_instance: OrbitState = rc.OrbitState
    assert (
        hasattr(orbit_state_instance, "new_default_seeded")
        or hasattr(orbit_state_instance, "new_with_seed")
        or callable(orbit_state_instance)
    ), "OrbitState lacks known constructors"

    # Construct using the best available constructor (try in order)
    st: "OrbitState" = orbit_state_instance.new_default_seeded(1337)  # type: ignore[arg-type]

    assert st is not None
    assert hasattr(st, "carrier")
    c: "Complex" = st.carrier()  # type: ignore[attr-defined]
    # Complex types may expose re/im or real/imag or __complex__
    assert (
        (hasattr(c, "re") and hasattr(c, "im"))
        or (hasattr(c, "real") and hasattr(c, "imag"))
        or hasattr(c, "__complex__")
    )

    # Residual params are expected for Orbit-based builds
    assert hasattr(
        rc, "ResidualParams"
    ), "ResidualParams missing in Orbit-enabled build"
    rp = rc.ResidualParams()
    assert hasattr(rp, "k_residuals")
