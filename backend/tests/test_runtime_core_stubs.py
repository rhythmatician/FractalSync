"""Validate runtime_core stubs against the actual bindings."""

from __future__ import annotations

import pytest

rc = pytest.importorskip("runtime_core")

EXPECTED_CLASSES = {
    "Complex": [],
    "FeatureExtractor": [
        "feature_mean",
        "feature_std",
        "num_features_per_frame",
        "extract_windowed_features",
        "compute_normalization_stats",
        "normalize_features",
    ],
    "ResidualParams": [
        "k_residuals",
        "residual_cap",
        "radius_scale",
    ],
    "OrbitState": [
        "lobe",
        "sub_lobe",
        "theta",
        "omega",
        "s",
        "alpha",
        "k_residuals",
        "residual_omega_scale",
        "step",
        "synthesize",
        "clone",
    ],
}


def _get_class(cls_name: str):
    if not hasattr(rc, cls_name):
        pytest.fail(f"runtime_core missing expected class {cls_name}")
    return getattr(rc, cls_name)


def test_classes_have_expected_members():
    """For each class in our stubs check the expected attributes/methods exist.

    We attempt a conservative check: look for the member on the class object
    and (when possible) on a default instance. For certain classes we use
    safe fallbacks to obtain an instance (e.g. `lobe_point_at_angle` for
    `Complex`, explicit constructor args for `OrbitState`). This makes the
    test tolerant of minor ABI differences while verifying presence of the
    API surface described in the stubs.
    """
    all_missing = {}
    for cls_name, members in EXPECTED_CLASSES.items():
        cls = _get_class(cls_name)

        # Try to obtain an instance for instance-level checks. Be conservative
        # and avoid letting exceptions fail the entire test suite.
        inst = None
        try:
            inst = cls()
        except Exception:
            # Fallbacks for commonly non-default-constructible types
            try:
                if cls_name == "OrbitState":
                    inst = cls(
                        1,
                        0,
                        0.0,
                        float(getattr(rc, "DEFAULT_BASE_OMEGA", 0.15)),
                        1.02,
                        0.3,
                        int(getattr(rc, "DEFAULT_K_RESIDUALS", 6)),
                        float(getattr(rc, "DEFAULT_RESIDUAL_OMEGA_SCALE", 1.0)),
                    )
                elif cls_name == "Complex" and hasattr(rc, "lobe_point_at_angle"):
                    inst = rc.lobe_point_at_angle(1, 0, 0.1, 1.0)
                elif cls_name == "ResidualParams":
                    inst = cls()
                elif cls_name == "FeatureExtractor":
                    inst = cls()
            except Exception:
                inst = None

        # Complex has several acceptable access patterns; treat as alternative group
        if cls_name == "Complex":
            has_re_im = (hasattr(cls, "re") and hasattr(cls, "im")) or (
                inst is not None and hasattr(inst, "re") and hasattr(inst, "im")
            )
            has_real_imag = (hasattr(cls, "real") and hasattr(cls, "imag")) or (
                inst is not None and hasattr(inst, "real") and hasattr(inst, "imag")
            )
            has_complex = hasattr(cls, "__complex__") or (
                inst is not None and hasattr(inst, "__complex__")
            )
            if not (has_re_im or has_real_imag or has_complex):
                pytest.fail(
                    f"Complex type lacks any expected accessor patterns (re/im, real/imag, "
                    f"or __complex__) on class {cls} or its instances"
                )
            continue

        missing = []
        for m in members:
            present = False
            # Member on the class object (e.g., classmethod, attribute)
            if hasattr(cls, m):
                present = True
            # Member on an instance (e.g., property/method bound to instance)
            if not present and inst is not None and hasattr(inst, m):
                present = True
            if not present:
                missing.append(m)

        if missing:
            all_missing[cls_name] = missing

    if all_missing:
        messages = [f"{k}: missing {sorted(v)}" for k, v in all_missing.items()]
        pytest.fail("Runtime core API differs from stubs:\n" + "\n".join(messages))
