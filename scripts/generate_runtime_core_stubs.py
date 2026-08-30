"""Generate runtime_core Python stubs by introspecting the live module.

Replaces the previous metadata-based generator (which consumed a
hand-written ``export_binding_metadata()`` copy of every signature in
pybindings.rs — a third copy of the API surface that could drift
unchecked). Structure (names, parameters, defaults, member presence) now
comes from the compiled module itself, so it can never drift from the
bindings. Only Python-facing type annotations live in the table below,
because PyO3 0.21 does not expose Python types through introspection.

Usage:
    python scripts/generate_runtime_core_stubs.py -o backend/stubs/runtime_core

Requires the ``runtime_core`` extension to be importable (run
``maturin develop --release`` in runtime-core/ first).
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Callable

import runtime_core as rc

# ---------------------------------------------------------------------------
# Type-annotation table.
#
# Keys are "callable or member" identifiers; values are (params, return)
# annotation strings where params maps parameter name -> annotation. A
# parameter present in the live signature but absent here gets no
# annotation (rendered bare, which mypy treats as Unknown). Defaults are
# always taken from the live module, never from this table.
#
# This table is the ONLY hand-maintained part. It changes rarely (types
# are stable; names/params/defaults are not) and any mismatch between
# table keys and the live module is reported by the generator.
# ---------------------------------------------------------------------------

RET = "__return__"

FUNCTION_TYPES: dict[str, dict[str, str]] = {
    "set_distance_field_py": {
        "data": "Sequence[Sequence[float]]",
        "xmin": "float",
        "xmax": "float",
        "ymin": "float",
        "ymax": "float",
        RET: "None",
    },
    "sample_distance_field_py": {
        "coords": "Sequence[complex]",
        RET: "list[float]",
    },
    "get_builtin_distance_field_py": {
        "name": "str",
        RET: "tuple[int, int, float, float, float, float]",
    },
    "residual_phases_for_seed_py": {
        "seed": "int",
        "k_residuals": "int",
        RET: "list[float]",
    },
    "load_mip_pyramid_py": {
        "f_bin_path": "str",
        "s_bin_path": "str",
        "meta_path": "str",
        RET: "tuple[int, float, float, float, float]",
    },
    "install_pyramid_py": {
        "levels_data": "Sequence[Sequence[float]]",
        "widths": "Sequence[int]",
        "heights": "Sequence[int]",
        "re_min": "float",
        "re_max": "float",
        "im_min": "float",
        "im_max": "float",
        RET: "int",
    },
    "clear_pyramid_py": {RET: "None"},
    "player_observation_py": {
        "c_re": "float",
        "c_im": "float",
        RET: "list[float]",
    },
    "minimap_slope_py": {
        "c_re": "float",
        "c_im": "float",
        "level": "int",
        RET: "tuple[float, float]",
    },
    "minimap_shore_proximity_batch_py": {
        "re": "Sequence[float]",
        "im": "Sequence[float]",
        "level": "int",
        RET: "list[float]",
    },
    "contour_biased_step_py": {
        "c_re": "float",
        "c_im": "float",
        "u_re": "float",
        "u_im": "float",
        "h": "float",
        "d_star": "float",
        "max_step": "float",
        "level": "int",
        "energy": "float",
        RET: "tuple[float, float]",
    },
    "mandelbrot_distance_estimate": {
        "coords": "Union[Sequence[complex], tuple[Sequence[float], Sequence[float]]]",
        "ys": "Optional[Sequence[float]]",
        RET: "list[float]",
    },
    "mandelbrot_distance_estimate_py": {
        "coords": "Union[Sequence[complex], tuple[Sequence[float], Sequence[float]]]",
        "ys": "Optional[Sequence[float]]",
        RET: "list[float]",
    },
    "mandelbrot_cardioid_proximity_batch": {
        "coords": "Sequence[complex]",
        RET: "list[float]",
    },
    "orbit_path_metrics_py": {
        "coords": "Sequence[complex]",
        RET: "tuple[float, float, float]",
    },
    "compute_runtime_visual_metrics": {
        "image": "Sequence[float]",
        "width": "int",
        "height": "int",
        "channels": "int",
        "c": "complex",
        "max_iter": "int",
        RET: "RuntimeVisualMetrics",
    },
    "lobe_point_at_angle": {
        "lobe": "int",
        "sub_lobe": "int",
        "theta": "float",
        "s": "float",
        RET: "complex",
    },
}

# Per-class member annotations. "__init__" uses the constructor's
# __text_signature__ parameter names. Getters (getset_descriptor) are
# rendered as @property with the annotation from ATTR_TYPES.
CLASS_METHOD_TYPES: dict[str, dict[str, dict[str, str]]] = {
    "FeatureExtractor": {
        "__init__": {
            "sr": "int",
            "hop_length": "int",
            "n_fft": "int",
            "include_delta": "bool",
            "include_delta_delta": "bool",
            RET: "None",
        },
        "num_features_per_frame": {RET: "int"},
        "extract_windowed_features": {
            "audio": "Union[Sequence[float], NDArray[np.floating]]",
            "window_frames": "int",
            RET: "NDArray",
        },
        "test_simple": {RET: "list[float]"},
        "compute_normalization_stats": {
            "all_features": "Union[Sequence[Sequence[float]], Sequence[NDArray[np.floating]]]",
            RET: "None",
        },
        "normalize_features": {
            "features": "Union[Sequence[float], NDArray[np.floating]]",
            RET: "list[float]",
        },
    },
    "ResidualParams": {
        "__init__": {
            "k_residuals": "int",
            "residual_cap": "float",
            "radius_scale": "float",
            RET: "None",
        },
    },
    "OrbitState": {
        "__init__": {
            "lobe": "int",
            "sub_lobe": "int",
            "theta": "float",
            "omega": "float",
            "s": "float",
            "alpha": "float",
            "k_residuals": "int",
            "residual_omega_scale": "float",
            "seed": "Optional[int]",
            RET: "None",
        },
        "new_with_seed": {
            "lobe": "int",
            "sub_lobe": "int",
            "theta": "float",
            "omega": "float",
            "s": "float",
            "alpha": "float",
            "k_residuals": "int",
            "residual_omega_scale": "float",
            "seed": "int",
            RET: "OrbitState",
        },
        "new_default_seeded": {"seed": "int", RET: "OrbitState"},
        "advance": {"dt": "float", RET: "None"},
        "carrier": {RET: "complex"},
        "residual_phases": {RET: "list[float]"},
        "residual_omegas": {RET: "list[float]"},
        "synthesize": {
            "residual_params": "ResidualParams",
            "band_gates": "Optional[list[float]]",
            RET: "complex",
        },
        "step": {
            "dt": "float",
            "residual_params": "ResidualParams",
            "band_gates": "Optional[list[float]]",
            RET: "complex",
        },
    },
    "PlayerState": {
        "__init__": {
            "lobe": "int",
            "sub_lobe": "int",
            "s": "float",
            "alpha": "float",
            RET: "None",
        },
        "apply_controls": {
            "s": "float",
            "alpha": "float",
            "omega_scale": "float",
            RET: "None",
        },
        "set_lobe": {"lobe": "int", "sub_lobe": "int", RET: "None"},
        "set_level": {"level": "int", RET: "None"},
        "set_d_star": {"d_star": "float", RET: "None"},
        "set_max_step": {"max_step": "float", RET: "None"},
        "set_energy": {"energy": "float", RET: "None"},
        "step": {
            "dt": "float",
            "h": "float",
            "band_gates": "Optional[list[float]]",
            RET: "tuple[float, float]",
        },
    },
    "OrbitController": {
        "__init__": {
            "s": "float",
            "alpha": "float",
            "omega": "float",
            RET: "None",
        },
        "apply_controls": {"s": "float", "alpha": "float", RET: "None"},
        "set_momentum": {"on": "bool", RET: "None"},
        "set_drag": {"drag": "float", RET: "None"},
        "set_thrust": {"thrust": "float", RET: "None"},
        "set_energy": {"energy": "float", RET: "None"},
        "set_shore_bias": {"on": "bool", RET: "None"},
        "set_d_star": {"d_star": "float", RET: "None"},
        "set_max_step": {"max_step": "float", RET: "None"},
        "set_level": {"level": "int", RET: "None"},
        "set_c": {"re": "float", "im": "float", RET: "None"},
        "step": {
            "dt": "float",
            "band_gates": "Optional[list[float]]",
            "h": "float",
            RET: "tuple[float, float]",
        },
    },
}

# Attribute annotations for getset_descriptor members (rendered as
# @property). Types inferred from a live instance where possible; this
# table is the fallback / authority.
ATTR_TYPES: dict[str, dict[str, str]] = {
    "ResidualParams": {
        "k_residuals": "int",
        "residual_cap": "float",
        "radius_scale": "float",
    },
    "OrbitState": {
        "lobe": "int",
        "sub_lobe": "int",
        "theta": "float",
        "omega": "float",
        "s": "float",
        "alpha": "float",
    },
    "PlayerState": {"c_re": "float", "c_im": "float", "speed": "float"},
    "OrbitController": {"theta": "float"},
    "FeatureExtractor": {
        "feature_mean": "Optional[list[float]]",
        "feature_std": "Optional[list[float]]",
    },
    "RuntimeVisualMetrics": {
        "edge_density": "float",
        "color_uniformity": "float",
        "brightness_mean": "float",
        "brightness_std": "float",
        "brightness_range": "float",
        "mandelbrot_membership": "bool",
    },
}

# Constants rendered as module-level typed names. Values come from the
# live module; only the annotation is declared here.
CONSTANT_TYPES: dict[str, str] = {
    "SAMPLE_RATE": "int",
    "HOP_LENGTH": "int",
    "N_FFT": "int",
    "WINDOW_FRAMES": "int",
    "DEFAULT_K_RESIDUALS": "int",
    "DEFAULT_RESIDUAL_CAP": "float",
    "DEFAULT_RESIDUAL_OMEGA_SCALE": "float",
    "DEFAULT_BASE_OMEGA": "float",
    "DEFAULT_ORBIT_SEED": "int",
    "CONTROLLER_VERSION": "str",
    "FEATURE_VERSION": "str",
    "NORM_EPS": "float",
}

# Classes to document, in stable order.
CLASS_ORDER = [
    "FeatureExtractor",
    "ResidualParams",
    "OrbitState",
    "PlayerState",
    "OrbitController",
    "RuntimeVisualMetrics",
]

# Functions to document, in stable order.
FUNCTION_ORDER = [
    "set_distance_field_py",
    "sample_distance_field_py",
    "get_builtin_distance_field_py",
    "residual_phases_for_seed_py",
    "load_mip_pyramid_py",
    "install_pyramid_py",
    "clear_pyramid_py",
    "player_observation_py",
    "minimap_slope_py",
    "minimap_shore_proximity_batch_py",
    "contour_biased_step_py",
    "mandelbrot_distance_estimate",
    "mandelbrot_distance_estimate_py",
    "mandelbrot_cardioid_proximity_batch",
    "orbit_path_metrics_py",
    "compute_runtime_visual_metrics",
    "lobe_point_at_angle",
]

HEADER = '''"""Type stubs for the ``runtime_core`` native extension.

Generated by ``scripts/generate_runtime_core_stubs.py`` — do not edit by
hand. Structure (names, parameters, defaults) is introspected from the
compiled module; Python-facing type annotations come from the generator's
annotation tables. Regenerate with:

    maturin develop --release   # in runtime-core/
    python scripts/generate_runtime_core_stubs.py
"""

from typing import Optional, Sequence, Union
from numpy.typing import NDArray
import numpy as np

'''

# Instance-construction recipes for attribute type inference. Classes
# absent from this map fall back to the annotation table. Lookups go
# through getattr because Pylance resolves `runtime_core` to the
# installed stub, which does not declare these classes.
INSTANCE_RECIPES: dict[str, Callable[[], Any]] = {
    "ResidualParams": lambda: getattr(rc, "ResidualParams")(),
    "OrbitState": lambda: getattr(rc, "OrbitState").new_default_seeded(42),
    "FeatureExtractor": lambda: getattr(rc, "FeatureExtractor")(),
}


def _format_default(value: Any) -> str:
    """Render a live default value as stub syntax."""
    if value is Ellipsis:
        # PyO3 renders Optional params with no explicit default as
        # Ellipsis; the stub convention is `...`.
        return "..."
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return repr(value)


def _signature_parts(
    func: Any,
    annotations: dict[str, str] | None,
    skip_first_self: bool,
) -> tuple[list[str], str | None]:
    """Build (param strings, return annotation) from a live callable."""
    annotations = annotations or {}
    sig = inspect.signature(func)
    params: list[str] = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        if skip_first_self and i == 0 and name == "self":
            continue
        ann = annotations.get(name)
        if p.default is inspect.Parameter.empty:
            piece = f"{name}: {ann}" if ann else name
        else:
            default = _format_default(p.default)
            piece = f"{name}: {ann} = {default}" if ann else f"{name} = {default}"
        params.append(piece)
    ret = annotations.get(RET)
    return params, ret


def _render_function(name: str, func: Any, annotations: dict[str, str] | None) -> str:
    params, ret = _signature_parts(func, annotations, skip_first_self=False)
    ret_part = f" -> {ret}" if ret else ""
    return f"def {name}({', '.join(params)}){ret_part}: ...\n"


def _render_method(
    cls_name: str, name: str, func: Any, annotations: dict[str, str] | None
) -> str:
    # PyO3 staticmethods surface as plain builtin functions (no `self`
    # first param); instance methods surface as method_descriptor with a
    # leading `self`. Instance methods KEEP `self` in the stub; the
    # skip flag only exists for the (unused) case where a live signature
    # carries an implicit self that the stub must not show.
    is_static = "self" not in inspect.signature(func).parameters
    params, ret = _signature_parts(func, annotations, skip_first_self=False)
    ret_part = f" -> {ret}" if ret else ""
    prefix = "    @staticmethod\n" if is_static else ""
    return f"{prefix}    def {name}({', '.join(params)}){ret_part}: ...\n"


def _render_property(name: str, annotation: str | None, settable: bool) -> str:
    ann = f" -> {annotation}" if annotation else ""
    if settable:
        # Descriptor has fset: render as a mutable attribute so mypy
        # accepts writes (e.g. live_controller mutating OrbitState.s).
        return f"    {name}: {annotation or 'Any'}\n"
    return f"    @property\n    def {name}(self){ann}: ...\n"


def _descriptor_is_settable(cls: type, name: str, descriptor: Any) -> bool:
    """Return True if a getset_descriptor accepts writes.

    PyO3 getset_descriptors do not expose ``fset``, so the only reliable
    test is to attempt a write against a throwaway instance and catch the
    AttributeError that read-only descriptors raise. Instances come from
    INSTANCE_RECIPES (``cls.__new__`` routes to PyO3's ``#[new]`` and
    needs full constructor args).
    """
    recipe = INSTANCE_RECIPES.get(cls.__name__)
    if recipe is None:
        return False
    try:
        instance = recipe()
    except Exception:
        return False
    try:
        descriptor.__set__(instance, None)
    except AttributeError:
        return False
    except Exception:
        # Writable but rejected the None probe value (e.g. type check):
        # the attribute is still settable.
        return True
    return True


def _render_class(cls_name: str, cls: type) -> str:
    lines = [f"class {cls_name}:", ""]
    method_types = CLASS_METHOD_TYPES.get(cls_name, {})
    attr_types = ATTR_TYPES.get(cls_name, {})

    # Constructor from __text_signature__ (PyO3 exposes it on the class).
    init_ann = method_types.get("__init__")
    text_sig = getattr(cls, "__text_signature__", None)
    if text_sig:
        params: list[str] = []
        for raw in text_sig.strip("()").split(","):
            raw = raw.strip()
            if not raw:
                continue
            if "=" in raw:
                pname, default = raw.split("=", 1)
                pname = pname.strip()
                ann = (init_ann or {}).get(pname)
                rendered = _render_text_sig_default(default)
                params.append(
                    f"{pname}: {ann} = {rendered}" if ann else f"{pname} = {rendered}"
                )
            else:
                ann = (init_ann or {}).get(raw)
                params.append(f"{raw}: {ann}" if ann else raw)
        ret = (init_ann or {}).get(RET, "None")
        lines.append(f"    def __init__(self, {', '.join(params)}) -> {ret}: ...")
        lines.append("")

    # Members from the live class, in dir() order (alphabetical, stable).
    # PyO3 `#[pyo3(get)]` fields and `#[getter]` methods surface as
    # getset_descriptors; a descriptor with fset (getter+setter pair or
    # `#[pyo3(get, set)]` field) is rendered as a mutable attribute so
    # mypy accepts writes. Read-only descriptors render as @property.
    for member_name in dir(cls):
        if member_name.startswith("_"):
            continue
        member = getattr(cls, member_name)
        kind = type(member).__name__
        if kind == "getset_descriptor" or not callable(member):
            settable = kind == "getset_descriptor" and _descriptor_is_settable(
                cls, member_name, member
            )
            lines.append(
                _render_property(member_name, attr_types.get(member_name), settable)
            )
        elif callable(member):
            lines.append(
                _render_method(
                    cls_name, member_name, member, method_types.get(member_name)
                )
            )

    lines.append("")
    return "\n".join(lines)


def _coerce_default(raw: str) -> Any:
    """Convert a __text_signature__ default string to a Python value."""
    raw = raw.strip()
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    if raw == "...":
        return Ellipsis
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _render_text_sig_default(raw: str) -> str:
    """Render a __text_signature__ default token directly as stub syntax."""
    raw = raw.strip()
    # PyO3 renders Optional params without explicit defaults as `...`;
    # numeric/bool/None tokens pass through unchanged.
    return raw


def _infer_attr_type(cls_name: str, attr: str) -> str | None:
    """Infer an attribute annotation from a live instance value."""
    recipe = INSTANCE_RECIPES.get(cls_name)
    if recipe is None:
        return None
    try:
        inst = recipe()
        value = getattr(inst, attr)
        return type(value).__name__
    except Exception:
        return None


def generate_pyi() -> str:
    out = [HEADER]

    # Constants (values verified live; annotation from table).
    out.append("# Module-level constants (values from the compiled module).\n")
    for name, ann in CONSTANT_TYPES.items():
        if not hasattr(rc, name):
            raise RuntimeError(
                f"Constant {name!r} in annotation table is missing from the live module; "
                "update CONSTANT_TYPES."
            )
        out.append(f"{name}: {ann}\n")
    out.append("\n")

    # Classes.
    for cls_name in CLASS_ORDER:
        if not hasattr(rc, cls_name):
            raise RuntimeError(
                f"Class {cls_name!r} in CLASS_ORDER is missing from the live module; "
                "update CLASS_ORDER."
            )
        out.append(_render_class(cls_name, getattr(rc, cls_name)))
        out.append("\n")

    # Module-level functions.
    for name in FUNCTION_ORDER:
        if not hasattr(rc, name):
            raise RuntimeError(
                f"Function {name!r} in FUNCTION_ORDER is missing from the live module; "
                "update FUNCTION_ORDER."
            )
        out.append(_render_function(name, getattr(rc, name), FUNCTION_TYPES.get(name)))

    # Report table keys that no longer match the live module (drift guard
    # for the annotation tables themselves).
    live_funcs = {
        n for n in dir(rc) if not n.startswith("_") and callable(getattr(rc, n))
    }
    stale = sorted(set(FUNCTION_TYPES) - live_funcs)
    if stale:
        raise RuntimeError(
            f"FUNCTION_TYPES has stale entries not on the module: {stale}"
        )

    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default="backend/stubs/runtime_core")
    args = parser.parse_args()

    pyi_text = generate_pyi()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runtime_core.pyi"
    out_path.write_text(pyi_text, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
