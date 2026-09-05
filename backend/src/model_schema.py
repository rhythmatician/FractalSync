"""Typed adapter for Rust-owned model output schemas."""

from dataclasses import dataclass
import json
from typing import cast

import runtime_core
import torch


@dataclass(frozen=True)
class OutputDescriptor:
    name: str
    group: str
    activation: str
    minimum: float | None
    maximum: float | None
    scale: float
    offset: float


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"model schema {field} must be numeric")
    return float(value)


def parse_schema(payload: str) -> tuple[OutputDescriptor, ...]:
    raw = cast(object, json.loads(payload))
    if not isinstance(raw, list):
        raise ValueError("model schema must be an array")
    descriptors: list[OutputDescriptor] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("model schema entries must be objects")
        name = entry.get("name")
        activation = entry.get("activation")
        group = entry.get("group")
        if not isinstance(name, str) or not isinstance(activation, str) or not isinstance(group, str):
            raise ValueError("model schema name, group, and activation must be strings")
        minimum_raw = entry.get("min")
        maximum_raw = entry.get("max")
        descriptors.append(
            OutputDescriptor(
                name=name,
                group=group,
                activation=activation,
                minimum=None if minimum_raw is None else _number(minimum_raw, "min"),
                maximum=None if maximum_raw is None else _number(maximum_raw, "max"),
                scale=_number(entry.get("scale"), "scale"),
                offset=_number(entry.get("offset"), "offset"),
            )
        )
    return tuple(descriptors)


def output_schema(model_type: str, k_bands: int) -> tuple[OutputDescriptor, ...]:
    if model_type == "controls_v2":
        return parse_schema(runtime_core.controls_v2_schema_json())
    if model_type == "orbit_control":
        return parse_schema(runtime_core.orbit_control_schema_json(k_bands))
    return parse_schema(runtime_core.legacy_visual_schema_json())


def legacy_visual_export_schema() -> tuple[OutputDescriptor, ...]:
    """Return the historical browser metadata contract owned by Rust."""
    return parse_schema(runtime_core.legacy_visual_export_ranges_json())


def apply_schema(raw: torch.Tensor, schema: tuple[OutputDescriptor, ...]) -> torch.Tensor:
    if raw.shape[-1] != len(schema):
        raise ValueError(f"expected {len(schema)} outputs, got {raw.shape[-1]}")
    columns: list[torch.Tensor] = []
    for index, descriptor in enumerate(schema):
        value = raw[..., index]
        if descriptor.activation in ("sigmoid", "scaled_sigmoid"):
            value = torch.sigmoid(value)
        elif descriptor.activation == "tanh":
            value = torch.tanh(value)
        elif descriptor.activation == "scaled_softplus_clamped":
            value = torch.nn.functional.softplus(value)
        elif descriptor.activation != "identity":
            raise ValueError(f"unsupported activation {descriptor.activation!r}")
        value = value * descriptor.scale + descriptor.offset
        if descriptor.minimum is not None or descriptor.maximum is not None:
            value = torch.clamp(
                value, min=descriptor.minimum, max=descriptor.maximum
            )
        columns.append(value)
    return torch.stack(columns, dim=-1)


def apply_named_schema(
    raw_by_name: dict[str, torch.Tensor], schema: tuple[OutputDescriptor, ...]
) -> torch.Tensor:
    missing = [descriptor.name for descriptor in schema if descriptor.name not in raw_by_name]
    if missing:
        raise ValueError(f"missing model heads for schema fields: {missing}")
    raw = torch.stack([raw_by_name[descriptor.name] for descriptor in schema], dim=-1)
    return apply_schema(raw, schema)
