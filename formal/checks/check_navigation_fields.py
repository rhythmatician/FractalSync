"""Compare four interior navigation paths without changing runtime or Controls."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from cmath import exp, phase, sqrt
from dataclasses import dataclass
import json
from math import atanh, degrees, isfinite, pi, tanh
from pathlib import Path
from statistics import median
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import runtime_core


LENGTH = 0.8
STEPS = 256
EPSILON = 0.01
STARTS = (0.25 + 0j, 0.48 * exp(1j * pi / 3), 0.66 * exp(-2j * pi / 3))
NAMES = ("Geodesic", "Multiplier", "Map iso-scale", "P2 trap")
COLORS = ("#c8cde1", "#65deaf", "#f5b56c", "#cfa6ff")
TYPE = ImageFont.load_default(size=14)


class Refusal(ValueError):
    """The named field or its local chart cannot supply a contour direction."""


class MapJet(TypedDict):
    D: float | None
    grad_D: tuple[float | None, float | None]
    validity: str
    singularity: str
    is_bridge: bool


@dataclass(frozen=True)
class Chart:
    name: str
    place: Callable[[complex], complex]
    derivative: Callable[[complex], complex]
    ruler_factor_diff: Callable[[complex], complex]
    view: tuple[complex, float]

    def ruler(self, u: complex) -> float:
        return (1 - abs(u)**2) * abs(self.derivative(u)) / 2

    def multiplier_normal(self, u: complex) -> complex:
        return 2 * u / self.derivative(u).conjugate()

    def candidate_normal(self, u: complex) -> complex:
        d = self.derivative(u)
        grad_u = abs(d) * u - (1 - abs(u)**2) * self.ruler_factor_diff(u) / 2
        return grad_u / d.conjugate()


CHARTS = (
    Chart(
        "Main cardioid", lambda u: u * (2-u) / 4, lambda u: (1-u)/2,
        lambda u: (u-1)/(2*abs(1-u)), (-0.17 + 0j, 0.66),
    ),
    Chart(
        "Period-two bulb", lambda u: -1 + u/4, lambda _u: 0.25 + 0j,
        lambda _u: 0j, (-1 + 0j, 0.30),
    ),
)


def trap(c: complex) -> float:
    return abs(c * (c + 1))**2


def trap_normal(c: complex) -> complex:
    return 2 * (2*c + 1).conjugate() * c * (c + 1)


def map_sample(c: complex) -> tuple[float, complex]:
    """Return signed D and its derivative from the installed regular Rust jet."""
    try:
        jet: MapJet = json.loads(runtime_core.geometry_provider_query(c, EPSILON))
    except RuntimeError as error:
        raise Refusal(str(error)) from error
    if jet["validity"] != "regular" or jet["is_bridge"]:
        raise Refusal(f'provider {jet["validity"]}, {jet["singularity"]}')
    distance = jet["D"]
    if distance is None or not isfinite(distance) or distance >= 0:
        raise Refusal("signed-distance sample is outside the sampled interior")
    dx, dy = jet["grad_D"]
    if dx is None or dy is None:
        raise Refusal("provider gradient is unavailable")
    return distance, complex(dx, dy)


def unit(normal: complex) -> complex:
    size = abs(normal)
    if not isfinite(size) or size <= 1e-8:
        raise Refusal("gradient below the numerical threshold")
    return normal / size


def disk_direction(chart: Chart, u: complex, normal: complex) -> complex:
    return 1j * unit(normal) * abs(chart.derivative(u)) / chart.derivative(u)


def rk4_contour(
    chart: Chart, start: complex, normal: Callable[[complex], complex], steps: int,
) -> list[complex]:
    def velocity(u: complex) -> complex:
        if abs(u) >= 1:
            raise Refusal("trajectory left its multiplier chart")
        return (1 - abs(u)**2) / 2 * disk_direction(chart, u, normal(chart.place(u)))

    h = LENGTH / steps
    points = [start]
    for _ in range(steps):
        u = points[-1]
        k1 = velocity(u)
        k2 = velocity(u + h*k1/2)
        k3 = velocity(u + h*k2/2)
        k4 = velocity(u + h*k3)
        new_u = u + h * (k1 + 2*k2 + 2*k3 + k4)/6
        velocity(new_u)
        points.append(new_u)
    return points


def paths(chart: Chart, u0: complex, steps: int) -> dict[str, list[complex] | str]:
    initial = disk_direction(chart, u0, trap_normal(chart.place(u0)))
    radius = abs(u0)
    omega = (1-radius**2)/(2*radius)
    geodesic, multiplier = [], []
    for j in range(steps + 1):
        s = LENGTH * j/steps
        z = tanh(s/2) * initial
        geodesic.append((u0+z)/(1+u0.conjugate()*z))
        multiplier.append(u0 * exp(1j * omega*s))
    result: dict[str, list[complex] | str] = {"Geodesic": geodesic, "Multiplier": multiplier}
    for label, normal in (
        ("Map iso-scale", lambda c: map_sample(c)[1]), ("P2 trap", trap_normal),
    ):
        try:
            result[label] = rk4_contour(chart, u0, normal, steps)
        except Refusal as error:
            result[label] = str(error)
    return result


def disk_distance(u: complex, v: complex) -> float:
    return 2 * atanh(abs((u-v)/(1-v.conjugate()*u)))


def angle(u: complex, v: complex) -> float:
    return abs(degrees(phase(unit(u).conjugate()*unit(v))))


def gradient_ablation(chart: Chart) -> None:
    comparisons: dict[str, list[float]] = {
        "trap/mult": [], "Map/mult": [], "trap/Map": [], "candidate ruler/mult": [],
    }
    for radius in (0.2, 0.4, 0.65):
        for k in range(12):
            u = radius * exp(2j * pi*k/12)
            c = chart.place(u)
            natural, trap_grad = chart.multiplier_normal(u), trap_normal(c)
            comparisons["trap/mult"].append(angle(trap_grad, natural))
            comparisons["candidate ruler/mult"].append(angle(chart.candidate_normal(u), natural))
            try:
                _, map_grad = map_sample(c)
                map_angle = angle(map_grad, natural)
            except Refusal:
                continue
            comparisons["Map/mult"].append(map_angle)
            comparisons["trap/Map"].append(angle(trap_grad, map_grad))
    for name, angles in comparisons.items():
        if angles:
            near = sum(x <= 10 for x in angles)
            near_line = sum(min(x, 180-x) <= 10 for x in angles)
            print(f"  {name}: {len(angles)}/36 valid, median {median(angles):.3f} deg, "
                  f"max {max(angles):.3f} deg, {near}/{len(angles)} within 10 deg "
                  f"({near_line}/{len(angles)} ignoring orientation)")
        else:
            print(f"  {name}: no valid direction pairs")


def trace_ablation(
    chart: Chart, u0: complex, fine: dict[str, list[complex] | str],
) -> None:
    coarse = paths(chart, u0, STEPS//2)
    natural = fine["Multiplier"]
    assert isinstance(natural, list)
    c0 = chart.place(u0)
    initial_trap = trap(c0)
    print(f"  start u={u0.real:+.4f}{u0.imag:+.4f}i, c={c0.real:+.4f}{c0.imag:+.4f}i")
    for name in NAMES:
        points = fine[name]
        if isinstance(points, str):
            print(f"    {name}: refused ({points})")
            continue
        assert all(abs(u) < 1 for u in points)
        chord_length = sum(disk_distance(u,v) for u,v in zip(points, points[1:]))
        assert abs(chord_length-LENGTH) < 1e-5
        delta = max(disk_distance(u,v) for u,v in zip(points, natural, strict=True))
        trap_drift = max(abs(trap(chart.place(u))-initial_trap) for u in points)
        map_drift = "n/a"
        if name == "Map iso-scale":
            d0 = map_sample(c0)[0]
            map_drift = f"{max(abs(map_sample(chart.place(u))[0]-d0) for u in points):.2g}"
        if name == "P2 trap":
            assert trap_drift < 1e-8
        if name == "Multiplier":
            assert max(abs(abs(u)-abs(u0)) for u in points) < 1e-14
        if name == "Geodesic":
            assert abs(disk_distance(u0,points[-1])-LENGTH) < 1e-12
        old = coarse[name]
        refine = "coarse refused" if isinstance(old, str) else f"{disk_distance(points[-1],old[-1]):.2g}"
        print(f"    {name}: chord {chord_length:.6f}, max vs mult {delta:.4f}, "
              f"trap drift {trap_drift:.2g}, Map drift {map_drift}, endpoint refinement {refine}")
    map_path, trap_path = fine["Map iso-scale"], fine["P2 trap"]
    if isinstance(map_path, list) and isinstance(trap_path, list):
        separation = max(disk_distance(u,v) for u,v in zip(map_path, trap_path, strict=True))
        print(f"    max trap vs Map {separation:.4f}")


def panel_point(chart: Chart, c: complex, origin: tuple[int, int], size: int) -> tuple[int, int]:
    center, half_width = chart.view
    delta = (c-center)/half_width
    return round(origin[0]+size*(1+delta.real)/2), round(origin[1]+size*(1-delta.imag)/2)


def draw_map(
    draw: ImageDraw.ImageDraw, chart: Chart, origin: tuple[int, int], size: int,
    traces: list[dict[str, list[complex] | str]], frame: int | None = None,
) -> None:
    x, y = origin
    draw.rectangle((x,y,x+size,y+size), fill="#172236", outline="#52637b")
    outline = [panel_point(chart, chart.place(exp(2j*pi*k/240)), origin, size) for k in range(241)]
    draw.line(outline, fill="#62758c", width=1)
    for trace in traces:
        for name, color in zip(NAMES, COLORS, strict=True):
            points = trace[name]
            if isinstance(points, str):
                continue
            path = [panel_point(chart, chart.place(u), origin, size) for u in points]
            draw.line(path, fill=color, width=2)
            j = 0 if frame is None else frame
            px, py = path[j]
            draw.ellipse((px-3,py-3,px+3,py+3), fill=color)


def julia(c: complex, size: int) -> Image.Image:
    axis = np.linspace(-1.55, 1.55, size)
    z = axis[None,:] + 1j*axis[::-1,None]
    escape = np.zeros(z.shape, dtype=np.uint8)
    alive = np.ones(z.shape, dtype=bool)
    for k in range(1, 81):
        z[alive] = z[alive]**2+c
        leaving = alive & (np.abs(z) > 2)
        escape[leaving] = k
        alive &= ~leaving
    levels = np.sqrt(escape.astype(float)/80)
    rgb = np.stack((40+200*levels, 66+130*levels, 104+135*levels), axis=-1)
    rgb[alive] = (9,17,31)
    return Image.fromarray(rgb.astype(np.uint8))


def visual_evidence(
    directory: Path, traces: list[list[dict[str, list[complex] | str]]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    maps = Image.new("RGB", (640, 365), "#101827")
    draw = ImageDraw.Draw(maps)
    for k, chart in enumerate(CHARTS):
        x = 25 + 310*k
        draw.text((x,20), chart.name, font=TYPE, fill="white")
        draw_map(draw, chart, (x,48), 275, traces[k])
    for k, (name, color) in enumerate(zip(NAMES, COLORS, strict=True)):
        draw.text((25+150*k,337), name, font=TYPE, fill=color)
    maps.save(directory/"185-navigation-maps.png")
    # Animate the real positive start in both charts. The plate includes all starts.
    size, pad = 152, 12
    width, height = 5*size + 6*pad, 2*(size+24) + 58
    frames = []
    for frame in range(0, STEPS+1, 8):
        image = Image.new("RGB", (width,height), "#101827")
        draw = ImageDraw.Draw(image)
        draw.text((pad,12), f"Intrinsic s={LENGTH*frame/STEPS:.3f} / {LENGTH:.2f}  |  "
                  "Julia viewport x,y in [-1.55, 1.55]  |  80 iterations, bailout 2",
                  font=TYPE, fill="white")
        for row, chart in enumerate(CHARTS):
            y = 40+row*(size+24)
            trace = traces[row][0]
            draw.text((pad,y), chart.name, font=TYPE, fill="white")
            draw_map(draw, chart, (pad,y+18), size, [trace], frame)
            for k, (name, color) in enumerate(zip(NAMES, COLORS, strict=True), 1):
                x = pad+k*(size+pad)
                draw.text((x,y), name, font=TYPE, fill=color)
                path = trace[name]
                if isinstance(path, str):
                    draw.text((x,y+30), "Trajectory refused", font=TYPE, fill=color)
                    continue
                image.paste(julia(chart.place(path[frame]),size), (x,y+18))
        frames.append(image)
    frames[0].save(directory/"185-navigation-julia.gif", save_all=True,
                   append_images=frames[1:], duration=110, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="Write a Map PNG and Julia GIF to this directory")
    args = parser.parse_args()
    asset = runtime_core.get_builtin_distance_field_py("mandelbrot_1024")
    print(f"Rust {runtime_core.geometry_provider_version()}, asset {asset}, epsilon {EPSILON}")
    print(f"Intrinsic speed 1, length {LENGTH}, RK4 steps {STEPS} / {STEPS//2}")
    traces = []
    for chart in CHARTS:
        print(chart.name)
        gradient_ablation(chart)
        chart_traces = [paths(chart, u0, STEPS) for u0 in STARTS]
        for u0, fine in zip(STARTS, chart_traces, strict=True):
            trace_ablation(chart, u0, fine)
        traces.append(chart_traces)
    if args.images:
        visual_evidence(args.images, traces)


if __name__ == "__main__":
    main()
