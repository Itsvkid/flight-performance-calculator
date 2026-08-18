"""Figures for the performance model.

Palette and mark specs follow a validated categorical set — slots 1-3 of the
reference data-viz palette, checked for colour-vision separation and contrast
before use rather than picked by eye. Every series is direct-labelled as well
as legended, so identity never rests on colour alone.

Each function returns the Matplotlib figure and, given `path`, writes a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt  # noqa: E402

from . import performance as perf  # noqa: E402
from .aircraft import Aircraft  # noqa: E402

# ── Themes ──────────────────────────────────────────────────────────────────
#
# Two selected palettes, not one flipped. The dark steps are the same hues
# re-stepped for a dark surface and validated against it — an automatic
# inversion of the light set fails both the lightness band and the contrast
# floor. The dark surface, ink and grid are the portfolio site's own tokens, so
# a figure dropped onto that page sits flush with it instead of floating on a
# rectangle of a slightly different black.

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_muted": "#52514e",
        "grid": "#e4e3df",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "fill_alpha": 0.10,
    },
    "dark": {
        "surface": "#101316",
        "ink": "#edeef0",
        "ink_muted": "#788087",
        "grid": "#23282e",
        "series": ("#3987e5", "#d95926", "#199e70"),
        "fill_alpha": 0.16,   # a wash needs more weight to read on a dark ground
    },
}

_ACTIVE = THEMES["light"]

LINE_W = 2.0
DPI = 160


def use_theme(name: str) -> None:
    """Select the palette subsequent figures are drawn with."""
    global _ACTIVE
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; expected one of {list(THEMES)}")
    _ACTIVE = THEMES[name]


def _c(role: str):
    return _ACTIVE[role]


def _axes(title: str, xlabel: str, ylabel: str, size=(8.0, 5.0)):
    """A figure with recessive grid and axes, so the data carries the ink."""
    fig, ax = plt.subplots(figsize=size, dpi=DPI)
    fig.patch.set_facecolor(_c("surface"))
    ax.set_facecolor(_c("surface"))

    ax.set_title(title, color=_c("ink"), fontsize=12, fontweight="600",
                 loc="left", pad=14)
    ax.set_xlabel(xlabel, color=_c("ink_muted"), fontsize=10)
    ax.set_ylabel(ylabel, color=_c("ink_muted"), fontsize=10)

    ax.grid(True, color=_c("grid"), linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_c("grid"))
    ax.tick_params(colors=_c("ink_muted"), labelsize=9, length=0)
    return fig, ax


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=_c("surface"), bbox_inches="tight")
    return fig


def thrust_curves(aircraft: Aircraft, altitude: float = 10000.0,
                  path=None):
    """Thrust required and thrust available against true airspeed.

    Where the curves cross bounds level flight; the gap between them is what
    pays for climb. The minimum of the required curve is V_md — marked,
    because it is the speed every other result in the module is anchored to.
    """
    v_stall = perf.stall_speed(aircraft, altitude)
    # Stop a little past the thrust-limited maximum. Plotting out to four times
    # stall speed drew a supersonic region this aircraft cannot reach, which
    # invites the reader to believe a part of the curve that means nothing.
    v_top = perf.max_level_speed(aircraft, altitude)
    v_end = (v_top * 1.12) if v_top else v_stall * 2.6
    v = np.linspace(v_stall, v_end, 400)

    required = np.array([perf.drag(aircraft, x, altitude) for x in v]) / 1000
    available = np.array(
        [aircraft.thrust_available(altitude, x) for x in v]
    ) / 1000

    fig, ax = _axes(
        f"Thrust required and available — {aircraft.name} at "
        f"{altitude / 1000:.0f} km",
        "True airspeed (m/s)", "Thrust (kN)",
    )

    ax.plot(v, required, color=_c("series")[0], linewidth=LINE_W,
            label="Thrust required (drag)", zorder=3)
    ax.plot(v, available, color=_c("series")[1], linewidth=LINE_W,
            label="Thrust available", zorder=3)

    v_md = perf.v_min_drag(aircraft, altitude)
    d_md = perf.drag(aircraft, v_md, altitude) / 1000
    ax.plot([v_md], [d_md], "o", color=_c("series")[0], markersize=8, zorder=4)
    ax.annotate(
        f"$V_{{md}}$ = {v_md:.0f} m/s\nL/D = {aircraft.lift_to_drag_max:.1f}",
        xy=(v_md, d_md), xytext=(12, -34), textcoords="offset points",
        color=_c("ink"), fontsize=9,
    )

    # Direct labels supply the relief the contrast check asks for, and mean the
    # chart still reads if it is printed in greyscale.
    ax.annotate("Available", xy=(v[-1], available[-1]), xytext=(-4, 8),
                textcoords="offset points", color=_c("ink"), fontsize=9,
                ha="right", fontweight="600")
    ax.annotate("Required", xy=(v[-1], required[-1]), xytext=(-4, -16),
                textcoords="offset points", color=_c("ink"), fontsize=9,
                ha="right", fontweight="600")

    ax.legend(frameon=False, labelcolor=_c("ink_muted"), fontsize=9, loc="upper center")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def flight_envelope(aircraft: Aircraft, path=None):
    """Altitude against true airspeed — where level flight is possible.

    Bounded on the left by stall and on the right by available thrust. The two
    close on each other with altitude; where they meet is the absolute ceiling,
    and the narrowing gap is why cruise altitude is a compromise rather than
    simply 'as high as possible'.
    """
    ceiling = perf.absolute_ceiling(aircraft)
    # Sampled with a square-law bias so the points bunch toward the ceiling,
    # where the two boundaries turn hardest.
    heights = ceiling * np.linspace(0.0, 1.0, 120) ** 0.7

    slowest, fastest, valid = [], [], []
    for h in heights:
        v_max = perf.max_level_speed(aircraft, float(h))
        v_min = perf.min_level_speed(aircraft, float(h))
        if v_max is None or v_min is None:
            continue
        slowest.append(v_min)
        fastest.append(v_max)
        valid.append(h / 1000)

    # At the absolute ceiling thrust equals drag at exactly one speed, so both
    # boundaries meet there. Sampling alone never lands on it, which left the
    # envelope drawn open at the top — closing it is the honest shape.
    apex = perf.max_rate_of_climb(aircraft, ceiling)[1]
    slowest.append(apex)
    fastest.append(apex)
    valid.append(ceiling / 1000)

    fig, ax = _axes(
        f"Flight envelope — {aircraft.name}",
        "True airspeed (m/s)", "Altitude (km)",
    )

    ax.fill_betweenx(valid, slowest, fastest, color=_c("series")[0], alpha=_c("fill_alpha"),
                     zorder=1, label="Level flight possible")
    ax.plot(slowest, valid, color=_c("series")[0], linewidth=LINE_W, zorder=3)
    ax.plot(fastest, valid, color=_c("series")[1], linewidth=LINE_W, zorder=3)

    service = perf.service_ceiling(aircraft) / 1000
    ax.axhline(service, color=_c("ink_muted"), linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.annotate(f"Service ceiling {service:.1f} km", xy=(slowest[0], service),
                xytext=(6, 6), textcoords="offset points",
                color=_c("ink_muted"), fontsize=9)

    mid = len(valid) // 2
    ax.annotate("Low-speed limit", xy=(slowest[mid], valid[mid]),
                xytext=(8, -22), textcoords="offset points",
                color=_c("ink"), fontsize=9, fontweight="600")
    ax.annotate("Thrust limit", xy=(fastest[mid], valid[mid]),
                xytext=(10, 0), textcoords="offset points",
                color=_c("ink"), fontsize=9, fontweight="600")

    ax.set_ylim(0, ceiling / 1000 * 1.12)
    return _save(fig, path)


def payload_range(aircraft: Aircraft, velocity: float | None = None,
                  altitude: float = 10000.0, path=None):
    """The payload-range trade.

    Flat while the tanks still have room, then sloped once fuel is the binding
    constraint and every extra kilometre costs payload.
    """
    if velocity is None:
        velocity = perf.v_max_range(aircraft, altitude)

    points = perf.payload_range_points(aircraft, velocity, altitude)
    ranges = [r / 1000 for r, _ in points]
    payloads = [p / 1000 for _, p in points]

    fig, ax = _axes(
        f"Payload-range — {aircraft.name} at {altitude / 1000:.0f} km, "
        f"{velocity:.0f} m/s",
        "Range (km)", "Payload (t)",
    )

    ax.fill_between(ranges, payloads, color=_c("series")[0], alpha=_c("fill_alpha"), zorder=1)
    ax.plot(ranges, payloads, color=_c("series")[0], linewidth=LINE_W,
            marker="o", markersize=8, zorder=3)

    for i, (label, r, p) in enumerate(zip("ABCD", ranges, payloads)):
        last = i == len(ranges) - 1
        ax.annotate(f"{label}  {r:.0f} km / {p:.1f} t", xy=(r, p),
                    xytext=(-8, 14) if last else (8, 10),
                    textcoords="offset points", color=_c("ink"), fontsize=9,
                    ha="right" if last else "left")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def generate_all(aircraft: Aircraft, directory="figures", theme="light",
                 suffix=""):
    """Write every figure in one theme. Returns the paths written."""
    use_theme(theme)
    directory = Path(directory)
    written = []
    for name, fn in (
        (f"thrust-curves{suffix}.png",
         lambda p: thrust_curves(aircraft, path=p)),
        (f"flight-envelope{suffix}.png",
         lambda p: flight_envelope(aircraft, path=p)),
        (f"payload-range{suffix}.png",
         lambda p: payload_range(aircraft, path=p)),
    ):
        target = directory / name
        fig = fn(target)
        plt.close(fig)
        written.append(target)
    return written
