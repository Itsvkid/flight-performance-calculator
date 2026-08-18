"""Smoke tests for the figures.

These do not check that a plot looks right — no test can. They check it is
produced without error and written to disk, so a refactor that breaks figure
generation fails here rather than silently the next time a report is built.
"""

import matplotlib.pyplot as plt

from src import plotting
from src.aircraft import NARROWBODY_TWIN


def test_generates_every_figure(tmp_path):
    written = plotting.generate_all(NARROWBODY_TWIN, tmp_path)
    assert len(written) == 3
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 10_000  # a real PNG, not a blank canvas


def test_figures_use_the_validated_palette():
    """Series colours are the validated slots, not matplotlib defaults."""
    assert plotting.THEMES["light"]["series"] == ("#2a78d6", "#eb6834", "#1baf7a")
    assert plotting.THEMES["dark"]["series"] == ("#3987e5", "#d95926", "#199e70")


def test_dark_theme_is_selected_not_flipped():
    """The dark steps must differ from the light ones, not be an inversion.

    An automatic flip of the light palette fails both the dark lightness band
    and the contrast floor, so the two sets are chosen independently.
    """
    light = plotting.THEMES["light"]
    dark = plotting.THEMES["dark"]
    assert light["series"] != dark["series"]
    assert light["surface"] != dark["surface"]


def test_generates_both_themes(tmp_path):
    light = plotting.generate_all(NARROWBODY_TWIN, tmp_path, theme="light")
    dark = plotting.generate_all(NARROWBODY_TWIN, tmp_path, theme="dark",
                                 suffix="-dark")
    assert len(light) == len(dark) == 3
    assert all(p.exists() for p in light + dark)
    # Distinct files, not the same render written twice.
    assert {p.name for p in light}.isdisjoint({p.name for p in dark})


def test_unknown_theme_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown theme"):
        plotting.use_theme("solarized")


def test_each_figure_builds_without_a_path():
    for fn in (plotting.thrust_curves, plotting.flight_envelope,
               plotting.payload_range):
        fig = fn(NARROWBODY_TWIN)
        assert fig is not None
        plt.close(fig)
