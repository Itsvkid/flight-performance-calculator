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
    assert plotting.SERIES[:3] == ("#2a78d6", "#eb6834", "#1baf7a")


def test_each_figure_builds_without_a_path():
    for fn in (plotting.thrust_curves, plotting.flight_envelope,
               plotting.payload_range):
        fig = fn(NARROWBODY_TWIN)
        assert fig is not None
        plt.close(fig)
