"""ISA validation against published ISO 2533 table values.

These are not self-consistency checks — every expected number below comes from
the standard atmosphere tables, not from this implementation. That is the whole
point: a model checked only against itself cannot be shown to be wrong.
"""

import math

import pytest

from src import atmosphere as atm

# Altitude, T (K), p (Pa), rho (kg/m^3) — ISO 2533 tabulated values.
ISA_TABLE = [
    (0, 288.15, 101325.0, 1.22500),
    (1000, 281.65, 89874.6, 1.11164),
    (2000, 275.15, 79495.2, 1.00649),
    (5000, 255.65, 54019.9, 0.73643),
    (8000, 236.15, 35599.8, 0.52579),
    (11000, 216.65, 22632.1, 0.36392),
    (15000, 216.65, 12044.6, 0.19367),
    (20000, 216.65, 5474.9, 0.08803),
]


@pytest.mark.parametrize("h, t_ref, p_ref, rho_ref", ISA_TABLE)
def test_matches_isa_table(h, t_ref, p_ref, rho_ref):
    """Matches the published tables at every tabulated altitude.

    Temperature and pressure hold to 0.1%. Density is allowed 0.2% because
    published ISA tables disagree with each other in the fourth significant
    figure of density — they round rho0 to 1.225 and carry different numbers
    of digits in R, and some tabulate against geometric rather than
    geopotential altitude. The tighter internal check is
    `test_ideal_gas_law_holds`, which no rounding of a reference table can
    excuse.
    """
    assert atm.temperature(h) == pytest.approx(t_ref, rel=1e-3)
    assert atm.pressure(h) == pytest.approx(p_ref, rel=1e-3)
    assert atm.density(h) == pytest.approx(rho_ref, rel=2e-3)


def test_sea_level_sound_speed():
    """340.29 m/s at sea level — the value every textbook quotes."""
    assert atm.sound_speed(0) == pytest.approx(340.294, rel=1e-4)


def test_ideal_gas_law_holds():
    """p = rho*R*T must hold at every altitude, both layers included.

    Catches a whole class of error the table comparison alone would miss: if
    the tropospheric and stratospheric branches were ever made inconsistent
    with each other, this fails even where the tables still pass.
    """
    for h in range(0, 20001, 500):
        s = atm.at(h)
        assert s.pressure == pytest.approx(s.density * atm.R * s.temperature, rel=1e-6)


def test_continuous_across_tropopause():
    """No step change at 11 km where the two branches meet."""
    below = atm.at(atm.H_TROPOPAUSE - 0.001)
    above = atm.at(atm.H_TROPOPAUSE + 0.001)
    assert below.pressure == pytest.approx(above.pressure, rel=1e-6)
    assert below.density == pytest.approx(above.density, rel=1e-6)
    # 2 mm of altitude either side, so a 1.3e-5 K step is the lapse rate
    # doing its job, not a discontinuity.
    assert below.temperature == pytest.approx(above.temperature, rel=1e-7)


def test_monotonic_decrease():
    """Pressure and density fall monotonically through the whole range."""
    heights = list(range(0, 20001, 250))
    pressures = [atm.pressure(h) for h in heights]
    densities = [atm.density(h) for h in heights]
    assert all(a > b for a, b in zip(pressures, pressures[1:]))
    assert all(a > b for a, b in zip(densities, densities[1:]))


def test_sigma_at_sea_level():
    assert atm.at(0).sigma == pytest.approx(1.0, rel=1e-9)


def test_rejects_altitude_outside_model_range():
    """Fails loudly rather than extrapolating a model that no longer applies."""
    with pytest.raises(ValueError, match="outside this model's valid range"):
        atm.temperature(25000)
    with pytest.raises(ValueError, match="outside this model's valid range"):
        atm.temperature(-1000)
