"""International Standard Atmosphere (ISA).

Implements ISO 2533 from sea level to 20 km — the troposphere with its
6.5 K/km lapse, and the isothermal lower stratosphere above 11 km.

The stratosphere matters more than it first appears: a jet's service ceiling
and its cruise altitude both sit above the tropopause, so a troposphere-only
model quietly returns nonsense exactly where cruise performance is computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── ISA constants (ISO 2533) ────────────────────────────────────────────────
T0 = 288.15          # sea-level temperature, K
P0 = 101325.0        # sea-level pressure, Pa
RHO0 = 1.225         # sea-level density, kg/m^3
LAPSE = 0.0065       # tropospheric lapse rate, K/m
G0 = 9.80665         # standard gravity, m/s^2
R = 287.05287        # specific gas constant for dry air, J/(kg*K)
GAMMA = 1.4          # ratio of specific heats

H_TROPOPAUSE = 11000.0   # m
H_MAX = 20000.0          # m — upper limit of this model

# Conditions at the tropopause, derived once at import rather than recomputed
# per call, and used as the base for the stratospheric exponential.
_T_TROP = T0 - LAPSE * H_TROPOPAUSE
_EXP = G0 / (R * LAPSE)
_P_TROP = P0 * (_T_TROP / T0) ** _EXP
_RHO_TROP = RHO0 * (_T_TROP / T0) ** (_EXP - 1.0)


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric properties at a single altitude."""

    altitude: float      # m
    temperature: float   # K
    pressure: float      # Pa
    density: float       # kg/m^3
    sound_speed: float   # m/s

    @property
    def sigma(self) -> float:
        """Density ratio rho/rho0 — the form most performance equations want."""
        return self.density / RHO0


def temperature(altitude: float) -> float:
    """Static air temperature in K."""
    _check(altitude)
    if altitude <= H_TROPOPAUSE:
        return T0 - LAPSE * altitude
    return _T_TROP


def pressure(altitude: float) -> float:
    """Static pressure in Pa."""
    _check(altitude)
    if altitude <= H_TROPOPAUSE:
        return P0 * (temperature(altitude) / T0) ** _EXP
    # Isothermal layer: the power law collapses to an exponential because the
    # lapse rate is zero and the exponent g0/(R*L) is undefined.
    return _P_TROP * math.exp(-G0 * (altitude - H_TROPOPAUSE) / (R * _T_TROP))


def density(altitude: float) -> float:
    """Air density in kg/m^3."""
    _check(altitude)
    if altitude <= H_TROPOPAUSE:
        return RHO0 * (temperature(altitude) / T0) ** (_EXP - 1.0)
    return _RHO_TROP * math.exp(-G0 * (altitude - H_TROPOPAUSE) / (R * _T_TROP))


def sound_speed(altitude: float) -> float:
    """Speed of sound in m/s."""
    return math.sqrt(GAMMA * R * temperature(altitude))


def at(altitude: float) -> AtmosphereState:
    """Full atmospheric state at `altitude` metres."""
    return AtmosphereState(
        altitude=altitude,
        temperature=temperature(altitude),
        pressure=pressure(altitude),
        density=density(altitude),
        sound_speed=sound_speed(altitude),
    )


def _check(altitude: float) -> None:
    if not -610.0 <= altitude <= H_MAX:
        raise ValueError(
            f"altitude {altitude} m is outside this model's valid range "
            f"(-610 to {H_MAX:.0f} m). Above 20 km the lapse rate turns "
            f"positive and this implementation would silently be wrong."
        )
