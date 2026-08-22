"""Aircraft definition: geometry, mass, drag polar and engine.

Everything downstream reads from this one object, so a performance number can
always be traced back to the inputs that produced it.

Masses are in kg and converted to weight internally. Mixing the two is the
classic way to end up with a plausible-looking answer that is wrong by 9.81.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .atmosphere import G0, RHO0, at


@dataclass(frozen=True)
class Aircraft:
    """A fixed-wing aircraft with a parabolic drag polar and a jet engine.

    Attributes
    ----------
    name          identifier used in plots and reports
    mass          maximum take-off mass, kg
    fuel_mass     usable fuel, kg — the mass burned between Wi and Wf
    wing_area     reference wing area S, m^2
    span          wing span b, m
    cd0           zero-lift (parasite) drag coefficient
    oswald        Oswald span efficiency factor e, typically 0.7-0.85
    thrust_sl     total sea-level static thrust, N
    tsfc          thrust specific fuel consumption, kg/(N*s) — mass of fuel
                  per newton of thrust per second. Note this is MASS based;
                  Breguet needs it weight based and multiplies by g itself
    cl_max        maximum lift coefficient, clean
    lapse_exp     thrust lapse exponent n in T(h) = T_sl * sigma**n
    oew           operating empty mass, kg — needed for payload-range
    max_payload   maximum structural payload, kg
    mach_dd       drag-divergence Mach number — where compressibility drag
                  starts rising steeply (see performance.wave_drag_coefficient).
                  Like cd0 and oswald, this is not something manufacturers
                  publish; it is a class-typical estimate, not measured data.
    """

    name: str
    mass: float
    fuel_mass: float
    wing_area: float
    span: float
    cd0: float
    oswald: float
    thrust_sl: float
    tsfc: float
    cl_max: float = 1.4
    lapse_exp: float = 0.7
    oew: float | None = None
    max_payload: float | None = None
    mach_dd: float = 0.80

    def __post_init__(self) -> None:
        for field, value in (
            ("mass", self.mass),
            ("wing_area", self.wing_area),
            ("span", self.span),
            ("cd0", self.cd0),
            ("thrust_sl", self.thrust_sl),
        ):
            if value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
        if not 0 < self.oswald <= 1:
            raise ValueError(f"oswald must be in (0, 1], got {self.oswald}")
        if not 0 < self.mach_dd < 1:
            raise ValueError(f"mach_dd must be in (0, 1), got {self.mach_dd}")
        if self.fuel_mass >= self.mass:
            raise ValueError("fuel_mass cannot equal or exceed total mass")
        if self.oew is not None and self.max_payload is not None:
            # The interesting case is a deliberately over-subscribed aircraft:
            # OEW + max payload + max fuel exceeding MTOW is what produces the
            # sloped middle segment of a payload-range diagram. Only the
            # degenerate case where payload alone busts MTOW is rejected.
            if self.oew + self.max_payload > self.mass:
                raise ValueError("oew + max_payload cannot exceed MTOW")

    # ── Derived geometry ────────────────────────────────────────────────────

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S."""
        return self.span**2 / self.wing_area

    @property
    def k(self) -> float:
        """Induced drag factor, CD = CD0 + k*CL^2."""
        return 1.0 / (math.pi * self.aspect_ratio * self.oswald)

    @property
    def weight(self) -> float:
        """Take-off weight in N."""
        return self.mass * G0

    @property
    def empty_weight(self) -> float:
        """Weight with all usable fuel burned, N."""
        return (self.mass - self.fuel_mass) * G0

    @property
    def wing_loading(self) -> float:
        """W/S in N/m^2."""
        return self.weight / self.wing_area

    # ── Drag polar optima (closed form) ─────────────────────────────────────
    #
    # These have exact analytical solutions, so they double as the reference
    # the numerical routines in performance.py are checked against. If a
    # search and its closed form disagree, one of them is wrong.

    @property
    def cl_min_drag(self) -> float:
        """CL at minimum drag: sqrt(CD0/k).

        At this CL the induced and parasite terms are equal — the identity
        the tests assert directly.
        """
        return math.sqrt(self.cd0 / self.k)

    @property
    def lift_to_drag_max(self) -> float:
        """(L/D)_max = 1 / (2*sqrt(CD0*k))."""
        return 1.0 / (2.0 * math.sqrt(self.cd0 * self.k))

    def thrust_available(self, altitude: float,
                         velocity: float | None = None) -> float:
        """Installed thrust at altitude and speed, N.

        T = T_sl * sigma**n * (1 - 0.49*sqrt(M))

        The Mach term is not optional decoration. A high-bypass turbofan loses
        a large fraction of its static thrust to ram drag as speed builds, and
        omitting it inflates every climb and ceiling number downstream — this
        model produced a 16.8 km ceiling and a 12 degree climb angle before the
        term was added, against roughly 12 km and 5-8 degrees in reality.

        The correlation is Mattingly's for high-bypass turbofans. Passing no
        velocity returns static thrust, which is the right answer only on the
        runway brakes-off.
        """
        lapse = self.thrust_sl * at(altitude).sigma**self.lapse_exp
        if velocity is None:
            return lapse
        mach = velocity / at(altitude).sound_speed
        return lapse * max(0.0, 1.0 - 0.49 * math.sqrt(mach))


# A reference aircraft loosely in the class of a narrow-body twin, used in
# examples and tests. The numbers are representative, not any real type's
# certified data — anything published as a result must state its own source.
NARROWBODY_TWIN = Aircraft(
    name="Reference narrow-body twin",
    mass=70000.0,
    fuel_mass=18000.0,
    wing_area=124.6,
    span=34.3,
    cd0=0.020,
    oswald=0.80,
    thrust_sl=2 * 110000.0,
    tsfc=1.7e-5,   # ~0.6 lb/(lbf*h), typical high-bypass turbofan
    cl_max=1.5,
    oew=41500.0,
    max_payload=18000.0,
    mach_dd=0.80,
)
