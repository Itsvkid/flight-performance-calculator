"""Flight performance: drag, climb, ceilings, range and endurance.

Every routine here takes an `Aircraft` and an altitude, and every one of them
has either a closed-form answer or a physical identity it must satisfy. That
is what `tests/test_performance.py` checks — not that the numbers look
plausible, but that they agree with the algebra they came from.

Jet (thrust-producing) formulations throughout. Propeller aircraft use
power-based equivalents and are not implemented.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq, minimize_scalar

from .aircraft import Aircraft
from .atmosphere import G0, H_MAX, at, density

# Service ceiling is conventionally where the best climb rate falls to
# 100 ft/min. Stated here rather than buried as a literal.
SERVICE_CEILING_ROC = 0.508  # m/s


def _weight(aircraft: Aircraft, weight: float | None) -> float:
    return aircraft.weight if weight is None else weight


# ── Aerodynamics ────────────────────────────────────────────────────────────

def lift_coefficient(aircraft, velocity, altitude, weight=None) -> float:
    """CL required for level flight: CL = 2W / (rho*V^2*S)."""
    w = _weight(aircraft, weight)
    return 2.0 * w / (density(altitude) * velocity**2 * aircraft.wing_area)


def drag_coefficient(aircraft: Aircraft, cl: float) -> float:
    """Parabolic polar: CD = CD0 + k*CL^2."""
    return aircraft.cd0 + aircraft.k * cl**2


def drag(aircraft, velocity, altitude, weight=None) -> float:
    """Total drag in N for level flight at the given condition."""
    cl = lift_coefficient(aircraft, velocity, altitude, weight)
    cd = drag_coefficient(aircraft, cl)
    return 0.5 * density(altitude) * velocity**2 * aircraft.wing_area * cd


def lift_to_drag(aircraft, velocity, altitude, weight=None) -> float:
    cl = lift_coefficient(aircraft, velocity, altitude, weight)
    return cl / drag_coefficient(aircraft, cl)


def stall_speed(aircraft, altitude, weight=None) -> float:
    """V_stall = sqrt(2W / (rho*S*CL_max))."""
    w = _weight(aircraft, weight)
    return math.sqrt(
        2.0 * w / (density(altitude) * aircraft.wing_area * aircraft.cl_max)
    )


# ── Speeds with closed-form solutions ───────────────────────────────────────

def v_min_drag(aircraft, altitude, weight=None) -> float:
    """Minimum drag speed, V_md = sqrt( 2W/(rho*S) * sqrt(k/CD0) ).

    At this speed induced and parasite drag are equal and L/D is at its
    maximum. Both facts are asserted in the tests.
    """
    w = _weight(aircraft, weight)
    return math.sqrt(
        2.0 * w / (density(altitude) * aircraft.wing_area)
        * math.sqrt(aircraft.k / aircraft.cd0)
    )


def v_max_range(aircraft, altitude, weight=None) -> float:
    """Best-range speed for a jet, where CL = sqrt(CD0 / (3k)).

    Maximising range means maximising V*(L/D), not (L/D) — which is why this
    sits above V_md by a factor of 3**0.25, roughly 32% faster.
    """
    w = _weight(aircraft, weight)
    cl = math.sqrt(aircraft.cd0 / (3.0 * aircraft.k))
    return math.sqrt(2.0 * w / (density(altitude) * aircraft.wing_area * cl))


# ── Climb ───────────────────────────────────────────────────────────────────

def rate_of_climb(aircraft, velocity, altitude, weight=None) -> float:
    """RoC = V*(T - D)/W, m/s. Negative means it cannot hold that condition."""
    w = _weight(aircraft, weight)
    excess = aircraft.thrust_available(altitude, velocity) - drag(
        aircraft, velocity, altitude, weight
    )
    return velocity * excess / w


def max_rate_of_climb(aircraft, altitude, weight=None) -> tuple[float, float]:
    """Best climb rate at an altitude and the speed that achieves it.

    Returns
    -------
    (roc, velocity) in (m/s, m/s)

    Searched numerically between stall and a high-speed bound. There is a
    closed form for the parabolic-polar case, but it assumes thrust is
    independent of speed; keeping the search means the lapse model can be
    replaced later without invalidating this function.
    """
    v_stall = stall_speed(aircraft, altitude, weight)
    lower, upper = v_stall * 1.05, v_stall * 6.0

    result = minimize_scalar(
        lambda v: -rate_of_climb(aircraft, v, altitude, weight),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return -result.fun, result.x


def absolute_ceiling(aircraft, weight=None) -> float:
    """Altitude where the best rate of climb reaches zero, m.

    Raises if the aircraft cannot climb at sea level, rather than returning a
    negative altitude that would propagate silently into a range calculation.
    """
    def best_roc(h: float) -> float:
        return max_rate_of_climb(aircraft, h, weight)[0]

    if best_roc(0.0) <= 0.0:
        raise ValueError(
            f"{aircraft.name} cannot climb at sea level — thrust is below "
            f"minimum drag. Check thrust_sl, mass and cd0."
        )
    if best_roc(H_MAX) > 0.0:
        return H_MAX  # still climbing at the model's ceiling

    return brentq(best_roc, 0.0, H_MAX, xtol=1.0)


def service_ceiling(aircraft, weight=None) -> float:
    """Altitude where the best rate of climb falls to 100 ft/min, m."""
    def excess(h: float) -> float:
        return max_rate_of_climb(aircraft, h, weight)[0] - SERVICE_CEILING_ROC

    if excess(0.0) <= 0.0:
        raise ValueError(f"{aircraft.name} cannot reach its service ceiling.")
    if excess(H_MAX) > 0.0:
        return H_MAX

    return brentq(excess, 0.0, H_MAX, xtol=1.0)


# ── Range and endurance ─────────────────────────────────────────────────────

def breguet_range(aircraft, velocity, altitude, weight_initial=None,
                  weight_final=None) -> float:
    """Breguet range for a jet, R = (V/ct)*(L/D)*ln(Wi/Wf), in metres.

    Cruise is assumed at constant altitude and speed with L/D fixed at its
    initial value. That is the textbook form and it overestimates slightly,
    because L/D actually improves as fuel burns off. Quote it as such.
    """
    wi = _weight(aircraft, weight_initial)
    wf = aircraft.empty_weight if weight_final is None else weight_final
    if wf <= 0 or wi <= wf:
        raise ValueError("initial weight must exceed final weight")

    ld = lift_to_drag(aircraft, velocity, altitude, wi)
    # tsfc is mass based, kg/(N*s); Breguet in weights needs 1/s, so scale by g.
    # Skipping this is a silent factor-of-9.81 error in the answer.
    ct = aircraft.tsfc * G0
    return (velocity / ct) * ld * math.log(wi / wf)


def endurance(aircraft, velocity, altitude, weight_initial=None,
              weight_final=None) -> float:
    """Jet endurance, E = (1/ct)*(L/D)*ln(Wi/Wf), in seconds.

    Maximised at V_md — where range is maximised at V_max_range. The two
    optima are different speeds, which is the whole reason both exist.
    """
    wi = _weight(aircraft, weight_initial)
    wf = aircraft.empty_weight if weight_final is None else weight_final
    if wf <= 0 or wi <= wf:
        raise ValueError("initial weight must exceed final weight")

    ld = lift_to_drag(aircraft, velocity, altitude, wi)
    ct = aircraft.tsfc * G0
    return (1.0 / ct) * ld * math.log(wi / wf)


def payload_range_points(aircraft, velocity, altitude):
    """The four corners of a payload-range diagram.

    Returns [(range_m, payload_kg), ...] for:

    A. Max payload, no fuel                       — range zero, the y-intercept
    B. Max payload, fuel filling the rest of MTOW — the flat segment ends here
    C. Max fuel, payload cut to stay at MTOW      — the sloped trade segment
    D. Max fuel, zero payload                     — ferry range

    The B-to-C slope is the whole point of the diagram: once the tanks are the
    binding constraint, every extra kilometre is bought by offloading payload.
    An aircraft whose OEW, max payload and max fuel sum below MTOW has no such
    segment, and B and C collapse onto each other.
    """
    if aircraft.oew is None or aircraft.max_payload is None:
        raise ValueError(
            f"{aircraft.name} needs oew and max_payload for a payload-range "
            f"diagram; both are None."
        )

    def _range(tow_kg: float, fuel_kg: float) -> float:
        if fuel_kg <= 0:
            return 0.0
        wi = tow_kg * G0
        wf = (tow_kg - fuel_kg) * G0
        return breguet_range(aircraft, velocity, altitude, wi, wf)

    mtow, oew = aircraft.mass, aircraft.oew
    max_payload, max_fuel = aircraft.max_payload, aircraft.fuel_mass

    fuel_at_max_payload = min(mtow - oew - max_payload, max_fuel)
    payload_at_max_fuel = max(0.0, min(max_payload, mtow - oew - max_fuel))

    return [
        (0.0, max_payload),
        (_range(oew + max_payload + fuel_at_max_payload, fuel_at_max_payload),
         max_payload),
        (_range(oew + payload_at_max_fuel + max_fuel, max_fuel),
         payload_at_max_fuel),
        (_range(oew + max_fuel, max_fuel), 0.0),
    ]


def max_level_speed(aircraft, altitude, weight=None) -> float | None:
    """Fastest speed where thrust still matches drag, m/s.

    Returns None when the aircraft cannot sustain level flight at all at this
    altitude — which is what defines the top of the flight envelope.
    """
    def excess(v: float) -> float:
        return aircraft.thrust_available(altitude, v) - drag(
            aircraft, v, altitude, weight
        )

    v_md = v_min_drag(aircraft, altitude, weight)
    if excess(v_md) <= 0:
        return None

    upper = v_md
    for _ in range(60):
        upper *= 1.05
        if excess(upper) < 0:
            return brentq(excess, v_md, upper, xtol=1e-3)
    return None


def min_level_speed(aircraft, altitude, weight=None) -> float | None:
    """Slowest speed where level flight is actually sustainable, m/s.

    Not the same as stall speed. Low and slow, induced drag climbs steeply, and
    above roughly 10 km a jet runs out of thrust before it runs out of wing —
    the low-speed boundary of the envelope becomes thrust-limited, not
    stall-limited. Returning the stall speed there would claim flight is
    possible where the aircraft cannot hold height.

    Returns None when level flight is impossible at any speed.
    """
    def excess(v: float) -> float:
        return aircraft.thrust_available(altitude, v) - drag(
            aircraft, v, altitude, weight
        )

    v_md = v_min_drag(aircraft, altitude, weight)
    if excess(v_md) <= 0:
        return None

    v_stall = stall_speed(aircraft, altitude, weight)
    if excess(v_stall) >= 0:
        return v_stall          # wing gives out before the engines do
    return brentq(excess, v_stall, v_md, xtol=1e-3)
