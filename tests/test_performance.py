"""Performance validation against closed-form results.

Every test below checks the code against algebra that can be done by hand, not
against numbers this code produced earlier. Where a routine searches
numerically, the test confirms the search lands on the analytical optimum —
if the two disagree, one of them is wrong and the test says so.
"""

import math

import pytest
from scipy.optimize import minimize_scalar

from src import performance as perf
from src.aircraft import NARROWBODY_TWIN, Aircraft
from src.atmosphere import G0, density

AC = NARROWBODY_TWIN


# ── The drag polar identity ─────────────────────────────────────────────────

@pytest.mark.parametrize("altitude", [0, 5000, 11000])
def test_induced_equals_parasite_at_min_drag(altitude):
    """The defining property of V_md: k*CL^2 == CD0.

    This is the single most useful check in the module. If the minimum-drag
    speed is wrong in any way, this identity breaks.
    """
    v = perf.v_min_drag(AC, altitude)
    cl = perf.lift_coefficient(AC, v, altitude)
    assert AC.k * cl**2 == pytest.approx(AC.cd0, rel=1e-10)


@pytest.mark.parametrize("altitude", [0, 5000, 11000])
def test_lift_to_drag_peaks_at_v_min_drag(altitude):
    """L/D at V_md equals the closed form 1/(2*sqrt(CD0*k))."""
    v = perf.v_min_drag(AC, altitude)
    assert perf.lift_to_drag(AC, v, altitude) == pytest.approx(
        AC.lift_to_drag_max, rel=1e-10
    )


def test_numerical_search_finds_the_same_optimum():
    """An independent numerical maximisation of L/D lands on V_md."""
    altitude = 8000
    result = minimize_scalar(
        lambda v: -perf.lift_to_drag(AC, v, altitude),
        bounds=(60, 400), method="bounded", options={"xatol": 1e-8},
    )
    assert result.x == pytest.approx(perf.v_min_drag(AC, altitude), rel=1e-5)


def test_cl_at_min_drag_matches_closed_form():
    v = perf.v_min_drag(AC, 6000)
    assert perf.lift_coefficient(AC, v, 6000) == pytest.approx(
        AC.cl_min_drag, rel=1e-10
    )


# ── Best-range speed ────────────────────────────────────────────────────────

def test_range_speed_is_3_to_the_quarter_above_min_drag():
    """V_range / V_md = 3**0.25 for a jet on a parabolic polar."""
    ratio = perf.v_max_range(AC, 10000) / perf.v_min_drag(AC, 10000)
    assert ratio == pytest.approx(3.0**0.25, rel=1e-10)


def test_range_is_actually_maximised_at_that_speed():
    """Numerically maximise V*(L/D) and confirm it agrees."""
    altitude = 10000
    result = minimize_scalar(
        lambda v: -perf.breguet_range(AC, v, altitude),
        bounds=(120, 500), method="bounded", options={"xatol": 1e-8},
    )
    assert result.x == pytest.approx(perf.v_max_range(AC, altitude), rel=1e-5)


def test_endurance_is_maximised_at_min_drag_speed():
    """Endurance peaks at V_md, range at V_range — different speeds."""
    altitude = 10000
    result = minimize_scalar(
        lambda v: -perf.endurance(AC, v, altitude),
        bounds=(120, 500), method="bounded", options={"xatol": 1e-8},
    )
    assert result.x == pytest.approx(perf.v_min_drag(AC, altitude), rel=1e-5)
    assert perf.v_max_range(AC, altitude) > perf.v_min_drag(AC, altitude)


# ── Level flight consistency ────────────────────────────────────────────────

@pytest.mark.parametrize("velocity", [150.0, 220.0, 300.0])
def test_lift_equals_weight_in_level_flight(velocity):
    """Whatever CL the code returns must actually hold the aircraft up."""
    altitude = 9000
    cl = perf.lift_coefficient(AC, velocity, altitude)
    lift = 0.5 * density(altitude) * velocity**2 * AC.wing_area * cl
    assert lift == pytest.approx(AC.weight, rel=1e-10)


def test_stall_speed_scales_with_inverse_root_density():
    """V_stall ∝ 1/sqrt(rho): the classic altitude effect."""
    v0, v10 = perf.stall_speed(AC, 0), perf.stall_speed(AC, 10000)
    expected = math.sqrt(density(0) / density(10000))
    assert v10 / v0 == pytest.approx(expected, rel=1e-10)


# ── Climb and ceilings ──────────────────────────────────────────────────────

def test_rate_of_climb_is_zero_at_absolute_ceiling():
    """The definition of the absolute ceiling."""
    h = perf.absolute_ceiling(AC)
    assert perf.max_rate_of_climb(AC, h)[0] == pytest.approx(0.0, abs=1e-2)


def test_service_ceiling_sits_below_absolute_and_climbs_100fpm():
    service = perf.service_ceiling(AC)
    absolute = perf.absolute_ceiling(AC)
    assert service < absolute
    assert perf.max_rate_of_climb(AC, service)[0] == pytest.approx(
        perf.SERVICE_CEILING_ROC, abs=1e-2
    )


def test_climb_rate_falls_with_altitude():
    rates = [perf.max_rate_of_climb(AC, h)[0] for h in (0, 3000, 6000, 9000)]
    assert all(a > b for a, b in zip(rates, rates[1:]))


def test_underpowered_aircraft_raises_rather_than_returning_nonsense():
    """A ceiling below sea level is not a number worth propagating."""
    brick = Aircraft(
        name="Underpowered", mass=70000.0, fuel_mass=18000.0,
        wing_area=124.6, span=34.3, cd0=0.020, oswald=0.80,
        thrust_sl=5000.0, tsfc=1.7e-5,
    )
    with pytest.raises(ValueError, match="cannot climb at sea level"):
        perf.absolute_ceiling(brick)


# ── Range and endurance ─────────────────────────────────────────────────────

def test_breguet_scales_with_log_mass_ratio():
    """Doubling ln(Wi/Wf) doubles the range, all else equal."""
    v, h = 250.0, 10000
    wi = AC.weight
    r1 = perf.breguet_range(AC, v, h, wi, wi / math.e)
    r2 = perf.breguet_range(AC, v, h, wi, wi / math.e**2)
    assert r2 / r1 == pytest.approx(2.0, rel=1e-10)


def test_breguet_uses_weight_based_tsfc():
    """Guards the factor-of-g that mass-based TSFC would silently introduce.

    The stored tsfc is kg/(N*s); Breguet needs 1/s. Recomputing the closed
    form by hand here means a future refactor cannot quietly drop the g.
    """
    v, h = 250.0, 10000
    wi, wf = AC.weight, AC.empty_weight
    ld = perf.lift_to_drag(AC, v, h, wi)
    expected = (v / (AC.tsfc * G0)) * ld * math.log(wi / wf)
    assert perf.breguet_range(AC, v, h) == pytest.approx(expected, rel=1e-12)


def test_range_is_physically_plausible():
    """A coarse bound that would have caught the factor-of-g bug immediately.

    Before the tsfc units were fixed this returned 67 000 km. No narrow-body
    flies a third of that, and a check this crude is what catches it.
    """
    r_km = perf.breguet_range(AC, perf.v_max_range(AC, 10000), 10000) / 1000
    assert 3000 < r_km < 9000


def test_rejects_impossible_weight_ratio():
    with pytest.raises(ValueError, match="must exceed final weight"):
        perf.breguet_range(AC, 250.0, 10000, AC.weight, AC.weight * 1.1)


# ── Envelope boundaries ─────────────────────────────────────────────────────

def test_low_speed_limit_is_stall_down_low():
    """Near sea level the wing gives out before the engines do."""
    for h in (0, 5000, 8000):
        assert perf.min_level_speed(AC, h) == pytest.approx(
            perf.stall_speed(AC, h), rel=1e-9
        )


def test_low_speed_limit_becomes_thrust_limited_up_high():
    """Above ~12 km the boundary is thrust, not stall.

    Drawing the envelope's left edge as stall speed at altitude claims level
    flight where the aircraft cannot hold height — this is the check that
    stops that from coming back.
    """
    h = 13000
    assert perf.min_level_speed(AC, h) > perf.stall_speed(AC, h) * 1.05


def test_envelope_closes_at_the_absolute_ceiling():
    """Both boundaries meet at the ceiling: one speed, zero excess thrust."""
    h = perf.absolute_ceiling(AC) - 1.0
    assert perf.min_level_speed(AC, h) == pytest.approx(
        perf.max_level_speed(AC, h), rel=5e-2
    )


def test_no_level_flight_above_the_ceiling():
    h = perf.absolute_ceiling(AC) + 200.0
    assert perf.max_level_speed(AC, h) is None
    assert perf.min_level_speed(AC, h) is None


def test_max_level_speed_exceeds_min_drag_speed():
    for h in (0, 6000, 11000):
        assert perf.max_level_speed(AC, h) > perf.v_min_drag(AC, h)


# ── Payload-range ───────────────────────────────────────────────────────────

def test_payload_range_has_the_right_shape():
    points = perf.payload_range_points(AC, 250.0, 10000)
    ranges = [r for r, _ in points]
    payloads = [p for _, p in points]

    assert ranges[0] == 0.0                      # A: no fuel, no range
    assert payloads[0] == AC.max_payload         # A and B at max payload
    assert payloads[1] == AC.max_payload
    assert payloads[-1] == 0.0                   # D: ferry
    assert all(a < b for a, b in zip(ranges, ranges[1:]))       # range grows
    assert all(a >= b for a, b in zip(payloads, payloads[1:]))  # payload falls


def test_payload_range_respects_mtow():
    """No corner may exceed maximum take-off mass."""
    for _, payload in perf.payload_range_points(AC, 250.0, 10000):
        fuel = min(AC.fuel_mass, AC.mass - AC.oew - payload)
        assert AC.oew + payload + fuel <= AC.mass + 1e-6


def test_payload_range_needs_mass_breakdown():
    bare = Aircraft(
        name="No breakdown", mass=70000.0, fuel_mass=18000.0,
        wing_area=124.6, span=34.3, cd0=0.020, oswald=0.80,
        thrust_sl=220000.0, tsfc=1.7e-5,
    )
    with pytest.raises(ValueError, match="needs oew and max_payload"):
        perf.payload_range_points(bare, 250.0, 10000)
