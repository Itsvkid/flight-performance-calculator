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


# ── Compressibility (wave) drag ─────────────────────────────────────────────

def test_wave_drag_is_zero_at_and_below_mach_dd():
    assert perf.wave_drag_coefficient(AC, AC.mach_dd) == 0.0
    assert perf.wave_drag_coefficient(AC, AC.mach_dd - 0.1) == 0.0


def test_wave_drag_matches_the_closed_form_above_mach_dd():
    mach = AC.mach_dd + 0.05
    expected = 20.0 * (mach - AC.mach_dd) ** 4
    assert perf.wave_drag_coefficient(AC, mach) == pytest.approx(expected)


def test_wave_drag_rises_with_mach_above_mach_dd():
    """Not just nonzero — actually rising, which is the entire point of a
    'divergence' term."""
    values = [perf.wave_drag_coefficient(AC, m)
              for m in (AC.mach_dd, AC.mach_dd + 0.02, AC.mach_dd + 0.05,
                        AC.mach_dd + 0.10)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] > 0.0


def test_drag_coefficient_default_mach_is_the_pure_parabolic_polar():
    """No mach argument (the default) must reproduce exactly what the
    module produced before wave drag existed — this is the regression
    guard for every closed-form aerodynamic identity elsewhere in this
    file that still assumes an incompressible polar."""
    cl = 0.5
    assert perf.drag_coefficient(AC, cl) == pytest.approx(AC.cd0 + AC.k * cl**2)


def test_drag_below_mach_dd_matches_the_incompressible_polar():
    """Regression guard for drag() itself, not just drag_coefficient(): at
    a subsonic condition well below mach_dd, adding the mach-aware wave-drag
    term must not have changed anything."""
    altitude = 3000
    velocity = 150.0
    cl = perf.lift_coefficient(AC, velocity, altitude)
    cd_incompressible = AC.cd0 + AC.k * cl**2
    expected = 0.5 * density(altitude) * velocity**2 * AC.wing_area * cd_incompressible
    assert perf.drag(AC, velocity, altitude) == pytest.approx(expected)


def test_drag_above_mach_dd_matches_an_independent_recomputation():
    """An independent-route check, not a re-run of the same code path: drag
    is recomputed here from the raw dynamic-pressure formula with the wave
    term added by hand, rather than by calling drag_coefficient at all."""
    from src.atmosphere import at

    altitude = 10000
    velocity = 0.9 * at(altitude).sound_speed  # comfortably past AC.mach_dd
    mach = velocity / at(altitude).sound_speed
    assert mach > AC.mach_dd

    cl = perf.lift_coefficient(AC, velocity, altitude)
    cd = AC.cd0 + AC.k * cl**2 + 20.0 * (mach - AC.mach_dd) ** 4
    expected = 0.5 * density(altitude) * velocity**2 * AC.wing_area * cd
    assert perf.drag(AC, velocity, altitude) == pytest.approx(expected)


def test_aircraft_rejects_mach_dd_out_of_range():
    with pytest.raises(ValueError, match="mach_dd"):
        Aircraft(
            name="Bad", mass=70000.0, fuel_mass=18000.0, wing_area=124.6,
            span=34.3, cd0=0.020, oswald=0.80, thrust_sl=2 * 110000.0,
            tsfc=1.7e-5, mach_dd=1.2,
        )


def test_max_level_speed_no_longer_approaches_mach_one():
    """The regression test for the bug this whole feature exists to fix
    (see VALIDATION.md, 'Maximum speed'): before wave drag, this model
    predicted a maximum level speed around Mach 0.98-1.09 for aircraft that
    actually cruise at 0.78-0.84, because a parabolic polar alone never
    stops the solver finding a transonic thrust-drag intersection. It
    still isn't perfect — see VALIDATION.md for the honest remaining gap —
    but it must no longer be able to reach transonic/supersonic speeds at
    all."""
    from src.atmosphere import at

    altitude = 11000
    v_max = perf.max_level_speed(AC, altitude)
    assert v_max is not None
    mach_max = v_max / at(altitude).sound_speed
    assert mach_max < 0.98


# ── Best-range speed ────────────────────────────────────────────────────────

def test_range_speed_is_3_to_the_quarter_above_min_drag():
    """V_range / V_md = 3**0.25 for a jet on a parabolic polar."""
    ratio = perf.v_max_range(AC, 10000) / perf.v_min_drag(AC, 10000)
    assert ratio == pytest.approx(3.0**0.25, rel=1e-10)


def test_range_is_actually_maximised_at_that_speed():
    """Numerically maximise V*(L/D) and confirm it agrees.

    altitude=8000, not 10000: v_max_range's closed form is an incompressible-
    polar optimum (see drag_coefficient's docstring), valid only where wave
    drag is actually negligible. At 10000 m, V_max_range for AC sits at Mach
    0.865 — past AC.mach_dd (0.80) — so wave drag genuinely pulls the true
    optimum down from the incompressible closed form there, and the search
    correctly disagrees with it. At 8000 m, V_max_range's Mach is 0.75,
    comfortably below mach_dd, which is the regime this identity actually
    describes.
    """
    altitude = 8000
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


# ── Validation module ───────────────────────────────────────────────────────

def test_every_reference_aircraft_is_self_consistent():
    """Each published aircraft must survive the Aircraft validators."""
    from src.validation import REFERENCES
    for ref in REFERENCES:
        ac = ref.aircraft
        assert ac.oew + ac.max_payload <= ac.mass
        assert 6.0 < ac.aspect_ratio < 12.0        # jet transport band
        assert 14.0 < ac.lift_to_drag_max < 22.0   # jet transport band


def test_sensitivity_band_brackets_the_point_estimate():
    """The swept band must contain the nominal result, or the sweep is broken."""
    from src.validation import REFERENCES, evaluate, sensitivity_band
    for ref in REFERENCES:
        lo, hi = sensitivity_band(ref, "range")
        assert lo <= evaluate(ref)["range_model"] <= hi


def test_worse_drag_polar_always_shortens_range():
    """A monotonicity check the sweep would otherwise hide."""
    from dataclasses import replace
    from src.validation import B737_800, cruise_velocity, _range_at_payload
    v = cruise_velocity(B737_800)
    clean = replace(B737_800.aircraft, cd0=0.017)
    dirty = replace(B737_800.aircraft, cd0=0.024)
    args = (B737_800.published_range_payload, B737_800.cruise_altitude, v)
    assert _range_at_payload(clean, *args) > _range_at_payload(dirty, *args)


def test_report_states_the_compressibility_limitation():
    """The known failure must stay in the report, not get tidied away."""
    from src.validation import report
    text = report()
    assert "compressibility" in text.lower()
    assert "0.8" in text
