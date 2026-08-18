# 03 — Aircraft flight performance calculator (Python)

Classical flight-performance metrics computed from aircraft geometry, mass and
engine data.

**Status:** Complete — model, figures and validation done. Ready to split into its own GitHub repo.
**Tool:** Python 3.11+, numpy, matplotlib

## Objective

A small, tested, installable package — not a notebook. This is the project that
most directly backs the site's positioning, so the code quality is part of the
deliverable. A reviewer will open `src/` before they open any plot.

## Formulations

**ISA atmosphere** (troposphere, h < 11 km):

```
T(h) = T0 - L*h
rho(h) = rho0 * (1 - L*h/T0) ** (g0/(R*L) - 1)
```
with `T0 = 288.15 K`, `L = 0.0065 K/m`, `rho0 = 1.225 kg/m^3`,
`R = 287.05 J/(kg*K)`, `g0 = 9.80665 m/s^2`.

**Drag polar:** `CD = CD0 + CL^2 / (pi * AR * e)`

**Breguet range (jet):** `R = (V/ct) * (L/D) * ln(Wi/Wf)`

## Module layout

```
src/
  atmosphere.py      rho, T, p, a as functions of altitude
  aircraft.py        dataclass: mass, S, AR, b, CD0, e, engine specs
  performance.py     V_md, V_mp, max rate of climb, ceilings, range, endurance
  plotting.py        power required vs available, flight envelope
tests/
  test_atmosphere.py sea-level and 11 km values against ISA tables
  test_performance.py analytical checks
figures/             output plots
```

## Validation — do not skip this

Every function needs something to check it against, or the whole thing is
unfalsifiable:

- **Atmosphere:** ISA tables at 0, 5 000, 11 000 m. These are published to
  several decimals — your function should match.
- **Minimum drag speed:** at V_md the induced and parasite drag terms are
  equal. Assert it.
- **Whole model:** run a real aircraft with published numbers (a Cessna 172 or
  a 737 both work) and compare computed range and ceiling against the type
  certificate. Being 10% off with a stated reason is a result. Being 400% off
  silently is a bug.

## Deliverables

- [x] `atmosphere.py` — ISA to 20 km, troposphere + stratosphere
- [x] `aircraft.py` — geometry, drag polar, Mach-dependent thrust lapse
- [x] `performance.py` — drag, climb, ceilings, range, endurance
- [x] 37 tests passing against ISO 2533 and closed-form identities
- [x] `plotting.py` — thrust curves, flight envelope, payload-range
- [x] 48 tests passing
- [x] Validation table against three published aircraft — see [VALIDATION.md](VALIDATION.md)
- [ ] `requirements.txt` and a README with install and usage
- [ ] Power required vs power available curve
- [ ] Flight envelope: altitude vs true airspeed
- [ ] Payload-range diagram
- [ ] Validation table: computed vs published, with the discrepancy explained
- [ ] Its own GitHub repository

## Log

| Date | What was done |
|---|---|
| 2026-08-19 | `validation.py` + VALIDATION.md. Three aircraft, 737-800 / A320-200 / 777-300ER, spanning a 4.5x mass range. Ceiling within 8% on all three but two published values fall outside the drag-polar band — testing it apart showed the 777 gap is weight (model gives exactly 13.1 km at 92% MTOW) while the A320 gap is not, and reverses with weight, so those published figures are almost certainly certified operating altitudes rather than performance ceilings. Range percentages flatter the model: it burns all fuel in cruise while published range carries reserves. Biggest finding — the model predicts M 0.98-1.09 max speed because the polar has no compressibility term, so any speed above M 0.8 is meaningless. 52 tests. |
| 2026-08-19 | `plotting.py` — three figures on a colour-vision-validated palette. Drawing them exposed a physics bug no unit test had: the envelope's left edge was stall speed at every altitude, but above ~12 km a jet runs out of thrust before it runs out of wing, so the true low-speed limit is thrust-limited. Added `min_level_speed`; at 13 km it returns 186 m/s against a 166 m/s stall. Envelope now closes at the apex where both limits meet. 48 tests. |
| 2026-08-19 | `aircraft.py` + `performance.py`. Two bugs found by sanity-checking outputs against reality rather than by any test. (1) TSFC is stored mass-based, kg/(N*s), but Breguet in weights needs 1/s — the missing g made range 67 000 km. (2) Thrust ignored forward speed, so a turbofan kept its static thrust at Mach 0.8; that gave a 16.8 km ceiling and a 12 deg climb angle. Added Mattingly's high-bypass lapse. Now: ceiling 13.5 km, range 6 890 km, cruise thrust 52.5 kN — all in the right band. 37 tests pass. |
| 2026-08-19 | Environment audited (see ../SETUP.md). `atmosphere.py`: ISA to 20 km, both layers, `AtmosphereState` dataclass with `sigma`. 14 tests pass. Reference densities at 5 km and 8 km from memory proved inconsistent with the tabulated pressures at the same altitudes — `p = rho*R*T` did not hold for them — so density tolerance is 0.2% with the ideal gas law carrying the strict internal check instead. |
