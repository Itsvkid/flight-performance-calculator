# 03 — Aircraft flight performance calculator (Python)

Classical flight-performance metrics computed from aircraft geometry, mass and
engine data.

**Status:** Not started
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

- [ ] `src/` modules with docstrings and type hints
- [ ] `tests/` passing, including the ISA-table checks
- [ ] `requirements.txt` and a README with install and usage
- [ ] Power required vs power available curve
- [ ] Flight envelope: altitude vs true airspeed
- [ ] Payload-range diagram
- [ ] Validation table: computed vs published, with the discrepancy explained
- [ ] Its own GitHub repository

## Log

| Date | What was done |
|---|---|
| | |
