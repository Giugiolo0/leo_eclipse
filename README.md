# leo_eclipse

Eclipse duration and beta angle for **circular low Earth orbits**, in one
self-contained Python script.

The purpose of the code is to show that a dedicated mission-analysis package is
not needed to size the eclipse budget of a LEO mission: the problem is
elementary geometry plus a low-precision solar ephemeris, and the result matches
NASA GMAT R2025a (SPICE conical `EclipseLocator`, DE421 ephemerides, J2 gravity
field, RK8(9) propagator) over a 61-day reference run at h = 411 km, i = 37 deg:

```
beta angle        RMS 0.38 deg over an 83 deg sweep   (5780 samples)
eclipse duration  +1.57 % with k = 1.02 (default)     (948 orbits)
                  -0.42 % with k = 1.00
```

---

## Use

Requirements: Python >= 3.8 and NumPy. Matplotlib for the figures and openpyxl
for the `.xlsx` table are both optional — without openpyxl the table is written
as `.csv`.

Run it:

```bash
python leo_eclipse.py
```

and pick a mission:

```
==================================================================
 leo_eclipse -- choose a mission
==================================================================
  1) WINK / Space Rider     h =   411 km, i =  37.00 deg,  61 days
  2) ISS                    h =   418 km, i =  51.64 deg, 365 days
  3) Sun-synchronous        h =   600 km, i =  97.79 deg, 365 days
  4) Starlink shell         h =   550 km, i =  53.00 deg, 365 days
  5) Custom                 h =   411 km, i =  37.00 deg   (configuration block)
  0) All of them, with a side-by-side comparison
  q) Quit
==================================================================
```

That is the whole workflow: one file, one command, no other script to run. Each
choice runs the self-tests, prints the mission summary and writes its own
subfolder of `results`:

```
results/<mission>/eclipse_table.xlsx     beta angle, eclipse duration, eclipsed fraction
results/<mission>/beta_and_eclipse.png   beta angle and eclipse duration over the mission
results/<mission>/eclipse_vs_beta.png    the universal curve D(beta), with the mission orbits on it
results/<mission>/shadow_profile.png     the geometric shadow test over one orbit
results/mission_comparison.png           only with choice 0: all the orbits on the same axes
```

Choice `0` also prints the comparison table, which is the point of the exercise
— the same code and the same formulas produce every line, only the orbital
parameters change:

```
Mission                 h [km]  i [deg]  T [min]    dOm/dt  beta min  beta max   D min   D max  D mean  sunlit
WINK / Space Rider       411.0    37.00    92.79    -6.395    -48.36     35.08   31.00   36.78   35.22   62.0%
ISS                      417.5    51.64    92.92    -4.953    -73.01     74.67    0.00   36.76   32.16   65.4%
Sun-synchronous          600.0    97.79    96.69     0.986     57.08     83.48    0.00   22.86    5.56   94.3%
Starlink shell           550.0    53.00    95.65    -4.489    -76.31     74.45    0.00   36.32   30.01   68.6%
```

The reference orbits are the published ones: the ISS box is 413 x 422 km at
51.64 deg, the main Starlink Gen1 shell is 550 km at 53 deg, and 97.79 deg is
the inclination that makes a 600 km orbit sun-synchronous.

## Your own mission

Entry `5` of the menu is the **MISSION CONFIGURATION** block at the top of the
file. Edit it and nothing else:

```python
ALTITUDE_KM     = 411.0          # orbital altitude above R_E              [km]
INCLINATION_DEG = 37.0           # orbital inclination                    [deg]
RAAN_DEG        = 0.0            # right ascension of the node AT EPOCH   [deg]
ARG_LAT0_DEG    = 0.0            # argument of latitude at epoch          [deg]

EPOCH           = "2028-02-01"   # mission start, UTC, ISO format
END             = "2028-04-02"   # mission end, UTC  (set to None to use DAYS)
DAYS            = 61.0           # mission length, used only if END is None

PENUMBRA_FACTOR = 1.02           # 1.02 = literature penumbra allowance
```

To add a permanent entry to the menu, add a line to the `MISSIONS` dictionary
just below that block; it appears in the menu and in the comparison
automatically.

The same choices are available without the menu, which is what a batch run or
another script would use:

```bash
python leo_eclipse.py --mission 2
python leo_eclipse.py --mission all
python leo_eclipse.py --altitude 700 --inclination 98 --days 120
python leo_eclipse.py --penumbra-factor 1.0
python leo_eclipse.py --self-test
```

## Use as a library

```python
from datetime import datetime
from leo_eclipse import OrbitConfig, EclipseModel

cfg = OrbitConfig(altitude_km=411.0, inclination_deg=37.0, raan_deg=0.0,
                  epoch=datetime(2028, 2, 1), duration_days=61)
model = EclipseModel(cfg)

print(model.summary())
for date, beta, eclipse_min in model.daily_table():
    print(date.date(), f"{beta:+6.2f} deg", f"{eclipse_min:5.2f} min")
```

| Method | Returns |
|---|---|
| `model.beta_deg(t)` | beta angle at time `t` [deg] |
| `model.eclipse_duration_min(beta)` | closed-form eclipse duration per orbit [min] |
| `model.eclipse_fraction(beta)` | eclipsed fraction of the orbit [-] |
| `model.in_eclipse(t)` | geometric shadow test at time `t` (bool) |
| `model.shadow_distance_km(t)` | `(r . s_hat, distance from the Earth-Sun axis)` [km] |
| `model.eclipse_duration_numeric_min(t0)` | eclipse duration by direct sampling [min] |
| `model.position_eci(t)` | satellite position in ECI [km] |
| `model.orbit_normal_eci(t)` | unit normal to the orbital plane in ECI |
| `model.sample()` / `daily_table()` / `orbit_table()` / `write_table()` | time series |
| `model.summary()` | printable mission report |
| `make_figures(model, out_dir)` | the three figures of one mission |
| `compare_missions(models, out_dir)` / `comparison_table(models)` | several missions side by side |

Derived attributes: `period_s`, `n`, `a`, `raan_dot`, `rho`, `beta_crit_deg`,
`umbra_length`, `umbra_radius`, `penumbra_radius`, `v_circ`.

## Scope

The script computes the eclipse, and only the eclipse. Payload illumination,
detector faces and thermal budgets are mission-specific and are not part of it;
what they need from here is `beta_deg()`, `in_eclipse()` and
`eclipse_duration_min()`.

## The penumbra factor

`PENUMBRA_FACTOR` (`k`) defaults to **1.02**, the value used in the literature
and retained as the baseline in the accompanying paper, so that durations stay
comparable with published analyses and because its bias is conservative
(slightly longer eclipses) for cold-case thermal and battery sizing.

The validation against GMAT quantifies that choice: `k = 1.02` biases the
duration by **+1.57 %**, `k = 1.00` by **-0.42 %** with a threefold smaller RMS
residual. For sub-percent accuracy set `PENUMBRA_FACTOR = 1.0`.

`k` scales only the closed-form duration. The geometric shadow test
`in_eclipse()` is independent of it, which is why the two can be compared.

## Self-tests

They run at each start (`SELF_TEST = True`); `python leo_eclipse.py
--self-test` runs them alone. All of them are configuration-independent:

1. Kepler period against reference LEO orbits.
2. Beta angle inside the theoretical bounds +/-(eps + i), on every orbit of
   the catalogue.
3. `D(+beta) = D(-beta)`.
4. `D` maximal at beta = 0, zero beyond beta_crit.
5. Geometric (sampled) duration against the closed form at `k = 1`, on orbits
   spread over the beta sweep — the two independent implementations of the
   shadow crossing must agree.
6. The nodal regression, inverted at the sun-synchronous rate of 360 deg per
   sidereal year, against the published sun-synchronous inclinations: the model
   returns 97.66, 99.01 and 100.72 deg at 567, 894 and 1262 km, against the
   tabulated 97.7, 99.0 and 100.7 deg. This is the only check that compares
   against external numbers rather than against the model's own arithmetic.

## Assumptions

Circular orbit; secular J2 only; low-precision solar ephemeris (~0.01 deg in
longitude); spherical Earth; cylindrical umbra with a scalar penumbra factor;
uniform UTC. Each is documented, with its quantified impact, in the module
docstring and in Table 1 of the accompanying paper.

## License

MIT — see `LICENSE`.
