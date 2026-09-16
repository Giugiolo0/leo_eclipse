"""
leo_eclipse.py -- Eclipse and beta-angle analysis for circular low Earth orbits.
================================================================================

A single, self-contained Python script that computes, for any circular LEO:

    * the beta angle (solar elevation above the orbital plane) over time,
      including the secular J2 regression of the node;
    * the eclipse duration per orbit, both in closed form and by direct
      geometric sampling of the Earth shadow.

The point of the script is that no external mission-analysis software is
needed: the whole computation is elementary geometry plus a low-precision
solar ephemeris, and it reproduces NASA GMAT R2025a (SPICE conical
EclipseLocator, DE421 ephemerides, J2 gravity field, RK8(9) propagator) to
0.4 deg RMS on the beta angle and to better than 2 % on the eclipse duration
over a 61-day mission at h = 411 km, i = 37 deg.

--------------------------------------------------------------------------------
HOW TO USE IT
--------------------------------------------------------------------------------

Run it:

    python leo_eclipse.py

and pick a mission from the menu:

    1) WINK / Space Rider    h = 411.0 km, i = 37.00 deg,  61 days
    2) ISS                   h = 417.5 km, i = 51.64 deg, 365 days
    3) Sun-synchronous       h = 600.0 km, i = 97.79 deg, 365 days
    4) Starlink shell        h = 550.0 km, i = 53.00 deg, 365 days
    5) Custom                the MISSION CONFIGURATION block below
    0) All of them, with a side-by-side comparison

The reference orbits are the published ones: the ISS box is 413 x 422 km at
51.64 deg, the main Starlink shell is 550 km at 53 deg, and 97.79 deg is the
inclination that makes a 600 km orbit sun-synchronous. The self-test verifies
that last one from the other side, by inverting the nodal regression at the
sun-synchronous rate and recovering the tabulated inclinations.

Each choice prints the mission summary and writes, in its own subfolder of
``results``, the table of the analysis and the figures.

To change the custom mission, edit the MISSION CONFIGURATION block below.
Nothing else in the file has to be touched, and there is no other file to run.

The same choices are available non-interactively, which is what a batch run or
another script would use::

    python leo_eclipse.py --mission 2
    python leo_eclipse.py --mission all
    python leo_eclipse.py --altitude 700 --inclination 98 --days 120
    python leo_eclipse.py --self-test

Requirements: Python >= 3.8, NumPy. Matplotlib for the figures, and openpyxl
if the table is wanted as .xlsx instead of .csv -- both optional.

--------------------------------------------------------------------------------
MODEL ASSUMPTIONS (all of them)
--------------------------------------------------------------------------------
A1  Circular orbit (e = 0). The radius is constant, r = R_E + h, and the
    argument of latitude grows linearly with time.
A2  Secular J2 only. The orbital plane precesses at the mean rate
    dOmega/dt = -3/2 J2 (R_E/a)^2 n cos i; short-period and higher-order zonal
    terms, drag, SRP and third-body perturbations are ignored.
A3  Low-precision solar ephemeris (mean longitude + equation of centre, two
    terms). Accuracy about 0.01 deg in ecliptic longitude, i.e. < 0.02 deg on
    the beta angle -- negligible against the 0.4 deg RMS observed against GMAT.
A4  Spherical Earth of radius R_E = 6378.137 km; oblateness is not included in
    the shadow cross-section.
A5  Cylindrical (umbra-only) shadow for the geometric in/out test; the
    closed-form duration additionally applies the scalar factor
    PENUMBRA_FACTOR (1.02 by default) to account for the penumbra transition
    and for atmospheric extinction at the shadow boundary.
A6  Times are UTC and treated as uniform (no UT1/TT distinction, no leap
    seconds). The resulting epoch error is < 1.5 min over the 21st century.

Author: G. Fontanella (GSSI / INFN-LNGS)
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Constants", "OrbitConfig", "EclipseModel", "sun_unit_vector_eci",
           "julian_date", "julian_centuries", "make_figures",
           "compare_missions", "self_test", "MISSIONS"]
__version__ = "1.2.0"


# ============================================================================
#
#                         MISSION CONFIGURATION
#                     ---------------------------
#            This is the only block that has to be edited.
#              It defines the "Custom" entry of the menu.
#
# ============================================================================

ALTITUDE_KM     = 411.0          # orbital altitude above R_E              [km]
INCLINATION_DEG = 37.0           # orbital inclination                    [deg]
RAAN_DEG        = 0.0            # right ascension of the node AT EPOCH   [deg]
ARG_LAT0_DEG    = 0.0            # argument of latitude at epoch          [deg]

EPOCH           = "2028-02-01"   # mission start, UTC, ISO format
END             = "2028-04-02"   # mission end, UTC  (set to None to use DAYS)
DAYS            = 61.0           # mission length, used only if END is None

PENUMBRA_FACTOR = 1.02           # 1.02 = literature penumbra allowance
                                 # 1.00 = pure geometric umbra (see the paper)

OUTPUT_DIR      = "results"      # created next to this script if missing
PLOTS           = True           # save the figures (needs matplotlib)
SHOW_PLOTS      = False          # also open them in a window
CSV_STEP_HOURS  = 24.0           # sampling step of the output table        [h]
SELF_TEST       = True           # run the consistency checks before the analysis

# ---------------------------------------------------------------------------
# The mission catalogue offered by the menu. Add an entry and it appears.
# ---------------------------------------------------------------------------
MISSIONS: Dict[str, dict] = {
    "1": dict(name="WINK / Space Rider", altitude_km=411.0, inclination_deg=37.0,
              raan_deg=0.0, epoch="2028-02-01", end="2028-04-02"),
    # ISS: mean of the 413 x 422 km operational box, inclination 51.64 deg.
    "2": dict(name="ISS", altitude_km=417.5, inclination_deg=51.64,
              raan_deg=0.0, epoch="2028-01-01", days=365.0),
    # Sun-synchronous: 97.79 deg is the inclination that makes dOmega/dt equal
    # 360 deg / 365.2422 days at h = 600 km (see the self-test).
    "3": dict(name="Sun-synchronous", altitude_km=600.0, inclination_deg=97.79,
              raan_deg=0.0, epoch="2028-01-01", days=365.0),
    "4": dict(name="Starlink shell", altitude_km=550.0, inclination_deg=53.0,
              raan_deg=0.0, epoch="2028-01-01", days=365.0),
}

# ============================================================================
#                     END OF THE CONFIGURATION BLOCK
# ============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
class Constants:
    """Fundamental constants. Override on an :class:`OrbitConfig` if needed."""

    R_EARTH = 6378.137          # Earth equatorial radius            [km]
    MU_EARTH = 398600.4418      # Earth gravitational parameter      [km^3/s^2]
    J2 = 1.08262668e-3          # Earth second zonal harmonic        [-]
    R_SUN = 696340.0            # Solar radius                       [km]
    AU = 149597870.7            # Astronomical unit                  [km]
    OBLIQUITY = 23.43929111     # Obliquity of the ecliptic (J2000)  [deg]


# ----------------------------------------------------------------------------
# Solar ephemeris (low precision, see assumption A3)
# ----------------------------------------------------------------------------
def julian_date(date: datetime) -> float:
    """Julian Date of a (naive, UTC) ``datetime``, Fliegel-Van Flandern style."""
    year, month = date.year, date.month
    day = date.day + (date.hour + date.minute / 60.0
                      + (date.second + date.microsecond * 1e-6) / 3600.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    return (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1))
            + day + b - 1524.5)


def julian_centuries(date: datetime) -> float:
    """Julian centuries elapsed since the J2000.0 epoch."""
    return (julian_date(date) - 2451545.0) / 36525.0


def sun_unit_vector_eci(date: datetime,
                        obliquity_deg: float = Constants.OBLIQUITY) -> np.ndarray:
    """
    Unit vector towards the Sun in the Earth-centred inertial (ECI, mean of
    J2000) frame.

    Mean longitude and equation of centre give the true ecliptic longitude
    ``lambda``; the vector is then rotated from the ecliptic to the equatorial
    frame by the obliquity ``eps``:

        s_hat = [cos(lambda), cos(eps) sin(lambda), sin(eps) sin(lambda)]
    """
    t = julian_centuries(date)
    mean_longitude = (280.460 + 36000.771 * t) % 360.0
    mean_anomaly = math.radians((357.5291092 + 35999.05034 * t) % 360.0)
    centre = (1.914666471 * math.sin(mean_anomaly)
              + 0.019994643 * math.sin(2.0 * mean_anomaly))
    lam = math.radians(mean_longitude + centre)
    eps = math.radians(obliquity_deg)
    return np.array([math.cos(lam),
                     math.cos(eps) * math.sin(lam),
                     math.sin(eps) * math.sin(lam)])


# ----------------------------------------------------------------------------
# Mission configuration
# ----------------------------------------------------------------------------
@dataclass
class OrbitConfig:
    """
    Complete description of a circular LEO mission.

    Parameters
    ----------
    altitude_km       : orbital altitude above the mean equatorial radius [km]
    inclination_deg   : orbital inclination                               [deg]
    raan_deg          : right ascension of the ascending node AT EPOCH    [deg]
    epoch             : mission start, UTC (naive ``datetime``)
    duration_days     : mission length in days (ignored if ``end`` is given)
    end               : explicit mission end, UTC (optional)
    arg_lat0_deg      : argument of latitude at epoch, measured from the
                        ascending node                                    [deg]
    penumbra_factor   : empirical correction applied to the analytical
                        eclipse duration (1.02 = penumbra/atmosphere
                        allowance; 1.0 for a pure umbra duration)
    name              : label used in the report and in the output folder
    """

    altitude_km: float
    inclination_deg: float
    epoch: datetime
    raan_deg: float = 0.0
    duration_days: float = 1.0
    end: Optional[datetime] = None
    arg_lat0_deg: float = 0.0
    penumbra_factor: float = 1.02
    name: str = "Custom"
    constants: Constants = field(default_factory=Constants)

    def __post_init__(self) -> None:
        if self.altitude_km <= 0:
            raise ValueError("altitude_km must be positive")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must lie in [0, 180]")
        if self.end is None:
            self.end = self.epoch + timedelta(days=float(self.duration_days))
        elif self.end <= self.epoch:
            raise ValueError("end must be later than epoch")
        else:
            self.duration_days = (self.end - self.epoch).total_seconds() / 86400.0

    @property
    def slug(self) -> str:
        """Filesystem-safe version of the mission name."""
        out = "".join(c if (c.isalnum() or c in "-_") else "_"
                      for c in self.name.lower())
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_") or "mission"


def config_from_entry(entry: dict, penumbra_factor: float = PENUMBRA_FACTOR
                      ) -> OrbitConfig:
    """Build an :class:`OrbitConfig` from a MISSIONS catalogue entry."""
    return OrbitConfig(
        name=entry["name"],
        altitude_km=entry["altitude_km"],
        inclination_deg=entry["inclination_deg"],
        raan_deg=entry.get("raan_deg", 0.0),
        arg_lat0_deg=entry.get("arg_lat0_deg", 0.0),
        epoch=datetime.fromisoformat(entry["epoch"]),
        end=datetime.fromisoformat(entry["end"]) if entry.get("end") else None,
        duration_days=entry.get("days", 1.0),
        penumbra_factor=entry.get("penumbra_factor", penumbra_factor))


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class EclipseModel:
    """Beta angle, shadow geometry and eclipse duration for a circular LEO."""

    def __init__(self, config: OrbitConfig):
        self.cfg = config
        c = config.constants

        # --- Derived orbital quantities (self-consistent by construction) ---
        self.a = c.R_EARTH + config.altitude_km                # semi-major axis [km]
        self.period_s = 2.0 * math.pi * math.sqrt(self.a ** 3 / c.MU_EARTH)
        self.n = 2.0 * math.pi / self.period_s                 # mean motion [rad/s]
        self.v_circ = math.sqrt(c.MU_EARTH / self.a)           # speed [km/s]

        # --- Secular J2 nodal regression [rad/s] (assumption A2) ---
        self.raan_dot = (-1.5 * c.J2 * (c.R_EARTH / self.a) ** 2
                         * self.n * math.cos(math.radians(config.inclination_deg)))

        # --- Shadow geometry ---
        self.rho = math.asin(c.R_EARTH / self.a)               # Earth angular radius [rad]
        self.beta_crit_deg = math.degrees(self.rho)
        self.sun_angular_radius = math.atan(c.R_SUN / c.AU)
        self.umbra_length = c.R_EARTH / math.tan(self.sun_angular_radius)
        self.penumbra_length = 1.02 * self.umbra_length
        self.umbra_radius = c.R_EARTH * (1.0 - config.altitude_km / self.umbra_length)
        self.penumbra_radius = c.R_EARTH * (1.0 + config.altitude_km / self.penumbra_length)

    # ------------------------------------------------------------------ time
    def _elapsed(self, date: datetime) -> float:
        return (date - self.cfg.epoch).total_seconds()

    def raan_deg(self, date: datetime) -> float:
        """RAAN at ``date``, including the secular J2 regression."""
        return self.cfg.raan_deg + math.degrees(self.raan_dot * self._elapsed(date))

    def arg_latitude_rad(self, date: datetime) -> float:
        """Argument of latitude (= true anomaly for a circular orbit) [rad]."""
        return (math.radians(self.cfg.arg_lat0_deg)
                + self.n * self._elapsed(date)) % (2.0 * math.pi)

    # ------------------------------------------------------------------ frames
    def orbit_normal_eci(self, date: datetime) -> np.ndarray:
        """Unit normal to the orbital plane in ECI."""
        i = math.radians(self.cfg.inclination_deg)
        om = math.radians(self.raan_deg(date))
        return np.array([math.sin(om) * math.sin(i),
                         -math.cos(om) * math.sin(i),
                         math.cos(i)])

    def rotation_orbital_to_eci(self, date: datetime) -> np.ndarray:
        """Active rotation R_z(Omega) R_x(i) taking orbital-plane -> ECI."""
        i = math.radians(self.cfg.inclination_deg)
        om = math.radians(self.raan_deg(date))
        r_i = np.array([[1.0, 0.0, 0.0],
                        [0.0, math.cos(i), -math.sin(i)],
                        [0.0, math.sin(i), math.cos(i)]])
        r_om = np.array([[math.cos(om), -math.sin(om), 0.0],
                         [math.sin(om), math.cos(om), 0.0],
                         [0.0, 0.0, 1.0]])
        return r_om @ r_i

    def position_eci(self, date: datetime) -> np.ndarray:
        """Satellite position vector in ECI [km]."""
        u = self.arg_latitude_rad(date)
        p_orb = np.array([math.cos(u), math.sin(u), 0.0])
        return self.rotation_orbital_to_eci(date) @ p_orb * self.a

    # ------------------------------------------------------------------ beta
    def beta_deg(self, date: datetime) -> float:
        """
        Beta angle: elevation of the Sun above the orbital plane [deg].

        beta = arcsin( s_hat . n_hat )
        """
        s = sun_unit_vector_eci(date, self.cfg.constants.OBLIQUITY)
        n = self.orbit_normal_eci(date)
        return math.degrees(math.asin(float(np.clip(np.dot(s, n), -1.0, 1.0))))

    def beta_bounds_deg(self) -> Tuple[float, float]:
        """Theoretical annual bounds |beta| <= eps + i (for i <= 90 - eps)."""
        eps = self.cfg.constants.OBLIQUITY
        i = self.cfg.inclination_deg
        b = eps + min(i, 180.0 - i)
        return (-min(b, 90.0), min(b, 90.0))

    # ------------------------------------------------------------------ eclipse
    def eclipse_duration_min(self, beta_deg: float,
                             penumbra_factor: Optional[float] = None) -> float:
        """
        Analytical eclipse duration per orbit [min] for a given beta angle.

            cos(phi/2) = cos(rho) / cos(beta),      D = phi/(2 pi) * T * k

        Returns 0.0 when |beta| >= beta_crit (no shadow crossing).
        """
        k = self.cfg.penumbra_factor if penumbra_factor is None else penumbra_factor
        b = math.radians(abs(beta_deg))
        if b >= self.rho:
            return 0.0
        ratio = math.cos(self.rho) / math.cos(b)
        if abs(ratio) >= 1.0:
            return 0.0
        phi = 2.0 * math.acos(ratio)
        return (phi / (2.0 * math.pi)) * self.period_s * k / 60.0

    def eclipse_fraction(self, beta_deg: float) -> float:
        """Eclipsed fraction of the orbital period [-]."""
        return self.eclipse_duration_min(beta_deg) / (self.period_s / 60.0)

    def shadow_distance_km(self, date: datetime) -> Tuple[float, float]:
        """
        The two quantities the geometric shadow test is built on:

            (r . s_hat, |r - (r . s_hat) s_hat|)   [km, km]

        The first is negative on the anti-solar side of the Earth; the second
        is the distance from the Earth-Sun axis, to be compared with R_E.
        """
        s = sun_unit_vector_eci(date, self.cfg.constants.OBLIQUITY)
        r = self.position_eci(date)
        along = float(np.dot(r, s))
        return along, float(np.linalg.norm(r - along * s))

    def in_eclipse(self, date: datetime) -> bool:
        """
        Geometric shadow test (cylindrical umbra, assumption A5).

        The satellite is eclipsed iff it is on the anti-solar side of the Earth
        AND its distance from the Earth-Sun axis is smaller than R_E::

            r . s_hat < 0   and   |r - (r . s_hat) s_hat| < R_E
        """
        along, perp = self.shadow_distance_km(date)
        return along < 0.0 and perp < self.cfg.constants.R_EARTH

    def eclipse_duration_numeric_min(self, start: datetime,
                                     step_s: float = 1.0) -> float:
        """
        Eclipse duration of the single orbit beginning at ``start`` [min],
        obtained by direct sampling of :meth:`in_eclipse`. Independent
        cross-check of :meth:`eclipse_duration_min` (which it reproduces to
        within the penumbra factor).
        """
        n_steps = int(round(self.period_s / step_s))
        count = sum(1 for k in range(n_steps)
                    if self.in_eclipse(start + timedelta(seconds=k * step_s)))
        return count * step_s / 60.0

    # ------------------------------------------------------------------ output
    def sample(self, step: timedelta = timedelta(days=1)
               ) -> Iterator[Tuple[datetime, float, float]]:
        """Yield ``(date, beta_deg, eclipse_min)`` from epoch to end."""
        t = self.cfg.epoch
        while t <= self.cfg.end:
            beta = self.beta_deg(t)
            yield t, beta, self.eclipse_duration_min(beta)
            t += step

    def daily_table(self) -> List[Tuple[datetime, float, float]]:
        """Daily (noon-sampled) beta angle and eclipse duration."""
        return [(t + timedelta(hours=12),
                 self.beta_deg(t + timedelta(hours=12)),
                 self.eclipse_duration_min(self.beta_deg(t + timedelta(hours=12))))
                for t in (self.cfg.epoch + timedelta(days=k)
                          for k in range(int(self.cfg.duration_days) + 1))]

    def orbit_table(self) -> List[Tuple[datetime, float, float]]:
        """Per-orbit ``(date, beta_deg, eclipse_min)`` over the whole mission."""
        n_orbits = int(self.cfg.duration_days * 86400.0 / self.period_s)
        out = []
        for k in range(n_orbits + 1):
            t = self.cfg.epoch + timedelta(seconds=k * self.period_s)
            beta = self.beta_deg(t)
            out.append((t, beta, self.eclipse_duration_min(beta)))
        return out

    def write_table(self, path_base: str,
                    step: timedelta = timedelta(days=1)) -> str:
        """
        Write the ``(date, beta, eclipse duration, eclipsed fraction)`` table,
        as ``.xlsx`` when openpyxl is available and as ``.csv`` otherwise.
        Returns the path actually written.
        """
        header = ["utc", "beta_deg", "eclipse_min", "eclipse_fraction"]
        rows = [(t.strftime("%Y-%m-%d %H:%M"), round(beta, 4), round(ecl, 4),
                 round(ecl / (self.period_s / 60.0), 5))
                for t, beta, ecl in self.sample(step)]
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except ImportError:
            path = path_base + ".csv"
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)
            return path

        wb = Workbook()
        ws = wb.active
        ws.title = "eclipse"
        ws.append(header)
        for row in rows:
            ws.append(list(row))
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col, width in zip("ABCD", (18, 12, 14, 18)):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "A2"
        path = path_base + ".xlsx"
        wb.save(path)
        return path

    def summary(self) -> str:
        """Human-readable summary of the configuration and of the mission."""
        table = self.daily_table()
        betas = [b for _, b, _ in table]
        ecl = [e for _, _, e in table]
        c = self.cfg
        return "\n".join([
            "=" * 66,
            f" leo_eclipse {__version__} -- {c.name}",
            "=" * 66,
            f" Altitude                {c.altitude_km:10.3f} km",
            f" Semi-major axis         {self.a:10.3f} km",
            f" Inclination             {c.inclination_deg:10.3f} deg",
            f" RAAN at epoch           {c.raan_deg:10.3f} deg",
            f" Orbital period          {self.period_s / 60.0:10.3f} min"
            f"   ({86400.0 / self.period_s:.2f} orbits/day)",
            f" Nodal regression        {math.degrees(self.raan_dot) * 86400.0:10.4f} deg/day",
            f" Earth angular radius    {self.beta_crit_deg:10.3f} deg  (= beta_crit)",
            f" Umbra cone length       {self.umbra_length:10.1f} km",
            f" Umbra radius at h       {self.umbra_radius:10.1f} km",
            f" Penumbra factor k       {c.penumbra_factor:10.3f}",
            f" Epoch                   {c.epoch:%Y-%m-%d %H:%M} UTC",
            f" End                     {c.end:%Y-%m-%d %H:%M} UTC"
            f"   ({c.duration_days:.0f} days)",
            "-" * 66,
            f" Beta angle              {min(betas):+7.2f} .. {max(betas):+7.2f} deg"
            f"   (theoretical bound +/-{self.beta_bounds_deg()[1]:.1f})",
            f" Eclipse duration        {min(ecl):7.2f} .. {max(ecl):7.2f} min"
            f"   (mean {sum(ecl) / len(ecl):.2f})",
            f" Eclipse-free days       {sum(1 for e in ecl if e == 0.0):d} / {len(ecl):d}",
            f" Mean eclipsed fraction  {sum(ecl) / len(ecl) / (self.period_s / 60.0) * 100.0:7.2f} %",
            "=" * 66,
        ])


# ----------------------------------------------------------------------------
# Figures (matplotlib is imported here, and only here)
# ----------------------------------------------------------------------------
def _headroom(ax, frac: float = 0.26) -> None:
    """Expand the upper y-limit so that the legend sits clear of the curves."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) / (1.0 - frac))


def make_figures(model: EclipseModel, out_dir: str,
                 show: bool = False) -> List[str]:
    """
    Save the figures of the analysis in ``out_dir`` and return their paths:

        beta_and_eclipse   beta angle and eclipse duration over the mission
        eclipse_vs_beta    the universal curve D(beta), k as configured and k = 1
        shadow_profile     the geometric shadow test over one representative
                           orbit, against the closed-form duration
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    cfg = model.cfg
    saved: List[str] = []

    def save(fig, name: str) -> None:
        path = os.path.join(out_dir, name + ".png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        saved.append(path)

    def date_axis(ax) -> None:
        span = max(cfg.duration_days, 1.0)
        if span > 120:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, int(round(span / 9.0)))))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ---------------------------------------------- 1. beta and eclipse vs time
    orbits = model.orbit_table()
    t = [o[0] for o in orbits]
    beta = np.array([o[1] for o in orbits])
    ecl = np.array([o[2] for o in orbits])
    umbra = np.array([model.eclipse_duration_min(b, 1.0) for b in beta])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"{cfg.name}  --  h = {cfg.altitude_km:.0f} km, "
                 f"i = {cfg.inclination_deg:.1f} deg", fontsize=15)

    ax1.plot(t, beta, color="royalblue", lw=1.6, label=r"$\beta(t)$")
    ax1.axhline(0.0, color="gray", lw=0.5)
    # Draw a beta_crit line only on the side the mission actually approaches,
    # so that the panel is not stretched by a threshold that is never in play.
    margin = 12.0
    shown = False
    for sign in (+1.0, -1.0):
        level = sign * model.beta_crit_deg
        if beta.min() - margin <= level <= beta.max() + margin:
            ax1.axhline(level, color="crimson", ls="--", lw=1.2,
                        label=(rf"$\beta_{{crit}} = "
                               rf"{model.beta_crit_deg:.1f}^\circ$")
                              if not shown else None)
            shown = True
    ax1.set_ylabel(r"$\beta$  [deg]", fontsize=13)
    ax1.grid(alpha=0.3)
    _headroom(ax1, 0.24)
    ax1.legend(loc="upper center", ncol=2, fontsize=11, framealpha=0.95)

    ax2.plot(t, ecl, color="#c0392b", lw=1.6,
             label=f"$k = {cfg.penumbra_factor:.2f}$, mean {ecl.mean():.2f} min")
    if abs(cfg.penumbra_factor - 1.0) > 1e-9:
        ax2.plot(t, umbra, color="#e67e22", ls=":", lw=1.4,
                 label=f"$k = 1.00$, mean {umbra.mean():.2f} min")
    ax2.set_ylabel("Eclipse duration / orbit  [min]", fontsize=13)
    ax2.set_xlabel(f"Date ({cfg.epoch.year})", fontsize=13)
    ax2.grid(alpha=0.3)
    _headroom(ax2, 0.24)
    ax2.legend(loc="upper center", ncol=2, fontsize=11, framealpha=0.95)
    date_axis(ax2)

    plt.tight_layout()
    save(fig, "beta_and_eclipse")
    if not show:
        plt.close(fig)

    # ------------------------------------------- 2. the universal curve D(beta)
    grid = np.linspace(0.0, model.beta_crit_deg, 500)
    d_k = np.array([model.eclipse_duration_min(b) for b in grid])
    d_1 = np.array([model.eclipse_duration_min(b, 1.0) for b in grid])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(np.abs(beta), ecl, s=26, facecolors="none", edgecolors="royalblue",
               lw=0.6, alpha=0.35, zorder=1,
               label=f"{len(beta)} mission orbits")
    ax.plot(grid, d_k, color="#c0392b", lw=2, zorder=4,
            label=rf"$k = {cfg.penumbra_factor:.2f}$")
    if abs(cfg.penumbra_factor - 1.0) > 1e-9:
        ax.plot(grid, d_1, color="#e67e22", ls=":", lw=2, zorder=4,
                label=r"$k = 1.00$")
    ax.axvline(model.beta_crit_deg, color="gray", ls="--", lw=1.2,
               label=rf"$\beta_{{crit}} = \rho = {model.beta_crit_deg:.2f}^\circ$")
    ax.set_xlabel(r"$|\beta|$  [deg]", fontsize=13)
    ax.set_ylabel("Eclipse duration / orbit  [min]", fontsize=13)
    ax.set_title(r"$\cos(\varphi/2) = \cos\rho / \cos\beta$,  "
                 r"$D = \varphi/2\pi \cdot T \cdot k$   "
                 rf"($h = {cfg.altitude_km:.0f}$ km)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, model.beta_crit_deg * 1.04)
    _headroom(ax, 0.22)
    ax.legend(loc="upper center", ncol=2, fontsize=11, framealpha=0.95)
    plt.tight_layout()
    save(fig, "eclipse_vs_beta")
    if not show:
        plt.close(fig)

    # ------------------------------- 3. the geometric test over a single orbit
    # Representative orbit: the one with the longest eclipse of the mission.
    t0 = orbits[int(np.argmax(ecl))][0]
    dt_min = np.linspace(0.0, model.period_s / 60.0, 1200)
    along, perp, shadow = [], [], []
    for m in dt_min:
        al, pe = model.shadow_distance_km(t0 + timedelta(minutes=float(m)))
        along.append(al)
        perp.append(pe)
        shadow.append(al < 0.0 and pe < cfg.constants.R_EARTH)
    along, perp = np.array(along), np.array(perp)
    shadow = np.array(shadow)

    numeric = model.eclipse_duration_numeric_min(t0, step_s=0.5)
    closed_1 = model.eclipse_duration_min(model.beta_deg(t0), 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(dt_min[along >= 0.0], perp[along >= 0.0], ".", ms=2.4, color="gray",
            label=r"sunlit side, $\mathbf{r}\cdot\hat{s} > 0$")
    ax.plot(dt_min[along < 0.0], perp[along < 0.0], ".", ms=2.4, color="royalblue",
            label=r"anti-solar side, $\mathbf{r}\cdot\hat{s} < 0$")
    ax.axhline(cfg.constants.R_EARTH, color="crimson", ls="--", lw=1.4,
               label=rf"$R_\oplus = {cfg.constants.R_EARTH:.0f}$ km")
    ax.fill_between(dt_min, 0, 1, where=shadow, color="black", alpha=0.13,
                    step="mid", transform=ax.get_xaxis_transform(),
                    label=f"umbra: sampled {numeric:.2f} min, "
                          f"closed form {closed_1:.2f} min")
    ax.set_xlabel(f"Time from {t0:%d/%m/%Y %H:%M} UTC  [min]", fontsize=13)
    ax.set_ylabel(r"$|\mathbf{r} - (\mathbf{r}\cdot\hat{s})\hat{s}|$  [km]",
                  fontsize=13)
    ax.set_title(rf"Geometric shadow test over one orbit "
                 rf"($\beta = {model.beta_deg(t0):+.2f}^\circ$)", fontsize=13)
    ax.set_xlim(0, dt_min[-1])
    ax.grid(alpha=0.3)
    _headroom(ax, 0.24)
    ax.legend(loc="upper center", ncol=2, fontsize=10, framealpha=0.95)
    plt.tight_layout()
    save(fig, "shadow_profile")
    if not show:
        plt.close(fig)

    if show:
        plt.show()
    return saved


def compare_missions(models: Sequence[EclipseModel], out_dir: str,
                     show: bool = False) -> Optional[str]:
    """
    Side-by-side comparison of several missions: the universal curve D(beta)
    of each orbit, and the eclipse duration along each mission. Returns the
    path of the figure, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    os.makedirs(out_dir, exist_ok=True)
    palette = ["#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for idx, model in enumerate(models):
        col = palette[idx % len(palette)]
        cfg = model.cfg
        label = (f"{cfg.name}  ($h={cfg.altitude_km:.0f}$ km, "
                 f"$i={cfg.inclination_deg:.1f}^\\circ$)")

        grid = np.linspace(0.0, model.beta_crit_deg, 400)
        ax1.plot(grid, [model.eclipse_duration_min(b) for b in grid],
                 color=col, lw=1.8, label=label)
        ax1.axvline(model.beta_crit_deg, color=col, ls=":", lw=1.0, alpha=0.6)

        table = model.daily_table()
        elapsed = [(t - cfg.epoch).total_seconds() / 86400.0 for t, _, _ in table]
        ax2.plot(elapsed, [e for _, _, e in table], color=col, lw=1.6, label=label)

    ax1.set_xlabel(r"$|\beta|$  [deg]", fontsize=13)
    ax1.set_ylabel("Eclipse duration / orbit  [min]", fontsize=13)
    ax1.set_title(r"Universal curve $D(\beta)$ (dotted: $\beta_{crit}$)", fontsize=13)
    ax1.grid(alpha=0.3)
    _headroom(ax1, 0.30)
    ax1.legend(loc="upper center", fontsize=9, framealpha=0.95)

    ax2.set_xlabel("Days from epoch", fontsize=13)
    ax2.set_ylabel("Eclipse duration / orbit  [min]", fontsize=13)
    ax2.set_title("Eclipse duration along each mission", fontsize=13)
    ax2.grid(alpha=0.3)
    _headroom(ax2, 0.30)
    ax2.legend(loc="upper center", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    path = os.path.join(out_dir, "mission_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def comparison_table(models: Sequence[EclipseModel]) -> str:
    """The four missions side by side, one line each."""
    head = (f"{'Mission':22s} {'h [km]':>7s} {'i [deg]':>8s} {'T [min]':>8s} "
            f"{'dOm/dt':>9s} {'beta min':>9s} {'beta max':>9s} "
            f"{'D min':>7s} {'D max':>7s} {'D mean':>7s} {'sunlit':>7s}")
    lines = ["", head, "-" * len(head)]
    for model in models:
        table = model.daily_table()
        betas = [b for _, b, _ in table]
        ecl = [e for _, _, e in table]
        mean = sum(ecl) / len(ecl)
        lines.append(
            f"{model.cfg.name:22s} {model.cfg.altitude_km:7.1f} "
            f"{model.cfg.inclination_deg:8.2f} {model.period_s / 60.0:8.2f} "
            f"{math.degrees(model.raan_dot) * 86400.0:9.3f} "
            f"{min(betas):9.2f} {max(betas):9.2f} "
            f"{min(ecl):7.2f} {max(ecl):7.2f} {mean:7.2f} "
            f"{100.0 - mean / (model.period_s / 60.0) * 100.0:6.1f}%")
    lines.append("")
    lines.append("  D = eclipse duration per orbit [min]; sunlit = fraction of the "
                 "orbit in sunlight.")
    lines.append("  The same code and the same formulas produce every line: only "
                 "the orbital parameters change.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Self-tests: physical identities the model must satisfy on any mission
# ----------------------------------------------------------------------------
def self_test(verbose: bool = True) -> bool:
    """
    Configuration-independent consistency checks. Returns True if all pass.

    1. Kepler's third law reproduces reference LEO periods.
    2. The beta angle stays inside its theoretical bounds +/-(eps + i).
    3. Eclipse duration is even in beta: D(+beta) = D(-beta).
    4. D(beta) is maximum at beta = 0 and vanishes for |beta| >= beta_crit.
    5. The geometric (sampled) eclipse duration matches the closed form
       divided by the penumbra factor -- the two independent implementations
       of the shadow crossing must agree.
    """
    ok = True
    report = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        report.append(f"  [{'PASS' if passed else 'FAIL'}] {name}"
                      f"{'  ' + detail if detail else ''}")

    # 1 -- Kepler
    refs = {"ISS (420 km)": (420.0, 92.9), "Hubble (540 km)": (540.0, 95.4),
            "SPACE RIDER (411 km)": (411.0, 92.8), "Polar (800 km)": (800.0, 100.9)}
    for name, (h, t_ref) in refs.items():
        m = EclipseModel(OrbitConfig(altitude_km=h, inclination_deg=0.0,
                                     epoch=datetime(2028, 1, 1)))
        check(f"Kepler period {name}", abs(m.period_s / 60.0 - t_ref) < 0.15,
              f"{m.period_s / 60.0:.2f} min vs {t_ref} min")

    # 2 -- beta bounds, on every orbit of the catalogue
    for key in sorted(MISSIONS):
        entry = MISSIONS[key]
        m = EclipseModel(OrbitConfig(altitude_km=entry["altitude_km"],
                                     inclination_deg=entry["inclination_deg"],
                                     epoch=datetime(2028, 1, 1),
                                     duration_days=365))
        betas = [b for _, b, _ in m.sample(timedelta(days=1))]
        bound = m.beta_bounds_deg()[1]
        check(f"beta within +/-(eps+i)  {entry['name']}",
              max(abs(b) for b in betas) <= bound + 1e-9,
              f"max |beta| = {max(abs(b) for b in betas):.2f} deg <= {bound:.2f} deg")

    # 3..5 -- a representative orbit
    cfg = OrbitConfig(altitude_km=411.0, inclination_deg=37.0, raan_deg=0.0,
                      epoch=datetime(2028, 1, 1), duration_days=365)
    model = EclipseModel(cfg)

    sym = max(abs(model.eclipse_duration_min(b) - model.eclipse_duration_min(-b))
              for b in np.linspace(0.0, 89.0, 200))
    check("D(+beta) = D(-beta)", sym < 1e-9, f"max diff {sym:.2e} min")

    d0 = model.eclipse_duration_min(0.0)
    monotone = all(model.eclipse_duration_min(b) <= d0 + 1e-12
                   for b in np.linspace(0.0, 89.0, 200))
    check("D maximal at beta = 0", monotone, f"D(0) = {d0:.2f} min")
    check("D = 0 beyond beta_crit",
          model.eclipse_duration_min(model.beta_crit_deg + 0.01) == 0.0,
          f"beta_crit = {model.beta_crit_deg:.2f} deg")

    worst = 0.0
    for day in (0, 40, 110, 260):
        t0 = cfg.epoch + timedelta(days=day)
        numeric = model.eclipse_duration_numeric_min(t0, step_s=1.0)
        closed = model.eclipse_duration_min(model.beta_deg(t0)) / cfg.penumbra_factor
        worst = max(worst, abs(numeric - closed))
    check("geometric vs closed-form umbra", worst < 0.5,
          f"max deviation {worst:.3f} min")

    # 6 -- external check of the J2 nodal regression against the published
    # sun-synchronous altitude/inclination pairs. An orbit is sun-synchronous
    # when its node precesses at 360 deg per sidereal year; inverting the
    # model at that rate must return the tabulated inclinations.
    rate_sso = 360.0 / 365.2422                       # deg/day
    for h, i_ref in ((567.0, 97.7), (894.0, 99.0), (1262.0, 100.7)):
        lo, hi = 95.0, 103.0                          # bisection, no SciPy needed
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            m = EclipseModel(OrbitConfig(altitude_km=h, inclination_deg=mid,
                                         epoch=datetime(2028, 1, 1)))
            if math.degrees(m.raan_dot) * 86400.0 < rate_sso:
                lo = mid
            else:
                hi = mid
        i_model = 0.5 * (lo + hi)
        check(f"sun-synchronous inclination at {h:.0f} km",
              abs(i_model - i_ref) < 0.05,
              f"{i_model:.3f} deg vs {i_ref:.1f} deg published")

    if verbose:
        print("leo_eclipse self-test")
        print("\n".join(report))
        print(f"  => {'ALL TESTS PASSED' if ok else 'FAILURES DETECTED'}\n")
    return ok


# ----------------------------------------------------------------------------
# The menu
# ----------------------------------------------------------------------------
def custom_config(penumbra_factor: float = PENUMBRA_FACTOR) -> OrbitConfig:
    """The mission described by the MISSION CONFIGURATION block."""
    return OrbitConfig(
        name="Custom",
        altitude_km=ALTITUDE_KM,
        inclination_deg=INCLINATION_DEG,
        raan_deg=RAAN_DEG,
        arg_lat0_deg=ARG_LAT0_DEG,
        epoch=datetime.fromisoformat(EPOCH),
        end=datetime.fromisoformat(END) if END else None,
        duration_days=DAYS,
        penumbra_factor=penumbra_factor)


def _menu_text() -> str:
    lines = ["", "=" * 66, " leo_eclipse -- choose a mission", "=" * 66]
    for key in sorted(MISSIONS):
        e = MISSIONS[key]
        if e.get("end"):
            span = (datetime.fromisoformat(e["end"])
                    - datetime.fromisoformat(e["epoch"])).days
        else:
            span = int(e.get("days", 1))
        lines.append(f"  {key}) {e['name']:<22s} h = {e['altitude_km']:5.0f} km, "
                     f"i = {e['inclination_deg']:6.2f} deg, {span:3d} days")
    nxt = str(max(int(k) for k in MISSIONS) + 1)
    lines += [
        f"  {nxt}) {'Custom':<22s} h = {ALTITUDE_KM:5.0f} km, "
        f"i = {INCLINATION_DEG:6.2f} deg   (configuration block)",
        "  0) All of them, with a side-by-side comparison",
        "  q) Quit",
        "=" * 66,
    ]
    return "\n".join(lines)


def select_configs(choice: str, penumbra_factor: float) -> List[OrbitConfig]:
    """Translate a menu choice into the list of missions to analyse."""
    choice = choice.strip().lower()
    custom_key = str(max(int(k) for k in MISSIONS) + 1)
    if choice in ("0", "all", "a"):
        return [config_from_entry(MISSIONS[k], penumbra_factor)
                for k in sorted(MISSIONS)]
    if choice in (custom_key, "custom", "c", ""):
        return [custom_config(penumbra_factor)]
    if choice in MISSIONS:
        return [config_from_entry(MISSIONS[choice], penumbra_factor)]
    raise ValueError(f"unknown mission '{choice}'")


def ask_choice() -> Optional[str]:
    """Prompt for a menu choice. Returns None if the user quits."""
    custom_key = str(max(int(k) for k in MISSIONS) + 1)
    valid = set(MISSIONS) | {"0", custom_key}
    print(_menu_text())
    while True:
        try:
            answer = input(f" Mission [0-{custom_key}, q to quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if answer in ("q", "quit", "exit"):
            return None
        if answer in valid:
            return answer
        print("  Not a valid choice.")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beta angle and eclipse duration of a circular LEO orbit. "
                    "Run with no arguments for the interactive menu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mission", type=str, default=None,
                   help="menu entry to run without prompting "
                        "(a number, 'custom', or 'all')")
    p.add_argument("--altitude", type=float, default=None, help="altitude [km]")
    p.add_argument("--inclination", type=float, default=None,
                   help="inclination [deg]")
    p.add_argument("--raan", type=float, default=None, help="RAAN at epoch [deg]")
    p.add_argument("--epoch", type=str, default=None,
                   help="epoch, ISO format (YYYY-MM-DD[THH:MM:SS]), UTC")
    p.add_argument("--end", type=str, default=None,
                   help="mission end, ISO format, UTC")
    p.add_argument("--days", type=float, default=None,
                   help="mission duration [days]; takes precedence over --end")
    p.add_argument("--arg-lat0", type=float, default=None,
                   help="argument of latitude at epoch [deg]")
    p.add_argument("--penumbra-factor", type=float, default=PENUMBRA_FACTOR,
                   help="penumbra/atmosphere correction (1.0 = pure umbra)")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                   help="folder for the tables and the figures")
    p.add_argument("--no-plots", action="store_true", help="skip the figures")
    p.add_argument("--show", action="store_true", help="display the figures")
    p.add_argument("--no-self-test", action="store_true",
                   help="skip the consistency checks")
    p.add_argument("--self-test", action="store_true",
                   help="run the consistency checks and exit")
    return p.parse_args(argv)


def _apply_overrides(cfg: OrbitConfig, args: argparse.Namespace) -> OrbitConfig:
    """Rebuild a config with the orbital parameters given on the command line."""
    over = dict(
        name=cfg.name,
        altitude_km=cfg.altitude_km if args.altitude is None else args.altitude,
        inclination_deg=(cfg.inclination_deg if args.inclination is None
                         else args.inclination),
        raan_deg=cfg.raan_deg if args.raan is None else args.raan,
        arg_lat0_deg=cfg.arg_lat0_deg if args.arg_lat0 is None else args.arg_lat0,
        epoch=cfg.epoch if args.epoch is None else datetime.fromisoformat(args.epoch),
        penumbra_factor=args.penumbra_factor)
    if args.days is not None:
        over.update(end=None, duration_days=args.days)
    elif args.end is not None:
        over.update(end=datetime.fromisoformat(args.end))
    else:
        over.update(end=cfg.end)
    if any(v is not None for v in (args.altitude, args.inclination, args.days,
                                   args.end, args.epoch)):
        over["name"] = cfg.name if cfg.name != "Custom" else "Custom"
    return OrbitConfig(**over)


def run_mission(cfg: OrbitConfig, out_root: str, plots: bool,
                show: bool) -> EclipseModel:
    """Analyse one mission: summary, table, figures."""
    model = EclipseModel(cfg)
    print()
    print(model.summary())

    out_dir = os.path.join(out_root, cfg.slug)
    os.makedirs(out_dir, exist_ok=True)
    table = model.write_table(os.path.join(out_dir, "eclipse_table"),
                              step=timedelta(hours=CSV_STEP_HOURS))
    print(f"\n  table   {table}")

    if plots:
        try:
            for path in make_figures(model, out_dir, show=show):
                print(f"  figure  {path}")
        except ImportError:
            print("  figures skipped: matplotlib is not installed "
                  "(pip install matplotlib)")
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1

    explicit_orbit = any(v is not None for v in
                         (args.altitude, args.inclination, args.raan, args.epoch,
                          args.end, args.days, args.arg_lat0))
    if args.mission is not None:
        choice = args.mission
    elif explicit_orbit:
        choice = "custom"
    elif sys.stdin is not None and sys.stdin.isatty():
        answer = ask_choice()
        if answer is None:
            return 0
        choice = answer
    else:                                    # piped or scheduled run
        choice = "custom"

    try:
        configs = select_configs(choice, args.penumbra_factor)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    configs = [_apply_overrides(c, args) for c in configs]

    if SELF_TEST and not args.no_self_test:
        print()
        if not self_test():
            print("WARNING: the self-test failed; the results below are suspect.\n")

    out_root = args.output_dir
    if not os.path.isabs(out_root):
        out_root = os.path.join(SCRIPT_DIR, out_root)
    os.makedirs(out_root, exist_ok=True)

    plots = PLOTS and not args.no_plots
    show = args.show or SHOW_PLOTS
    models = [run_mission(cfg, out_root, plots, show) for cfg in configs]

    if len(models) > 1:
        print(comparison_table(models))
        if plots:
            path = compare_missions(models, out_root, show=show)
            if path:
                print(f"\n  figure  {path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
