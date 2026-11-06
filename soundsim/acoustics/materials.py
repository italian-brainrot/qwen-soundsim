"""Material databases: absorption coefficients, panel (structural) properties and
porous-medium flow resistivities.

Three separate physical descriptions are kept apart deliberately, because the
original model mixes them at its peril:

* **absorption** (``MATERIALS``) -- how much energy a surface removes from an
  incident diffuse field; used by Sabine/Eyring and the image-source method.
* **transmission** (``PANELS``) -- density, Young's modulus, Poisson ratio,
  loss factor and thickness; used to *compute* a transmission-loss curve from
  the damped-plate equations (see :mod:`soundsim.acoustics.transmission`).
* **porous grounds** (``GROUNDS``) -- flow resistivity + structure factor for
  the Allard-Johnson equivalent-fluid model of the outdoor ground.

Absorption coefficients are quoted at 1/1-octave centre frequencies
125, 250, 500, 1k, 2k, 4k Hz.  Values are the diffuse-field ( reverberation-room,
ISO 354) figures compiled in Beranek & Mellow, *Engineering Acoustics*, and
Cremer & Muller, *Structural Acoustics*; where a source quotes normal incidence
it is marked in the notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np

from .constants import THIRD_OCTAVE_CENTRES

BAND_CENTRES = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
_BAND_LOG = np.log2(BAND_CENTRES)


@dataclass(frozen=True)
class Absorber:
    """A room surface material: frequency-dependent absorption coefficient."""
    name: str
    alpha: tuple[float, ...]        # 6 values at 125..4k Hz
    notes: str = ""

    def alpha_at(self, freq_hz) -> np.ndarray:
        """Log-log interpolation of alpha onto arbitrary frequencies (clipped)."""
        f = np.asarray(freq_hz, dtype=np.float64)
        lf = np.clip(np.log2(np.maximum(f, 1e-3)), _BAND_LOG[0] - 1, _BAND_LOG[-1] + 1)
        a = np.interp(lf, _BAND_LOG, np.asarray(self.alpha, dtype=np.float64))
        return np.clip(a, 1e-3, 1.0)


# ---------------------------------------------------------------------------
# Reverberation-room (diffuse field) absorption coefficients
# ---------------------------------------------------------------------------
MATERIALS: dict[str, Absorber] = {}


def _add(name, alpha, notes=""):
    MATERIALS[name] = Absorber(name, tuple(float(x) for x in alpha), notes)


_add("concrete_painted", (0.01, 0.01, 0.02, 0.02, 0.02, 0.02), "hard, reflective")
_add("concrete_rough", (0.01, 0.02, 0.03, 0.04, 0.05, 0.07))
_add("brick_painted", (0.03, 0.03, 0.03, 0.04, 0.05, 0.07))
_add("brick_unpainted", (0.03, 0.03, 0.04, 0.05, 0.07, 0.14))
_add("plaster_rough", (0.10, 0.08, 0.07, 0.05, 0.04, 0.03), "moulded, ornamental")
_add("plaster_smooth", (0.01, 0.01, 0.02, 0.02, 0.02, 0.02))
_add("gypsum_board_painted", (0.29, 0.10, 0.05, 0.04, 0.07, 0.09), "on battens, cavity")
_add("glass_3mm", (0.18, 0.06, 0.04, 0.03, 0.02, 0.02), "large pane, mounted at 90 deg")
_add("glass_thick_large", (0.35, 0.25, 0.18, 0.07, 0.04, 0.03))
_add("marble_tile", (0.01, 0.01, 0.01, 0.02, 0.02, 0.02))
_add("ceramic_tile", (0.01, 0.01, 0.01, 0.02, 0.02, 0.02))
_add("wood_floor_on_joists", (0.28, 0.22, 0.17, 0.09, 0.09, 0.10), "empty cavity below")
_add("wood_parquet_concrete", (0.04, 0.04, 0.06, 0.06, 0.06, 0.06))
_add("plywood_12mm_airgap_100", (0.40, 0.25, 0.15, 0.10, 0.09, 0.09))
_add("carpet_thick_pad", (0.02, 0.06, 0.14, 0.37, 0.60, 0.65), "13 mm pile on foam pad")
_add("carpet_thin", (0.02, 0.03, 0.06, 0.14, 0.32, 0.37))
_add("vinyl_floor", (0.02, 0.02, 0.03, 0.03, 0.04, 0.05))
_add("curtain_heavy", (0.07, 0.31, 0.49, 0.75, 0.70, 0.60), "300 g/m2, 200 mm from wall")
_add("curtain_light", (0.07, 0.31, 0.49, 0.75, 0.70, 0.60), "gathered 1/10")
_add("upholstered_chair", (0.03, 0.14, 0.21, 0.26, 0.26, 0.21), "per chair, ~0.2 m2 proj.")
_add("seat_auditorium_upholstered", (0.02, 0.08, 0.17, 0.27, 0.25, 0.20), "unoccupied")
_add("seat_auditorium_empty", (0.01, 0.02, 0.03, 0.05, 0.05, 0.05))
_add("audience_crowded", (0.06, 0.21, 0.55, 0.74, 0.88, 0.88), "per m2 of floor")
_add("person_sitting", (0.03, 0.08, 0.13, 0.19, 0.28, 0.33), "per person (0.5 m2 proj.)")
_add("mineral_wool_50mm", (0.15, 0.55, 0.90, 0.93, 0.93, 0.93), "50 mm, 40 kg/m3, 13 mm facing")
_add("mineral_wool_25mm", (0.08, 0.24, 0.50, 0.63, 0.66, 0.66), "25 mm board")
_add("acoustic_ceiling_tile", (0.17, 0.40, 0.65, 0.80, 0.90, 0.90), "perforated mineral board")
_add("open_window", (0.10, 0.30, 0.50, 0.70, 0.80, 0.90), "opening into an unlit space")
_add("foam_wedge", (0.10, 0.40, 0.85, 0.95, 0.98, 0.99), "50 mm anechoic wedge")
_add("plywood_perforated", (0.12, 0.22, 0.30, 0.35, 0.40, 0.40), "perforated absorber panel")
_add("car_interior", (0.15, 0.20, 0.30, 0.40, 0.45, 0.45), "trim + seats + glass, averaged")
_add("fabric_padded_wall", (0.10, 0.30, 0.45, 0.55, 0.55, 0.50))
_add("snow", (0.10, 0.25, 0.45, 0.60, 0.60, 0.60), "200 mm fresh")
_add("grass_short", (0.05, 0.08, 0.12, 0.20, 0.25, 0.30), "short grass over hard ground")
_add("gravel", (0.05, 0.07, 0.10, 0.15, 0.18, 0.20))
_add("asphalt", (0.01, 0.01, 0.02, 0.02, 0.03, 0.03))
_add("water_calm", (0.00, 0.00, 0.01, 0.01, 0.01, 0.01))
_add("soil_forest", (0.10, 0.20, 0.35, 0.50, 0.60, 0.60), "forest floor, leaf litter")


@dataclass(frozen=True)
class Panel:
    """A limp/damped plate used for transmission loss.

    thickness_m : plate thickness h [m]
    density     : material density rho_s [kg/m^3]
    young_pa    : Young's modulus E [Pa]
    poisson     : Poisson ratio nu [-]
    loss_factor : structural (hysteretic) damping eta [-]; governs how deep the
                  coincidence dip is.  Damped gypsum eta ~ 0.05-0.10, bare metal
                  panels eta ~ 0.001-0.01, wood ~ 0.01.
    """
    name: str
    thickness_m: float
    density: float
    young_pa: float
    poisson: float = 0.25
    loss_factor: float = 0.01
    notes: str = ""

    @property
    def surface_density(self) -> float:
        return self.thickness_m * self.density

    @property
    def bending_stiffness(self) -> float:
        """Plate rigidity B = E h^3 / (12 (1 - nu^2)) [N m]."""
        return self.young_pa * self.thickness_m**3 / (12.0 * (1.0 - self.poisson**2))

    @property
    def damping(self) -> float:
        return self.loss_factor


PANELS: dict[str, Panel] = {}


def _addp(name, h, rho, e, nu=0.25, eta=0.01, notes=""):
    PANELS[name] = Panel(name, h, rho, e, nu, eta, notes)


_addp("gypsum_board_12mm", 0.0127, 800.0, 2.9e9, 0.25, 0.05,
      "1/2 in faced gypsum; f_c ~ 2.4 kHz")
_addp("gypsum_board_10mm", 0.0095, 800.0, 2.9e9, 0.25, 0.05)
_addp("plywood_12mm", 0.012, 550.0, 7.0e9, 0.20, 0.01)
_addp("osb_12mm", 0.012, 650.0, 3.5e9, 0.20, 0.01)
_addp("mdf_18mm", 0.018, 750.0, 3.3e9, 0.25, 0.02)
_addp("glass_4mm", 0.004, 2500.0, 7.25e10, 0.22, 0.001, "f_c ~ 11 kHz (above band)")
_addp("glass_6mm", 0.006, 2500.0, 7.25e10, 0.22, 0.001)
_addp("glass_10mm", 0.010, 2500.0, 7.25e10, 0.22, 0.001)
_addp("car_window_glass", 0.0035, 2500.0, 7.25e10, 0.22, 0.002,
      "automotive tempered glass, ~8 kg/m2")
_addp("steel_1mm", 0.001, 7850.0, 2.05e11, 0.29, 0.002, "thin sheet, high f_c")
_addp("steel_3mm", 0.003, 7850.0, 2.05e11, 0.29, 0.002)
_addp("aluminium_3mm", 0.003, 2700.0, 7.0e10, 0.33, 0.003)
_addp("car_body_panel", 0.0008, 7850.0, 2.05e11, 0.29, 0.02,
      "0.8 mm sheet with paint/damping layer")
_addp("concrete_100mm", 0.10, 2300.0, 2.1e10, 0.20, 0.01, "dense, f_c well above band")
_addp("concrete_200mm", 0.20, 2300.0, 2.1e10, 0.20, 0.01)
_addp("brick_100mm", 0.10, 1800.0, 1.0e10, 0.25, 0.015, "fired clay masonry")
_addp("brick_200mm", 0.20, 1800.0, 1.0e10, 0.25, 0.015)
_addp("aac_block_150mm", 0.15, 600.0, 2.2e9, 0.20, 0.03, "autoclaved aerated concrete")
_addp("timber_stud_wall", 0.013, 500.0, 9.0e9, 0.25, 0.02,
      "gypsum+stud idealised as one leaf, m' ~ 20 kg/m2 effective")
_addp("plywood_carpet_15mm", 0.015, 500.0, 5.0e9, 0.25, 0.05, "car floor panel")
_addp("lead_sheet_2mm", 0.002, 11340.0, 1.6e10, 0.45, 0.02)


@dataclass(frozen=True)
class Ground:
    """Porous outdoor ground, described by an equivalent fluid (Allard-Johnson).

    flow_resistivity : sigma [Pa s / m^2]  -- the dominant parameter
    structure_factor : multiplier on the high-frequency tortuosity alpha_inf
                       (>= 1); 1.0 for ordinary open-pored media.
    porosity         : open porosity phi [-]; 0.96 for fibrous/soil media,
                       lower for dense granular materials.
    """
    name: str
    flow_resistivity: float
    structure_factor: float = 1.0
    porosity: float = 0.96
    notes: str = ""


GROUNDS: dict[str, Ground] = {}


def _addg(name, sigma, kprime=1.0, notes=""):
    GROUNDS[name] = Ground(name, float(sigma), float(kprime), 0.96, notes)


_addg("asphalt_sealed", 1e9, 1.0, "effectively rigid/reflective")
_addg("concrete", 1e10, 1.0, "rigid")
_addg("grass_short", 5e6, 1.0, "50 mm grass over soil: behaves as a resistive layer")
_addg("grass_long", 2e5, 1.0, "meadow, deep root mat -- strongly absorbing")
_addg("soil_tilled", 1e5, 1.0)
_addg("forest_floor", 3e4, 1.0, "humus + leaf litter, very absorbing")
_addg("snow_fresh", 3e4, 1.0, "200 mm fresh snow")
_addg("gravel", 2e7, 1.0)
_addg("sand_dry", 1e5, 1.0)
_addg("water_still", 1e11, 1.0, "rigid pressure-release-free reflector")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def absorber(name: str) -> Absorber:
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown absorber {name!r}; options: "
                       f"{', '.join(sorted(MATERIALS))}") from None


def panel(name: str) -> Panel:
    try:
        return PANELS[name]
    except KeyError:
        raise KeyError(f"unknown panel {name!r}; options: "
                       f"{', '.join(sorted(PANELS))}") from None


def ground(name: str) -> Ground:
    try:
        return GROUNDS[name]
    except KeyError:
        raise KeyError(f"unknown ground {name!r}; options: "
                       f"{', '.join(sorted(GROUNDS))}") from None


def interpolate_bands(values, freqs_hz, band_centres=BAND_CENTRES):
    """Log-log interpolation of a band table onto arbitrary frequencies."""
    values = np.asarray(values, dtype=np.float64)
    lf = np.log2(np.asarray(freqs_hz, dtype=np.float64))
    return np.interp(lf, np.log2(band_centres), values)


def third_octave(values, band_centres=BAND_CENTRES):
    """Resample a 1/1-octave table onto the 1/3-octave centre frequencies."""
    return interpolate_bands(values, THIRD_OCTAVE_CENTRES, band_centres)
