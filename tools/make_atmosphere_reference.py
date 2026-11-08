"""Regenerate tests/data/atmosphere_alpha_reference.csv.

The fixture is the cross-check behind
``tests/test_atmosphere_reference.py``: the atmospheric absorption of
``soundsim.acoustics.atmosphere`` compared, over a grid of temperature, humidity
and frequency, against a *different* published implementation of the same
standard (ISO 9613-1:1993 Annex A).

That reference lives in the third-party ``acoustic-toolbox`` package, which is
deliberately **not** a dependency of this repo -- a check that silently skips
because a package is missing is how the model's headline verification stopped
running (AUDIT.md pass 1, finding F6).  So the reference is vendored here as
plain numbers, and this script is only needed when the fixture is regenerated.

Usage (needs the reference package installed somewhere throwaway)::

    pip download --no-deps -d /tmp/at acoustic-toolbox && unzip -q -d /tmp/at /tmp/at/*.whl
    REF_MODULE=/tmp/at/acoustic_toolbox/standards/iso_9613_1_1993.py python tools/make_atmosphere_reference.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from soundsim.acoustics import atmosphere          # noqa: E402

REF_MODULE = os.environ.get(
    "REF_MODULE", "/tmp/at/acoustic_toolbox/standards/iso_9613_1_1993.py")
OUT = os.path.join(os.path.dirname(HERE), "tests", "data",
                   "atmosphere_alpha_reference.csv")

TEMPS_C = [-20, -10, 0, 10, 20, 30, 40, 50]
RH_PCT = [10, 30, 50, 70, 90]
FREQS_HZ = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def load_reference(path: str):
    """Import one module by file path: its package __init__ pulls in matplotlib."""
    spec = importlib.util.spec_from_file_location("ref_iso_9613_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def header() -> str:
    return f"""\
# Reference table for atmospheric sound absorption, alpha in dB per metre.
#
# Generated from an independent published implementation of the same standard:
#   acoustic-toolbox 0.2.2, module acoustic_toolbox/standards/iso_9613_1_1993.py
#   (BSD 3-clause, (c) Acoustic Toolbox / Valentin Le Bescond & Andrew Mitchell;
#    a fork of Alexandre Schimmel's "acoustics"), at
#    https://pypi.org/project/acoustic-toolbox/
# Conditions: p_a = 101.325 kPa, reference 101.325 kPa / 293.15 K.
#
# That implementation writes ISO 9613-1:1993 Annex A in the Arrhenius form
#     alpha = 8.686 f^2 [ 1.84e-11 (T/T0)^1/2 / p_r
#                        + (T/T0)^-5/2 ( 0.01275 exp(-2239.1/T)/(f_rO (1+(f/f_rO)^2))
#                                      + 0.1068 exp(-3352/T)   /(f_rN (1+(f/f_rN)^2)) ) ]
# i.e. line-shape denominators and vibrational-temperature exponentials, while
# soundsim/acoustics/atmosphere.py writes the same physics as a peak line-shape
# amplitude (alpha*lambda)_max per relaxation process times 2x/(1+x^2), with the
# classical term separate and an explicit division by the speed of sound.  The two
# share no constants beyond those in the standard, so agreeing on this table is a
# real cross-check -- not a copy of one formula by two people.
#
# Regenerate with:  {os.path.relpath(__file__, os.path.dirname(HERE))}
# temp_c,rh_pct,freq_hz,alpha_db_per_m
"""


def main() -> int:
    ref = load_reference(REF_MODULE)
    f = np.asarray(FREQS_HZ)
    worst = (0.0, None)
    rows = []
    for tc in TEMPS_C:
        for rh in RH_PCT:
            tk, p = tc + 273.15, 101.325
            h = ref.molar_concentration_water_vapour(rh, ref.saturation_pressure(tk), p)
            a_ref = ref.attenuation_coefficient(
                pressure=p, temperature=tk,
                reference_pressure=101.325, reference_temperature=293.15,
                relaxation_frequency_nitrogen=ref.relaxation_frequency_nitrogen(p, tk, h),
                relaxation_frequency_oxygen=ref.relaxation_frequency_oxygen(p, h),
                frequency=f)
            a_mine = atmosphere.attenuation_coefficient(f, tc, rh)
            dev = np.abs(a_mine - a_ref) / a_ref
            if dev.max() > worst[0]:
                worst = (float(dev.max()), (tc, rh, float(f[int(np.argmax(dev))])))
            for fq, ar in zip(f, a_ref):
                rows.append((tc, rh, fq, ar))

    with open(OUT, "w") as fh:
        fh.write(header())
        for tc, rh, fq, ar in rows:
            fh.write(f"{tc},{rh},{fq:.0f},{ar:.8e}\n")
    print(f"{len(rows)} values -> {OUT}")
    print(f"soundsim deviates from the reference by at most {worst[0] * 100:.2f} % "
          f"at T={worst[1][0]} C, RH={worst[1][1]} %, f={worst[1][2]:.0f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
