#!/usr/bin/env python3
"""Render a scene to a WAV file without the GUI.

Useful for checking what a parameter does when you are not at a speaker, and for
listening on headphones later:

    python tools/render.py --preset parked_car_exterior --out car.wav
    python tools/render.py --preset far_field_park --distance 600 --rh 20 \\
                           --material Orchestra.wav --out park.wav
    python tools/render.py --list

The scene's own impulse response is written next to the rendered audio as
``<out>_ir.wav`` so it can be loaded into any other tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soundsim.engine import presets, renderer                    # noqa: E402
from soundsim.engine.player import demo_material, load_file     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="party_next_door",
                    help="scene name (see --list)")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--out", default="soundsim_render.wav")
    ap.add_argument("--material", default=None,
                    help="audio file to play through the scene (default: built-in demo)")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="seconds of the built-in demo material")
    ap.add_argument("--distance", type=float, default=None,
                    help="override the listener distance (m)")
    ap.add_argument("--level", type=float, default=None,
                    help="override the source level (dB SPL at 1 m)")
    ap.add_argument("--temp", type=float, default=None, help="air temperature (C)")
    ap.add_argument("--rh", type=float, default=None, help="relative humidity (%%)")
    ap.add_argument("--order", type=int, default=None, help="image-source order")
    ap.add_argument("--fs", type=int, default=48_000)
    ap.add_argument("--stereo", type=float, default=0.18,
                    help="spaced-pair separation in m (0 = mono)")
    args = ap.parse_args(argv)

    if args.list:
        for key, (title, sub) in presets.PRESETS.items():
            print(f"{key:26s} {title:32s} {sub}")
        return 0
    if args.preset not in presets.PRESETS:
        ap.error(f"unknown preset {args.preset!r}; try --list")

    scene = presets.make(args.preset)
    if args.distance is not None:
        s = np.asarray(scene.source.position, float)
        r = np.asarray(scene.receiver, float)
        d = r - s
        d = d / max(np.linalg.norm(d), 1e-9)
        scene.receiver = tuple(s + d * args.distance)
    if args.level is not None:
        scene.source.spl_at_1m_db = args.level
    if args.temp is not None:
        scene.weather.temp_c = args.temp
    if args.rh is not None:
        scene.weather.rh_pct = args.rh
    if args.order is not None:
        scene.ism_order = args.order

    print(f"scene: {scene.name} -- {scene.description}")
    if args.stereo > 0:
        irs, res = renderer.render_pair(scene, args.fs, separation_m=args.stereo)
    else:
        res = renderer.render(scene, args.fs)
        irs = [res.ir]
    m = res.metrics
    print(f"  paths {m['n_paths']}   response {len(res.ir)/args.fs:.2f} s   "
          f"render {_t(res)}")
    print(f"  free field at the listener {m['free_field_spl_db']:.1f} dB SPL -> "
          f"received {m['total_spl_db']:.1f} dB SPL "
          f"({m['environment_loss_db']:+.1f} dB)")
    print(f"  T60 mid-band {m['t60_mid_db']:.2f} s, C50 {m['c50']:+.1f} dB, "
          f"ITDG {m['itdg_ms']:.1f} ms")

    # the two channels' responses differ by a few samples (different geometry);
    # align them so the convolution below is rectangular
    length = max(len(ir) for ir in irs)
    irs = [np.pad(ir, (0, length - len(ir))) for ir in irs]

    material = (load_file(args.material, args.fs) if args.material
                else demo_material(args.duration, args.fs))
    n = min(len(material.samples), len(irs[0]) + 6 * args.fs)
    total = n + len(irs[0]) - 1            # length of a linear convolution
    out = np.zeros((total, material.samples.shape[1]))
    for ch in range(material.samples.shape[1]):
        ir = irs[min(ch, len(irs) - 1)]
        out[:, ch] = np.convolve(material.samples[:n, ch], ir)
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    sf.write(args.out, out, args.fs, subtype="FLOAT")
    ir_stack = np.stack(irs[:material.samples.shape[1]], axis=1)
    sf.write(str(Path(args.out).with_suffix("")) + "_ir.wav", ir_stack, args.fs,
             subtype="FLOAT")
    print(f"wrote {args.out} ({out.shape[0]/args.fs:.1f} s) and "
          f"{Path(args.out).with_suffix('')}_ir.wav")
    return 0


def _t(res) -> str:
    return f"{res.metrics.get('ir_length_ms', 0)/1000:.2f} s long"


if __name__ == "__main__":
    raise SystemExit(main())
