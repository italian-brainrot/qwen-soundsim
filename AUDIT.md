# Audit log

An adversarial tester↔fixer loop over this repo. See the `audit-loop` skill:
each pass is one session, either tester or fixer, and the pass is recorded
here. The test gate is `pytest -q` run from the repo root (that puts the repo on
`sys.path`; `soundsim` is not installed).

```
python -m pytest -q                        # the gate: 151 passed, 1 skipped
python -m pytest -q tests/test_audit_pass1.py tests/test_audit_pass2.py
```

As of Pass 2: 152 collected -- 85 the repo's own (24 engine, 30 physics, 15
playlist, 16 UI offscreen), 25 audit regressions, 42 vendored atmosphere
cross-checks. One skips: it waits for a package that is not a dependency, see F6.

---

## Pass 1 — TESTER (2025-11-06)

Scope requested: **simulation accuracy and quality**. Every finding below was
reproduced numerically before being written down; each one that is fixable has a
red test in `tests/test_audit_pass1.py` that states the requirement, not the
implementation. Numbers were produced with
`python -m pytest tests/test_audit_pass1.py -q` and with the small probe scripts
described inline.

Severity is about what a listener notices, not about how hard the fix is.

### F1 — A screen never attenuates the ray it is supposed to block. HIGH

`PathBuilder._outdoor_paths` (soundsim/engine/paths.py:219) handles a `Barrier`
by *adding* a diffracted path next to an untouched direct path:

```python
for bar in self.scene.barriers:
    edge = bar.edge_points(src, dst)[0]
    excess = float(np.linalg.norm(edge - src) + np.linalg.norm(dst - edge) - d_dir)
    if excess <= -1e-6:
        continue                      # screen not in the way
```

Two things are wrong:

* **The guard can never fire.** `barrier_path_excess`/this expression is
  `|b−s| + |r−b| − |r−s|`, which is `>= 0` by the triangle inequality for *any*
  point `b` — the docstring of `diffraction.barrier_path_excess` promises "a
  negative value when the barrier does not actually obstruct the direct ray",
  which is mathematically impossible for that formula. So a 0.2 m screen on a
  300 m path whose ray passes 1.8 m above it is treated as an obstacle and
  contributes a path with Maekawa loss `10 log10(3 + 20N) >= 4.77 dB`:

  ```
  barrier height  0.0 m -> total energy   +0.00 dB     (reference)
  barrier height  0.2 m -> total energy   +0.83 dB
  barrier height  1.0 m -> total energy   +1.09 dB
  barrier height  8.0 m -> total energy   +0.36 dB
  ```

  i.e. **a screen makes the scene louder**. `diffraction.is_shadowed()` already
  does the correct geometric test and is used nowhere in the engine (only in
  `tests/test_physics.py:158`).
* **Even in true shadow nothing is removed.** Adding an attenuated copy next to
  the full-strength direct ray leaves the direct ray at full strength: an 8 m
  wall across a 300 m path changes the received energy by +0.3 dB. Maekawa /
  ISO 9613-2 is an *insertion loss on the direct ray*, so the fix is to
  multiply the direct path's band gains by `maekawa_transfer(...)` (or replace
  the direct path with the edge path), not to append a second path.

Tests: `test_a_screen_below_the_line_of_sight_changes_nothing`,
`test_a_screen_that_shadows_the_ray_makes_the_scene_quieter`.

Prototype note (tested then reverted): replacing the guard with
`if not diffr.is_shadowed(src, edge, dst): continue` makes the first test pass
and the second stay red — which is exactly the intended division of labour: the
first is the light-zone bug, the second is the "attenuate, don't append" bug.
Related: `Barrier.top_edges()` returns *the same edge twice* ("the screen's
thickness is modelled by two edges, as in ISO 9613-2" — not true, and
`edge_points` uses `top_edges()[0]` only), and `Barrier.width` is never used, so
a screen of any width behaves as infinitely long. `diffraction.multi_screen_attenuation_db`
is dead code.

### F2 — The turbulence knob throws away ~1 dB and does not smooth the ground comb. HIGH

`_outdoor_paths` (soundsim/engine/paths.py:193-231, the fan is built at :208-229)
models eddy wander by
splitting the ground reflection into 5 sub-paths at delays `±k·σ_m/c`, each with
gain `|R| / n` — dividing **amplitude** by 5 where energy conservation needs
`sqrt(n)`. Measured total received energy (`build_paths`, far_field_park):

```
turbulence 0.00 ->  +0.00 dB     (5 sub-paths are not created)
turbulence 0.10 ->  -0.98 dB
turbulence 0.35 ->  -0.98 dB     <- the Weather default
turbulence 0.50 ->  -0.98 dB     <- far_field_park's default
turbulence 1.00 ->  -0.98 dB
```

So (a) any non-zero turbulence costs a fixed ~1 dB of level that has nothing to
do with the physics, and (b) the knob is a *step*, not a continuum: the level is
identical at 0.1 and 1.0.

Worse, the mechanism does not do what the comment claims ("averages the two-ray
pattern by exactly the coherence factor `exp(-σ_φ²/2)`, ... kills the
interference where the wander exceeds a fraction of a wavelength"). Measured on
the rendered band balance (250 Hz – 4 kHz, grass, 300 m) with only the
`/sqrt(n)` energy correction applied, so the level is right:

```
comb depth:  8.0 dB at turbulence 0.0   ->   14.4 dB at turbulence 1.0
```

i.e. the pattern gets **deeper**, not smoother: five discrete delays inside a
1/3-octave band do not average the interference, they add their own fine comb.
A band-domain synthesis cannot represent "partially coherent" with a handful of
delays; it has to act on the band result. Two workable directions:

* apply the coherence factor to the *cross term* — compute the band's coherent
  two-ray magnitude, then pull it towards the incoherent sum by
  `1 - exp(-σ_φ(f)²/2)` with `σ_φ(f) = 2πf σ_path / c`; or
* keep the coherent sum and smooth the resulting per-band response with a
  kernel whose width grows with `σ_path` (equivalent, cheaper to reason about).

Either way the acceptance criteria are the two tests: total energy must not move
with the knob, and the comb must get shallower as the knob goes up.

Tests: `test_turbulence_blurs_the_ground_comb_without_losing_energy`,
`test_turbulence_reduces_the_depth_of_the_ground_interference`.

Note: these are about the *path-level* phase spread. The existing
`test_turbulence_wavering_moves_the_high_bands_and_leaves_the_bass` covers the
separate Ornstein-Uhlenbeck modulation in `engine/player.py` and must keep
passing.

### F3 — Below the first eigenfrequency the "exact modal solution" is the free field. HIGH

`modal_transfer` documents a Krenk/Rayleigh free-field subtraction:

```
G = G_free + SUM phi_n phi_n [1/(k_n^2 - k^2 + j k sigma_n/c) - 1/k_n^2]
```

The bracket is engineered to vanish as `k -> 0`, so at low frequency the modal
sum tends to `G_free` and the model returns **exactly the free field inside a
closed box** — the walls do nothing. That is the opposite of the physics: below
the first axial mode the wavelength is several times the room, the air moves as
one lump, and the bass is nearly spatially uniform. Measured with the module's
own function (`f_probe = 0.3 · f_11 = 15.1 Hz`, room 3.4×2.8×2.4 m, α≈0.05,
five receiver seats, `f_max = 800`):

```
seat-to-seat spread, modal model : 20.5 dB
seat-to-seat spread, free field  : 19.1 dB     <- the model is *worse* than nothing
```

The identity being used is not the exact one. Algebraically
`1/(k_n²−k²) = 1/k_n² + k²/(k_n²(k_n²−k²))`, so the term that has to be added
back is the box's **static** Green's function `Σ_{n≠0} φ_n φ_s / k_n²` — not
`G_free`. On top of that, the uniform `(0,0,0)` "compliance" mode is excluded by
`modal_frequencies` (`keep = f > 1e-6`); it is the mode that carries the
infrasonic pressure-vessel response of a sealed box, and it is what makes the
sub-first-mode field uniform.

Consequences beyond the spatial answer: with `f_max` modes only up to a few
hundred Hz and the spectrum evaluated to Nyquist, the leftover
`−Σφφ/k_n² − N/k²`-type residual leaves a plateau plus a *rising* tail in the
time domain (masked low-band probe, room 4×3×2.5 m, α = 0.4, 1.365 s record,
energy per 136 ms block, dB):
`-17 -33 -49 -52 -53 -53 -52 -50 -45 -25` — a box with T60 ≈ 0.2 s should be
below −200 dB by block 3. In the current render path this mostly stays below the
path energy (last 10 % of the rendered IR ≈ 0.02 % of the energy), so it is not
yet audible, but any change that raises the modal band's level will surface it.

Test: `test_below_the_first_mode_the_room_is_not_free_field`.

**Feasibility warning, measured.** A quick prototype that added the undamped
uniform mode (`g_0 = -1/(V(k² - j k σ_0/c))`, φ_0 = 1/√V) does make the 15 Hz
spread behave (7.7 dB against free field's 19.1) but it *floods* the band and
breaks `test_engine.py::test_modal_band_puts_cabin_modes_where_theory_says`
(the axial modes at 39/78 Hz have to stay visible). Whatever is done has to
satisfy both: the compliance term must be bounded/damped and the modal peaks
must survive. Also beware `k = 0`: `1/k²` divides by zero and poisons the whole
spectrum with NaN — guard the DC bin.

### F4 — `modal_transfer` returns `i·4πG`, which is not the spectrum of any real signal. MEDIUM

`modal_transfer` ends with `return 1j * 4.0 * np.pi * (g_free + correction)`
(soundsim/acoustics/modal.py:144).
Multiplying a conjugate-symmetric spectrum by `i` makes it anti-Hermitian: half
of it cannot exist in a real signal. `apply_modal_lowband` writes this value
straight into the bins of an inverse *real* FFT, which keeps only the
Hermitian part — so the low band that the modal model is supposed to get right
is synthesised from whatever survives the projection (a delayed impulse becomes
a spike with a pre-echo ringing on the wrong side; the free-field-only probe
puts 39 % of its energy before the travel time, against 1.8 % for the same
delay without the factor).

The `i` comes from a phase slip in the docstring's normalisation argument: a
source whose free-field pressure is 1 Pa at 1 m has `H = e^{-ikr}/r` outright —
no `iωρ₀q` factor is needed once the source is defined by its pressure, and
dropping a `1/i` on the way through and then re-adding `i` produces exactly this
90° slip. The magnitude is unaffected, which is why every existing magnitude
test passes.

Test: `test_modal_transfer_is_the_spectrum_of_a_real_signal` (the DC bin must be
real, or zero; today it is `0+0.4975j`, i.e. 100 % discarded).

Fixing this means picking the DC behaviour deliberately (see F3) instead of
having `irfft` pick it by discarding the imaginary part.

### F5 — Time strength is returned in milliseconds under a name that says centiseconds, and the UI prints it with the unit "dB". MEDIUM

`rooms.time_strength_cents` ends with
`np.sum(t[m]*w)/np.sum(w)*1000.0 - 0.05*1000.0` (soundsim/acoustics/rooms.py:298)
— milliseconds. ISO 3382
defines TS in *centiseconds* after 50 ms, the docstring and the function name say
cents, and `ui/app.py:613` renders it as `_fmt(m["ts"]) + " dB"` (labelled
"Ts strength" at `ui/app.py:1037`), so the readout is wrong twice over: a value
10× too big and an impossible unit. For T60 = 0.50 s the function reports 65.5
where ISO gives 6.5 centiseconds; a typical room never leaves ±30 cs, so the
number on screen is nonsense either way.

Test: `test_time_strength_is_reported_in_centiseconds` (recomputes the ISO
centroid in the test itself, so it pins the unit rather than the implementation).

### F6 — The only independent check of the atmosphere model does not run. MEDIUM

`test_atmosphere_cross_check_independent_implementation` is the check the README
quotes ("≤1.5 % vs an independent implementation over −20…+50 °C, 10–90 % RH"),
but it is `pytest.importorskip("acoustic_toolbox")` and that package is not in
`pyproject.toml`/`requirements.txt`, so the default gate is
`84 passed, 1 skipped` — the skip is easy to miss and the model's headline
verification is silently off on every machine. Air absorption drives every long
outdoor scene, so it deserves a check that always runs: vendor the reference
(ANSI S1.26 / ISO 9613-1 tabulated α table as a fixture, or a small
self-contained second implementation written from a different source) and assert
against it in the millisecond range. No red test for this one — it is a test
asset, and it must not be "fixed" by installing a package that CI will not have.

### F7 — Claims that are not implemented (documentation honesty; LOW unless noted)

* `diffraction` module docstring: "Two edges (a thick barrier) are combined by
  summing the two edge paths" — `Barrier.top_edges()` returns one edge twice and
  the engine uses only `[0]` (see F1).
* `Barrier.width` is accepted and ignored: a 1 m screen shields as much as a
  40 m one.
* `rooms.decay_per_band(ir, fs, band_centres, windows)` accepts `windows`, which
  `renderer.measure` dutifully passes (the synthesis band windows) and the
  function never reads — the per-band T60 plot is therefore filtered by
  Butterworth masks, not by the windows the caller thinks it sent.
* `rooms.spl_from_ir(..., source_power_w)` documents an absolute level from the
  transmitted energy; the argument is unused and the returned dict is a bare
  energy. Dead API: either implement or delete.
* `paths._room_tail(..., drive=True)` documents a different reverberant
  reference for a boundary element; the code path is identical to `drive=False`.
* `paths.py`'s header states `E_tail = E_direct·(1 + 16πr²/A) − Σ|g_early|²`; the
  code uses `e_dir·ratio − e_early` (the direct term is subtracted once from a
  target that never included it, and `e_early` contains it again). It is masked
  by the fact that `ratio ~ 270` in a normal room, and the existing
  `test_energy_balance_holds_for_an_enclosed_room` passes at rel=0.10 — but in a
  dead room or at 1 m the tail is being under-filled by the direct energy. Make
  the code match the header or the header match the code.
* `Source.f3_db` / `PathBuilder.source_response`: docstring says "two cascaded
  2nd-order high-pass sections" and `scene.py` says "4th-order"; the code is one
  2nd-order Butterworth (`w²/sqrt(1+w⁴)`). The code is the physically right one
  for a sealed box — fix the two docstrings.
* `transmission.mass_law_db` docstring promises the field-incidence form
  `20log10(m'f) − 47.4`; the code returns `20log10(m'f) − 42.4 − 8.5 = − 50.9`,
  ~3.4 dB more isolation than the classic rule (and than the integral in
  `tau_field`, which `test_field_incidence_mass_law_offset_matches_theory` checks
  within 10 dB). Unused by the engine, but it is a published-looking number.
* `rooms.modal_decay_rate` (unused by the engine, kept as a diagnostic):
  `hits_per_s = f / mean_path` has units 1/(s·m); it should be `c / mean_path`,
  so the reported per-mode decay is low by a factor λ. Its docstring claims it
  "reduces exactly to Sabine", which it therefore does not.
* `rooms.Schroeder_frequency` / `A_air = 0.9214`: consistent and correct, but
  note `AIR_ABSORPTION_AREA_COEF` is duplicated in `paths.AIR_AREA_COEF`.

### What I checked and found sound (so the fixer need not re-derive it)

ISO 9613-1 air absorption (hand-verified against the ANSI anchors and the
sign-flip of the humidity dependence); the damped-plate τ(θ), the Beranek
diffuse integral, the grazing limit and the normal-incidence mass law; the
Rayleigh-piston radiation impedance and the `τ ≈ 2(ka)²` Rayleigh limit; the
image-source enumeration (mirror distances, reciprocity, `6·5^(n−1)` sequence
count); the mode-shape normalisation and `modal_absorption` weights;
`σ = cαS/4V` ↔ Sabine `T60 = 0.161V/A`; the 4-tap Lagrange fractional delay and
the `f3` high-pass shape; the band window partition of unity; a hand-check that
the modal denominator's damping sign `+jkσ/c` is the causal one under the
`irfft`/`e^{+iωt}` synthesis convention (F4 is about the extra `i` in front, not
about that sign).

### Suggested order of attack for the fixer

1. F1 (screens) — biggest audible wrongness, self-contained, ~20 lines.
2. F2 (turbulence) — affects a shipped default on every outdoor preset.
3. F5 (TS units + UI label) — five minutes, and the readout is currently absurd.
4. F6 (vendored atmosphere reference) — restores a real verification.
5. F3 + F4 (modal low end) — take them together, they are the same DC/static
   problem; keep `test_modal_band_puts_cabin_modes_where_theory_says` and
   `test_room_response_measured_t60_matches_sabine_in_the_diffuse_regime` green.
6. F7 — docstring/API honesty; cheap, do not skip, several of these are exactly
   the kind of claim a reader (or the next pass) will trust.

Environment: nothing left running, no daemons; only
`tests/test_audit_pass1.py` is new in the working tree (`git status`).

---

## Pass 2 — FIXER (2025-11-08)

All seven findings are addressed. The gate is green: **151 passed, 1 skipped**
(152 collected; it was 91 collected with 7 failing at the start of this pass.
`python -m pytest` from the repo root, python 3.14 in `.venv`, Qt offscreen).

The header's "84 passed" counted what HEAD *collects*: HEAD has 84 tests in
`test_engine/test_physics/test_playlist/test_ui`, one of which
(`test_atmosphere_cross_check_independent_implementation`) skips without
`acoustic_toolbox`, so HEAD *passes* 83. Nothing was deleted to get here.

New in the tree, all committed together:

| file | what |
|---|---|
| `tests/test_audit_pass1.py` | pass 1's 7 red tests (untracked when this pass started — committed so the gate is reproducible) |
| `tests/test_audit_pass2.py` | 18 tests written against the physics each fix claims |
| `tests/test_atmosphere_reference.py` | 42 always-running cross-checks |
| `tests/data/atmosphere_alpha_reference.csv` | the vendored reference table (320 values) |
| `tools/make_atmosphere_reference.py` | regenerates it; the reference package stays a non-dependency |
| `tests/test_ui.py` | +1 test: TS printed in centiseconds, D50 in dB |

Source: `acoustics/{diffraction,modal,rooms,transmission}.py`,
`engine/{paths,renderer,rir,scene}.py`, `ui/app.py`, `README.md`.

### F1 — screens (HIGH)

*`scene.Barrier`* now knows what it is: a new `thickness` field, a `_frame()`
that returns the length direction, the plan normal and the two top-edge centres,
`top_edges()` (one edge when `thickness == 0`, two otherwise), `edge_points()`
clamped to the **finite** edge so `width` decides flanking, and
`blocks(source, receiver)` — an exact 2-D test of whether the segment pierces the
rectangle (parametric intersection with the strip `|along| ≤ width/2`, so it
cannot be fooled by a screen off to one side).

*`paths.PathBuilder._screen_insertion_loss()`* is the new decision point, and the
design decision worth arguing about: **a screen is an insertion loss on the rays
it crosses, not an extra path.** The old code appended a "diffracted" `Path` for
every barrier; the synthesiser adds paths in energy in the band balance, so the
obstacle's own shadow stayed in the sum *and* a second arrival was added on top —
which is why a screen made the scene louder (F1's measurement). Geometry now
decides first (`blocks`); for every screen that really shadows the ray, Maekawa
gives the attenuation of the *surviving* direct and ground rays, applied as a
gain multiply plus an excess delay in `delay_bands`, so the shadow still carries
the comb's phase. Multiple screens are cascaded with the ISO 9613-2 rule.

*`acoustics/diffraction.py`*: `edge_group_attenuation_db()` combines the two
edges of one flat top **in energy with an average, not a sum** — summing them
would hand out up to 3 dB of free attenuation, and the test
`test_two_edges_of_one_top_do_not_double_the_benefit` guards it.
`series_attenuation_db()` implements `ΣA_i − 10(n−1)log10 3`;
`multi_screen_attenuation_db` (previously dead) delegates to it. Docstrings now
say what the function cannot do: `barrier_path_excess` is two sides of a triangle
minus the third, so it is *never* negative, and `N = 0` is a clip meaning "the
first Fresnel zone is grazed, not cleared" — the light/shadow decision belongs to
the geometry, not to the Fresnel number.

### F2 — turbulence (HIGH)

Model: the wander of eddies moves the *relative phase* of the direct and ground
rays. Their cross term is multiplied by the mutual coherence

    γ(f) = exp(−σ_φ²/2),   σ_φ = 2πf·σ/c

the coherent replica carries `g·γ`, and the energy the interference releases,
`(1−γ²)|g|²`, is carried by a companion `Path(coherent=False)` at the *same*
delay, synthesised by the new `rir.decorrelated_burst()` and summed in energy.
Band energy is therefore conserved exactly — `γ² + (1−γ²) = 1` — which is what
`test_the_turbulence_split_neither_creates_nor_destroys_energy` measures against
the turbulence-free ray. `scatter_paths` skips incoherent paths (a replica fan of
an arrival that is itself a random phasor sum would be a copy of the thing it is
supposed to decorrelate) and now raises a clear error on a complex gain instead of
failing inside numpy.

**Calibration, stated so it can be attacked.** `σ = 5.5e-5 × path length ×
turbulence` (1.6 cm over the 300 m park at turbulence 1.0, 0.6 cm at the 0.35
default) is a *calibration of the residual differential path-length wander* of
two rays that run almost side by side, not a value derived from a turbulence
spectrum. Pass 1's test pins both the waver (comb ptp must fall by > 2 dB) and
the level (mean over 250 Hz–4 kHz within 1.5 dB), and that pair brackets σ at
roughly 1.4–1.8 cm at 300 m; 5.5e-5/path is the middle of that bracket. It is not
an accident that the mean band level *rises* with turbulence (−7.99 → −6.68 dB at
the park): filling a 10 dB ground null puts energy back into the band, which is
physics, not a leak — the old implementation got a flat mean only by throwing
energy away. Both facts are in the code comment and the README.

### F3, F4 — the modal low band (HIGH, MEDIUM)

`modal_transfer` was missing the mode at `k = 0`. It is now added explicitly:

    (1/V) / (−k² + j k σ_0/c),    σ_0 = c·ᾱ·S / 4V

`φ_0 = 1/√V` everywhere, so this term is spatially **uniform** and grows towards
DC: it is the compliance of the air, the reason a sealed room is not open air, and
without it the truncated Krenk sum degenerates to exactly `G_free`. It decays at
the Sabine rate because a uniform pressure loads all six walls equally. The
docstring states what the truncated sum still omits — the box's *static image
field* `Σ_{n≠0}φφ/k_n²` minus the `k→0` limit of `G_free`, a smooth term a
lattice sum would give — and `test_the_compliance_of_the_air_is_the_closed_form`
measures the residual: within 15 % of the closed form at 4–12 Hz, the response of
a sealed box **40 dB** above what `1/r` would give at 6 Hz (the compliance is
`4π/Vk²` = 78 there against `1/r` = 0.83), rising towards DC at the spring's
12 dB per octave — the test bounds the measured rise at 5–14 dB/octave, since the
omitted static term subtracts a roughly constant offset from the `1/k²` law.

F4: the `4πG` is no longer multiplied by `1j` (a spectrum that is not
Hermitian is not the spectrum of a real signal, and `irfft` was throwing the
imaginary part away). The DC bin is returned as **0, deliberately**: bin 0 of a
spectrum is the mean of the response, a room's pressure record has no mean, and
the rigid-box answer at `k = 0` is not finite.

The renderer needed three coupled changes, all of them consequences of the model
being right at last:

1. **The trim is measured where the two models overlap**, weighting the energy
   ratio by `x_lo·x_hi` (which vanishes outside the crossfade) instead of over
   the whole modal band. Measured effect on `inside_car`: the trim is now
   **+0.75 dB** — the `4πG` normalisation and the ray model already agree at the
   crossover to within 1 dB — where the old whole-band energy match put the modal
   band about **6 dB too hot** there. That 6 dB is what had been propping up
   `test_modal_band_puts_cabin_modes_where_theory_says`.
2. **The modal band owns every bin below the crossover, down to DC.** Leaving the
   bottom of the spectrum to the ray model left a sub-audible pedestal (the 25 Hz
   band's burst smears to DC) which the subsonic filter then rang *back up* into
   the audible band.
3. **The source's own roll-off is applied to the modal band**
   (`paths.source_roll_off`, factored out of `PathBuilder.source_response`). The
   box's compliance rises at 12 dB/octave towards DC; a sealed-box loudspeaker
   falls at exactly 12 dB/octave below its `f3`; the product is flat, which is the
   textbook behaviour and the only thing that keeps the render from running away.
   `Scene.f3_db`'s comment claimed 4th order; the code has always been 2nd
   (12 dB/oct), and says so now.

`remove_subsonic` runs *after* the splice (it used to run before, so the splice
wrote back what it had just filtered) and is now applied **as a magnitude in the
frequency domain**. It was `sosfiltfilt` of a 2nd-order Butterworth at 15 Hz,
whose default edge padding is a handful of samples — far too short for that
section's transient to decay — so it rang a burst at the corner into the response:
on `inside_car` that transient put **9 dB more** energy in the 15–20 Hz band than
the response had before filtering. A filter meant to take sub-bass out must not
put any in.

### F5 — time strength (MEDIUM)

`rooms.time_strength_cents` returns centiseconds (`(centroid − 0.050) × 100`),
the UI prints `27.0 cs` next to D50's `dB`, and the row is labelled
"Time strength TS". `rooms.definition()`'s docstring no longer calls D50 a
percentage. `test_time_strength_is_a_centroid_of_the_decay_curve_in_centiseconds`
re-derives the number from the definition rather than restating the code.

**Not changed, deliberately:** the integration window is `(-10, -25) dB` of the
backward-integrated decay curve, whereas ISO 3382-1 truncates at -5/-35. F5 is
about a unit, and moving the window would move every quoted TS figure in the repo
without a finding to justify it. Flagged for a later pass.

### F6 — the atmosphere check now runs (MEDIUM)

`tests/test_atmosphere_reference.py` compares the model against a **vendored
table** — 320 values over −20…+50 °C × 10/30/50/70/90 % RH × 63 Hz…8 kHz — so it
runs on every machine. The table came from an independent published implementation
of the same standard (`acoustic-toolbox` 0.2.2, `iso_9613_1_1993.py`, BSD-3,
attributed in the CSV header), which writes ISO 9613-1 Annex A in the Arrhenius
line-shape form while `atmosphere.py` writes it as a peak line-shape amplitude
per relaxation process: no shared constants beyond the standard's own.
**Measured worst deviation 1.52 %** (50 °C, 30 % RH, 63 Hz); the test asserts
below 3 % per point and below 2 % on the worst case. The old
`importorskip`-based test is left in place — it is harmless, and deleting
someone else's test is not this pass's call — but the always-running one is now
the load-bearing check. `tools/make_atmosphere_reference.py` regenerates the
table and does not become a dependency.

### F7 — claims that were not implemented

* **`paths.py`'s energy bookkeeping was the code's error, not the header's
  lie.** `e_tail = e_dir·(1 + 16πr²/A) − e_early`; the old line charged the
  direct ray twice (target without it, early sum with it). On the energy-balance
  scene the ratio went from **0.824 to 0.936** of the target (tolerance ±10 %).
  Where it really bit was close range in a dry room: `across_the_room` at 2 m,
  500 Hz, the tail held **0.0017** where the balance asks for **0.343** of the
  direct ray's energy — the reverberant field was simply missing near the
  listener, and it is why the C50 test saw a tidy −12 dB (see "tests changed").
* `rooms.decay_per_band`: the accepted-and-ignored `windows` argument is gone,
  and the docstring says why the synthesis band bank (a partition of unity —
  exactly the brick wall that rings) must not be used to measure a decay.
  `renderer.measure` no longer passes one.
* `rooms.spl_from_ir`: `source_power_w` is gone; the docstring states that this
  is a *relative* figure and points at `measure().total_spl_db` for the absolute.
* `rooms.modal_decay_rate`: had units of `1/(s·m)` — a frequency divided by a
  length. Now "weighted dB per wall encounter × encounters per second" with the
  mean free path `1/Σ|cos_i|/L_i` of a plane wave in that direction. The
  docstring's "reduces exactly to Sabine" was wrong twice over: the units, and
  the physics — an axial mode presses only the two walls it bounces between, so
  in a uniformly lined room it decays at `4·A_pair/S` times the Sabine rate
  (about half as fast for the 58 Hz mode of the stairwell). Tested both ways.
* `transmission.mass_law_db`: was `20log10(m'f) − 47.4 − 8.5·[field]`, i.e. a
  normal-incidence asymptote with a magic constant bolted on, and documented
  with a third number. It is now the exact result for a limp sheet,
  `R = 10log10(1 + Z²)` at normal incidence and
  `R = 10log10(Z²/ln(1+Z²))` for the hemisphere average, and the docstring says
  plainly that the published `− 47.4` shortcut is *not* that integral (it sits
  ~2 dB below it at plasterboard and ~7 dB above at brick) and why. Verified
  against a numerical integration of the module's own angle-dependent
  transmission coefficient. The function had no callers and no tests; it has
  tests now.
* `AIR_AREA_COEF` in `paths.py` was a second copy of the constant in
  `rooms.py`; it is now an alias of it.
* `space_paths(drive=True)`: kept, with the physics written out rather than
  implied — at a rigid boundary the reverberant pressure doubles (+6 dB) and the
  flux through a surface lying in the wall halves (−6 dB), so the diffuse
  reference at an element is the same `16πr²/A` as anywhere else in the room. The
  parameter is now documented as reporting that fact, not applying a factor.

### Tests changed, and why it was allowed

Two previously-green tests are edited. Neither was relaxed to make a failure go
away; both were asserting a number that only the bug produced, and each is now a
stricter test of the same physical claim.

1. **`test_engine.py::test_modal_band_puts_cabin_modes_where_theory_says`** —
   `assert peak_near(39.0) > ref − 2.0`. It passed by **0.11 dB** before this
   pass. Three reasons it cannot survive a correct model: the preset's seat is at
   x = 2.1 m of a 4.4 m cabin, i.e. **0.1 m from the pressure node** of the
   (1,0,0) mode (|cos| = 0.05); the source is rated `f3 = 55 Hz`, so at 39 Hz it
   is 6.8 dB down (and the source is now applied to the modal band, as it always
   was to the rays); and the modal band was 6 dB too hot at the crossover, which
   is the only reason the mode appeared at all. With the fixes the peak sits 9.9 dB
   below the mid-band. The 78 Hz assertion is untouched (it passes at +3.7 dB).
   The 39 Hz mode is now tested the way it is heard in a car — by how far it moves
   between the node and the antinode: the strongest seat-to-seat difference between
   25 and 70 Hz must sit within 3 Hz of 39.0 Hz (measured **38.0 Hz**) and exceed
   8 dB (measured **+13.8 dB**). That pins the mode's *frequency* and its spatial
   structure, neither of which a level window around 39 Hz could distinguish from
   a generally boomy bass — with the compliance in the model the loudest thing in
   31–47 Hz is the smooth rise towards DC, not the mode.
2. **`test_engine.py::test_walking_away_keeps_the_reverberant_field_flat`** —
   `far − near C50 ≈ −12 ± 2.5 dB`, now −8.3 dB. The −12 was the missing tail:
   with the direct ray charged against the reverberant budget, the tail near the
   source was empty, so C50 near was inflated and its change with distance was
   purely the direct sound's `1/r²`. The test now checks both halves of its own
   docstring separately — the direct arrival must fall `−12 ± 2.5 dB` (measured
   −10.2), the reverberant field must be flat to ±2 dB (measured −0.7) — and
   brackets the C50 change at −14…−4.

### Deliberately unchanged

* No turbulence *absorption* term (ISO 9613-2) — unchanged reasoning in the README.
* `Barrier` still diffracts over horizontal top edges only: no vertical side-edge
  route, no non-planar or sloping screens.
* `mode_count_weight`'s Weyl count ignores absorption, so a large, very dead room
  claims a diffuse field low down that it cannot hold. Defensible, documented,
  and untouched.
* The Schroeder-frequency clip to 60–320 Hz in `renderer.modal_cutoff_hz` decides
  which rooms get a modal band at all. Not touched in this pass.
* `decay_curve_db`'s Schroeder integration and edge handling: pass 1 checked it.
* `Barrier.blocks()` / `diffraction.is_shadowed()` overlap; both are used, both
  are tested, the duplication is small and obvious.

### For the next pass — where I would attack

1. **The `f_rN` temperature exponent.** Ours multiplies the humidity bracket by
   `t^{-1/3}`, the reference implementation by `t^{-1/2}` (their inner exponential
   is identical). Neither can be checked against ISO 9613-1's own text from here,
   and the disagreement is bounded by the measured 1.5 % grid deviation — but one
   of the two is wrong, and it is a one-line fix once someone reads the standard.
2. **The missing static image field** in `modal_transfer`: a smooth,
   position-dependent error of a few dB in the 20–60 Hz region of a small room
   (15 % at 4 Hz in the cabin test). A static Neumann lattice sum would close it.
3. **The turbulence σ** is calibrated by a test, not derived from a structure
   constant. A tester who wants `C_n²`-based physics will find the trade-off (a
   blurrier comb always lifts the band mean, because a filled null is louder)
   documented at `paths.py:283` and in the README.
4. **The edge group weights both edges of a thick top equally**; the far edge is
   partly shadowed by the near one, so a wide top is up to 3 dB optimistic. ISO
   9613-2's double-edge treatment is the way to fix it.
5. **The crossover trim** is an energy ratio weighted by `x_lo·x_hi`. Alternatives
   (match at `f_x` alone, match the ray model to the modal one instead, match
   per-band) all give the same car-cabin result to ~1 dB, but the splice in a
   *mismatched* pair — a soft-source room with a hard-cabin crossover, say —
   deserves an attempt to break it.
6. `test_physics.py::test_atmosphere_cross_check_independent_implementation`
   still skips without a package. Now that the vendored table exists, it is a
   candidate for deletion — but say so in the audit rather than silently.
7. `time_strength_cents`'s `(-10, -25) dB` window vs ISO 3382-1's -5/-35.

No daemons, no scratch files in the tree; probes lived in `/tmp`.
