# SoundSim

A physically based **acoustic environment simulator that works like a media
player**.  Load a file (or use the built-in demo material), press play, and the
sound is rendered as it would actually be heard somewhere else: through a party
wall, from a parked car, from across a field, from the next room, or from the
street while the living room party continues behind the glass.

There are no impulse-response files and no fudge factors.  Everything the ear
hears is computed from geometry, construction and weather, at the resolution the
underlying physics is specified at — 1/3 octave bands.

```bash
python -m soundsim                 # GUI
python tools/render.py --list      # offline renders, no speaker needed
pytest -q                          # 84 physics, engine and interface tests
```

## What you can do

* 11 environments: party next door, through a closed window, open window at
  night, next room (partition + door), two doors down through a corridor, parked
  car heard outside, inside a car, distant stage in a park, across a room,
  concrete stairwell, small bedroom.  The one in force is named in the bar at
  the top, and it says so when you have edited it away from its own defaults
  (**reset** puts them back).
* Live parameters — source level and its own bass limit, listener distance,
  image-source order, temperature, humidity, wind, ground type — re-render the
  response and **crossfade it into the playing audio**, so a change is heard as
  a transition.
* Plots: impulse response, backward-integrated decay curve, 1/3-octave balance
  against free field, and T60 per band measured on the response plotted against
  the Sabine prediction.
* **Audio gets in however you like to work**: type or paste a path — a file, a
  folder, a `*.glob`, or a whole list of them on separate lines — into the path
  box and press Enter; or Choose… and paste into the chooser's own path line;
  or drop files on the window; or Ctrl+V with the track list focused.  The list
  itself is a splitter pane: drag the handle and it takes half the window if you
  want.  It has a filter box, the playing row is marked, and the header names
  the loaded audio with its path, length, rate and channel count.
* **Prev / Play / Stop / Next / Shuffle / Repeat** along the bottom, with a
  seek bar across the full width.  Space toggles play, `Alt+Left`/`Alt+Right`
  step through tracks, `Alt+S` shuffles, `Alt+R` cycles repeat, `Ctrl+O` opens
  the chooser, and the media keys work.  A track ending by itself hands over to
  the next one; repeat-one rewinds it instead.  Pausing lets the room's tail ring
  out but holds the position, so resume picks up where you left it.  The
  playlist, shuffle, repeat and the last folder are remembered for the next
  start.
* The received level and its difference from free field are read out next to the
  measured ISO parameters, together with the Schroeder frequency, so you can see
  when a space is in its modal regime rather than a diffuse one.
* Stereo is a spaced pair: each channel is an independent physical render with
  the receivers 18 cm apart, so inter-channel time/level differences come out of
  the geometry (ground interference, a barrier edge, a mode null) rather than
  from a widening effect.

## How the sound is built

```
Scene (rooms, walls, windows, ground, weather, screens)
  -> path graph:   space --(coupling element)--> space
  -> per 1/3-octave band:  direct, image-source reflections, modal low band,
     transmission through elements, ground images, screen diffraction
  -> hybrid synthesis: band-limited impulses + energy-balanced diffuse tail
  -> partitioned FFT convolution in the audio callback
```

**The cascade is the point.**  A wall or window is driven by the *whole* interior
field — direct sound, every interior reflection, and the interior reverberant
tail — and re-radiates into the receiver's space with `sqrt(S·τ / 2π)`.  So sound
from the next room carries both the panel's transmission loss *and* the source
room's reverberation: muffled, bass-shifted, with extra ringing whose decay time
belongs to the room you cannot see.

## Physics, and how each piece is checked

| Effect | Model | Test |
|---|---|---|
| Air absorption | ISO 9613-1 Annex A (Bass & King) | `tests/test_atmosphere_reference.py` — every value against a **vendored table** of an independent published implementation (acoustic-toolbox, BSD-3), −20…+50 °C × 10–90 % RH × 63 Hz…8 kHz; measured worst 1.5 %, asserted below 3 %. Runs everywhere: the reference is data, not an import. |
| Porous ground | JCA equivalent fluid (Johnson–Champoux–Allard), `α∞`, `Λ` from flow resistivity | `test_jca_matches_delany_bazley_where_db_is_valid`, `test_jca_stays_physical_for_resistive_grounds` |
| Wall / window | damped plate: mass law + coincidence + damping, diffuse field integral (Beranek); exact Rayleigh piston radiation | `test_normal_incidence_mass_law_is_exact`, `test_grazing_incidence_transmits_completely`, `test_coincidence_dip_exists_and_deepens_with_low_damping` |
| Double leaf | air-spring resonance `f0 = 60√((m1+m2)/(m1m2d))`, below it the mass law of the sum, above it leaf-by-leaf + cavity bonus | `test_double_leaf_mass_air_mass_resonance` |
| Openings | air-plug radiation impedance, `τ = 4Re z/|z+1|²`, Rayleigh limit `τ ≈ 2(ka)²` | `test_aperture_is_high_pass_relative_to_size` |
| Ground reflection | JCA plane-wave coefficient **with phase**, carried into the synthesis as a per-band delay, so direct and reflected rays interfere | `test_a_park_swallows_the_mids_that_still_water_carries`, `test_hard_ground_is_louder_than_no_ground_at_all` |
| Screens | Maekawa / ISO 9613-2 `10log10(3+20N)`, applied as **insertion loss on the rays the screen shadows**: the shadow is decided geometrically, the diffracting point is the shortest route over the *finite* top edge, the two edges of a flat top are averaged in energy, and a cascade of screens follows the ISO rule `ΣA_i − 10(n−1)log10 3` | `test_a_screen_below_the_line_of_sight_changes_nothing`, `test_a_screen_that_shadows_the_ray_makes_the_scene_quieter`, `test_a_screen_long_enough_to_cover_the_ray_shields_more_than_one_that_is_not`, `test_a_cascade_of_screens_is_the_iso_9613_2_cascade` |
| Rooms | Sabine + Fitzroy + atmospheric area `A_air = 0.9214·α·V`, Schroeder frequency | `test_sabine_matches_hand_computation`, `test_schroeder_frequency_small_room_high_large_room_low`, engine tests |
| Early reflections | image sources with per-wall `√(1−α)` per hit, 1/r, air absorption | `test_first_order_distances_are_mirror_distances`, `test_ism_is_reciprocal`, `test_reflection_sequence_count_matches_textbook` |
| Low-frequency regime | exact Rayleigh modal sum of the box (Krenk free-field subtraction) **including the uniform mode**, i.e. the compliance of the air — the term that makes a sealed room a sealed room; per-mode absorption ⇒ per-mode Q | `test_below_the_first_mode_the_room_is_not_free_field`, `test_the_compliance_of_the_air_is_the_closed_form`, `test_cube_axial_modes`, `test_modal_band_puts_cabin_modes_where_theory_says` (39 Hz mode located by its seat-to-seat structure) |
| Low-frequency validity of the diffuse field | Weyl's mode count per 1/3-oct band, `4πVf²Δf/c³`, weights the reverberant build-up: below about one mode per band the room cannot store the diffuse amount | (checked via the sub-60 Hz share of every rendered response) |
| Late field | diffuse energy balance `E/E_dir = 1 + 16πr²/A`, tail scaled to the remainder | `test_energy_balance_holds_for_an_enclosed_room`, `test_synth_tail_reproduces_its_own_decay_time`, `test_room_response_measured_t60_matches_sabine_in_the_diffuse_regime` |
| Real-time core | uniform partitioned FFT convolution, exact | `test_partitioned_convolver_is_exact_for_any_block_size` (rel. err ~3e-16) |

Measured against analysis (this repo, not literature values copied in):

* T60 of a concrete stairwell, 1/3 octave: **measured 5.82 / 5.00 / 4.02 s** at
  500 / 1 k / 2 k Hz against **Sabine 5.77 / 5.04 / 4.10 s**.
* A tail specified at 0.40 s measures 0.40 s in the mid bands.
* Walking from 2 m to 8 m inside one room: the direct arrival drops **12 dB** as
  `1/r²` demands, the room's reverberant field moves **0.7 dB**, and C50 follows
  with **8 dB** -- less than 12 because the early reflections arrive with the
  direct sound and the room's own field is there to be heard.
* Replacing a double-glazed window with 100 mm masonry of the same area buys
  18 dB; opening that window gives back more than 20 dB.
* 300 m of cold, dry air removes far more at 8 kHz than at 250 Hz, and the
  humidity dependence flips sign between 2 kHz and 8 kHz — the behaviour the
  relaxation model predicts.

## What distance actually sounds like

At 300 m with the source 4 m up and the listener 1.5 m up, the two rays meet at
about a degree of grazing. Over still water or asphalt `R = +1` and they add:
+5.9 dB relative to free field at 125 Hz. Over forest floor or dry sand the
coefficient rotates by more than 150 degrees and the ground *subtracts*: −21.5 dB
at the same frequency. That is a 27 dB difference in the low mids between two
scenes that are otherwise identical, and it is interference, not absorption --
which is why the band balance is computed as a coherent pressure sum, the same
way the response is synthesised.

Measured band balance for *distant stage in the park* (dB relative to free field
at the same distance, 8 °C, 80 % RH):

| ground | 125 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---|---|---|---|---|
| still water / asphalt | +5.9 | +5.4 | +4.8 | +2.5 | −6.6 |
| short grass | −1.1 | −7.0 | −11.4 | −8.6 | −8.1 |
| long grass | −12.6 | −17.8 | −10.9 | −5.8 | −6.8 |
| forest floor | −21.5 | −16.1 | −9.7 | −5.3 | −6.7 |

## Level staging (why "quiet scene" never means "crank the gain")

The response is absolutely calibrated, and the monitor path is deliberately
separate from the physics:

* **Full-scan peak prediction** — the whole material is convolved with the
  response in 30 s chunks and the monitor gain is set so the true worst sample
  fits with headroom.  Guessing from the first seconds clips later passages,
  and a clipped bass render is heard as buzzing.
* **Soft limiter** as a last resort: one-block attack, ~1 s release.  Hard
  clipping of a bass-heavy render is the single worst-sounding failure mode an
  auralizer has, so it is not used.
* **Sub-15 Hz removal** on the synthesised response.  A 1/3-octave band at 25 Hz
  cannot be represented in a response a few cycles long, and the smearing down to
  DC is synthesis artefact, not room acoustics: it was 2-3 % of the energy, all
  of it inaudible and all of it in the speaker cone.
* **Monitor low-cut** (default off): for speakers that cannot reach the lowest
  octave.  Off keeps the physics literal; the plots and dB figures never change.

Measured linearity of the render path: a 1 kHz sine through the "party next door"
response stays at 1 kHz — energy below 40 Hz is 67 dB down, block-rate artefacts
63 dB down, harmonic distortion 123 dB down.

## Honest limitations

* **Ground interference is resolved only to a third of an octave.**  The
  reflection coefficient's phase is carried as a per-band delay, which is exact
  at a band centre and approximate across the band -- fine for the broad soft-
  ground hollow, not for razor-thin combs.  Panel transmission is still applied
  as a magnitude: around a coincidence dip the fine comb structure is
  approximate.
* **Atmospheric turbulence blurs the interference, and is calibrated, not
  predicted.**  Eddies wander the *relative phase* of two rays that run almost
  side by side, so the engine carries the mutual coherence of the direct and
  ground rays, `γ(f) = exp(−σ_φ²/2)` with `σ_φ = 2πf·σ/c`: the interference term
  is multiplied by γ and the energy it releases, `(1−γ²)|g|²`, is carried as an
  incoherent arrival at the same delay — so the split conserves band energy
  exactly, and the pattern gets shallower while the mean level rises slightly,
  which is what filling a ground null does.  `σ = 5.5e-5 × path length ×
  turbulence` (1.6 cm over the 300 m park at turbulence 1.0) is a **calibration**
  of the residual differential path wander, not a value read off a turbulence
  spectrum; the depth also depends on wind, temperature gradient and site.  Eddies
  make distant treble waver over fractions of a second (the stage in the player:
  an Ornstein-Uhlenbeck modulation of the upper band).  The ISO 9613-2 turbulence
  *absorption* term was deliberately not implemented, because with plausible
  coefficients it predicts absurd figures (−90 dB at 4 kHz over 300 m) and needs
  site calibration.
* **Rectangular rooms only.**  Rooms are boxes; image sources are exact for a box
  and the modal solution is exact for a box.  L-shaped rooms, ceilings with
  reliefs, and furniture geometry are out of scope.
* **Box degeneracy.**  A rectangular box is an integrable geometry: many
  reflection sequences land on exactly the same image.  Counting them as
  independent energy makes the image sum diverge unless α > 0.8, so each image is
  counted once (Borish's convention) and the missing late energy is supplied by
  the energy-balanced tail.  `multiplicity` is kept only as diagnostic.
* **Modal region.**  Below the Schroeder frequency (shown in the UI) a ray
  description is the wrong model, so the exact modal sum replaces the low bands —
  but only when source and listener share one room.  In a cascade the modes of
  the *source* room are represented energetically, not as discrete resonances.
* **No head, no HRTF.**  Stereo is a spaced pair: correct inter-channel cues, but
  front/back and elevation cues are absent.  Mono downmix is honest "one
  microphone" output.
* **Bass through glass is genuinely bad news.**  Below the double-leaf resonance
  the two leaves move together and a window isolates only some dB, so a scene
  like *party next door* really is dominated by 25-120 Hz at the listener: 30-40
  dB more transmission there than at 2 kHz.  If that sounds like a buzzing rumble
  on your speakers, it is the cone failing to make 40 Hz, not the model — use the
  monitor low-cut or headphones, and the band-balance plot will show the tilt is
  what the construction predicts.
* **Cascaded-room levels carry real uncertainty** (a few dB).  The corridor /
  two-partition chain is built from the standard incoherent cascade of
  transmissions and reverberant energy balances, which is what building
  acoustics practice does, but flanking via structure-borne paths is not modelled.
* **Absolute playback level is not physics.**  The response is calibrated (a
  unit-rms signal is 1 Pa at 1 m), which is what makes every dB figure here
  meaningful — but your amplifier has gain too, so the app applies a bounded
  *monitor gain* to bring the predicted received level to a comfortable one.
  It is applied after the response, never inside it, and the amount is reported.
* **Source is a flat-SPL monopole above its own −3 dB point** (default 55 Hz),
  omnidirectional, falling at 12 dB per octave below it — the velocity, hence the
  volume velocity, of a sealed box. That roll-off is applied to the modal low band
  as well as to every ray, because a room cannot be louder than the loudspeaker
  driving it: it is also what keeps the box's own 12 dB-per-octave bass rise
  (compliance of the air) from running away below audibility.  Loudspeaker
  directivity is not modelled.
* Very live rooms are truncated at a configurable response length (default 6 s)
  for the live path; the *offline* renderer uses the same cap, adjustable.

## Layout

```
soundsim/acoustics/   constants, atmosphere, materials, transmission, ground,
                      diffraction, rooms, image_source, modal
soundsim/engine/      bands, scene, paths, rir, renderer, convolver, player,
                      presets, playlist
soundsim/ui/          Qt main window, playlist and path-first file chooser,
                      QPainter plots
tools/
  render.py           offline WAV rendering
  make_atmosphere_reference.py   regenerates the vendored air-absorption table
tests/                test_physics.py (30)  test_engine.py (24)
                      test_playlist.py (15)  test_ui.py (16, offscreen)
                      test_atmosphere_reference.py (42, data-driven)
                      test_audit_pass1.py, test_audit_pass2.py (regressions
                      written against the findings in AUDIT.md)
```

The track order and the path expansion live in `engine/` rather than in the
widgets: which track comes next under shuffle and repeat, and what a pasted line
of paths turns into, are rules worth pinning down without a display.
