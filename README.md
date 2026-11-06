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
pytest -q                          # 42 physics + engine tests
```

## What you can do

* 11 environments: party next door, through a closed window, open window at
  night, next room (partition + door), two doors down through a corridor, parked
  car heard outside, inside a car, distant stage in a park, across a room,
  concrete stairwell, small bedroom.
* Live parameters — source level and its own bass limit, listener distance,
  image-source order, temperature, humidity, wind, ground type — re-render the
  response and **crossfade it into the playing audio**, so a change is heard as
  a transition.
* Plots: impulse response, backward-integrated decay curve, 1/3-octave balance
  against free field, and T60 per band measured on the response plotted against
  the Sabine prediction.
* Space bar plays/pauses.  Add files or a folder, double-click to play.  The
  received level and its difference from free field are read out next to the
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
| Air absorption | ISO 9613-1 Annex A (Bass & King) | `test_atmosphere_cross_check_independent_implementation` — ≤1.5 % vs an independent implementation over −20…+50 °C, 10–90 % RH |
| Porous ground | JCA equivalent fluid (Johnson–Champoux–Allard), `α∞`, `Λ` from flow resistivity | `test_jca_matches_delany_bazley_where_db_is_valid`, `test_jca_stays_physical_for_resistive_grounds` |
| Wall / window | damped plate: mass law + coincidence + damping, diffuse field integral (Beranek); exact Rayleigh piston radiation | `test_normal_incidence_mass_law_is_exact`, `test_grazing_incidence_transmits_completely`, `test_coincidence_dip_exists_and_deepens_with_low_damping` |
| Double leaf | air-spring resonance `f0 = 60√((m1+m2)/(m1m2d))`, below it the mass law of the sum, above it leaf-by-leaf + cavity bonus | `test_double_leaf_mass_air_mass_resonance` |
| Openings | air-plug radiation impedance, `τ = 4Re z/|z+1|²`, Rayleigh limit `τ ≈ 2(ka)²` | `test_aperture_is_high_pass_relative_to_size` |
| Screens | Maekawa / ISO 9613-2, `10log10(3+20N)` with a numerically found edge point | `test_maekawa_zero_path_excess_is_4p8_db`, `test_barrier_geometry_detects_shadow` |
| Rooms | Sabine + Fitzroy + atmospheric area `A_air = 0.9214·α·V`, Schroeder frequency | `test_sabine_matches_hand_computation`, `test_schroeder_frequency_small_room_high_large_room_low`, engine tests |
| Early reflections | image sources with per-wall `√(1−α)` per hit, 1/r, air absorption | `test_first_order_distances_are_mirror_distances`, `test_ism_is_reciprocal`, `test_reflection_sequence_count_matches_textbook` |
| Low-frequency regime | exact Rayleigh modal sum of the box (Krenk free-field subtraction), per-mode absorption ⇒ per-mode Q | `test_cube_axial_modes`, modal decay = Sabine (7.11 s vs 7.11 s), cabin axial modes at 39.0 / 78.0 / 95.3 / 117.0 Hz |
| Late field | diffuse energy balance `E/E_dir = 1 + 16πr²/A`, tail scaled to the remainder | `test_energy_balance_holds_for_an_enclosed_room`, `test_synth_tail_reproduces_its_own_decay_time`, `test_room_response_measured_t60_matches_sabine_in_the_diffuse_regime` |
| Real-time core | uniform partitioned FFT convolution, exact | `test_partitioned_convolver_is_exact_for_any_block_size` (rel. err ~3e-16) |

Measured against analysis (this repo, not literature values copied in):

* T60 of a concrete stairwell, 1/3 octave: **measured 5.86 / 4.99 / 4.09 s** at
  500 / 1 k / 2 k Hz against **Sabine 5.77 / 5.04 / 4.10 s**.
* A tail specified at 0.40 s measures 0.40 s in the mid bands.
* Doubling the listener distance inside one room drops C50 by 12 dB, as the
  direct-only `1/r²` law demands.
* Replacing a double-glazed window with 100 mm masonry of the same area buys
  18 dB; opening that window gives back more than 20 dB.
* 300 m of cold, dry air removes far more at 8 kHz than at 250 Hz, and the
  humidity dependence flips sign between 2 kHz and 8 kHz — the behaviour the
  relaxation model predicts.

## Honest limitations

* **Magnitude-only transmission and ground phase.**  Panel and ground transfer
  are applied as per-band magnitudes with the path delay; their frequency-dependent
  *phase* is not.  Around a coincidence dip or a grass-interference notch the
  fine comb structure is therefore approximate.
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
  omnidirectional.  Loudspeaker directivity is not modelled.
* Very live rooms are truncated at a configurable response length (default 6 s)
  for the live path; the *offline* renderer uses the same cap, adjustable.

## Layout

```
soundsim/acoustics/   constants, atmosphere, materials, transmission, ground,
                      diffraction, rooms, image_source, modal
soundsim/engine/      bands, scene, paths, rir, renderer, convolver, player, presets
soundsim/ui/          Qt main window, QPainter plots
tools/render.py       offline WAV rendering
tests/                test_physics.py (30)  test_engine.py (12)
```
