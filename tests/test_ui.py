"""Offscreen checks of the widgets themselves.

Only the things a screenshot would otherwise have to prove: that a typed path
reaches the transport, that the track list can be made bigger than a strip,
that the transport buttons drive the playlist model, and that the header says
what audio and what environment are actually selected.
"""
import os

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"no usable Qt here: {exc}")
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"cannot start Qt without a display: {exc}")
    return app


@pytest.fixture
def tracks(tmp_path):
    """Three small real files, so the decoder is in the loop too."""
    d = tmp_path / "music"
    d.mkdir()
    names = ["alpha.wav", "bravo.wav", "charlie.wav"]
    for i, n in enumerate(names):              # 1 s, 2 s, 3 s
        sf.write(str(d / n), np.zeros((48_000 * (i + 1), 2), np.float32), 48_000)
    return d


def shutdown(w):
    """No loose render threads: the response render runs on its own QThread, and
    Qt takes the process with it if that thread is still live at exit."""
    w._debounce.stop()
    if w.worker is not None and w.worker.isRunning():
        w.worker.wait(30_000)
    w.close()


@pytest.fixture
def win(qapp, tmp_path):
    """A window with its own empty state file, so tests never inherit a
    playlist from the one before."""
    from PySide6.QtCore import QSettings
    from soundsim.ui.app import MainWindow
    w = MainWindow(QSettings(str(tmp_path / "state.ini"), QSettings.IniFormat))
    w.player.start = lambda: None         # never open a real output device here
    yield w
    shutdown(w)


def test_a_path_type_into_a_box_becomes_the_playing_audio(win, tracks):
    """The headline complaint: pasting a path must be enough."""
    target = str(tracks / "bravo.wav")
    win.panel.path_box.setText(target)
    win.panel.path_box.returnPressed.emit()             # as pressing Enter
    assert len(win.tracks) == 1
    assert os.path.basename(win.player.material.path) == "bravo.wav"
    assert "bravo.wav" in win.track_title.text()
    assert str(tracks) in win.track_sub.text()
    assert win.counter.text() == "1 / 1"


def test_a_pasted_folder_lands_in_the_list_in_order(win, tracks):
    win.panel.path_box.setText(str(tracks) + "\n" + str(tracks / "alpha.wav"))
    win.panel.path_box.returnPressed.emit()
    assert [os.path.basename(p) for p in win.tracks.items] == \
        ["alpha.wav", "bravo.wav", "charlie.wav"]


def test_the_track_list_is_a_pane_and_not_a_strip(win):
    """It used to be capped at 88 px with no way to grow it."""
    assert win.panel.view.maximumHeight() > 1000
    assert win.panel.view.minimumHeight() < 200


def test_transport_buttons_move_the_cursor(win, tracks):
    win._add_paths([str(tracks)])
    first = win.tracks.current()
    win.btn_next.click()
    assert win.tracks.current() != first
    assert os.path.basename(win.player.material.path) == \
        os.path.basename(win.tracks.current())
    win.btn_prev.click()
    assert win.tracks.current() == first
    win.btn_shuffle.click()
    assert win.tracks.shuffle and win.btn_shuffle.isChecked()
    win.btn_shuffle.click()
    assert not win.tracks.shuffle
    for want in ("all", "one", "off"):
        win.btn_repeat.click()
        assert win.tracks.repeat == want


def test_the_header_names_the_audio_and_the_environment(win):
    """No guessing which of the twelve environments is applied."""
    win._update_now_playing()
    win._update_env_label()
    assert win.env_name.text() == "Party next door"
    assert win.env_sub.text().startswith("environment")
    assert win.scene_list.currentRow() == 0
    assert win.player.material is not None            # the demo material
    assert "demo material" in win.track_title.text()
    assert "kHz" in win.track_sub.text()


def test_choosing_an_environment_shows_it_and_edits_are_marked(win):
    from soundsim.engine import presets
    row = list(presets.PRESETS).index("concrete_stairwell")
    win.scene_list.setCurrentRow(row)
    assert win.env_name.text() == "Concrete stairwell"
    assert win.env_sub.text().startswith("environment")
    win.in_distance.setValue(win.in_distance.value() + 3.0)
    assert win.env_name.text().endswith("edited")
    assert win.btn_reset_env.isEnabled()
    win.btn_reset_env.click()                 # back to the preset's own numbers
    assert win.env_name.text() == "Concrete stairwell"
    assert not win.btn_reset_env.isEnabled()


def test_the_chooser_takes_a_pasted_folder(win, tracks, qapp):
    from soundsim.ui.media import AudioChooser
    dlg = AudioChooser(win, start=str(tracks))
    dlg.path_edit.setText("here-is-nothing.wav")
    dlg._accept_path()
    assert not dlg.chosen                       # and it says so, quietly
    dlg.path_edit.setText(str(tracks))
    dlg._accept_path()
    assert [os.path.basename(p) for p in dlg.chosen] == \
        ["alpha.wav", "bravo.wav", "charlie.wav"]


def test_removing_rows_takes_them_out_of_the_list(win, tracks):
    win._add_paths([str(tracks)])
    assert len(win.tracks) == 3
    win._remove_rows([0, 1])
    assert [os.path.basename(p) for p in win.tracks.items] == ["charlie.wav"]
    win._clear_playlist()
    assert len(win.tracks) == 0 and win.counter.text() == "0 / 0"


def test_a_big_list_stays_responsive_and_marks_the_playing_row(win, tmp_path,
                                                               qapp):
    """Four hundred rows is an ordinary folder; the view must keep up."""
    import time

    names = [f"t{i:04d}.wav" for i in range(400)]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    win._add_paths([str(tmp_path / n) for n in names])
    assert win.panel.view.count() == 400
    t0 = time.perf_counter()
    for i in (0, 123, 399, 7):
        win.tracks.select(i)
        win._refresh_playlist_view()
    assert win.panel.view.item(7).text().startswith("\u25b6")
    assert time.perf_counter() - t0 < 1.0          # re-mirroring is not O(list)


def test_the_playlist_and_its_settings_survive_a_restart(qapp, tmp_path, tracks):
    from PySide6.QtCore import QSettings
    from soundsim.ui.app import MainWindow

    ini = str(tmp_path / "state.ini")
    w = MainWindow(QSettings(ini, QSettings.IniFormat))
    w._add_paths([str(tracks)])
    w.btn_shuffle.click()
    w.btn_repeat.click()                      # repeat: all
    w.tracks.select(1)
    w._save_session()
    shutdown(w)

    q = MainWindow(QSettings(ini, QSettings.IniFormat))
    q.player.start = lambda: None
    assert [os.path.basename(p) for p in q.tracks.items] == \
        ["alpha.wav", "bravo.wav", "charlie.wav"]
    assert q.tracks.index == 1 and q.tracks.shuffle and q.tracks.repeat == "all"
    assert q.btn_shuffle.isChecked() and q.btn_repeat.text() == "Repeat: all"
    assert os.path.basename(q.player.material.path) == "bravo.wav"
    shutdown(q)


def test_the_transport_reads_itself_back_into_the_window(win, tracks):
    """The poll tick used to be created and never started: the seek bar, the
    time, the level meter and the auto-advance to the next track all hang off
    it, and none of them moved."""
    win._add_paths([str(tracks / "bravo.wav")])
    assert win._tick.isActive()
    win.player.set_position_seconds(1.0)
    win._poll_player()
    assert win.time_label.text() == "0:01 / 0:02"
    assert win.seek.value() == 500


def test_a_finished_track_hands_over_to_the_next_one(win, tracks):
    """End of material -> next track, and with repeat-one the same one."""
    win.player.start = lambda: None         # no device needed for this one
    win._add_paths([str(tracks)])
    assert os.path.basename(win.player.material.path) == "alpha.wav"
    assert os.path.basename(win.tracks.current()) == "alpha.wav"
    win._was_playing = True
    win.player.playing = False              # what the callback does at the end
    win.player.index = len(win.player.material.samples)
    win._poll_player()
    assert os.path.basename(win.tracks.current()) == "bravo.wav"
    assert os.path.basename(win.player.material.path) == "bravo.wav"
    assert win.player.playing                         # resumed on its own
    win.btn_repeat.click()                  # repeat: all
    win.btn_repeat.click()                  # repeat: one
    win.player.index = len(win.player.material.samples)
    win._was_playing = True
    win.player.playing = False
    win._poll_player()
    assert os.path.basename(win.player.material.path) == "bravo.wav"


def test_filtering_the_list_does_not_lose_track_of_the_rows(win, tracks):
    win._add_paths([str(tracks)])
    win.panel.filter_box.setText("alph")
    assert [i for i in range(3) if not win.panel.view.item(i).isHidden()] == [0]
    note = win.panel.note.text()
    assert "1 / 3 shown" in note and "playing" in note
    win._activate_row(2)                      # the hidden rows are still there
    assert os.path.basename(win.player.material.path) == "charlie.wav"
    win.panel.filter_box.setText("")
    assert not win.panel.view.item(1).isHidden()
    assert "shown" not in win.panel.note.text()


def test_the_panes_keep_the_size_you_left_them(qapp, tmp_path):
    """The track list is resizable; that is only useful if it stays resized."""
    from PySide6.QtCore import QSettings
    from soundsim.ui.app import MainWindow

    ini = str(tmp_path / "s.ini")
    saved = QSettings(ini, QSettings.IniFormat)
    saved.setValue("split_audio", [120, 700])       # what a dragged handle leaves
    saved.sync()

    q = MainWindow(QSettings(ini, QSettings.IniFormat))
    q.player.start = lambda: None
    q.resize(1360, 860)
    q.show()
    qapp.processEvents()                            # a splitter splits only once
    sizes = q.stack.sizes()                         # it has been laid out
    assert len(sizes) == 2 and sizes[1] > sizes[0] * 2
    shutdown(q)


def test_time_strength_is_shown_in_centiseconds_and_d50_in_decibels(win):
    """AUDIT.md pass 1, finding F5: TS is a *time* (ISO 3382 centiseconds), but
    the window printed the number with "dB" after it, next to D50 which really is
    a level -- so the one column read as a level and the other as a time."""
    class _Result:
        ir = np.zeros(48_000)
        sample_rate = 48_000
        paths = []
        metrics = {"t60_iso": 0.6, "t60_mid_db": 0.55, "c50": 3.2, "c80": 1.1,
                   "d50": 62.4, "ts": 27.0, "itdg_ms": 5.2, "total_spl_db": 74.0,
                   "environment_loss_db": -12.0, "modal_cutoff_hz": 300.0,
                   "modal_modes_used": 40, "n_paths": 51, "schroeder_mid": 300.0}

    win._refresh_metrics(_Result(), 0.1)
    ts = win.metric_labels["ts"].text()
    d50 = win.metric_labels["d50"].text()
    assert ts.endswith("cs") and "dB" not in ts, f"TS rendered as {ts!r}"
    assert d50.endswith("dB"), f"D50 rendered as {d50!r}"
    assert ts.split()[0] == "27.0"


def test_the_seek_bar_does_not_move_while_paused(win, tracks):
    """What was seen: pressing pause left the slider sliding along."""
    import numpy as np
    win._add_paths([str(tracks / "bravo.wav")])          # two seconds long
    win.player.index = 48_000                            # one second in
    win._poll_player()
    assert win.seek.value() == 500 and win.time_label.text() == "0:01 / 0:02"
    out = np.zeros((256, 2))
    for _ in range(40):        # the output stream keeps calling back when paused
        win.player._callback(out, 256, None, None)
    win._poll_player()
    assert win.seek.value() == 500 and win.time_label.text() == "0:01 / 0:02"
