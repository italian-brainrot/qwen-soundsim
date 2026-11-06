"""The parts of the UI that are pure logic: which track comes next, and getting
audio in from a path somebody typed rather than from a tree somebody clicked.

Both are deliberately kept out of the Qt widgets so they can be checked here.
"""
import os

import pytest

from soundsim.engine.player import (expand_audio_paths, is_audio_file,
                                    iter_audio_files, iter_subdirs)
from soundsim.engine.playlist import Playlist


# ---------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------
def make(n=5):
    return Playlist([f"/music/track{i:02d}.wav" for i in range(n)])


def test_next_and_prev_walk_the_order_things_were_added_in():
    p = make()
    assert p.current().endswith("track00.wav")
    assert p.next().endswith("track01.wav")
    assert p.next().endswith("track02.wav")
    assert p.prev().endswith("track01.wav")
    assert p.prev().endswith("track00.wav")
    assert p.prev() is None                 # repeat off: the top is the top
    assert p.current().endswith("track00.wav")


def test_repeat_off_stops_at_the_end_and_repeat_all_wraps():
    p = Playlist(["a.wav", "b.wav"])
    p.next()
    assert p.next() is None and p.current() == "b.wav"
    p.set_repeat("all")
    assert p.next() == "a.wav"
    assert p.prev() == "b.wav"              # and backwards wraps too


def test_repeat_one_rewinds_only_when_the_track_ends_by_itself():
    p = Playlist(["a.wav", "b.wav"])
    p.set_repeat("one")
    assert p.next(auto=True) == "a.wav"     # reached the end: same track again
    assert p.next() == "b.wav"              # a deliberate skip still moves on
    assert p.cycle_repeat() == "off"


def test_shuffle_visits_every_track_exactly_once():
    p = make(9)
    p.seed(20250519)
    p.set_shuffle(True)
    seen = [p.current()]
    for _ in range(len(p) - 1):
        nxt = p.next()
        assert nxt is not None and nxt not in seen
        seen.append(nxt)
    assert p.next() is None                 # one full pass, then stop
    assert sorted(seen) == sorted(p.items)


def test_shuffling_keeps_the_playing_track_then_carries_on_in_order():
    p = make(6)
    p.seed(7)
    p.select(3)
    p.set_shuffle(True)
    assert p.current() == "/music/track03.wav"
    nxt = p.next()
    assert nxt is not None and nxt != "/music/track03.wav"
    p.set_shuffle(False)                    # carry on in file order from here
    assert p.current() == nxt
    cur = p.index
    assert p._order == list(range(cur, 6)) + list(range(0, cur))
    expected = "/music/track%02d.wav" % (cur + 1 if cur + 1 < 6 else 0)
    assert p.next() == expected


def test_files_added_while_shuffling_are_dealt_in_ahead_of_the_cursor():
    p = make(4)
    p.seed(3)
    p.set_shuffle(True)
    p.select(1)
    p.add(["/music/new.wav"])
    assert p._order[p._slot()] == 1                 # the cursor did not move
    assert 4 in p._order[p._slot() + 1:]            # the new track lies ahead
    assert p.current() == "/music/track01.wav"


def test_removing_the_playing_track_follows_what_takes_its_place():
    p = make(4)
    p.select(2)
    p.remove([0])
    assert p.current() == "/music/track02.wav"      # shifted up, still playing
    p.remove([p.index])                             # remove the playing one
    assert p.current() == "/music/track03.wav"
    assert [os.path.basename(x) for x in p.items] == \
        ["track01.wav", "track03.wav"]
    p.remove([0])                                   # the other one
    assert len(p) == 1 and p.current() == "/music/track03.wav"
    p.remove([0])
    assert len(p) == 0 and p.current() is None


def test_add_never_duplicates_and_a_cleared_list_has_no_cursor():
    p = Playlist(["a.wav"])
    assert p.add(["a.wav", "b.wav"]) == ["b.wav"]
    assert len(p) == 2
    p.clear()
    assert len(p) == 0 and p.current() is None
    assert p.next() is None and p.prev() is None
    assert p.add(["c.wav"]) == ["c.wav"]
    assert p.current() == "c.wav"


def test_snapshot_round_trip():
    p = make(5)
    p.seed(1)
    p.set_shuffle(True)
    p.next()
    state = p.snapshot()
    q = Playlist()
    q.restore(state)
    assert q.items == p.items
    assert q.current() == p.current()
    assert q.shuffle and q.repeat == p.repeat
    # the shuffled plan itself is regenerated: what must hold is that it is a
    # permutation and that continuing from the cursor visits everything
    assert sorted(q._order) == list(range(len(p)))
    rest = [q.next() for _ in range(len(p))]
    assert None in rest


# ---------------------------------------------------------------------------
# paths typed, pasted, dropped or globbed
# ---------------------------------------------------------------------------
@pytest.fixture
def library(tmp_path):
    rec = tmp_path / "records"
    rec.mkdir()
    (rec / "one.wav").write_bytes(b"x")
    (rec / "two.flac").write_bytes(b"x")
    (rec / "notes.txt").write_text("not audio")
    deep = rec / "box set"
    deep.mkdir()
    (deep / "three.mp3").write_bytes(b"x")
    (deep / "cover.jpg").write_bytes(b"x")
    hidden = rec / ".sync"
    hidden.mkdir()
    (hidden / "hidden.wav").write_bytes(b"x")
    (tmp_path / "with space.opus").write_bytes(b"x")
    return tmp_path


def test_a_typed_file_a_typed_glob_and_a_path_with_spaces_all_work(library):
    rec = str(library / "records")
    one = os.path.join(rec, "one.wav")
    assert expand_audio_paths(one) == [one]
    assert [os.path.basename(p) for p in expand_audio_paths(rec + "/*.wav")] == \
        ["one.wav"]
    assert expand_audio_paths(str(library / "with space.opus")) == \
        [str(library / "with space.opus")]


def test_a_typed_folder_expands_recursively_but_skips_hidden_ones(library):
    got = [os.path.basename(p)
           for p in expand_audio_paths(str(library / "records"))]
    assert got == ["one.wav", "two.flac", "three.mp3"]
    assert expand_audio_paths(str(library / "records"), recursive=False) == \
        [str(library / "records" / "one.wav"),
         str(library / "records" / "two.flac")]


def test_a_pasted_multiline_list_survives_spaces_in_names(library):
    pasted = "\n".join([
        str(library / "records" / "one.wav"),
        str(library / "with space.opus"),
        str(library / "records" / "notes.txt"),        # not audio: skipped
        "file://" + str(library / "records" / "two.flac"),
    ])
    got = expand_audio_paths(pasted)
    assert [os.path.basename(p) for p in got] == \
        ["one.wav", "with space.opus", "two.flac"]
    # and the same text pasted again adds nothing
    assert Playlist(got).add(expand_audio_paths(pasted)) == []


def test_expansion_is_absolute_deduplicated_and_capped(library):
    os.chdir(library)
    rel = os.path.join("records", "one.wav")
    assert expand_audio_paths([rel, rel, os.path.abspath(rel)]) == \
        [os.path.abspath(rel)]
    assert len(expand_audio_paths(str(library / "records"), limit=1)) == 1
    assert expand_audio_paths("'" + str(library / "with space.opus") + "'") == \
        [str(library / "with space.opus")]


def test_non_audio_and_missing_paths_yield_nothing():
    assert expand_audio_paths("/definitely/not/here.flac") == []
    assert expand_audio_paths("/etc/passwd") == []
    assert expand_audio_paths("") == []
    assert is_audio_file("/x/y.WAV") and not is_audio_file("/x/y.md")


def test_the_browser_walks_one_level_at_a_time(library):
    rec = str(library / "records")
    assert [os.path.basename(d) for d in iter_subdirs(rec)] == ["box set"]
    assert [os.path.basename(f)
            for f in iter_audio_files(rec, recursive=False)] == \
        ["one.wav", "two.flac"]
    assert [os.path.basename(f)
            for f in iter_audio_files(rec, recursive=True)] == \
        ["one.wav", "two.flac", "three.mp3"]
