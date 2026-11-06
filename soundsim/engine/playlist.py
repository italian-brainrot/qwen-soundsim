"""Track order: what plays next, and whether that is the neighbour or chance.

Deliberately Qt-free.  The main window mirrors this model in a list widget; the
rules for "next", "previous", "shuffle" and "repeat" live here so they can be
tested without a display, and so the visible list and the transport can never
drift into disagreeing about which track is playing.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

REPEAT_MODES = ("off", "all", "one")


class Playlist:
    """An ordered set of paths plus the position and the playback order.

    ``items`` is what the user sees (the order things were added in);
    ``_order`` is the order they are visited in, which differs from it only when
    shuffling.  Both stay free of duplicates: adding the same folder twice adds
    nothing.
    """

    def __init__(self, items: Iterable[str] | None = None):
        self.items: list[str] = [str(p) for p in (items or [])]
        self.index: int | None = 0 if self.items else None
        self.shuffle: bool = False
        self.repeat: str = "off"
        self._rng = random.Random()
        self._order: list[int] = list(range(len(self.items)))

    # ------------------------------------------------------------------
    # contents
    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.items)

    def __contains__(self, path: object) -> bool:
        return path in self.items

    def seed(self, seed) -> None:
        """Make the shuffle reproducible (used by the tests)."""
        self._rng = random.Random(seed)

    def add(self, paths: Iterable[str]) -> list[str]:
        """Append the paths not already here; returns what was actually added."""
        added = []
        for p in paths:
            p = str(p)
            if p in self.items:
                continue
            self.items.append(p)
            self._insert(len(self.items) - 1)
            added.append(p)
        if self.index is None and self.items:
            self.select(0)
        return added

    def remove(self, indices: Iterable[int]) -> list[str]:
        """Drop rows; the cursor follows the track that takes its place."""
        drop = {int(i) for i in indices if 0 <= int(i) < len(self.items)}
        if not drop:
            return []
        removed = [self.items[i] for i in sorted(drop)]
        # where each surviving old index lands now
        shift = {}
        j = 0
        for i in range(len(self.items)):
            if i in drop:
                continue
            shift[i] = j
            j += 1
        self.items = [self.items[i] for i in sorted(shift)]
        self._order = [shift[i] for i in self._order if i in shift]
        cur = self.index
        if cur is None:
            self.index = 0 if self.items else None
        elif cur in drop:
            nxt = cur + 1
            while nxt in drop:
                nxt += 1
            self.index = (shift[nxt] if nxt in shift
                          else (shift[max(shift, default=-1)] if shift else None))
        else:
            self.index = shift[cur]
        return removed

    def clear(self) -> None:
        self.items, self.index, self._order = [], None, []

    # ------------------------------------------------------------------
    # position
    def path_at(self, index: int) -> str | None:
        return self.items[index] if 0 <= index < len(self.items) else None

    def index_of(self, path: str) -> int | None:
        try:
            return self.items.index(str(path))
        except ValueError:
            return None

    def current(self) -> str | None:
        return self.path_at(self.index) if self.index is not None else None

    def select(self, index: int | None) -> str | None:
        """Point at a row (clamped) and return the path there."""
        if not self.items or index is None:
            self.index = 0 if self.items else None
        else:
            self.index = max(0, min(int(index), len(self.items) - 1))
        self._ensure_in_order()
        return self.current()

    def jump_to(self, path: str) -> str | None:
        """Point at *path* by name; None if it is not in the list."""
        i = self.index_of(path)
        return self.select(i) if i is not None else None

    # ------------------------------------------------------------------
    # order
    def set_shuffle(self, on: bool) -> None:
        """Shuffle the rest of the list around the current track.

        Turning it off mid-list leaves you where you are and continues in file
        order from that point, rather than jumping back to the top.
        """
        on = bool(on)
        if on == self.shuffle:
            return
        self.shuffle = on
        if on:
            rest = [i for i in self._order if i != self.index]
            self._rng.shuffle(rest)
            self._order = ([] if self.index is None else [self.index]) + rest
        else:
            cur = self.index or 0
            self._order = list(range(cur, len(self.items))) + \
                         list(range(0, cur)) if self.items else []
            self._ensure_in_order()

    def set_repeat(self, mode: str) -> str:
        self.repeat = mode if mode in REPEAT_MODES else "off"
        return self.repeat

    def cycle_repeat(self) -> str:
        """off -> all -> one -> off, like every other player."""
        self.repeat = REPEAT_MODES[(REPEAT_MODES.index(self.repeat) + 1)
                                   % len(REPEAT_MODES)]
        return self.repeat

    def _insert(self, idx: int) -> None:
        """Place a new track in the visit order.

        In shuffle mode it goes somewhere *ahead* of the cursor, so adding a
        folder while playing does not reorder what has already been heard.
        """
        if not self.shuffle:
            self._order.append(idx)
            return
        lo = self._slot() + 1 if self.index is not None else 0
        self._order.insert(self._rng.randint(lo, len(self._order)), idx)

    def _slot(self) -> int:
        try:
            return self._order.index(self.index)
        except ValueError:
            return 0

    def _ensure_in_order(self) -> None:
        if self.index is not None and self.index not in self._order:
            self._order.insert(self._slot(), self.index)

    def next(self, auto: bool = False) -> str | None:
        """The track to play after the current one; None means "stop here".

        ``auto`` is the end-of-file case, where repeat-one rewinds the same
        track; skipping forward manually always moves on.
        """
        if not self.items:
            return None
        if auto and self.repeat == "one":
            return self.current()
        self._ensure_in_order()
        slot = self._slot() + 1
        if slot >= len(self._order):
            if self.repeat == "all":
                slot = 0
            else:
                self.index = self._order[-1]
                return None
        self.index = self._order[slot]
        return self.current()

    def prev(self) -> str | None:
        """The track before the current one; None means "already at the start"."""
        if not self.items:
            return None
        self._ensure_in_order()
        slot = self._slot() - 1
        if slot < 0:
            if self.repeat == "all":
                slot = len(self._order) - 1
            else:
                self.index = self._order[0]
                return None
        self.index = self._order[slot]
        return self.current()

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Enough to put the list back next time the app starts."""
        return {"items": list(self.items), "index": self.index,
                "shuffle": self.shuffle, "repeat": self.repeat}

    def restore(self, state: dict) -> None:
        self.clear()
        self.items = [str(p) for p in state.get("items", [])]
        self._order = list(range(len(self.items)))
        self.shuffle = bool(state.get("shuffle", False))
        self.repeat = state.get("repeat", "off") if state.get("repeat") in \
            REPEAT_MODES else "off"
        if self.shuffle:
            rest = list(range(len(self.items)))
            self._rng.shuffle(rest)
            self._order = rest
        self.select(state.get("index", 0))
