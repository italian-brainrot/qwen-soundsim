"""Playlist widget, playlist panel and a file chooser that takes a typed path.

Everything here starts from the same complaint about the stock file dialog: it
makes you click through a folder tree to reach a file whose path you already
know.  So every entry point in this module takes text -- the path box above the
playlist, the chooser's own path line, Ctrl+V, and drag and drop -- and all of
them go through :func:`~soundsim.engine.player.expand_audio_paths`, which
understands files, folders, globs and pasted multi-line lists alike.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QKeySequence,
                           QPalette)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QVBoxLayout, QWidget,
)

from ..engine.player import (SCAN_LIMIT, expand_audio_paths, iter_audio_files,
                             iter_subdirs)

DIM = "#8a90a0"
BRIGHT = "#e6ebf2"
ACCENT = "#f0c070"


def _clip(text: str, width: int = 60) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------
def apply_dark_theme(target):
    """A dark palette, so the contrast the plots already assume holds everywhere.

    The plots paint themselves (#16181d) whatever the desktop says, and the
    labels are written in light greys -- on a light system theme that is white
    text on white.  The palette is set explicitly instead of leaving it to the
    desktop, and Fusion is asked for because the native GTK style ignores most
    of what a palette says.
    """
    pal = target.palette()
    setc = pal.setColor
    setc(QPalette.Window, QColor("#1e2127"))
    setc(QPalette.WindowText, QColor("#dfe4ec"))
    setc(QPalette.Base, QColor("#16181d"))
    setc(QPalette.AlternateBase, QColor("#1b1f26"))
    setc(QPalette.Text, QColor("#e6ebf2"))
    setc(QPalette.Button, QColor("#2b313b"))
    setc(QPalette.ButtonText, QColor("#dfe4ec"))
    setc(QPalette.Highlight, QColor("#3a6ea5"))
    setc(QPalette.HighlightedText, QColor("#ffffff"))
    setc(QPalette.ToolTipBase, QColor("#23262d"))
    setc(QPalette.ToolTipText, QColor("#e6ebf2"))
    setc(QPalette.PlaceholderText, QColor("#7c8492"))
    setc(QPalette.Link, QColor("#6fb3ff"))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        setc(QPalette.Disabled, role, QColor("#767d89"))
    target.setPalette(pal)
    app = QApplication.instance()
    if app is not None:
        # top-level dialogs take the application palette, not the parent
        # widget's, so a window-only palette leaves file dialogs in the light
        # theme with light-on-dark text everywhere else
        app.setPalette(pal)
        app.setStyle("Fusion")


def _paths_from_text(text: str) -> list[str]:
    """Rows of pasted/dropped text -> candidate paths (not yet validated)."""
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# the list itself
# ---------------------------------------------------------------------------
class PlaylistView(QListWidget):
    """The track list: selectable, filterable, and a drop/paste target.

    Rows carry their playlist index in ``Qt.UserRole``, so a filtered or
    partially hidden list still addresses the right tracks.
    """

    activate_row = Signal(int)          # double-click / Enter: play this row
    paths_dropped = Signal(list)        # raw text or URLs: files and/or folders
    rows_removed = Signal(list)         # playlist indices to delete
    cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setAcceptDrops(True)
        self.setUniformItemSizes(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setMinimumHeight(120)
        self._bold = QFont()
        self._bold.setBold(True)
        self._marked: int | None = None
        self._shown: list[str] = []
        self.itemActivated.connect(lambda it: self.activate_row.emit(
            it.data(Qt.UserRole)))

    # -- mirroring the model -------------------------------------------
    def set_paths(self, paths: list[str]) -> None:
        """Rebuild the rows from the playlist model.

        Cheap when nothing changed: a folder of four thousand tracks is a
        perfectly ordinary thing to have loaded, and the view is re-mirrored
        every time the cursor moves.
        """
        if list(paths) == self._shown:
            return
        sel = self.currentRow()
        self.blockSignals(True)
        self.clear()
        for i, p in enumerate(paths):
            name = os.path.basename(p) or p
            label = f"{i + 1:>4}  {name}"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, i)
            it.setData(Qt.UserRole + 1, label)      # the label without the marker
            it.setToolTip(p)
            it.setForeground(QColor(BRIGHT))
            self.addItem(it)
        if 0 <= sel < self.count():
            self.setCurrentRow(sel)
        self.blockSignals(False)
        self._shown = list(paths)
        self._marked = None

    def mark_current(self, index: int | None) -> None:
        """Point at the playing track: marker, colour, bold, selection.

        Rows are added in model order, so row number and playlist index agree;
        only the two rows that differ are touched.
        """
        if self._marked is not None and self._marked != index:
            it = self.item(self._marked)
            if it is not None:
                it.setFont(QFont())
                it.setForeground(QColor(BRIGHT))
                it.setText(it.data(Qt.UserRole + 1))
        self._marked = index
        if index is None or not (0 <= index < self.count()):
            return
        it = self.item(index)
        it.setFont(self._bold)
        it.setForeground(QColor(ACCENT))
        it.setText("\u25b6 " + it.data(Qt.UserRole + 1))
        self.blockSignals(True)
        self.setCurrentRow(index)
        self.scrollToItem(it, QAbstractItemView.PositionAtCenter)
        self.blockSignals(False)

    def set_filter(self, text: str) -> str:
        """Hide the rows that do not contain *text*; returns the count note."""
        text = text.strip().lower()
        if not text:
            for i in range(self.count()):
                self.item(i).setHidden(False)
            return ""
        shown = 0
        for i in range(self.count()):
            it = self.item(i)
            hit = text in it.text().lower() or text in (it.toolTip() or "").lower()
            it.setHidden(not hit)
            shown += int(hit)
        return f"{shown} / {self.count()} shown"

    def selected_rows(self) -> list[int]:
        return sorted(it.data(Qt.UserRole) for it in self.selectedItems())

    # -- input ----------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            rows = self.selected_rows()
            if rows:
                self.rows_removed.emit(rows)
                return
        if event.matches(QKeySequence.Paste):
            text = QApplication.clipboard().text() or ""
            if text.strip():
                self.paths_dropped.emit(_paths_from_text(text))
                return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = self._menu()
        if menu.isEmpty():
            return
        menu.exec(event.globalPos())
        menu.deleteLater()

    def _menu(self):
        menu = QMenu(self)
        rows = self.selected_rows()
        act = menu.addAction("Play" if len(rows) <= 1 else "Play selected")
        act.setEnabled(bool(rows))
        if rows:
            act.triggered.connect(lambda: self.activate_row.emit(rows[0]))
            rm = menu.addAction(f"Remove {len(rows)} track{'s' if len(rows) > 1 else ''}")
            rm.triggered.connect(lambda: self.rows_removed.emit(rows))
            if len(rows) == 1:
                path = self.item(self.row(self.currentItem())).toolTip()
                folder = os.path.dirname(path)
                menu.addAction("Open containing folder").triggered.connect(
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder)))
        menu.addSeparator()
        clear = menu.addAction("Clear playlist")
        clear.setEnabled(self.count() > 0)
        clear.triggered.connect(self.cleared.emit)
        return menu

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() or md.hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            text = "\n".join(u.toLocalFile() for u in md.urls()
                             if u.isLocalFile()) or md.text()
        else:
            text = md.text()
        if text.strip():
            self.paths_dropped.emit(_paths_from_text(text))
            event.acceptProposedAction()
            return
        super().dropEvent(event)


# ----------------------------------------------------------------------------
# panel: path box + list + the buttons that belong to them
# ---------------------------------------------------------------------------
class PlaylistPanel(QWidget):
    """Everything about *which audio*: add it, filter it, remove it.

    The list lives in a splitter on the main window, so it can be dragged to
    whatever size you want; nothing here clamps its height.
    """

    activate_row = Signal(int)
    add_paths = Signal(list)            # raw strings: files, folders or globs
    remove_rows = Signal(list)
    clear_list = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        head = QHBoxLayout()
        title = QLabel("<b>Audio</b>")
        self.note = QLabel("nothing loaded")
        self.note.setStyleSheet(f"color:{DIM}; font-size:11px;")
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("filter")
        self.filter_box.setClearButtonEnabled(True)
        self.filter_box.setMaximumWidth(200)
        self.filter_box.textChanged.connect(self._filter)
        head.addWidget(title)
        head.addWidget(self.note, 1)
        head.addWidget(self.filter_box)
        lay.addLayout(head)

        paths = QHBoxLayout()
        self.path_box = QLineEdit()
        self.path_box.setPlaceholderText(
            "type or paste a path — file, folder, glob or a list of them; "
            "Enter adds")
        self.path_box.setClearButtonEnabled(True)
        self.path_box.returnPressed.connect(self._submit_path)
        self.btn_add = QPushButton("Choose…")
        self.btn_add.setToolTip("Pick files or paste a path into the chooser")
        self.btn_add.clicked.connect(self._open_chooser)
        self.btn_folder = QPushButton("Folder…")
        self.btn_folder.setToolTip("Add every audio file in a folder "
                                   "(sub-folders included)")
        self.btn_folder.clicked.connect(self._open_folder)
        paths.addWidget(self.path_box, 1)
        paths.addWidget(self.btn_add)
        paths.addWidget(self.btn_folder)
        lay.addLayout(paths)

        self.view = PlaylistView(self)
        self.view.activate_row.connect(self.activate_row)
        self.view.paths_dropped.connect(self.add_paths)
        self.view.rows_removed.connect(self.remove_rows)
        self.view.cleared.connect(self.clear_list)
        lay.addWidget(self.view, 1)

        foot = QHBoxLayout()
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Delete the selected rows (Delete)")
        self.btn_remove.clicked.connect(self._remove)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_list.emit)
        hint = QLabel("double-click or Enter plays · drop files anywhere · "
                      "Ctrl+V pastes paths")
        hint.setStyleSheet(f"color:{DIM}; font-size:11px;")
        foot.addWidget(self.btn_remove)
        foot.addWidget(self.btn_clear)
        foot.addStretch(1)
        foot.addWidget(hint)
        lay.addLayout(foot)

        self.last_dir = os.path.expanduser("~")
        self.recents: list[str] = []
        self._note_base = ""
        self._note_filter = ""

    # -- kept in step with the model ------------------------------------
    def set_paths(self, paths: list[str]) -> None:
        self.view.set_paths(list(paths))
        self._filter(self.filter_box.text())

    def set_current(self, index: int | None) -> None:
        self.view.mark_current(index)

    def set_note(self, text: str) -> None:
        """The app's own summary ("10 tracks · playing 3"); the filter count is
        appended to it instead of replacing it."""
        self._note_base = text
        self._paint_note()

    def _paint_note(self):
        self.note.setText(" · ".join(p for p in (self._note_base,
                                                 self._note_filter) if p))

    def set_context(self, last_dir: str, recents: list[str]) -> None:
        """Where to open the chooser, and what it was recently."""
        if last_dir and os.path.isdir(last_dir):
            self.last_dir = last_dir
        self.recents = [d for d in recents if os.path.isdir(d)]

    def remember(self, path: str) -> None:
        """Note where the last thing added came from."""
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if not d or not os.path.isdir(d):
            return
        if d in self.recents:
            self.recents.remove(d)
        self.recents.insert(0, d)
        del self.recents[8:]
        self.last_dir = d

    # -- actions --------------------------------------------------------
    def _submit_path(self):
        spec = self.path_box.text().strip()
        if not spec:
            return
        self.add_paths.emit([spec])
        self.path_box.clear()

    def _remove(self):
        rows = self.view.selected_rows()
        if rows:
            self.remove_rows.emit(rows)

    def _filter(self, text: str):
        self._note_filter = self.view.set_filter(text)
        self._paint_note()

    def _open_chooser(self):
        dlg = AudioChooser(self, start=self.last_dir, recents=self.recents)
        if dlg.exec() == QDialog.Accepted and dlg.chosen:
            self.recents, self.last_dir = dlg.recents, dlg.cwd
            self.add_paths.emit(dlg.chosen)

    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Add a folder", self.last_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
        if d:
            self.last_dir = d
            self.add_paths.emit([d])


# ---------------------------------------------------------------------------
# the chooser
# ---------------------------------------------------------------------------
class AudioChooser(QDialog):
    """A file chooser built around a path line you can type or paste into.

    The list below is a single-level browser (folders first) so you can still
    walk somewhere you have never been, but the intended route is: paste the
    path, press Enter.  A folder name there means "everything audio inside it".
    """

    def __init__(self, parent=None, start: str | None = None,
                 recents: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add audio")
        self.setAcceptDrops(True)
        self.resize(760, 520)
        self.chosen: list[str] = []
        self.recents = [d for d in (recents or []) if os.path.isdir(d)]
        self.cwd = ""
        self.truncated = False

        lay = QVBoxLayout(self)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "paste or type a path: a file, a folder, a *.glob, or several "
            "lines at once — Enter adds")
        self.path_edit.setClearButtonEnabled(True)
        self.path_edit.returnPressed.connect(self._accept_path)
        top = QHBoxLayout()
        top.addWidget(self.path_edit, 1)
        self.btn_ok = QPushButton("Add")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._add_selected)
        top.addWidget(self.btn_ok)
        lay.addLayout(top)

        nav = QHBoxLayout()
        self.go_home = QPushButton("Home")
        self.go_home.clicked.connect(lambda: self._cd(os.path.expanduser("~")))
        self.go_music = QPushButton("Music")
        self.go_music.clicked.connect(self._go_music)
        self.go_up = QPushButton("Up")
        self.go_up.clicked.connect(lambda: self._cd(os.path.dirname(
            self.cwd.rstrip(os.sep) + os.sep) or os.sep))
        self.recent_box = QComboBox()
        self.recent_box.setToolTip("Folders you have used this session")
        self.recent_box.setMinimumWidth(220)
        self.recent_box.currentIndexChanged.connect(self._from_recent)
        nav.addWidget(self.go_home)
        nav.addWidget(self.go_music)
        nav.addWidget(self.go_up)
        nav.addWidget(self.recent_box, 1)
        lay.addLayout(nav)

        self.dir_label = QLabel()
        self.dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dir_label.setStyleSheet(f"color:{DIM}; font-size:11px;")
        lay.addWidget(self.dir_label)

        self.entries = QListWidget()
        self.entries.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.entries.setAlternatingRowColors(True)
        self.entries.itemActivated.connect(self._open_entry)
        self.entries.itemDoubleClicked.connect(self._open_entry)
        lay.addWidget(self.entries, 1)

        foot = QHBoxLayout()
        self.rec_box = QCheckBox("include sub-folders")
        self.rec_box.setChecked(True)
        self.rec_box.toggled.connect(lambda _: self._refresh())
        self.status = QLabel()
        self.status.setStyleSheet(f"color:{DIM}; font-size:11px;")
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        foot.addWidget(self.rec_box)
        foot.addWidget(self.status, 1)
        foot.addWidget(btn_close)
        lay.addLayout(foot)

        self.path_edit.setFocus()
        for d in self.recents:
            self.recent_box.addItem(d, d)
        self._cd(start or os.path.expanduser("~"))

    # -- browsing -------------------------------------------------------
    def _cd(self, d: str | None):
        d = os.path.abspath(os.path.expanduser(d or ""))
        if not d or not os.path.isdir(d):
            self._say("no such folder")
            return
        self.cwd = d
        if d not in self.recents:
            self.recents.insert(0, d)
            del self.recents[8:]
            cur = self.recent_box.currentData()
            self.recent_box.blockSignals(True)
            self.recent_box.clear()
            for r in self.recents:
                self.recent_box.addItem(r, r)
            if cur in self.recents:
                self.recent_box.setCurrentIndex(self.recents.index(cur))
            self.recent_box.blockSignals(False)
        self._refresh()

    def _from_recent(self, row: int):
        if row >= 0:
            self._cd(self.recent_box.itemData(row))

    def _go_music(self):
        for cand in (os.path.expanduser("~/Music"), "/music",
                     os.environ.get("XDG_MUSIC_DIR", "")):
            if cand and os.path.isdir(cand):
                self._cd(cand)
                return
        self._say("no Music folder here")

    def _refresh(self):
        self.entries.blockSignals(True)
        self.entries.clear()
        recursive = self.rec_box.isChecked()
        for d in iter_subdirs(self.cwd):
            it = QListWidgetItem(os.path.basename(d) + "/")
            it.setData(Qt.UserRole, d)
            it.setForeground(QColor("#8ab8ff"))
            it.setToolTip(d)
            self.entries.addItem(it)
        shown = 0
        for p in iter_audio_files(self.cwd, recursive=recursive,
                                  limit=SCAN_LIMIT + 1):
            label = (os.path.relpath(p, self.cwd) if recursive
                     else os.path.basename(p))
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            self.entries.addItem(it)
            shown += 1
        self.truncated = shown > SCAN_LIMIT
        self.entries.blockSignals(False)
        self._say(f"{min(shown, SCAN_LIMIT)} audio file"
                  f"{'s' if min(shown, SCAN_LIMIT) != 1 else ''} here"
                  f"{' (recursive)' if recursive else ''} · "
                  f"double-click a file to add it")
        self.dir_label.setText(_clip(self.cwd, 90))

    def _say(self, text: str):
        self.status.setText(text)

    def _open_entry(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if os.path.isdir(path):
            self._cd(path)
            return
        self._finish([p for p in
                      (it.data(Qt.UserRole) for it in self.entries.selectedItems())
                      if not os.path.isdir(p)] or [path])

    # -- typed paths ----------------------------------------------------
    def _accept_path(self):
        spec = self.path_edit.text().strip()
        if not spec:
            return
        paths = expand_audio_paths(spec, recursive=self.rec_box.isChecked())
        if paths:
            first = paths[0]
            self._cd(os.path.dirname(first))
            self._finish(paths)
            return
        # a folder with nothing playable in it: go there rather than say nothing
        for tok in _paths_from_text(spec):
            tok = os.path.expanduser(os.path.expandvars(
                tok.strip().strip('"\'')))
            if os.path.isdir(tok):
                self._cd(tok)
                self._say(f"no audio files inside {tok}")
                return
        self._say(f"nothing playable found for: {_clip(spec, 60)}")

    def _add_selected(self):
        files = [it.data(Qt.UserRole) for it in self.entries.selectedItems()
                 if not os.path.isdir(it.data(Qt.UserRole))]
        if files:
            self._finish(files)
        else:
            self._accept_path()

    def _finish(self, paths: list[str]):
        self.chosen = list(paths)
        self.accept()

    # -- drops on the dialog itself ------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        md = event.mimeData()
        text = ("\n".join(u.toLocalFile() for u in md.urls() if u.isLocalFile())
                if md.hasUrls() else md.text())
        if text.strip():
            self.path_edit.setText(text.strip())
            self._accept_path()
            event.acceptProposedAction()
