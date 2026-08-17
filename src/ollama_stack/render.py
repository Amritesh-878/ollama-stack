"""Terminal tidying for model output: markdown pipes and asterisks are noise in a shell."""

from __future__ import annotations

from collections.abc import Callable

SEPARATOR = set(":- ")
GAP = "  "


def is_row(line: str) -> bool:
    return line.strip().startswith("|")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    """The |:---|---| line under a header carries no content, only alignment."""
    return bool(cells) and all(cell and set(cell) <= SEPARATOR for cell in cells)


def table(rows: list[str]) -> str:
    """Aligned columns instead of raw pipes, which is the whole reason this module exists."""
    grid = [_cells(row) for row in rows]
    body = [cells for cells in grid if not _is_separator(cells)]
    if not body:
        return ""
    columns = max(len(cells) for cells in body)
    padded = [cells + [""] * (columns - len(cells)) for cells in body]
    widths = [max(len(row[index]) for row in padded) for index in range(columns)]
    lines = [
        GAP.join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in padded
    ]
    if len(lines) > 1:
        lines.insert(1, GAP.join("-" * width for width in widths))
    return "\n".join(lines)


class Pretty:
    """Streams prose through a character at a time, and holds only table rows back to align them."""

    def __init__(self, out: Callable[[str], object]) -> None:
        self._out = out
        self._line = ""
        self._prose = False
        self._star = False
        self._rows: list[str] = []

    def _emit(self, text: str) -> None:
        if text:
            self._out(text)

    def _drop_bold(self, char: str) -> str:
        """`**` is markup a terminal cannot render, so it never reaches the screen."""
        if self._star:
            self._star = False
            return "" if char == "*" else "*" + char
        if char == "*":
            self._star = True
            return ""
        return char

    def _flush_rows(self) -> None:
        if not self._rows:
            return
        rendered = table(self._rows)
        self._rows = []
        if rendered:
            self._emit(rendered + "\n")

    def _end_line(self) -> None:
        if is_row(self._line):
            self._rows.append(self._line)
        else:
            self._flush_rows()
            if not self._prose:
                self._emit(self._line)
            self._emit("\n")
        self._line = ""
        self._prose = False

    def _release_star(self) -> None:
        """A lone asterisk was held waiting for its pair, so it is real text after all."""
        if not self._star:
            return
        self._star = False
        if self._prose:
            self._emit("*")
        else:
            self._line += "*"

    def write(self, piece: str) -> None:
        for raw in piece:
            if raw == "\n":
                self._release_star()
                self._end_line()
                continue
            char = self._drop_bold(raw)
            if not char:
                continue
            if self._prose:
                self._emit(char)
                continue
            self._line += char
            # A line is prose the moment its first real character is not a pipe.
            if self._line.strip() and not is_row(self._line):
                self._prose = True
                self._emit(self._line)

    def close(self) -> None:
        """The caller owns the final newline, so a half-finished line is emitted without one."""
        self._release_star()
        self._flush_rows()
        if self._line and not self._prose:
            self._emit(self._line)
        self._line = ""
