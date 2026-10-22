"""Shared A4 memo PDF. Headings stay bold; the first line is the title."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HEADINGS = {
    "Analysis plan",
    "Blind check",
    "Channels",
    "Control check",
    "Decision power on the pre-test",
    "Errors",
    "Estimand",
    "Estimates",
    "Harness",
    "Holdout",
    "Lift test",
    "Next experiment.",
    "No feasible lift test on the available units.",
    "Operating characteristics",
    "Path log without the tactic",
    "Pretest-weighted CUPED total",
    "Prior check",
    "Read of the public panel",
    "Recommendation",
    "Recovery check.",
    "Tail, for reading only",
    "What changed after the first run",
    "What would overturn the scale decision",
    "What would overturn this",
    "meridian",
    "pymc",
    "robyn",
}


def write_memo_pdf(text: str, path: Path, chart_path: Path | None = None, *, chart_title: str = "Daily group path") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
        elif raw.startswith("- "):
            lines.extend(textwrap.wrap(raw, width=92, subsequent_indent="  ") or [raw])
        else:
            lines.extend(textwrap.wrap(raw, width=92) or [raw])
    pages: list[list[str]] = []
    page: list[str] = []
    for line in lines:
        page.append(line)
        if len(page) >= 44:
            pages.append(page)
            page = []
    if page:
        pages.append(page)
    with PdfPages(path) as pdf:
        for index, chunk in enumerate(pages):
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.patch.set_facecolor("white")
            y = 0.95
            for row, line in enumerate(chunk):
                heading = line in HEADINGS or (index == 0 and row == 0)
                figure.text(
                    0.08,
                    y,
                    line,
                    fontsize=13 if index == 0 and row == 0 else (11 if heading else 9.5),
                    fontweight="bold" if heading or (index == 0 and row == 0) else "normal",
                    family="DejaVu Sans",
                    va="top",
                    ha="left",
                )
                y -= 0.024 if heading else 0.019
            pdf.savefig(figure)
            plt.close(figure)
        if chart_path is not None and chart_path.is_file():
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.patch.set_facecolor("white")
            figure.text(0.08, 0.94, chart_title, fontsize=11, fontweight="bold", family="DejaVu Sans", va="top")
            axis = figure.add_axes((0.08, 0.22, 0.84, 0.66))
            axis.imshow(plt.imread(chart_path))
            axis.axis("off")
            pdf.savefig(figure)
            plt.close(figure)
