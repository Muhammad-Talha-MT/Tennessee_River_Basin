"""Shared style of the Section 5 figures: Okabe-Ito colors (one per model, the same in every figure), fonts, CSV location, a label-spacing helper
and an overlap check (fails if text overlaps text, leaves the figure, or sits on a line or dot)."""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "reports" / "section5"
FIG = REPO / "figures"
# Journal style: boxed axes, light grid, framed legends, black text, standard distinguishable colors (as in hydrology papers)
INK, MUTED, LIGHT = "#000000", "#555555", "#E6E6E6"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.8, "axes.edgecolor": "#000000",
                     "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.titlesize": 9, "axes.titleweight": "normal",
                     "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#BBBBBB", "legend.fancybox": False,
                     "legend.fontsize": 7.5, "legend.borderpad": 0.4, "legend.handlelength": 2.2,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
COLOR = {"M0": "#C62828", "M1": "#1F78B4", "M2": "#33A02C", "M4": "#6A3D9A", "LSTM": "#FF7F00", "NWM": "#000000", "OBS": "#000000"}
STYLE = {"M0": "-", "M1": "-", "M2": "-.", "M4": (0, (5, 1.5)), "LSTM": "-", "NWM": "--"}
WIDTH = {m: 1.3 for m in COLOR} | {"M0": 1.8, "NWM": 1.1}
DISPLAY = {"M0": "GNN", "M1": "GNN-noIns", "M2": "GNN-noMP", "M4": "GNN-Instat", "LSTM": "LSTM", "NWM": "NWM"}  # names in figures and tables; M0.. stay as keys in code and CSVs
LABEL = dict(DISPLAY, NWM="NWM (reference)")


def typo(text):
    """Typography of text drawn in figures (outside $...$): a hyphen between digits is an en dash (ranges: 1-5 -> 1-5 with an en dash), a hyphen standing
    alone between spaces (a difference: GNN - LSTM) or in front of a number (a negative value) is a true minus sign; hyphens inside names (GNN-noIns) stay."""
    parts = text.split("$")
    for i in range(0, len(parts), 2):  # even parts are outside math
        t = re.sub(r"(?<=\d)-(?=\d)", "\u2013", parts[i])
        t = re.sub(r"(?<=\s)-(?=\s)", "\u2212", t)
        t = re.sub(r"(?<![\w\u2013\u2212-])-(?=[\d.])", "\u2212", t)
        parts[i] = t
    return "$".join(parts)


_set_text = mtext.Text.set_text


def _typo_set_text(self, s):
    _set_text(self, typo(s) if isinstance(s, str) else s)


mtext.Text.set_text = _typo_set_text  # every text drawn by a Section 5 figure goes through typo()


def pretty(text):
    """Replace the model keys M0, M1, M2, M4 in a display string by their names (GNN, GNN-noIns, GNN-noMP, GNN-Instat)."""
    return re.sub(r"\bM([0124])\b", lambda m: DISPLAY["M" + m.group(1)], text)

NESTED, HEADWATER = "#C62828", "#1F78B4"
OBS = "#000000"
BAND = "#EFEFEF"
DIVERGING = "PuOr"  # map scale for a difference of two models: orange = second model higher (LSTM orange), purple = first model higher; one setting for every map
PROTOCOLS = ("gauged", "heldout", "blocked")
PNAME = {"gauged": "Gauged", "heldout": "Random hold-out", "blocked": "Blocked basins"}
PLONG = {"Gauged": "gauged", "Random hold-out": "heldout", "Blocked basins": "blocked"}


def read(name):
    return pd.read_csv(CSV / name)


def spread(values, gap):
    """Positions for labels at ``values`` with at least ``gap`` between neighbours (order kept, centred on the data)."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    out = np.empty(len(pos))
    out[order] = pos
    return out


def check(fig, axes, lines=True):
    """Text-text overlaps, text outside the figure, and (with ``lines``) data text on lines or dots of its axes."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []
    for ax in axes:
        items += [(t, ax, True) for t in ax.texts if t.get_text() and t.get_visible()]
        items += [(t, ax, False) for t in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title]
                  if t.get_text() and t.get_visible()]
    items += [(t, None, False) for t in fig.texts if t.get_text()]
    bbs = [t.get_window_extent(rend) for t, _, _ in items]
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if bbs[i].overlaps(bbs[j]):
                bad.append(("text-text", items[i][0].get_text()[:24], items[j][0].get_text()[:24]))
    fb = fig.bbox
    bad += [("outside", t.get_text()[:24], "") for (t, _, _), bb in zip(items, bbs) if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1]
    if lines:
        for (t, ax, data), bb in zip(items, bbs):
            if not data or getattr(t, "_skip_line_check", False):
                continue
            for ln in ax.lines:
                xy = ln.get_xydata()
                if len(xy) < 2 or getattr(ln, "_is_leader", False):
                    continue
                pts = ln.get_transform().transform(xy)
                pts = pts[np.isfinite(pts).all(1)]
                if len(pts) < 2:
                    continue
                seg = np.concatenate([np.linspace(pts[k], pts[k + 1], 12) for k in range(len(pts) - 1)])
                if ((seg[:, 0] > bb.x0) & (seg[:, 0] < bb.x1) & (seg[:, 1] > bb.y0) & (seg[:, 1] < bb.y1)).any():
                    bad.append(("text-line", t.get_text()[:24], str(ln.get_color())))
            for col in ax.collections:
                off = col.get_offsets()
                if len(off) == 0 or not hasattr(off, "shape") or off.shape[-1] != 2:
                    continue
                o = ax.transData.transform(off)
                if ((o[:, 0] > bb.x0) & (o[:, 0] < bb.x1) & (o[:, 1] > bb.y0) & (o[:, 1] < bb.y1)).any():
                    bad.append(("text-dot", t.get_text()[:24], ""))
    return bad


def save(fig, name, axes, lines=True):
    bad = check(fig, axes, lines)
    print(name, "overlaps:", bad or "none")
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white", **kw)
    if bad:
        sys.exit(f"{name}: overlap found")
