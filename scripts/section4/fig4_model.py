"""Figure 4: model schematic (full page width 7 in, under 3.5 in high). Names follow reports/section4_facts.md.
(a) information flow: inputs -> reach encoder -> message passing -> routing -> gauge insertion -> forecast. The gauge head (section 4 of the facts
sheet: it supplies the gauge forecast that the insertion uses after the issue) is the small neutral block above the chain. The three network
mechanisms are colored boxes; every other block is neutral.
(b) one example network, three copies (7 reaches: two headwater reaches joining a main stem; an upstream gauge, filled circle; a withheld reach below
it, open circle). Each copy draws what one mechanism does; the mechanism keeps its color from (a).
Run: python scripts/section4/fig4_model.py   Writes figures/fig4_model.pdf and .png (300 dpi). Fails if text overlaps text.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
TEXT, RIVER, LINE = "#1F1F1F", "#CFCFCF", "#7A7A7A"
NEUTRAL_FILL, NEUTRAL_EDGE = "#F0F0F0", "#8A8A8A"
MECH = {  # Okabe-Ito: one color per mechanism (edge and arrows), light fill
    "mp": dict(edge="#0072B2", fill="#D5E8F4"),
    "route": dict(edge="#E69F00", fill="#FBEBC8"),
    "insert": dict(edge="#CC79A7", fill="#F5E1EC"),
}
W, HA, HB = 7.0, 1.80, 1.62
fig = plt.figure(figsize=(W, HA + HB))
texts = []


def label(ax, x, y, s, **kw):
    t = ax.text(x, y, s, color=kw.pop("color", TEXT), **kw)
    texts.append(t)
    return t


def box(ax, x, y, w, h, s, fill=NEUTRAL_FILL, edge=NEUTRAL_EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec=edge, lw=lw, zorder=2))
    label(ax, x + w / 2, y + h / 2, s, ha="center", va="center", linespacing=1.15, zorder=3)


def arrow(ax, a, b, color=LINE, lw=1.3, rad=0.0, style="-|>", scale=9, shrink=0.0, z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=scale, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink, zorder=z))


def line(ax, xs, ys, color=LINE, lw=1.3):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=1)


# ---------------------------------------------------------------------------------------------------------------- (a) information flow
ax = fig.add_axes([0, HB / (HA + HB), 1, HA / (HA + HB)])
ax.set_xlim(0, W)
ax.set_ylim(0, HA)
ax.axis("off")
label(ax, 0.02, HA - 0.02, "(a) Information flow", fontweight="bold", ha="left", va="top")
IW, IH = 1.46, 0.30
names = ["Observed weather", "Forecast precipitation", "Static attributes", "Gauge observations"]
iy = [1.12, 0.77, 0.42, 0.07]
for s, y in zip(names, iy):
    box(ax, 0.02, y + 0.02, IW, IH, s)
cy, ch = 0.65, 0.64  # chain centre line and box height
pos, x = {}, 1.86
for key, w in (("enc", 0.78), ("mp", 0.86), ("route", 0.98), ("insert", 0.80), ("out", 0.70)):
    pos[key] = (x, w)
    x += w + 0.24
y0 = cy - ch / 2
box(ax, *pos["enc"][:1], y0, pos["enc"][1], ch, "Reach\nencoder")
box(ax, pos["mp"][0], y0, pos["mp"][1], ch, "Message\npassing", **MECH["mp"], lw=1.6)
box(ax, pos["route"][0], y0, pos["route"][1], ch, "Routing\n(Manning-type)", **MECH["route"], lw=1.6)
box(ax, pos["insert"][0], y0, pos["insert"][1], ch, "Gauge\ninsertion", **MECH["insert"], lw=1.6)
box(ax, pos["out"][0], y0, pos["out"][1], ch, "Forecast", lw=1.2)
# inputs -> encoder: a collector line (neutral)
ex = pos["enc"][0] - 0.13
for y in iy:
    line(ax, [0.02 + IW, ex], [y + 0.02 + IH / 2] * 2)
line(ax, [ex, ex], [iy[-1] + 0.02 + IH / 2, iy[0] + 0.02 + IH / 2])
arrow(ax, (ex, cy), (pos["enc"][0], cy))
# chain arrows
order = ["enc", "mp", "route", "insert", "out"]
for a, b in zip(order[:-1], order[1:]):
    arrow(ax, (pos[a][0] + pos[a][1], cy), (pos[b][0], cy))
# gauge observations -> insertion (insertion color: the observation enters the routed flow here)
gy = iy[-1] + 0.02 + IH / 2
ix = pos["insert"][0] + pos["insert"][1] / 2
line(ax, [0.02 + IW, ix], [gy, gy], color=MECH["insert"]["edge"], lw=1.6)
arrow(ax, (ix, gy), (ix, y0), color=MECH["insert"]["edge"], lw=1.6)
# gauge head (neutral): fed by routing, feeds the insertion (forecast after the issue) and the forecast at visible gauges
hx, hw, hy, hh = pos["route"][0] + 0.55, 0.78, y0 + ch + 0.22, 0.30
box(ax, hx, hy, hw, hh, "Gauge head")
arrow(ax, (pos["route"][0] + pos["route"][1] / 2, y0 + ch), (hx + 0.12, hy), rad=0.0)
arrow(ax, (hx + hw - 0.12, hy), (ix, y0 + ch), rad=0.0)

# ---------------------------------------------------------------------------------------------------------------- (b) one example network
axb_title = fig.add_axes([0, (HB - 0.25) / (HA + HB), 1, 0.25 / (HA + HB)])
axb_title.set_xlim(0, W)
axb_title.set_ylim(0, 0.25)
axb_title.axis("off")
label(axb_title, 0.02, 0.22, "(b) One small example network", fontweight="bold", ha="left", va="top")

AW, AH = 2.2, 1.33
stem = [0.52 + 0.31 * k for k in range(6)]  # main stem nodes
Y = 0.62
reaches = {  # name: (start, end)
    "h1": ((0.06, 1.00), (stem[0], Y)),
    "h2": ((0.06, 0.24), (stem[0], Y)),
    **{f"m{k + 1}": ((stem[k], Y), (stem[k + 1], Y)) for k in range(5)},
}
mid = {k: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for k, (a, b) in reaches.items()}
GAUGE, WITHHELD = "m2", "m5"  # upstream gauge reach and withheld reach (the last reach of the main stem)
neighbours = [("h1", "m1"), ("h2", "m1"), ("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("m4", "m5")]
axes_b = []
for i, (key, title) in enumerate((("mp", "Message passing"), ("route", "Routing"), ("insert", "Gauge insertion"))):
    a = fig.add_axes([i * (AW + 0.2) / W, 0.0, AW / W, AH / (HA + HB)])
    a.set_xlim(0, AW)
    a.set_ylim(0, AH)
    a.axis("off")
    axes_b.append(a)
    c = MECH[key]["edge"]
    a.add_patch(FancyBboxPatch((0.45, AH - 0.27), AW - 0.9, 0.25, boxstyle="round,pad=0,rounding_size=0.05", fc=MECH[key]["fill"], ec=c, lw=1.6, zorder=2))
    label(a, AW / 2, AH - 0.145, title, ha="center", va="center", zorder=3)
    for k, (p, q) in reaches.items():  # the river: light gray reaches
        a.plot([p[0], q[0]], [p[1], q[1]], color=RIVER, lw=5, solid_capstyle="butt", zorder=1)
    for x_, y_ in [reaches["h1"][0], reaches["h2"][0]] + [(s, Y) for s in stem]:
        a.plot([x_], [y_], "o", ms=2.5, color="#9A9A9A", zorder=2)
    if key == "mp":  # exchange between the states of neighbouring reaches (reach centres), both directions
        for u, v in neighbours:
            arrow(a, mid[u], mid[v], color=c, lw=1.5, style="<|-|>", scale=5.5, shrink=3.5)
        for k_, m_ in mid.items():
            if k_ not in (GAUGE, WITHHELD):
                a.plot(*m_, "o", ms=4, color=c, zorder=5)
    elif key == "route":  # lateral inflow moves along each reach, in flow direction (drawn just beside the reach)
        side = {"h1": (0.06, 0.075), "h2": (0.06, -0.075)}
        for k_, (p_, q_) in reaches.items():
            dx, dy = q_[0] - p_[0], q_[1] - p_[1]
            ox, oy = side.get(k_, (0.0, 0.11))
            arrow(a, (p_[0] + 0.15 * dx + ox, p_[1] + 0.15 * dy + oy), (p_[0] + 0.85 * dx + ox, p_[1] + 0.85 * dy + oy), color=c, lw=1.8, scale=9)
    else:  # the observation at the gauge reach is added to every reach below it
        g = mid[GAUGE]
        for k in ("m3", "m4", "m5"):
            arrow(a, (g[0], g[1] + 0.07), (mid[k][0], mid[k][1] + 0.09), color=c, lw=1.5, rad={"m3": -0.55, "m4": -0.45, "m5": -0.4}[k], scale=8, shrink=2)
    # markers: filled circle = gauge, open circle = withheld reach
    a.plot(*mid[GAUGE], "o", ms=7, color="black", zorder=6)
    a.plot(*mid[WITHHELD], "o", ms=7, mfc="white", mec="black", mew=1.4, zorder=6)
    label(a, mid[GAUGE][0], 0.30, "gauge", ha="center", va="center")
    label(a, mid[WITHHELD][0], 0.30, "withheld", ha="center", va="center")

# ---------------------------------------------------------------------------------------------------------------- checks and output
fig.canvas.draw()
rend = fig.canvas.get_renderer()
live = [t for t in texts if t.get_text()]
bbs = [t.get_window_extent(rend) for t in live]
bad = [(live[i].get_text(), live[j].get_text()) for i in range(len(live)) for j in range(i + 1, len(live)) if bbs[i].overlaps(bbs[j])]
fb = fig.bbox
bad += [(t.get_text(), "outside figure") for t, bb in zip(live, bbs) if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1]
print("text overlaps:", bad or "none")
OUT.mkdir(exist_ok=True)
for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
    fig.savefig(OUT / f"fig4_model.{ext}", facecolor="white", **kw)
print("size in:", fig.get_size_inches())
if bad:
    sys.exit("text overlap found")
