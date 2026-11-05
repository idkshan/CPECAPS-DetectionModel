"""Generate the figures used in docs/Chapter3 and docs/Chapter4 (PNG, print-ready).

    python scripts/make_figures.py            # writes docs/figures/*.png

All values are read from the code/config, so the figures stay in sync with
the implementation.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import _bootstrap  # noqa: F401,E402
from dms.config import Config  # noqa: E402
from dms.risk.rules import RISK_SETS, UNIT_SETS  # noqa: E402

OUT = Path("docs/figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated)
ORDINAL = {"Low": "#86b6ef", "Moderate": "#2a78d6", "High": "#104281"}  # one-hue ordinal ramp
RISK_ORD = {"NONE": "#86b6ef", "LOW": "#3987e5", "MEDIUM": "#1c5cab", "HIGH": "#0d366b"}
STATUS = {"NONE": "#0ca30c", "LOW": "#fab219", "MEDIUM": "#ec835a", "HIGH": "#d03b3b"}  # reserved status colours
AI_FILL, AI_EDGE = "#cde2fb", "#1c5cab"
RULE_FILL, RULE_EDGE = "#f0efec", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def box(ax, x, y, w, h, text, kind="rule", size=8.5, bold=False):
    fill, edge = (AI_FILL, AI_EDGE) if kind == "ai" else (RULE_FILL, RULE_EDGE) if kind == "rule" else ("white", INK2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.0))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = canvas(9.0, 8.6)
    rows = [  # (y, stage label, [(text, kind)], note)
        (0.855, "Stage 1\nAI\nperception", [("MediaPipe\nFace Landmarker\n(face detector, 478-pt\nmesh, 52 blendshapes)", "ai"),
                                           ("YOLO detector\nYOLOv8 / YOLO11 /\nYOLO26 (phone)", "ai"),
                                           ("Eye-state CNN\n(optional YOLO-cls,\nMRL Eye)", "ai")]),
        (0.665, "Stage 2\nvisual\nfeatures", [("EAR (eyes)\nMAR (mouth)", "rule"), ("Head pose\nyaw / pitch\n(solvePnP)", "rule"),
                                             ("eyeBlink,\njawOpen\n(learned)", "ai"), ("Phone conf.\nnear driver", "rule")]),
        (0.475, "Stage 3\nthreshold\nclassifier", [("Per-driver\ncalibration\n(adaptive baseline)", "rule"),
                                                        ("Evidence\nfusion\n(0.5 = threshold)", "rule"),
                                                        ("Temporal\npersistence\n(reject blinks,\nspeech, glances)", "rule")]),
        (0.285, "Stage 4\nseverity", [("Drowsiness\n(closure,\nPERCLOS)", "rule"), ("Yawning\n(count per\n5 min)", "rule"),
                                      ("Gaze\n(time away,\noff-road %)", "rule"), ("Phone use\n(time\nvisible)", "rule")]),
        (0.095, "Stage 5\ndriver\nrisk", [("Fuzzy blocks\nFatigue F,\nDistraction D,\nContext C", "rule"), ("Overall risk\nR = FIS(F, D, C)", "rule"),
                                         ("Alarm tier\nNONE / LOW /\nMEDIUM / HIGH", "rule")]),
    ]
    h = 0.12
    for y, label, items in rows:
        ax.text(0.005, y + h / 2, label, ha="left", va="center", fontsize=8.5, color=INK, weight="bold")
        n = len(items)
        x0, span, gap = 0.12, 0.70, 0.015
        w = (span - gap * (n - 1)) / n
        for i, (text, kind) in enumerate(items):
            box(ax, x0 + i * (w + gap), y, w, h, text, kind, size=7.6)
    for (y_top, *_), (y_bot, *_) in zip(rows[:-1], rows[1:]):
        arrow(ax, 0.47, y_top, 0.47, y_bot + h + 0.004)
    box(ax, 0.855, 0.095, 0.14, h, "Context\nGPS speed,\nMPU6050 motion,\ntrip duration", "other", size=7.4)
    arrow(ax, 0.855, 0.155, 0.822, 0.155)
    box(ax, 0.855, 0.855, 0.14, h, "Camera frame\n(ESP32-CAM,\n>= 10 FPS)", "other", size=7.4)
    arrow(ax, 0.855, 0.915, 0.822, 0.915)
    ax.add_patch(FancyBboxPatch((0.12, 0.018), 0.03, 0.025, boxstyle="round,pad=0.003", fc=AI_FILL, ec=AI_EDGE))
    ax.text(0.157, 0.03, "learned model (neural network)", va="center", fontsize=8.5, color=INK)
    ax.add_patch(FancyBboxPatch((0.47, 0.018), 0.03, 0.025, boxstyle="round,pad=0.003", fc=RULE_FILL, ec=RULE_EDGE))
    ax.text(0.507, 0.03, "deterministic math / rules / fuzzy logic", va="center", fontsize=8.5, color=INK)
    fig.savefig(OUT / "fig_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_membership():
    cfg = Config()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    x = np.linspace(0, 1, 501)
    for name, pts in UNIT_SETS.items():
        mu = np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
        a.plot(x, mu, color=ORDINAL[name], lw=2)
        peak = {"Low": 0.08, "Moderate": 0.5, "High": 0.9}[name]
        a.text(peak, 1.05, name, ha="center", fontsize=8.5, color=INK)
    arrowprops = dict(arrowstyle="-", color=INK2, lw=0.7)
    for c, mu, lab, xy_text in ((1 / 3, 1 / 6, "crossover 1/3", (0.03, 0.30)), (0.65, 0.25, "crossover 0.65", (0.76, 0.45))):
        a.axvline(c, color=INK2, ls=":", lw=0.9)
        a.annotate(lab, xy=(c, mu), xytext=xy_text, fontsize=7.5, color=INK2, arrowprops=arrowprops)
    a.set(xlim=(0, 1), ylim=(0, 1.15), xlabel="input value (severity, F, D, C)", ylabel="membership mu(x)",
          title="(a) Input and block-output sets (Table 1.7)")
    y = np.linspace(0, 100, 501)
    for name, pts in RISK_SETS.items():
        mu = np.interp(y, [p[0] for p in pts], [p[1] for p in pts])
        b.plot(y, mu, color=RISK_ORD[name], lw=2)
        peak = {"NONE": 6, "LOW": 40, "MEDIUM": 65, "HIGH": 93}[name]
        b.text(peak, 1.05, name, ha="center", fontsize=8.5, color=INK)
    for bound in cfg.risk.tier_bounds:
        b.axvline(bound, color=INK2, ls="--", lw=0.9)
        b.text(bound + 1, 0.93, f"{bound:g}", fontsize=7, color=INK2, va="center")
    b.set(xlim=(0, 100), ylim=(0, 1.15), xlabel="overall risk R (dashed = tier bounds)", ylabel="membership mu(R)", title="(b) Overall-risk output sets and tier bounds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_membership.png", dpi=200)
    plt.close(fig)


def fig_evidence_ramp():
    cfg = Config()
    r = cfg.thresholds.ear_closed_ratio
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ear = np.linspace(0.0, 0.40, 400)
    for i, (label, base) in enumerate((("Driver A (baseline EAR 0.30)", 0.30), ("Driver B, narrower eyes (baseline EAR 0.22)", 0.22))):
        thr = base * r
        ev = np.clip(0.5 * (ear - base) / (thr - base), 0, 1)
        ax.plot(ear, ev, color=SERIES[i], lw=2, label=label)
        ax.plot([thr], [0.5], "o", color=SERIES[i], ms=6, mec="white", mew=1.5)
        xy_text = (0.265, 0.62) if i == 0 else (0.015, 0.30)
        ax.annotate(f"threshold {thr:.3f}", xy=(thr, 0.5), xytext=xy_text, fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.axhline(0.5, color=INK2, ls=":", lw=0.9)
    ax.text(0.395, 0.46, "decision level 0.5", fontsize=7.5, color=INK2, ha="right", va="top")
    ax.set(xlabel="Eye Aspect Ratio (EAR)", ylabel="eye-closure evidence", xlim=(0, 0.4), ylim=(-0.02, 1.05),
           title=f"Evidence ramp: 0 at the driver's baseline, 0.5 at {r:.2f} x baseline")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_evidence_ramp.png", dpi=200)
    plt.close(fig)


def fig_hierarchical_fis():
    fig, ax = canvas(7.2, 3.6)
    inputs = [("eye closure", 0.86), ("yawning", 0.74), ("trip duration", 0.62), ("gaze deviation", 0.46), ("phone use", 0.34),
              ("speed", 0.18), ("erratic motion", 0.06)]
    for text, y in inputs:
        box(ax, 0.0, y, 0.2, 0.09, text, "other", size=8)
    blocks = [("Fatigue FIS\n27 rules -> F", 0.74, [0.86, 0.74, 0.62]), ("Distraction FIS\n9 rules -> D", 0.40, [0.46, 0.34]),
              ("Context FIS\n9 rules -> C", 0.12, [0.18, 0.06])]
    for text, y, srcs in blocks:
        box(ax, 0.33, y, 0.22, 0.12, text, "rule", size=8.5)
        for s in srcs:
            arrow(ax, 0.2, s + 0.045, 0.33, y + 0.06)
        arrow(ax, 0.55, y + 0.06, 0.66, 0.46)
    box(ax, 0.66, 0.40, 0.17, 0.14, "Overall FIS\n27 rules -> R\n(0-100)", "rule", size=8.5)
    arrow(ax, 0.83, 0.47, 0.87, 0.47)
    box(ax, 0.87, 0.40, 0.13, 0.14, "alarm tier\nfrom R", "other", size=8.5)
    ax.text(0.33, 0.97, "72 rules in total (a flat rule base for 7 inputs would need 3^7 = 2,187)", fontsize=8, color=INK2)
    fig.savefig(OUT / "fig_hierarchical_fis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_calibration_flow():
    fig, ax = canvas(7.6, 4.8)
    x, w, h = 0.33, 0.38, 0.12
    steps = [
        (0.86, "Session start / new driver:\nlook ahead with a relaxed face"),
        (0.69, "Collect frames for 5 s\n(face found, no phone)"),
        (0.52, "Medians of EAR, MAR, yaw, pitch,\neyeBlink, jawOpen; IQR of pose"),
        (0.35, "Valid? >= 25 frames, head steady,\neyes open, mouth neutral"),
        (0.13, "CALIBRATED: thresholds set\nrelative to personal baseline"),
    ]
    for y, t in steps:
        box(ax, x, y, w, h, t, "rule", size=8)
    for (y1, _), (y2, _) in zip(steps[:-2], steps[1:-1]):
        arrow(ax, x + w / 2, y1, x + w / 2, y2 + h)
    arrow(ax, x + w / 2, 0.35, x + w / 2, 0.25)
    ax.text(x + w / 2 + 0.015, 0.29, "yes", fontsize=8, color=INK)
    box(ax, 0.0, 0.35, 0.26, h, "no: keep defaults\n(pose limits x1.5)\nand retry", "other", size=7.8)
    arrow(ax, x, 0.41, 0.26, 0.41)
    arrow(ax, 0.13, 0.47, x, 0.75)
    box(ax, 0.76, 0.10, 0.25, 0.18, "While driving:\nslow EMA update\n(tau = 120 s) only when\nalert; clamped near\ncalibrated values", "other", size=7.6)
    arrow(ax, x + w, 0.19, 0.76, 0.19)
    fig.savefig(OUT / "fig_calibration_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ear_mar():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.axis("off")
    # (a) eye: corners p1, p4; upper lid p2, p3; lower lid p6, p5
    th = np.linspace(0, np.pi, 100)
    for sign in (1, -1):
        a.plot(np.cos(th) * 1.0, sign * np.sin(th) * 0.38, color=INK2, lw=1.2)
    a.add_patch(plt.Circle((0, 0), 0.26, fc="#b7d3f6", ec=INK2, lw=0.8))
    pts = {"p1": (-1.0, 0.0), "p2": (-0.35, 0.356), "p3": (0.35, 0.356), "p4": (1.0, 0.0), "p5": (0.35, -0.356), "p6": (-0.35, -0.356)}
    for k, (x, y) in pts.items():
        a.plot(x, y, "o", color=SERIES[0], ms=6, mec="white", mew=1)
        a.text(x + (0.08 if x >= 0 else -0.08) * (1 if abs(y) < 0.1 else 0), y + (0.12 if y > 0.05 else -0.2 if y < -0.05 else 0.1),
               k, ha="center", fontsize=8.5, color=INK)
    for (p, q) in (("p2", "p6"), ("p3", "p5")):
        a.plot(*zip(pts[p], pts[q]), color=SERIES[1], lw=1.2, ls="--")
    a.plot(*zip(pts["p1"], pts["p4"]), color=SERIES[2], lw=1.2, ls="--")
    a.set_title("(a) Eye landmarks for EAR (Eq. 3.6)", fontsize=9)
    a.set_xlim(-1.3, 1.3)
    a.set_ylim(-0.7, 0.7)
    # (b) mouth, open: inner-lip corners q1, q5; upper q2-q4; lower q6-q8
    th = np.linspace(0, np.pi, 100)
    b.plot(np.cos(th) * 1.0, np.sin(th) * 0.35, color=INK2, lw=1.2)
    b.plot(np.cos(th) * 1.0, -np.sin(th) * 0.55, color=INK2, lw=1.2)
    q = {"q1": (-1.0, 0.0), "q2": (-0.5, 0.303), "q3": (0.0, 0.35), "q4": (0.5, 0.303), "q5": (1.0, 0.0),
         "q6": (0.5, -0.476), "q7": (0.0, -0.55), "q8": (-0.5, -0.476)}
    for k, (x, y) in q.items():
        b.plot(x, y, "o", color=SERIES[0], ms=6, mec="white", mew=1)
        dy = 0.12 if y > 0.05 else -0.2 if y < -0.05 else 0.1
        dx = -0.14 if k == "q1" else 0.14 if k == "q5" else 0
        b.text(x + dx, y + dy, k, ha="center", fontsize=8.5, color=INK)
    for (m, n) in (("q2", "q8"), ("q3", "q7"), ("q4", "q6")):
        b.plot(*zip(q[m], q[n]), color=SERIES[1], lw=1.2, ls="--")
    b.plot(*zip(q["q1"], q["q5"]), color=SERIES[2], lw=1.2, ls="--")
    b.set_title("(b) Inner-lip landmarks for MAR (Eq. 3.7)", fontsize=9)
    b.set_xlim(-1.3, 1.3)
    b.set_ylim(-0.85, 0.6)
    fig.text(0.5, 0.1, "orange dashed = vertical distances, green dashed = horizontal distance; "
             "MediaPipe indices: eye 33,160,158,133,153,144 / 362,385,387,263,373,380; mouth 78,81,13,311,308,402,14,178",
             ha="center", fontsize=6.8, color=INK2)
    fig.savefig(OUT / "fig_ear_mar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_ear_mar()
    fig_membership()
    fig_evidence_ramp()
    fig_hierarchical_fis()
    fig_calibration_flow()
    from simulate_scenarios import CAL, SCENARIOS, plot_timeline
    from dms.simulation import ScenarioBuilder, run_scenario

    for key, name in (("phone_4s", "stage_timeline_phone.png"), ("closure_2s", "stage_timeline_microsleep.png")):
        label, build, kw = SCENARIOS[key]
        b = ScenarioBuilder().alert(CAL)
        build(b)
        frames = b.build()
        plot_timeline(frames, run_scenario(frames, **kw), label, str(OUT / name))
    for p in sorted(OUT.glob("*.png")):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
