"""Show how Stages 3-5 react to typical driver behaviours (no camera needed).

    python scripts/simulate_scenarios.py                    # table of scenarios
    python scripts/simulate_scenarios.py --plot docs/figures/stage_timeline_phone.png --scenario phone_4s

Useful for explaining the threshold/fuzzy logic and for checking the effect
of a config change before a field test (``--config configs/my.yaml``).
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from dms.config import load_config
from dms.simulation import ScenarioBuilder, max_tier, run_scenario

CAL = 8.0


def _repeat(b, n, fn):
    for _ in range(n):
        fn(b)
    return b


SCENARIOS = {
    "alert_2min": ("Alert driver, 2 minutes", lambda b: b.alert(120), {}),
    "blinking": ("Normal blinking", lambda b: _repeat(b, 20, lambda x: x.alert(3.0).blink()), {}),
    "talking": ("Talking for 30 s", lambda b: b.talk(30).alert(5), {}),
    "yawn_1": ("One yawn", lambda b: b.yawn(3).alert(10), {}),
    "yawn_3": ("Three yawns in ~70 s", lambda b: _repeat(b, 3, lambda x: x.yawn(3).alert(20)), {}),
    "closure_0.6s": ("Eyes closed 0.6 s", lambda b: b.eyes_closed(0.6).alert(8), {}),
    "closure_1s": ("Eyes closed 1.0 s", lambda b: b.eyes_closed(1.0).alert(8), {}),
    "closure_2s": ("Eyes closed 2.0 s (microsleep)", lambda b: b.eyes_closed(2.0).alert(8), {}),
    "glance_1.5s": ("Mirror glance 1.5 s", lambda b: b.look_away(1.5).alert(8), {}),
    "away_2.5s": ("Looking away 2.5 s", lambda b: b.look_away(2.5).alert(8), {}),
    "away_4s": ("Looking away 4 s", lambda b: b.look_away(4).alert(8), {}),
    "down_4s": ("Looking down 4 s", lambda b: b.look_down(4).alert(8), {}),
    "phone_2.5s": ("Phone visible 2.5 s", lambda b: b.phone(2.5).alert(8), {}),
    "phone_4s": ("Phone use 4 s", lambda b: b.phone(4).alert(8), {}),
    "phone_4s_stopped": ("Phone use 4 s, vehicle stopped", lambda b: b.phone(4).alert(8), {"speed_kmh": 0.0}),
    "face_lost_4s": ("Head turned far, face lost 4 s", lambda b: b.turn_away_and_lose_face(4).alert(8), {}),
    "camera_covered_4s": ("Camera covered 4 s (facing forward)", lambda b: b.face_lost(4).alert(8), {}),
    "away_twice_6s": ("Looking away 6 s, twice (alert must stop)", lambda b: b.look_away(6).alert(2).look_away(6).alert(30), {}),
    "harsh_brake": ("Harsh braking 0.5 g, alert driver", lambda b: b.alert(15), {"motion_at": (CAL, CAL + 1.0, 0.5)}),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenario", default=None, choices=sorted(SCENARIOS))
    ap.add_argument("--plot", default=None, help="save a stage-by-stage timeline PNG of --scenario")
    args = ap.parse_args()
    cfg = load_config(args.config)

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    print(f"{'scenario':38s} {'max tier':8s}  alarms (time after onset: tier)")
    for key in names:
        label, build, kw = SCENARIOS[key]
        b = ScenarioBuilder().alert(CAL)
        build(b)
        frames = b.build()
        out = run_scenario(frames, cfg, **kw)
        alarms = [f"{a.t - CAL:+.1f}s:{a.event} {a.tier}{'+push' if a.push else ''}" for o in out for a in o.alerts
                  if a.event in ("start", "escalate", "deescalate", "stop", "push")]
        shown = ", ".join(alarms[:6]) + (f" ... ({len(alarms)} total)" if len(alarms) > 6 else "")
        print(f"{label:38s} {max_tier(out, CAL, frames):8s}  {shown or '-'}")
        if args.plot:
            plot_timeline(frames, out, label, args.plot)
            print(f"saved {args.plot}")
    return 0


def plot_timeline(frames, out, title, path):
    """Stage-by-stage response. One y-axis per panel; colour follows the behaviour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#d9d8d4"
    colors = {"drowsiness": "#2a78d6", "yawning": "#eb6834", "gaze_deviation": "#1baf7a", "phone_usage": "#eda100"}
    status = {"NONE": "#0ca30c", "LOW": "#fab219", "MEDIUM": "#ec835a", "HIGH": "#d03b3b"}
    tiers = ("NONE", "LOW", "MEDIUM", "HIGH")
    t = [f.t - CAL for f in frames]
    fig, axes = plt.subplots(5, 1, figsize=(8, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2.5, 3, 1.4]})
    for ax in axes:
        ax.grid(True, color=grid, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0]
    ev = [("eyes closed", "drowsiness", [o.states.eye_evidence or 0 for o in out]),
          ("mouth open", "yawning", [o.states.mouth_evidence or 0 for o in out]),
          ("head away", "gaze_deviation", [o.states.pose_evidence or 0 for o in out]),
          ("phone visible (YOLO)", "phone_usage", [1.0 if o.states.phone_visible else 0.0 for o in out])]
    for label, key, ys in ev:
        ax.plot(t, ys, color=colors[key], lw=1.6, label=label)
    ax.axhline(0.5, color=ink2, ls="--", lw=0.8)
    ax.set_ylabel("Stage 3\nevidence")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    ax = axes[1]
    for key in colors:
        ax.plot(t, [getattr(o.severity, key).score for o in out], color=colors[key], lw=1.6, label=key.replace("_", " "))
    for y, lab in ((1 / 3, "MODERATE"), (0.65, "SEVERE")):
        ax.axhline(y, color=ink2, ls=":", lw=0.8)
        ax.text(-2.9, y + 0.02, lab, fontsize=6.5, color=ink2)
    ax.set_ylabel("Stage 4\nseverity")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    ax = axes[2]
    for label, ys, ls in (("F fatigue", [o.risk.fatigue for o in out], "-"), ("D distraction", [o.risk.distraction for o in out], "--"),
                          ("C context", [o.risk.context for o in out], ":")):
        ax.plot(t, ys, color=ink, ls=ls, lw=1.4, label=label)
    ax.set_ylabel("Stage 5\nF, D, C")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)

    ax = axes[3]
    bounds = [0, 25, 52.5, 77.5, 100]
    for i, tier in enumerate(tiers):
        ax.axhspan(bounds[i], bounds[i + 1], color=status[tier], alpha=0.12, lw=0)
        ax.text(t[-1], (bounds[i] + bounds[i + 1]) / 2, f" {tier}", fontsize=7, color=ink2, va="center")
    ax.plot(t, [o.risk.risk for o in out], color=ink, lw=1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Stage 5\nrisk R")

    ax = axes[4]
    ax.step(t, [tiers.index(o.active_tier) for o in out], where="post", color=ink, lw=1.6)
    ax.set_yticks(range(4), tiers, fontsize=7)
    ax.set_ylim(-0.3, 3.3)
    ax.set_ylabel("alarm")
    ax.set_xlabel("time since behaviour onset (s)")
    ax.set_xlim(-3, t[-1])
    fig.suptitle(title, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
