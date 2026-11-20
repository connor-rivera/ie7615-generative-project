"""Summarize the sweeps and choose the 'best settings' by a fixed, written-down rule.

Rule (agreed 2026-11-20, applied to mean CLIP score over the 15 sweep captions):
  guidance: the LOWEST value within 1% of the best guidance mean (avoids the over-saturated end
            when higher guidance buys almost nothing)
  steps:    the FEWEST steps within 1% of the 50-step mean (speed for the 1,000-image runs)
  sampler:  the highest mean

Each sweep moved one knob from the baseline (DDIM, 30, 7.5), so the combination is new.
src.sweep's 'best' condition generates it on the 15 captions; confirm() accepts it only if its
mean CLIP is within 1% of the baseline's, otherwise falls back to the baseline.

  python -m src.select_settings          # summary + plots + best_settings.json (proposed)
  python -m src.select_settings confirm  # after `python -m src.sweep best`
"""
import csv
import json
import sys
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sweep import BASELINE, LOG_PATH, OUT_DIR

BEST_PATH = OUT_DIR / "best_settings.json"
SUMMARY_PATH = OUT_DIR / "sweep_summary.csv"
TOL = 0.01


def load_rows():
    return [r for r in csv.DictReader(open(LOG_PATH)) if r["clip_score"]]


def settings_key(r):
    return (r["sampler"], int(r["steps"]), float(r["guidance"]))


def by_settings(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[settings_key(r)].append(r)
    return groups


def stats(group):
    return {"n": len(group), "clip": mean(float(r["clip_score"]) for r in group),
            "sec": mean(float(r["seconds"]) for r in group)}


def summarize():
    groups = by_settings(load_rows())
    base = (BASELINE["sampler"], BASELINE["steps"], float(BASELINE["guidance"]))
    sweeps = {
        "guidance": sorted((k[2], stats(g)) for k, g in groups.items()
                           if k[0] == base[0] and k[1] == base[1]),
        "sampler": sorted((k[0], stats(g)) for k, g in groups.items()
                          if k[1] == base[1] and k[2] == base[2]),
        "steps": sorted((k[1], stats(g)) for k, g in groups.items()
                        if k[0] == base[0] and k[2] == base[2]),
    }
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["knob", "value", "n_images", "mean_clip_score", "mean_seconds"])
        for knob, vals in sweeps.items():
            for v, s in vals:
                w.writerow([knob, v, s["n"], f"{s['clip']:.2f}", f"{s['sec']:.2f}"])

    plots = OUT_DIR / "plots"
    plots.mkdir(exist_ok=True)
    for knob, ylab, key in [("guidance", "mean CLIP score", "clip"),
                            ("steps", "mean CLIP score", "clip"),
                            ("steps", "seconds per image", "sec")]:
        xs = [v for v, _ in sweeps[knob]]
        ys = [s[key] for _, s in sweeps[knob]]
        plt.figure(figsize=(5, 3.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel(knob)
        plt.ylabel(ylab)
        plt.title(f"{ylab} vs {knob} (15 held-out captions)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / f"{key}_vs_{knob}.png", dpi=150)
        plt.close()
    return sweeps


def choose(sweeps):
    g = sweeps["guidance"]
    g_best = max(s["clip"] for _, s in g)
    guidance = min(v for v, s in g if s["clip"] >= g_best * (1 - TOL))

    st = dict(sweeps["steps"])
    ref = st[max(st)]["clip"]
    steps = min(v for v, s in st.items() if s["clip"] >= ref * (1 - TOL))

    sampler = max(sweeps["sampler"], key=lambda vs: vs[1]["clip"])[0]

    why = {
        "guidance": f"best mean {g_best:.2f}; lowest value within 1% = {guidance}",
        "steps": f"50-step mean {ref:.2f}; fewest steps within 1% = {steps}",
        "sampler": "highest mean: " + ", ".join(f"{v} {s['clip']:.2f}"
                                               for v, s in sweeps["sampler"]),
    }
    return {"sampler": sampler, "steps": steps, "guidance": guidance}, why


def main():
    sweeps = summarize()
    best, why = choose(sweeps)
    BEST_PATH.write_text(json.dumps({"status": "proposed (needs confirm)", "settings": best,
                                     "baseline": BASELINE, "rule": __doc__.split("Rule")[1]
                                     .split("Each sweep")[0].strip(), "why": why}, indent=1))
    for knob, vals in sweeps.items():
        print(knob, "  ".join(f"{v}: {s['clip']:.2f} ({s['sec']:.1f}s)" for v, s in vals))
    print("proposed:", best)
    for k, v in why.items():
        print(f"  {k}: {v}")


def confirm():
    d = json.loads(BEST_PATH.read_text())
    best = d["settings"]
    groups = by_settings(load_rows())
    base = stats(groups[(BASELINE["sampler"], BASELINE["steps"], float(BASELINE["guidance"]))])
    combo = groups.get((best["sampler"], int(best["steps"]), float(best["guidance"])))
    if not combo:
        sys.exit("Combination not generated yet: run `python -m src.sweep best` first.")
    c = stats(combo)
    ok = c["clip"] >= base["clip"] * (1 - TOL)
    d["confirm"] = {"combo_clip": round(c["clip"], 2), "combo_sec": round(c["sec"], 2),
                    "baseline_clip": round(base["clip"], 2), "baseline_sec": round(base["sec"], 2),
                    "accepted": ok}
    if not ok:
        d["settings"] = dict(BASELINE)
    d["status"] = "confirmed" if ok else "fell back to baseline (combo scored >1% lower)"
    BEST_PATH.write_text(json.dumps(d, indent=1))
    print(d["status"], d["confirm"], "->", d["settings"])


if __name__ == "__main__":
    confirm() if sys.argv[1:] == ["confirm"] else main()
