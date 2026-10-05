#!/usr/bin/env python
"""Overlay the learning curves of two runs -- e.g. the hand-built pathway
against the whole MaleCNS connectome."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from flydrive import viz

PANELS = [
    ("ep_distance", "distance per episode (m)"),
    ("crash_rate", "crash rate"),
    ("abs_lateral", "|lane offset| (m)"),
    ("speed", "mean speed (m/s)"),
    ("ep_return", "episode return"),
    ("explained_var", "value explained var"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under runs/")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="runs/comparison.png")
    ap.add_argument("--title", default="connectome structure vs. the whole connectome")
    args = ap.parse_args()

    hists, labels = [], []
    for i, r in enumerate(args.runs):
        path = r if r.endswith(".json") else os.path.join(r, "history.json")
        with open(path) as f:
            hists.append(json.load(f))
        labels.append(args.labels[i] if args.labels and i < len(args.labels)
                      else os.path.basename(r.rstrip("/")))

    colours = [viz.ACCENT, viz.WARM, viz.GOOD, viz.VIOLET]
    fig = viz.new_figure(15, 7.4)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.26,
                          left=0.055, right=0.985, top=0.84, bottom=0.10)

    for i, (key, label) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        viz.style_axes(ax, label)
        for h, lab, col in zip(hists, labels, colours):
            steps = np.asarray(h["step"], float) / 1e3
            y = np.asarray(h.get(key, []), float)
            n = min(len(y), len(steps))
            if n < 2:
                continue
            x, y = steps[:n], y[:n]
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            ax.plot(x[ok], y[ok], color=col, lw=0.6, alpha=0.25)
            ax.plot(x[ok], viz.smooth(y[ok]), color=col, lw=1.8, label=lab)
        if i == 0:
            ax.legend(fontsize=8, facecolor=viz.PANEL, edgecolor=viz.GRID,
                      labelcolor=viz.FG, framealpha=0.9, loc="lower right")
        if i // 3 == 1:
            ax.set_xlabel("env steps (k)", color=viz.MUTED, fontsize=7)

    fig.suptitle(args.title, color=viz.FG, fontsize=14, x=0.055, ha="left", y=0.965)
    sub = "   |   ".join(
        f"{lab}: {int(h['step'][-1]):,} steps" for h, lab in zip(hists, labels))
    fig.text(0.055, 0.905, sub, color=viz.MUTED, fontsize=9, ha="left")
    fig.savefig(args.out, dpi=105, facecolor=viz.BG)
    print("wrote", args.out)

    print()
    print(f"{'metric':<24}" + "".join(f"{l:>18}" for l in labels))
    print("-" * (24 + 18 * len(labels)))
    for key, label in PANELS:
        row = f"{label:<24}"
        for h in hists:
            y = np.asarray(h.get(key, []), float)
            y = y[np.isfinite(y)]
            row += f"{(y[-10:].mean() if len(y) else float('nan')):>18.3f}"
        print(row)


if __name__ == "__main__":
    main()
