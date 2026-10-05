"""Stacked outcome bars from eval_city --out JSON: how each driver's episodes end."""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["finished", "off_road", "wrong_turn", "collision", "stalled", "timeout"]
LABEL = {"finished": "finished route", "off_road": "off road", "wrong_turn": "wrong turn / lane", "collision": "collision",
         "stalled": "stalled", "timeout": "timeout"}
COLOR = {"finished": "#2a9d8f", "off_road": "#e76f51", "wrong_turn": "#f4a261", "collision": "#9b2226", "stalled": "#8d99ae", "timeout": "#adb5bd"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--out", default="runs/city_results.png")
    ap.add_argument("--names", nargs="*", default=None, help="display names, in the JSON's order")
    ap.add_argument("--title", default="How the drive ends — city, same routes, signals and traffic for every driver")
    a = ap.parse_args()
    rows = json.load(open(a.json))
    names = a.names or [r["run"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 1.3 + 0.75 * len(rows)))
    left = [0.0] * len(rows)
    for k in ORDER:
        vals = [100 * r[k] for r in rows]
        ax.barh(names, vals, left=left, height=0.58, color=COLOR[k], label=LABEL[k], edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    for i, r in enumerate(rows):
        ax.text(102, i, f"{r['red_per_episode']:.2f} red/ep · {r['distance']:.0f} m · {r['speed']:.1f} m/s", va="center", fontsize=8.5, color="#333", clip_on=False)
    ax.set_xlim(0, 100); ax.set_xlabel("episodes (%)"); ax.invert_yaxis()
    fig.subplots_adjust(right=0.72)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.35 / max(len(rows), 1) - 0.25), ncol=6, frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.2)
    ax.set_title(a.title, fontsize=10.5, loc="left")
    fig.savefig(a.out, dpi=150, bbox_inches="tight"); print("wrote", a.out)


if __name__ == "__main__":
    main()
