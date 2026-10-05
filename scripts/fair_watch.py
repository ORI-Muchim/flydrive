"""Evaluate a training run's latest checkpoint on the fixed seed every few
minutes and keep the best one by *driving outcome*, not training return.

Score = finished - 0.5 * red_runs_per_episode - off_road - collision - stalled.
Writes runs/<name>/fair_eval.log and copies the best checkpoint to
runs/<name>/best_fair.pt.
"""
import argparse, os, shutil, subprocess, sys, time, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("name"); ap.add_argument("--every", type=float, default=15.0, help="minutes")
    ap.add_argument("--steps", type=int, default=3400); ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123], help="evaluate on each seed and average (less selection bias)")
    a = ap.parse_args()
    run = f"runs/{a.name}"; best = -9.0; seen = None; t_end = time.time() + a.hours * 3600
    os.makedirs(run, exist_ok=True)          # the trainer may not have created it yet
    log = open(f"{run}/fair_eval.log", "a")
    while time.time() < t_end:
        ck = f"{run}/last.pt"
        if os.path.exists(ck) and os.path.getmtime(ck) != seen:
            seen = os.path.getmtime(ck)
            snap = f"{run}/eval_snapshot.pt"; shutil.copy(ck, snap)   # evaluate a frozen copy, not a file being rewritten
            out = f"{run}/eval_tmp.json"
            spec = f"{a.model}:{a.name}"
            # eval_city looks for runs/<name>/best.pt first; point it at the snapshot through a temp run dir
            tmp = f"runs/_fair_{a.name}"; os.makedirs(tmp, exist_ok=True); shutil.copy(snap, f"{tmp}/last.pt")
            rows_s = []
            for sd in a.seeds:
                r = subprocess.run([sys.executable, "scripts/eval_city.py", f"{a.model}:_fair_{a.name}", "--steps", str(a.steps), "--seed", str(sd), "--out", out],
                                   capture_output=True, text=True)
                try:
                    rows_s.append(json.load(open(out))[0])
                except Exception:
                    log.write(f"{time.strftime('%H:%M')} eval failed (seed {sd}): {r.stderr[-300:]}\n"); log.flush()
            if not rows_s:
                time.sleep(60); continue
            keys = ("finished", "off_road", "wrong_turn", "collision", "stalled", "red_per_episode", "distance")
            row = {k: sum(x[k] for x in rows_s) / len(rows_s) for k in keys}
            score = row["finished"] - 0.5 * row["red_per_episode"] - row["off_road"] - row["collision"] - row["stalled"]
            line = (f"{time.strftime('%H:%M')} finish {row['finished']:.2f} off {row['off_road']:.2f} wrong {row['wrong_turn']:.2f} "
                    f"coll {row['collision']:.2f} stall {row['stalled']:.2f} red {row['red_per_episode']:.2f} dist {row['distance']:.0f} score {score:+.2f}")
            if score > best:
                best = score; shutil.copy(snap, f"{run}/best_fair.pt"); line += "  <- best"
            print(line, flush=True); log.write(line + "\n"); log.flush()
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
