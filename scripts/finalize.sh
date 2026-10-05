#!/usr/bin/env bash
# One-shot finalisation: pick the fair-watched best checkpoints, evaluate them
# on the table seed in both cities, redraw the figures, and re-render the
# all-in-one videos with the current colour look.  Safe to run repeatedly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=${SCRATCH:-/tmp/fly_finalize}; mkdir -p "$S"
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

mkdir -p runs/final_fly_plain runs/final_fly_v3
# The representatives are chosen by hand (two-seed comparison, see the memory
# notes); only re-pick them from the watchers when explicitly asked.
if [ "${FINALIZE_PICK:-0}" = "1" ]; then
  [ -f runs/city_fly_ft/best_fair.pt ]    && cp runs/city_fly_ft/best_fair.pt    runs/final_fly_plain/last.pt
  [ -f runs/city_fly_v3_ft/best_fair.pt ] && cp runs/city_fly_v3_ft/best_fair.pt runs/final_fly_v3/last.pt
fi

echo "== plain grid =="
$PY scripts/eval_city.py expert:final_fly_plain fly:final_fly_plain cns:bc_cns_v10 --force-cfg --fillet 9 14 --city-vmax 10 --steps 3400 --out "$S/final_plain.json" 2>&1 | grep -vE "beta state|to_sparse_csr|weights_only|FutureWarning" | tail -4
$PY scripts/plot_city_results.py "$S/final_plain.json" --names "scripted expert (privileged)" "hand-built fly pathway: cloned + anchored PPO" "MaleCNS whole brain, frozen: ridge readout" --title "How the drive ends — plain 4×4 grid, same routes, signals and traffic for every driver" --out runs/city_results.png
echo "== v3 city =="
$PY scripts/eval_city.py expert:final_fly_v3 fly:final_fly_v3 cns:bc_cns_v3city --steps 3400 --out "$S/final_v3.json" 2>&1 | grep -vE "beta state|to_sparse_csr|weights_only|FutureWarning" | tail -4
$PY scripts/plot_city_results.py "$S/final_v3.json" --names "scripted expert (privileged)" "hand-built fly pathway: cloned + anchored PPO" "MaleCNS whole brain, frozen: ridge readout" --title "How the drive ends — v3 city (irregular blocks + buildings), same routes, signals and traffic for every driver" --out runs/city_v3_results.png

for spec in "final_fly_plain:city_final:plain grid" "final_fly_v3:city_v3_final:v3 city with buildings"; do
  run=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; label=${rest#*:}
  $PY scripts/watch.py --ckpt runs/$run/last.pt --model fly --env city --view neural --steps 1000 --seed 5 --title "hand-built fly pathway, cloned + anchored PPO - $label" --out videos/${name}_neural.mp4 2>&1 | grep -E "wrote|Traceback" | tail -1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/fly_brain_map.py --ckpt runs/$run/last.pt --steps 1000 --seed 5 --out videos/${name}_brain_regions.mp4 --title "hand-built pathway (83k params), cloned from the expert + anchored PPO, $label  |  somas from the MaleCNS connectome" 2>&1 | grep -E "wrote|Traceback" | tail -1
  $FF -y -loglevel error -i videos/${name}_neural.mp4 -i videos/${name}_brain_regions.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2,scale=1200:-2:flags=lanczos,scale=trunc(iw/2)*2:trunc(ih/2)*2[out]" -map "[out]" -c:v libx264 -profile:v main -level 4.0 -crf 25 -preset medium -pix_fmt yuv420p -movflags +faststart -shortest videos/${name}_all_small.mp4
  echo "composed videos/${name}_all_small.mp4"
done
echo "done: runs/city_results.png runs/city_v3_results.png videos/*_all_small.mp4"
