#!/usr/bin/env bash
# Reproduce the environment from scratch on a clean machine.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip setuptools wheel
.venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
.venv/bin/python -m pip install -q "numpy<2" matplotlib imageio imageio-ffmpeg tensorboard rich tqdm scipy pillow pandas pyarrow

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"vram: {free/2**30:.1f} / {total/2**30:.1f} GiB")
PY
echo
echo "ready.  try:"
echo "  .venv/bin/python scripts/demo_render.py      # what the fly sees"
echo "  .venv/bin/python scripts/validate_t4t5.py    # motion-detector tuning curves"
echo "  .venv/bin/python scripts/train.py --name fly01 --compile"
