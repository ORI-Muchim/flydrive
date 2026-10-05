#!/usr/bin/env bash
# Download the MaleCNS v1.0 tables (Janelia FlyEM, public bucket) into data/.
# They are too large for the git repository (the weights table alone is 479 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
get() { [ -s "data/$2" ] && { echo "have data/$2"; return; }; echo "fetching $2"; curl -fL --retry 3 -o "data/$2" "$B/$1"; }
get body-annotations-male-cns-v1.0-minconf-0.5.feather body-annotations.feather
get body-neurotransmitters-male-cns-v1.0.feather body-neurotransmitters.feather
get connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather connectome-weights.feather
