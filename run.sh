#!/usr/bin/env bash
# Run the whole pilot from this folder:
#   1) extract frozen AlexNet + DINO features (skipped if ./features already holds them; ~15 min on a laptop GPU)
#   2) train the weak supervisor, the two weak-to-strong students and the ceiling (linear probes, <1 min on GPU)
#   3) draw the figures and the summary table (CPU)
# Usage:  bash run.sh        (with the Python environment from README.md active)
set -e
cd "$(dirname "$0")"
[ -f features/dino_test.npy ] || python extract_features.py
python w2s_pilot.py
python make_figures.py
