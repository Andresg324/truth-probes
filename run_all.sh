#!/bin/bash
PY=$(command -v python3 || command -v python)
echo "Using: $PY"; $PY --version

for TAG in 0.5B 1.5B 3B gemma-2b 7B gemma-9b llama-3b llama-8b 14B; do
    mkdir -p results/$TAG figures/$TAG
    $PY deception_probe.py $TAG 2>&1 | tee results/$TAG/run_probe.txt
    $PY phase2.py $TAG 2>&1 | tee results/$TAG/run_phase2.txt
done
$PY inlp_calib.py 2>&1 | tee results/inlp_calib.txt