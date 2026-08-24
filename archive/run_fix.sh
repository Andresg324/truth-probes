#!/bin/bash
PY=$(command -v python3 || command -v python)
# Four gated models that failed
for TAG in gemma-2b llama-3b gemma-9b llama-8b; do
    mkdir -p results/$TAG figures/$TAG
    $PY deception_probe.py $TAG 2>&1 | tee results/$TAG/run_probe.txt
    $PY phase2.py $TAG 2>&1 | tee results/$TAG/run_phase2.txt
done

#Re-run phase2 on QWEN
for TAG in 0.5B 1.5B 3B 7B 14B; do
    $PY phase2.py $TAG 2>&1 | tee results/$TAG/run_phase2.txt
done
$PY inlp_calib.py 2>&1 | tee results/inlp_calib.txt