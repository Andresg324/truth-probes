#!/bin/bash
# Full release run. Regenerates every number in the paper plus all paper figures.
#
#   bash run_all.sh
#
# Order matters: analyses that call report() must precede make_results_md.py, and
# make_figures.py must follow it (it scrapes CI bands from logs/*phase2*.log).
#
# VERIFY ONCE: position_sensitivity.py and check_data_splits.py are invoked with a
# model argument, matching every other per-model script. If either loops over models
# internally, drop the loop.

set -euo pipefail
export MPLBACKEND=Agg
PY=$(command -v python3 || command -v python)
echo "Using: $PY"; $PY --version
STAMP=$(date +%F)

: "${HF_TOKEN:?HF_TOKEN is not set. Gemma and Llama are gated on Hugging Face}"

# Don't destroy data, just move it aside
if [ -d results ]; then mv results "results_pre_release_$STAMP"; fi
if [ -d figures ]; then mv figures "figures_pre_release_$STAMP"; fi
mkdir -p logs

FULL="0.5B 1.5B gemma-2b 3B"                  # full battery: probes, curves, transfer
CENSUS="llama-3b 7B llama-8b gemma-9b 14B"    # census battery only
CURVE="0.5B 1.5B 3B gemma-2b"                 # models with per-layer curves

# ---------------------------------------------------------------- extraction ----
for M in $FULL; do
    echo "===== $M (full) ====="
    mkdir -p results/$M figures/$M
    $PY scripts/deception_probe.py $M 2>&1 | tee logs/${M}_probe.log
    $PY scripts/phase2.py $M          2>&1 | tee logs/${M}_phase2.log
    $PY scripts/phase4_fixed.py $M    2>&1 | tee logs/${M}_phase4.log
done

for M in $CENSUS; do
    echo "===== $M (census only) ====="
    mkdir -p results/$M figures/$M
    $PY scripts/deception_probe.py $M --probe-only 2>&1 | tee logs/${M}_probe.log
    $PY scripts/phase4_fixed.py $M                 2>&1 | tee logs/${M}_phase4.log
done

# ---------------------------------------------------------------- controls ------
# Control 3: second weight-processing basis (feeds standardize_diag.py below)
for M in gemma-2b 3B; do
    $PY scripts/phase4_fixed.py $M noproc          2>&1 | tee logs/${M}_noproc.log
done

# Control 1: paraphrase triangle (format stability)
for M in 1.5B gemma-2b; do
    for P in tf claim; do
        $PY scripts/paraphrase_triangle.py $M $P   2>&1 | tee logs/${M}_para_${P}.log
    done
done

# ---------------------------------------------------------------- analyses ------
$PY scripts/leak_control.py                        2>&1 | tee logs/leak_census.log
for M in gemma-2b llama-8b; do
    $PY scripts/regress_transfer.py $M             2>&1 | tee logs/${M}_regress.log
done
$PY scripts/regress_census.py                      2>&1 | tee logs/regress_census.log
$PY scripts/denial_bias.py                         2>&1 | tee logs/denial_bias.log
for M in gemma-2b 3B; do
    $PY scripts/standardize_diag.py $M             2>&1 | tee logs/${M}_std.log
done

# Readout-position de-confound and split-integrity audit (both cited in the paper)
for M in gemma-2b llama-8b; do
    $PY scripts/position_sensitivity.py $M         2>&1 | tee logs/${M}_position.log
done
$PY scripts/check_data_splits.py                   2>&1 | tee logs/check_splits.log

# ---------------------------------------------------------------- addendum ------
# PROTOCOL section 6: retrained monitor under upstream steering. Pre-registered
# separately from the canonical battery; adds roughly 1 GPU-hour. Comment out this
# block to reproduce only the numbers in the paper as submitted.
for M in $CURVE; do
    $PY scripts/retrain_under_steering.py $M       2>&1 | tee logs/${M}_retrain_steer.log
done

# ---------------------------------------------------------------- outputs -------
$PY scripts/make_results_md.py                     2>&1 | tee logs/results_md.log
$PY scripts/make_figures.py                        2>&1 | tee logs/make_figures.log

tar czf "release_$STAMP.tgz" results figures logs MEASURED.md
printf '\n===== DONE in %dh %dm =====\n' $((SECONDS/3600)) $((SECONDS%3600/60))
echo "Download release_$STAMP.tgz and push before killing the pod"