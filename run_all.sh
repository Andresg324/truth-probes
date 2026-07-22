#!/bin/bash
set -euo pipefail
export MPLBACKEND=Agg
PY=$(command -v python3 || command -v python)
echo "Using: $PY"; $PY --version
STAMP=$(date +%F)

# Don't destroy data, just overwrite
[ -d results ] && mv results "results_pre_release_$STAMP"
[ -d figures ] && mv figures "figures_pre_release_$STAMP"
mkdir -p logs

FULL="0.5B 1.5B gemma-2b 3B" # Full battery: probes, curves, and the regress transfer
CENSUS="llama-3b 7B llama-8b gemma-9b 14B" # Census battery only

for M in $FULL; do
    echo "===== $M (full) ====="
    mkdir -p results/$M figures/$M
    $PY deception_probe.py $M 2>&1 | tee logs/${M}_probe.log
    $PY phase2.py $M          2>&1 | tee logs/${M}_phase2.log
    $PY phase4_fixed.py $M    2>&1 | tee logs/${M}_phase4.log
done

for M in $CENSUS; do
    echo "===== $M (census only) ====="
    mkdir -p results/$M figures/$M
    $PY deception_probe.py $M --probe-only 2>&1 | tee logs/${M}_probe.log
    $PY phase4_fixed.py $M                 2>&1 | tee logs/${M}_phase4.log
done

# Control 3 - second basis
for M in gemma-2b 3B; do
    $PY phase4_fixed.py $M noproc          2>&1 | tee logs/${M}_noproc.log
done

#Control 1 paraphrase triangle
for M in 1.5B gemma-2b; do
    for P in tf claim; do
        $PY paraphrase_triangle.py $M $P  2>&1 | tee logs/${M}_para_${P}.log
    done
done

# Analyses
$PY leak_control.py                       2>&1 | tee logs/leak_census.log
for M in gemma-2b llama-8b; do $PY regress_transfer.py $M 2>&1 | tee logs/${M}_regress.log; done
$PY regress_census.py                     2>&1 | tee logs/regress_census.log
$PY denial_bias.py                        2>&1 | tee logs/denial_bias.log
for M in gemma-2b 3B; do $PY standardize_diag.py $M 2>&1 | tee logs/${M}_std.log; done

# Write down the results
$PY make_results_md.py                    2>&1 | tee logs/results_md.log

# Package and get it off the pod
tar czf "release_$STAMP.tgz" results figures logs
echo "===== DONE - download release_$STAMP.tgz AND push to HF before killing the pod ====="