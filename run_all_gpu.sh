#!/bin/bash
set -u
SMALL_FULL="0.5B 1.5B 3B gemma-2b"                  # probe + mechanism + transfer (from_pretrained)
TRANSFER_ONLY="llama-3b 7B 14B gemma-9b llama-8b"   # transfer census only

for M in $SMALL_FULL; do
  echo "===== $M (full) ====="
  python deception_probe.py $M || echo "FAILED deception_probe $M"
  python phase2.py          $M || echo "FAILED phase2 $M"
  python phase4_fixed.py    $M || echo "FAILED phase4_fixed $M"
done
for M in $TRANSFER_ONLY; do
  echo "===== $M (transfer only) ====="
  python phase4_fixed.py $M || echo "FAILED phase4_fixed $M"
done
for M in gemma-2b 3B; do                            # Control 3 no_processing basis
  echo "===== $M noproc ====="; python phase4_fixed.py $M noproc || echo "FAILED noproc $M"
done
for M in gemma-2b 1.5B; do                          # Control 1 paraphrase triangle
  python paraphrase_triangle.py $M tf    || echo "FAILED paraphrase $M tf"
  python paraphrase_triangle.py $M claim || echo "FAILED paraphrase $M claim"
done
echo "===== ANALYSIS ====="
python leak_control.py
for M in gemma-2b 3B; do python regress_transfer.py $M || echo "FAILED regress $M"; done
for M in gemma-2b 3B; do python standardize_diag.py $M || echo "FAILED std $M"; done
echo "===== ALL DONE ====="