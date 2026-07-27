# What Deception Probes Read and How Evaluations of Them Fail

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21632023.svg)](https://doi.org/10.5281/zenodo.21632023)

Code, data, and regenerable results for the paper *What Deception Probes Read and How Evaluations of Them Fail* (Andres Garcia, 2026); [read the paper (PDF)](paper/deception-probes-paper.pdf). arXiv link will be added on release.

Linear probes on LLM activations are the leading proposal for monitoring deceptive generation. This repository contains the full pipeline, data, and regenerable results for a study asking three questions: **what do such probes read, where is that signal computed, and are the standard experiments used to answer those questions sound?**

Short answers, from a pre-specified pipeline on Qwen2.5-0.5B/1.5B/3B and Gemma-2-2b (with a nine-model transfer census):

- **Probes read truth-consistency**, and the computation behind it has **no canonical shape**: deception-specific ablation damage is band-localized at 1.5B (L15-L17), multi-site at Gemma-2b (L9/L11/L13/L14), fully absorbed by the deployed readout at 3B (AUC ≥ 0.93 under every single-MLP ablation), and absent at 0.5B.
- **Point interventions misidentify the mechanism in every model.** The strongest disruptor at 1.5B has zero deception-specificity, and 0.5B's "critical" layer is the readout itself. Only per-layer curves with a deception/truth/polarity decomposition adjudicate.
- **Instructed-transfer evaluation collapses without controls**: naive transfer AUCs span 0.030-1.000 across nine models (once exceeding the positive-control ceiling); denial bias (all five Qwen models lie only by denying truths) leaves 7/9 models unidentifiable; in the two identifiable models the instruction coefficient dominates a small, sign-inconsistent deception coefficient.
- Answer polarity survives all 43 ablation conditions (AUC 1.00 throughout), and clean-patching restores 95-98% of what ablation removes.

<p align="center">
  <img src="figures/paper/damage_grid.png" width="720" alt="Per-layer ablation damage curves across four models: same intervention, four different topologies">
</p>
<p align="center">
  <img src="figures/paper/nine_verdicts.png" width="720" alt="Nine-model transfer census: naive statistics span the full interval; the protocol leaves two models identifiable">
</p>

## The evaluation protocol

Six controls, each of which reversed a conclusion in this study before we adopted it:

1. Match probe readout position between training and evaluation.
2. Gate transfer tests on behavioral non-degeneracy; exclude, don't average.
3. Require a positive-control ceiling; any transfer number at or above it is likely a confound.
4. Use polarity-controlled regression, never subgroup AUCs.
5. Report causal effects threshold-free (AUC), never as recall on one class.
6. Localize with per-layer curves plus label decomposition, never point interventions at selected components.

## Repository layout

```
data/                      400-statement primary corpus + 100-statement second-source transfer set
deception_probe.py         Probe training, layer selection, steering, behavioral sweeps, figures
phase2.py                  Per-layer ablation curves, specificity curves, sufficiency (conduit test)
phase4_fixed.py            Instructed-transfer battery (behavioral pipeline)
paraphrase_triangle.py     Format-stability runs (tf / claim paraphrases)
position_sensitivity.py    Readout-position de-confound
regress_transfer.py        Polarity-controlled regression, single model
regress_census.py          Regression and identifiability gate, all nine models
leak_control.py            Instruction-leak census
denial_bias.py             Denial-bias table
standardize_diag.py        Preprocessing and position interaction
reporting.py               Structured results logging (writes results/*/RESULTS.json)
make_results_md.py         Regenerates MEASURED.md from RESULTS.json, the source of truth
make_figures.py            Regenerates the combined paper figures from MEASURED.md and saved curve data
run_all.sh                 Full release run, in order
paper/                     The paper (PDF) and combined paper figures
results/                   Per-model RESULTS.json, manifests, .npz curve data
figures/                   Per-model figures; figures/paper/ holds the combined paper figures
MEASURED.md                Generated source of truth; every number in the paper regenerates from here
```

## Reproducing

```bash
pip install -r requirements.txt          # exact versions from the release run's manifest
export HF_TOKEN=...                      # Gemma and Llama models are gated on Hugging Face
bash run_all.sh                          # full pipeline (A100-class GPU; ~15-25 GPU-hours)
python make_results_md.py                # regenerate MEASURED.md
```

Every number in the paper regenerates from `MEASURED.md`; if a number is not there, it was not measured. All published results come from a single clean-room run on one device and precision (A100, fp16); per-run manifests (seeds, package versions, model revisions) are in `results/*/MANIFEST*.json`. Expect decimal-level drift on other hardware; verdicts should not change. If one does, we would genuinely like to hear about it as the paper's evaluation section is about exactly this phenomenon.

Total compute: the release run reproduces for approximately $20-30 in rented GPU time; total project compute, including development and reruns, was under $100.

The complete results tree, including the large activation caches excluded from this repository, is archived at [doi:10.5281/zenodo.21632102](https://doi.org/10.5281/zenodo.21632102).

## A note on process

This project's findings were revised six times by its own controls before publication: a recall-based metric inverted the strongest necessity effect, an accuracy-based metric manufactured a critical layer at 3B, a restoration formula with a missing condition reversed the sufficiency verdict, a degenerate control looked like a discovery, a clean-room rerun flipped a regression coefficient's sign, and a control selected under one metric failed under another. Appendix F of the paper documents all six. The released pipeline is the version that survives them.

## Citation

See [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button uses it). BibTeX:

```bibtex
@misc{garcia2026deceptionprobes,
  title  = {What Deception Probes Read and How Evaluations of Them Fail},
  author = {Garcia, Andres},
  year   = {2026},
  url    = {https://github.com/Andresg324/truth-probes},
}
```

arXiv eprint and Zenodo DOI fields will be added once minted.

## Disclosure

The primary 400-statement corpus was drafted with LLM assistance (Claude Opus 4.8, May 2026) and reviewed by the author for factual accuracy and length. The 100-statement second-source transfer set was authored by a non-author with ChatGPT assistance (default model, June 2026), delivered as a document and converted to JSON by the author; it therefore tests generalization across author, model family, and collection pipeline, not AI-versus-human provenance. Manuscript preparation used AI writing assistance; all experimental design decisions, code review, and claims are the author's. See LICENSE for terms.