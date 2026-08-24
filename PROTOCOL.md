# PROTOCOL: Deception probe evaluation study

**Status of this document.** Sections 1–4 are *retrospective*: they record rules that were fixed
before the canonical release run and are evidenced by run artifacts, environment manifests
(`results/*/MANIFEST*.json`; one per model, see the scope note in §2), and the dated Zenodo
deposit. They were not lodged in a timestamped public registry before the fact. This is a process
limitation of this project and is stated as such here rather than glossed. Section 5 is
*prospective*: it is committed to the repository before any data for that analysis is collected,
and its git commit date is the timestamp.

Nothing in Sections 1–4 should be read as a pre-registration in the strict sense. They are a
faithful record of decision rules that were fixed in advance of measurement, published so that the
rules can be checked against the code and the results file.

---

## 1. Question and scope

Three questions, on Qwen2.5-0.5B/1.5B/3B/7B/14B, Gemma-2-2b/9b, Llama-3.2-3B, Llama-3.1-8B:

1. What do linear deception probes read?
2. Where is that signal computed?
3. Are the standard experiments used to answer (1) and (2) sound?

Construct: "deception" here is **teacher-forced knowledge-inconsistency**; asserting a statement
the model's own probe representation encodes as false. Not strategic deception. All claims scoped
accordingly.

**Scope split, fixed in advance:** per-layer curve analysis on the four models where the compute was
affordable (0.5B, 1.5B, 3B, gemma-2b); single-readout census on all nine. Reported separately
throughout, never pooled.

## 2. Frozen pipeline

| Decision | Rule |
|---|---|
| Splits | Statement-grouped; both members of a statement pair on one side of every split |
| Layer selection | Nested: accuracy averaged over 10 validation splits; earliest layer within 0.02 of peak (knee rule); test fold never used for selection |
| Critical MLP | Layer whose ablation most reduces the probe's **continuous score** on training-fold deceptive items |
| Causal metric | AUC (threshold-free) is the citable metric; balanced accuracy reported alongside in appendix |
| Intervals | Grouped bootstrap over statements, 1000 draws |
| Probe | L2 logistic regression, C = 0.1, max_iter = 2000 |
| Readout | Answer token (`resid_post`, final position); behavioral pipeline matches this position |
| Environment | Single device and precision for all published numbers (A100, fp16); loader fixed per experiment. Environment provenance recorded in `results/<model>/MANIFEST*.json`; see the manifest scope note below |
| Source of truth | `MEASURED.md`, generated from `results/*/RESULTS.json`. If a number is not there, it was not measured |

**Manifest scope (known limitation).** The release run wrote a single `MANIFEST.json` per model
directory, so each of those files records only the last script to run against that model. As released:
0.5B, 7B, 14B, gemma-9b, and llama-3b from `phase4_fixed.py`; 1.5B from `paraphrase_triangle.py`;
3B and gemma-2b from `standardize_diag.py`; llama-8b from `regress_transfer.py`. No manifest
survives from `deception_probe.py` or `phase2.py`. Because every release script ran in one session
on one device with one environment, the recorded device, precision, package versions, and model
revisions are valid for all analyses on that model; what the manifests do not recover is which
script and seed produced which individual number. `reporting.py` now derives the manifest filename from the calling script, so runs after
2026-08-24 write `MANIFEST_<script>.json` and carry per-analysis provenance; it also records the
GPU model rather than only `cuda`. The release-run manifests are left unchanged as the record of
what was actually written.

## 3. Pre-specified decision rules

| Rule | Threshold | Purpose |
|---|---|---|
| Answer-word leak control | Must sit near chance | Design-level check that answer wording carries no signal |
| De-confound leak flag | \|AUC − 0.5\| > 0.15 | Probe reads the instruction, not the answer |
| Readout-position sensitivity | Δ > 0.15 (same threshold, reused) | Position materially changes the verdict |
| Identifiability gate | Minority answer cell < 20 obs, or < 0.20 of the set, or condition number > 30 | Transfer regression is unidentifiable; exclude, do not average |
| Positive-control ceiling | Any transfer AUC at or above the in-format ceiling | Confound, not a result |
| Scale-trend falsification | Max deployed damage at 0.5B: > 0.33 supports the trend; 0.16–0.33 partial; < 0.16 retires it | Fixed before the 0.5B run; observed 0.235 → trend retired |
| Quasi-separation guard | \|coefficient\| > 5 on standardized probe scores | Transfer regression fit is unstable; report as quasi-separation, do not cite. Never fired |

## 4. Amendment log

Dated changes to the above, each made before the measurement it affected.

| # | Date | Change |
|---|---|---|
| 1 | 2026-07-19 | Causal effects moved from recall-on-deceptive to balanced accuracy (recall read boundary shifts as signal) |
| 2 | 2026-07-19 | Balanced accuracy supplemented by AUC as the citable metric after 3B decalibration was identified |
| 3 | 2026-07-20 | Sufficiency redefined: restoration measured against the ablated floor, not the steered baseline |
| 4 | 2026-07-20 | Point controls replaced by full per-layer ablation curves; selected-point controls retained only in the paraphrase runs and not cited |
| 5 | 2026-07-21 | Bootstrap changed from item-level to statement-grouped throughout |
| 6 | 2026-07-21 | Layer selection in the transfer pipeline corrected to nested validation folds (had used the test fold) |
| 7 | 2026-07-23 | Canonical clean-room run: all published numbers regenerated on one device and precision |
| 8 | 2026-07-23 | Position-sensitivity measurement added; reused the existing 0.15 leak threshold rather than choosing a new one |
| 9 | 2026-08-24 | `reporting.py` manifest filename derived from the calling script (`MANIFEST_<script>.json`) and GPU model recorded; applies to runs after this date, release-run manifests unchanged. Made before the §6 addendum was run |

Six documented conclusion reversals arising from these amendments are catalogued in the paper's
Appendix F.

---

## 5. PRE-REGISTRATION: External-corpus replication (Azaria & Mitchell)

**Committed before any data is collected. The commit date is the timestamp.**

### Motivation
The primary corpus is author-assembled (400 statements, minimal-pair construction, a bounded
near-duplicate leak). The open question is not whether the probe generalizes to another corpus,
but whether the **measurement pitfalls documented in this paper are properties of the method or of
this corpus**.

### Data
Azaria & Mitchell (2023) true/false statement set, six topics. Subsample to ~1,500 statements
balanced across topics and across true/false. Same forced-answer template as the primary corpus.
Same statement-grouped splits, same nested layer selection, same probe hyperparameters.

### Analyses (fixed)
1. Within-corpus probe accuracy per model (10 seeds).
2. Cross-corpus transfer, both directions (primary → Azaria, Azaria → primary), read at the layer
   selected on the *training* corpus.
3. Topic-held-out: train on five topics, test on the sixth, all six rotations.
4. Pitfall replication: single-split accuracy vs. ten-seed mean (the gap), and the answer-word leak
   control, computed on Azaria exactly as on the primary corpus.

### Interpretation rules (fixed in advance)
- **Source-generality:** cross-corpus transfer ≥ 0.70 in both directions = the signal is
  source-general. Below 0.60 in either direction means it is not, and the within-format positive claim
  in the paper is narrowed accordingly.
- **Semantic generality:** topic-held-out mean within 0.05 of within-topic accuracy means it generalizes
  across semantic domains. A gap above 0.10 = it does not, and is reported as such.
- **Pitfall replication (the primary question):** if the single-split minus ten-seed gap on Azaria
  is ≥ 0.03 at any model, the inflation pitfall is a property of the method. If the gap is < 0.03
  at every model, it is corpus-specific and the paper's claim is narrowed.
- Null and negative outcomes are reported with the same prominence as positive ones. No analysis
  below is added or removed after seeing any result; anything added later is labelled exploratory.

### Scope
Probe-level only. No ablation, patching, steering, or transfer-census analyses are re-run on this
corpus. Results enter §3 and the appendix of the paper; no claim in §4–§6 depends on them.

### Compute
Activation extraction on Qwen2.5-0.5B/1.5B/3B and Gemma-2-2b, all four in a single session on
one rented A100 (80 GB, RunPod), fp16, TransformerLens from_pretrained (LN-folded basis), the
same loader and precision as the canonical release run of 2026-07-23. Probe training, splits,
and all analysis run offline on CPU from cached activations. Estimated ~2–3 GPU-hours total.
A MANIFEST_azaria.json is written per model recording date, invocation, package versions,
model revision, device, dtype, and seed, as for every other run in this project.
Deviation from these settings — different loader, precision, or device — is itself a
deviation from protocol and must be recorded as an amendment, since the paper's own
Appendix E documents preprocessing choices changing a verdict.

## 6. Pre-registered addendum (2026-08-25): retrained monitor under upstream steering

**Committed before any data is collected. The commit date is the timestamp.**
Primary corpus, the four curve models; independent of the section 5 replication.

### Motivation
The steering result in the paper injects at the readout layer, where the intervention is a
translation and its effect on a linear probe is analytic: every decision score shifts by
-alpha*||w||, rank order is preserved, AUC is unchanged. That result therefore establishes
de-calibration, not destruction. This addendum asks the empirical question the analytic one
cannot: does an attack injected upstream, where the perturbation propagates through
nonlinearities, remove the signal itself?

### Procedure (fixed)
Injection at E = best_layer // 2, the same early point used by the sufficiency test.
0. Recalibrate alpha at E: sweep [0,2,4,6,8,10,15,20,30,40], take the first value driving
   frozen-probe detection on held-out deceptive items below 0.10. This reuses the existing
   min_alpha rule rather than introducing a new threshold. The published alpha was calibrated
   at the readout layer and does not transfer.
1. Extract activations at the frozen readout layer under that alpha, applied to EVERY item
   regardless of label.
2. Report frozen-probe detection rate and AUC on the steered activations.
3. Refit a probe (L2, C = 0.1, max_iter = 2000) on steered activations under the canonical
   split, and over ten GroupShuffleSplit seeds.

### Interpretation rules (fixed in advance)
- **Attack fails:** no alpha in the grid drives detection below 0.10. Reported as recomputation
  defeating upstream steering; the refit is not run.
- **Signal intact:** retrained AUC within 0.05 of clean AUC at every model. Headline narrows to
  "steering evades the deployed monitor, not the signal."
- **Signal removed:** retrained AUC below 0.60 at every model. Evasion claim strengthens to
  monitor-independent.
- **Mixed:** anything else, reported per model without aggregation.
- Independently: if detection falls below 0.5 while frozen AUC stays within 0.05 of clean, the
  upstream attack is also de-calibrating, and this is reported as a further instance of the
  metric confound of Appendix F.

Null and negative outcomes reported at equal prominence. Nothing added or removed after seeing
any result; anything added later is labelled exploratory.

### Compute
Forward passes only, four models, one session, one device and precision (A100, fp16,
TransformerLens from_pretrained), matching the canonical run. MANIFEST_retrain_under_steering.json
per model. Under 1 GPU-hour. Deviation from these settings is a protocol deviation and must be
recorded as an amendment.