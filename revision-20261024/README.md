# Retained revision artifacts — ICDEES 2026, paper 1571319789

**Hybrid Machine Learning versus Fine-Tuned Transformers for Sentiment Analysis of Indonesian Milk-Tea Outlet Reviews on Google Maps: A Performance and Deployment-Cost Benchmark**

This is a bounded release of the retained, non-text artifacts supporting the revised benchmark. It is separate from the older single-split baseline in the repository root. The baseline files and its historical training behavior have not been changed by this release.

## Contents and verification

- `results/main_individual_runs.csv`: 80 retained main-benchmark records (16 models × five split/training seeds, 42–46), including measured accuracy, macro-F1, AUC and recorded computational costs. Local filesystem paths have been removed. Transformer rows are retained aggregate records, not a fresh reconstruction of their missing item predictions.
- `results/recomputed_model_summary.csv`: means and sample standard deviations recomputed from those 80 records. Overlapping splits are dependent; these SDs are not independent-replication confidence intervals.
- `predictions/`: all 30 locally retained main-benchmark **CPU hybrid** prediction files, each with 507 examples. Only true binary labels, predictions, positive-class scores and fresh random example IDs are released. The original text, original row indices and source metadata are removed, and row order is shuffled. The same random IDs permit pairing the six hybrid models within a split; IDs are not linked between splits. SVM scores are decision scores and are not calibrated probabilities.
- `results/prediction_verification.csv`: macro-F1, accuracy and AUC recomputed from each of the 30 prediction files; all match the retained main run records within `1e-10`.
- `results/latency_pooled.csv`: retained pooled latency measurements from three passes. Timing is hardware-specific; this release does not remeasure performance.
- `PROTOCOL.json`, `SOURCE_MANIFEST.json`, `MANIFEST.json`: scope, transformations, source checksums and released-file checksums.
- `code/verify_results.py`: dependency-free verification code for released hashes, prediction metrics and model summaries.

Run from a checkout with Python 3.10 or newer:

```bash
python revision-20261024/code/verify_results.py
```

The root's historical training code is not the complete revised training pipeline. This folder reproduces the released numerical summaries and CPU prediction metrics; it cannot independently retrain the full revised benchmark.

## Deliberate exclusions and known limits

**This revision artifact release excludes review texts and reviewer identifiers.** It also excludes listing IDs, profile URLs, location/collection telemetry, raw exports, original source row IDs, credentials, model weights, and manuscripts. Older baseline files elsewhere in this repository have a different release scope; this statement applies only to this revision folder.

Transformer per-item predictions and fine-tuned weights were not present in the retained local archive and are **not supplied** here. The corrected main IndoBERT-lite aggregate records are retained, but cannot be independently recomputed from item predictions in this release.

The archived supplementary PDF and its derived supplementary tables are excluded. Two IndoBERT-lite rows in archived Tables S16 and S18 have not been reconciled with Table S12. This release does not imply that those rows, or every archived supplementary statistic, have been corrected. It does not change any scientific result in the approved manuscript.

Exact re-collection of source reviews may be impossible as listings and reviews change. Rating-derived labels, concentrated geography, shared preprocessing resources, unequal tuning budgets, overlapping splits and hardware-specific timing remain study limitations. No additional claims about historical scraper identity, independent human adjudication or hyperparameter provenance are established by these files.

## Use and attribution

The newly written verification code in `code/` is licensed under the MIT License in that directory. Please cite the paper and identify this release date when using the numerical artifacts. No license is granted here for third-party review text, pretrained models, or files outside this revision folder.
