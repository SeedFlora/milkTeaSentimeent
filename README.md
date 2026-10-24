# milkTeaSentimeent

The retained non-text artifacts for the revised five-split benchmark are available in [revision-20261024](revision-20261024/README.md). That folder includes its own scope, verification code, privacy exclusions and known limitations. The historical baseline below remains separate.

Historical dataset and training code for the conference paper:

**Hybrid Machine Learning versus Fine-Tuned Transformers for Indonesian Milk Tea Review Sentiment Analysis: A Performance and Computational Cost Benchmark**

This release contains the original binary sentiment benchmark for Indonesian milk tea outlet reviews. It corresponds to the paper's original single-split experiment, using historical files archived in June 2026. The collection made on 23 October 2026 is a separate dataset.

## Dataset

| File | Rows | Positive | Negative |
|---|---:|---:|---:|
| `dataset_clean.csv` | 5,063 | 4,223 | 840 |
| `data/train.csv` | 6,756 | 3,378 | 3,378 |
| `data/val.csv` | 506 | 422 | 84 |
| `data/test.csv` | 507 | 423 | 84 |

Each CSV contains:

| Column | Description |
|---|---|
| `text_clean` | Normalized review text used by the models |
| `label` | `positif` or `negatif` |
| `label_id` | `1` for positive; `0` for negative |

Labels derive from ratings: 4-5 stars are positive and 1-3 stars are negative. The released `dataset_clean.csv` retains only these three columns from the historical cleaned corpus, preserving its row order, cleaned text, and labels. Reviewer identifiers, original unprocessed text, and source metadata are excluded.

## Original experimental protocol

The 5,063 cleaned reviews were split using stratified sampling with seed **42**, approximately **80/10/10**:

- Training: 4,050 original reviews.
- Validation: 506 reviews.
- Test: 507 reviews.

Synonym replacement added **2,706 negative training examples**, producing the saved training set of 6,756 rows. Validation and test sets retain their original class distributions and contain no augmented examples. There is no exact cleaned-text overlap between the three splits.

**Use the saved files in `data/` as the canonical experimental splits.** These three files preserve the historical split files byte for byte.

## Code and reproducibility

Install dependencies and run the historical training scripts:

```bash
python -m pip install -r requirements.txt
python train_hybrid.py
python train_transformers.py
```

Transformer training requires CUDA with bfloat16 support and downloads pretrained models. Dependencies are not a frozen reconstruction of the historical training environment.

The scripts retain their historical behavior, with file paths changed to be repository-relative:

- `clean_dataset.py`: original cleaning and augmentation resources. Cleaning from scratch requires the unreleased raw source export.
- `prepare_data.py`: seed-42 splitting and training augmentation; running it overwrites the saved splits.
- `train_hybrid.py`: classical feature and classifier benchmarks.
- `train_transformers.py`: transformer fine-tuning. Epoch selection uses validation macro F1; the overall saved model is selected using test macro F1, as in the historical script.

Regenerating augmentation can depend on the software environment; the saved splits remain the reference inputs. `manifest.json` records dataset counts, transformations, and SHA-256 checksums.

The globally balanced exploratory corpus, later five-seed revision experiments, new scraping and annotation work, manuscripts, and confidential materials are outside this release.
