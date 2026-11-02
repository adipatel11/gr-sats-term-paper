# GR-SATS code

Code for the experiments in the paper. Right now it covers step 1: loading the two datasets with their splits, and training linear probes on frozen DINOv2 and BiomedCLIP features.

## Folders

```
code/
├── grsats/                    reusable code that the scripts and notebooks import
│   ├── config.py              paths, seeds, probe settings
│   ├── data/
│   │   ├── ham10000.py        HAM10000 + DermaMNIST-C splits, label, sex and age groups
│   │   ├── fitzpatrick17k.py  Fitzpatrick17k-C splits, label, skin-type groups
│   │   ├── splits.py          10% dev slice from train (later: cross-fitting folds)
│   │   └── image_dataset.py   PyTorch Dataset over either table
│   ├── models/
│   │   └── backbones.py       DINOv2 and BiomedCLIP plus their preprocessing
│   ├── features.py            run a backbone once and save the features
│   ├── probes.py              logistic regression probe, pick strength by dev NLL
│   └── metrics.py             AUC and NLL, overall and per group
├── scripts/                   what you actually run
│   ├── check_data.py
│   ├── extract_features.py
│   └── train_probes.py
├── notebooks/
│   └── 01_data_and_probes.ipynb   Colab driver for the scripts
├── tests/
│   └── test_data.py
├── outputs/                   features, logits, results (not committed)
└── requirements.txt
```

## Order to run

1. `scripts/check_data.py`: counts should match the data table in the paper (HAM10000 9,948 images, Fitzpatrick17k-C 11,098).
2. `scripts/extract_features.py`: once per dataset and backbone (4 runs, GPU).
3. `scripts/train_probes.py`: 3 seeds each, so 12 linear probes in total (CPU is fine).

## Data

The raw images are not in the repo. See the main README for where to download HAM10000 and Fitzpatrick17k. The split CSVs are in `data/` at the project root.
