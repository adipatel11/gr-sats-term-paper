# GR-SATS code

Code for the experiments in the paper. It loads the two datasets, trains the linear probes and LoRA models on DINOv2 and BiomedCLIP, runs the MedMNIST-C corruptions through them, fits the calibrators, and makes the paper's tables and figures.

## Files

- `config.py`: paths and settings
- `data.py`: loads HAM10000 and Fitzpatrick17k-C with their splits and groups
- `models.py`: the two backbones and the LoRA layers
- `metrics.py`: AUC and NLL, overall and per group
- `check_data.py`: prints image counts, these should match the data table in the paper
- `extract_features.py`: saves frozen features (GPU)
- `train_probes.py`: linear probes on the saved features (CPU is fine)
- `train_lora.py`: LoRA models (GPU)
- `corrupt_extract.py`: runs the 15 MedMNIST-C corruptions at 5 severities over the val + test images and saves every model's outputs (GPU)
- `calibrate.py`: fits TS, GR-SATS and the other calibrators on the corrupted outputs with cross-fitting and compares them (CPU is fine)
- `calibrate_extra.py`: the extra baselines, ablations and oracles, refits the `calibrate.py` methods too and checks them against `results/` (CPU is fine)
- `rank_tests.py`: Friedman + Nemenyi, H2, and the pooled bootstrap success test over all 8 settings (local)
- `make_tables.py`: the LaTeX tables in `tables/` and the matplotlib figures in `figures/`, all read from `results/` (local)
- `results/`: result CSVs for the 8 settings, `rank_tests.txt` is the saved output of `rank_tests.py`
- `test_code.py`: tests, run with `python -m pytest -q`
- `notebooks/`: what I run on Colab

## How to run

Run everything from this folder. There are 4 dataset and backbone combinations, each with 3 seeds, so 12 probes and 12 LoRA models.

```
pip install -r requirements.txt
python check_data.py
python extract_features.py --dataset ham --backbone dinov2
python train_probes.py --dataset ham --backbone dinov2
python train_lora.py --dataset ham --backbone dinov2
python corrupt_extract.py --dataset ham --models frozen
python calibrate.py --dataset ham --models frozen
python calibrate_extra.py --dataset ham --models frozen
```

Once all 8 settings are in `results/`, these run on a laptop:

```
python rank_tests.py
python make_tables.py
```

`--dataset` is `ham` or `fitz` and `--backbone` is `dinov2` or `biomedclip`. If Colab disconnects during `train_lora.py`, run the same command again and it picks up from the last epoch. `corrupt_extract.py` works the same way, finished corruptions get skipped.

`corrupt_extract.py` needs MedMNIST-C, which pins an old scikit-image that won't build on Colab, so install it with `pip install --no-deps medmnistc`. Its motion blur also needs ImageMagick (`apt-get install libmagickwand-dev` on Colab, `brew install imagemagick` on a Mac).

## Data

The images are not in the repo. See the main README for where to download them. The split CSVs are in `data/` at the project root.
