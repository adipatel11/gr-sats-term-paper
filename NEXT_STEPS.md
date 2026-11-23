# Where things stand (Nov 22, 2026)

## Done

- Data loading and splits, checked against the paper's data table (`code/check_data.py`).
- 12 linear probes and 12 LoRA models (2 datasets x 2 backbones x 3 seeds). Notebooks 1 and 2.
- All 15 MedMNIST-C corruptions at 5 severities, run through every model. Notebook 3.
- Calibration for all 8 settings with every method: none, TS clean, TS corr, TS-WG, DAC, AAR, SATS-avg, GR-SATS, GR-SATS + bias.
- Results are in `code/results/` (8 CSVs). Last commit: f3df551.
- Model weights and corrupted outputs are only on Google Drive (`MyDrive/grsats/outputs`). You don't need Colab again unless we add new experiments.

## Main results (worst-group NLL, held-out corruption families, severity 3 to 5, mean of 3 seeds)

| Setting | none | TS clean | TS corr | TS-WG | DAC | AAR | SATS-avg | GR-SATS |
|---|---|---|---|---|---|---|---|---|
| HAM DINOv2 LP | .550 | .638 | .528 | .488 | .487 | .503 | .487 | **.461** |
| HAM BiomedCLIP LP | .569 | .628 | .518 | .474 | .457 | .469 | .461 | **.442** |
| HAM DINOv2 LoRA | .488 | .521 | .450 | .425 | .443 | .435 | .445 | **.423** |
| HAM BiomedCLIP LoRA | .510 | .537 | .459 | **.434** | .440 | .462 | .461 | .435 |
| Fitz DINOv2 LP | .387 | .389 | .367 | .362 | .337 | **.332** | .334 | **.332** |
| Fitz BiomedCLIP LP | .511 | .521 | .471 | .463 | .412 | .421 | .405 | **.396** |
| Fitz DINOv2 LoRA | .385 | .342 | .333 | .332 | .332 | .330 | .329 | **.328** |
| Fitz BiomedCLIP LoRA | .512 | .447 | .412 | .407 | .378 | .412 | .372 | **.369** |

Bootstrap (24 tests per comparison, GR-SATS wins / ties / losses):
TS clean 24/0/0, TS corr 24/0/0, SATS-avg 22/2/0, DAC 22/1/1, TS-WG 19/3/2, AAR 18/5/1 (recounted from the bootstrap CSVs, this used to say 16/7/1).
All losses are on HAM BiomedCLIP LoRA.

## Points for the Results and Discussion sections

1. The two parts of GR-SATS help in different places. Shift part = TS-WG minus GR-SATS, group part = SATS-avg minus GR-SATS (DINOv2 / BiomedCLIP):
   - HAM LP: shift .027 / .032, group .026 / .019 (both matter)
   - HAM LoRA: shift .002 / -.001, group .023 / .026 (group only)
   - Fitz LP: shift .030 / .067, group .002 / .010 (shift mostly)
   - Fitz LoRA: shift .004 / .039, group .002 / .004 (shift mostly)
2. When the shift score stops pointing at the worst group, GR-SATS falls back to TS-WG (HAM BiomedCLIP LoRA seeds 0 and 2: 60+ is no longer more atypical than <60).
3. H2 holds: LoRA shrinks the gain over TS corr (HAM about .07 to .025, Fitz DINOv2 .034 to .006, Fitz BiomedCLIP .075 to .043).
4. On Fitz the worst group is always I-II, not V-VI. V-VI has the lowest NLL because its cancer rate is lowest (8.8%). Explain this in the discussion.
5. Costs to report honestly:
   - Clean ECE is within 0.01 of clean TS in only 2 of 8 settings (the paper's criterion fails overall).
   - The best-off group gets worse on HAM (<60 NLL about .17 to .20).
   - Clean worst-group NLL goes up about .01 on HAM.
6. TS fit on clean data is worse than no calibration on corrupted HAM images (all 4 HAM settings). This supports the motivation.
7. The bias variant lowers accuracy and isn't clearly better, so the main method is GR-SATS without bias.
8. AAR had one bad seed (HAM BiomedCLIP LoRA seed 1) because its quadratic is unconstrained and extrapolates on unseen corruption families. That's how the original AAR works.
9. AUC of the global TS rows moves slightly only because cross-fitting gives each fold its own temperature. Quote AUC from the "none" row or explain it in one sentence.

## Edits still needed in main.tex on Overleaf

- Exact kNN in PyTorch instead of FAISS.
- SciPy SLSQP (epigraph form) instead of CVXPY.
- LoRA early stopping patience is 5.
- MedMNIST-C black_corner severities are [1.10, 1.05, 1.00, 0.90, 0.95], so level 4 is stronger than level 5 (official setting, kept).
- DAC uses only the final layer (the official ViT version uses 13 layers), and T >= 1e-3 is a constraint instead of the official clip.
- GR-SATS limitation: the inverse temperature floor (0.05) can't soften very large wrong logits as much as DAC or AAR can.
- Replace "I will write the code" wording. The LLM section must say Claude wrote most of the code.
- Fill the planned Results tables with the numbers above.
- Cut from the paper (not run): TS with textures, multi-layer DAC, the ImageNet ViT and full fine-tuning stretch goals, and the Week 3 check against the official DAC code. Say so in the paper.
- DAC-WG is fit with SLSQP in epigraph form (from DAC's start and from the average-NLL fit, keeping the better one), not GroupDRO's alternating updates. It's not convex, so it's a local optimum.
- TS with Gaussian noise uses MedMNIST-C's gaussian_noise at all 5 severities, so compare it with the "wg_nll no noise" column (noise family left out of the test pool).
- Multi-domain TS: least squares from features to each domain's temperature (Yu et al. Alg. 1), with the predicted temperature clipped to the range of the domain temperatures (my addition).
- Cluster-wise TS: k-means (8 clusters) on 20,000 random fitting rows of L2-normalized features.
- Oracles use group cells: sex x age on HAM (4 cells), skin type on Fitz.

## Added Nov 22 (after the full run)

- `code/calibrate_extra.py`: TS with Gaussian noise, GR-SATS with excess NLL, k = 1 and 50, a group-balanced reference bank, DAC fit for average NLL, DAC-WG, cluster-wise TS, multi-domain TS, group-wise TS oracle, per-group GR-SATS oracle. It refits the old methods too and prints how far they are from `results/`.
- It writes 5 files per setting: `_extra.csv` (main metrics for all methods, plus debiased ECE, group ECE and ECE gaps), `_detail.csv` (NLL and Brier for every group, corruption and severity, for H1 and the per-family results), `_reliability.csv` (reliability diagram bins), `_scores.csv` (shift score per group, clean vs severity 3 to 5, for each k and bank), `_perimage.npz` (for the pooled bootstrap).
- `code/rank_tests.py` (runs locally): Friedman + Nemenyi, H2, and the success test pooled over all 8 settings.
- `code/notebooks/04_extra.ipynb` runs it on Colab.

## Added Nov 23 (extra run finished)

- Drive was full, so notebook 05 remade the lost corruption files on Colab's disk. All 4 settings of `calibrate_extra.py` match `results/*_calibration.csv` (5.6e-17). The big corruption files were not kept (not needed for the paper).
- `calibrate.fit` falls back to a 1-d search when SLSQP gives up on a one-parameter fit (happened 4 times in multi-domain TS on Fitz LoRA DINOv2 seed 2).
- `rank_tests.py` output saved in `code/results/rank_tests.txt`:
  - Friedman chi2 = 53.2, p = 1e-8, Nemenyi CD = 4.25. Mean ranks: GR-SATS 1.75, TS Gaussian 2.00, DAC 3.38, AAR 4.00, cluster TS 4.88, MD-TS 5.88, TS corr 6.12, none 8.25, TS clean 8.75. GR-SATS is significantly better than TS corr, none and TS clean, not than DAC/AAR/MD-TS by Nemenyi.
  - H3 success test passes against all 3: GR-SATS minus TS corr / DAC / AAR, WG NLL -.044 / -.013 / -.022, WG Brier -.0049 / -.0017 / -.0031, all 95% CIs exclude 0.
  - H2: clean TS harms the worst group from severity 0 or 1 in all 4 HAM settings and Fitz LP (tiny on Fitz DINOv2 LP), and never harms on Fitz LoRA.
- `make_tables.py` writes `code/tables/` (results, settings, ablation, success, families) and `code/figures/` (severity, groups, reliability, cd), all matplotlib from `results/`. Dr. Kurma: no AI-generated images.
- Things the paper must say honestly:
  - DAC-WG (.397) and GR-SATS (.398) are tied on average, and GR-SATS matches the per-group oracle (.398). Most of the gain is the worst-group objective.
  - TS with Gaussian noise ranks 2nd, but it saw the noise family. In the no-noise column GR-SATS .354 vs TS Gaussian .365, and TS Gaussian has the worst clean ECE (.073).
  - k = 1/10/50 and the balanced bank change WG NLL by at most .001.
  - Per family: GR-SATS gains most on noise (HAM .572 vs TS corr .728), while TS corr is best on HAM color and task-specific.
  - Reliability: without calibration HAM 60+ risk is underestimated under corruption, GR-SATS is closest to the diagonal.

## What wasn't run (for the poster and for questions)

Poster text:

**Not run:** official DAC and AAR code (both are reimplementations, DAC uses the final layer only), TS with unrelated textures, ImageNet ViT, full fine-tuning, a from-scratch rerun of one setting, per-severity reliability diagrams, and per-group ECE by severity.

If asked:

- Official code: I reimplemented DAC and AAR from the papers and DAC's published code and checked DAC with a unit test (constant distance gives plain TS). I never compared numbers with the official code, so a full 13-layer DAC could be stronger.
- Textures: Roschewitz et al. found this the most robust simple method, so it's the missing baseline that matters most.
- ImageNet ViT and full fine-tuning: stretch goals, dropped for time.
- From-scratch rerun: not done. The probe refit and the calibrator refits (5.6e-17) matched. LoRA was trained once per seed.
- Everything else in the results is finished: 8 settings, 3 seeds, 15 corruptions, 5 severities.

## Next steps, in order

Phase 3 poster is due Thursday, Nov 26, 2026 at 11:59pm (Phase 3 Canvas Discussions).

1. Done Nov 23: main.tex and the 4 figures updated on Overleaf, recompiled, and the new PDF is here. (Three claims fixed in the abstract, Contribution 1, 6.1, 6.4 and 7.2, legend added to `groups`, axis title added to `cd`.)
2. Make the poster (single-slide PPTX), figures from `code/figures/` only.
3. Write and rehearse the 15-minute talk.

`make_tables.py` was rerun on Nov 23. The tables came out identical, and the new figures and main.tex are on Overleaf.
