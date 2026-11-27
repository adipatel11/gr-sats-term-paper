# Where things stand (Nov 23, 2026)

The experiments are finished and the paper is written. What's left for Phase 3 is the poster and the talk. Poster numbers should come from the paper's tables (`code/tables/`), and this file now matches them.

## Done

- Data loading and splits, checked against the paper's data table (`code/check_data.py`).
- 12 linear probes and 12 LoRA models (2 datasets x 2 backbones x 3 seeds). Notebooks 1 and 2.
- All 15 MedMNIST-C corruptions at 5 severities, run through every model. Notebook 3.
- Calibration for all 8 settings with every method, plus the extra baselines, ablations and oracles (`calibrate.py`, `calibrate_extra.py`, notebooks 3 to 5).
- Results are in `code/results/` (CSVs, per-image files for the bootstrap, and `rank_tests.txt`).
- Model weights and the original outputs are only on Google Drive (`MyDrive/grsats/outputs`). Drive was full, so notebook 05 remade the lost corruption files on Colab's disk. They were not kept. You don't need Colab again unless we add new experiments.
- Paper: results, discussion, limitations, AI use and conclusion are all in `main.tex`.
- Full check on Nov 23: the numbers in Sections 6 and 7 were recomputed from the CSVs, the 5 tables came out identical, all 18 unit tests pass, and the PDF has no broken references.

## Main results (worst-group NLL, held-out corruption families, severity 3 to 5, mean of 3 seeds)

TS-WG and SATS-avg per setting are not in the paper, the other columns match Table 7 (it was Table 6 before the complexity table was added on Nov 26). The paper also has TS with Gaussian noise, multi-domain TS, cluster-wise TS and the oracles.

| Setting | none | TS clean | TS Gauss | TS corr | TS-WG | DAC | AAR | SATS-avg | GR-SATS |
|---|---|---|---|---|---|---|---|---|---|
| HAM DINOv2 LP | .550 | .638 | .464 | .528 | .488 | .487 | .503 | .487 | **.461** |
| HAM BiomedCLIP LP | .569 | .628 | .460 | .518 | .473 | .457 | .469 | .461 | **.442** |
| HAM DINOv2 LoRA | .488 | .521 | **.418** | .450 | .425 | .443 | .435 | .445 | .423 |
| HAM BiomedCLIP LoRA | .510 | .537 | **.428** | .459 | .434 | .440 | .462 | .461 | .435 |
| Fitz DINOv2 LP | .387 | .388 | .350 | .367 | .362 | .337 | **.332** | .334 | **.332** |
| Fitz BiomedCLIP LP | .511 | .521 | **.394** | .471 | .463 | .412 | .421 | .405 | .396 |
| Fitz DINOv2 LoRA | .385 | .342 | **.321** | .333 | .332 | .332 | .330 | .329 | .328 |
| Fitz BiomedCLIP LoRA | .512 | .447 | .374 | .412 | .407 | .377 | .412 | .372 | **.368** |

Bootstrap, each setting and seed separately (24 tests per comparison, GR-SATS wins / ties / losses):
TS clean 24/0/0, TS corr 24/0/0, SATS-avg 22/2/0, DAC 22/1/1, TS-WG 19/3/2, AAR 18/5/1.
All losses are on HAM BiomedCLIP LoRA.

## Hypotheses (same definitions as the paper)

- H1 (clean-fit TS gets worse with severity, and faster for the worst group than for the best one): holds in absolute terms in all 8 settings. In relative terms the difference is small (HAM 2.2 times for 60+ against 2.0 times for under 60).
- H2 (the severity where clean-fit TS starts doing harm is different for LP and LoRA): holds on Fitz but not on HAM. On HAM clean TS does harm from severity 0 or 1 in all 4 settings. On Fitz it does a little harm for LP (at most .012) and never for LoRA.
- H3 (GR-SATS passes the success test): passes against all 3. GR-SATS minus TS corr / DAC / AAR is -.044 / -.013 / -.022 in WG NLL and -.0049 / -.0017 / -.0031 in WG Brier, and all 95% CIs exclude 0.
- Not part of H2: the gain over TS corr is smaller for LoRA than for LP (HAM .067 and .076 to .027 and .024, Fitz DINOv2 .034 to .006, Fitz BiomedCLIP .075 to .043).
- Friedman chi2 = 53.2, p = 1e-8, Nemenyi CD = 4.25. Mean ranks: GR-SATS 1.75, TS Gaussian 2.00, DAC 3.38, AAR 4.00, cluster TS 4.88, MD-TS 5.88, TS corr 6.12, none 8.25, TS clean 8.75. GR-SATS is significantly better than TS corr, none and TS clean, but not than the other five.

## Points for the poster and talk

1. The two parts of GR-SATS help in different places. Shift part = TS-WG minus GR-SATS, group part = SATS-avg minus GR-SATS (DINOv2 / BiomedCLIP):
   - HAM LP: shift .027 / .032, group .026 / .019 (both matter)
   - HAM LoRA: shift .002 / -.001, group .023 / .026 (group only)
   - Fitz LP: shift .029 / .067, group .002 / .010 (shift mostly)
   - Fitz LoRA: shift .004 / .039, group .002 / .004 (shift mostly)
2. DAC-WG (.397) and GR-SATS (.398) are tied, and GR-SATS matches the per-group oracle (.398). The gain over DAC comes from the worst-group objective and not from the form of the calibrator.
3. Most of the gain is on noise (HAM .572 vs TS corr .728, Fitz .447 vs .555). Without noise the gain over TS corr drops from .044 to .009.
4. TS with Gaussian noise is better in 4 of 8 settings, but it saw the noise family. Without noise GR-SATS is .354 and TS Gaussian .365, and TS Gaussian has the worst clean ECE (.073).
5. On Fitz the worst group is always I-II and not V-VI. V-VI has the lowest NLL mostly because its cancer rate is lowest (8.8% against 16.0%).
6. Costs to report honestly:
   - Clean ECE is .028 against .013 for clean TS, and within 0.01 of it in only 2 of 8 settings, so that criterion fails. The other two secondary criteria hold.
   - Patients under 60 get worse on HAM (NLL .152 to .174 against TS corr, mean risk 11.4% when 5.1% are malignant).
   - Clean worst-group NLL goes up about .01 on HAM.
   - The ECE gap only shrinks for age (.058 to .038 against TS corr). It goes up for sex (.007 to .012) and skin type (.012 to .020), and V-VI ECE goes from .038 to .040. Overall ECE under corruption is .030, DAC and AAR are .024. Don't say GR-SATS makes the groups more equal in general.
7. Clean TS is worse than no calibration on corrupted HAM images in all 4 settings. This supports the motivation.
8. HAM BiomedCLIP LoRA is the weakest setting. In seeds 0 and 2, 60+ has lower clean scores than under 60, but the seeds where GR-SATS loses to TS-WG (1 and 2) don't line up with that, so don't claim it's the cause.
9. AAR had one bad seed there (seed 1, .536) because its quadratic is unconstrained and extrapolates. That's how the original AAR works.
10. The bias variant lowers accuracy and isn't clearly better, so the main method has no bias. k = 1/10/50 and the balanced bank change WG NLL by at most .001.

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

1. Done Nov 23: `main.tex` updated on Overleaf and the new PDF is here. Last edits: 5.4 (TS with corruptions is no longer called the strongest simple baseline), Table 2 row 2, 7.1 (NLL is still higher under corruption than on clean images), and the Figure 4 alt text and Figure 6 caption. Later the same day: 6.3 (excess NLL gives up all of the group-loss gain) and 7.5 (debiased ECE and per-attribute worst groups were planned but not reported). After that, 6.4 (the ECE gap between the sexes and between skin types goes up compared with TS corr) and y-axis labels on every panel of Figures 4, 5 and 6. The tables did not change.
   - Done Nov 24, on Overleaf and in the PDF here (15 pages): Dr. Kurma asked for at least four result figures with trends against the baselines, plus one process figure. Added `brier`, `families` and `tradeoff` (Figures 3, 7 and 9, Sections 6.1 and 6.4). With the severity figure (now Figure 5) that makes four. The process figure is Figure 2. Old Figures 3 to 6 are now 4, 5, 6 and 8.
   - Done Nov 26, in `main.tex` and the new PDF here: Dr. Kurma's items 6 and 7. New Section 4.4 "Computational Complexity" with Table 3 (cost of each calibrator). Table 1 now has a GR-SATS (this work) row and Shift / Groups / Post-hoc columns, and GR-SATS is the only row with all three. Every later table moved up by one (old Tables 3 to 9 are now 4 to 10). Table 1 is `[tp]` now, so check on Overleaf that it still lands near Section 2.
2. Make the poster (single-slide PPTX), figures from `code/figures/` only. `framework.png` is the method diagram and `taxonomy.png` is the literature figure. Its numbers in brackets are the paper's references.
3. Write and rehearse the 15-minute talk. Check on Canvas if it's live or recorded.

Phase 4 ZIP, later: follow `Phase 4/task.md`, put the 9 PNGs in `06_Figures/`, and leave out `Reference papers/`. If Figure 1 or 2 changes in `main.tex`, the two crops have to be remade.
