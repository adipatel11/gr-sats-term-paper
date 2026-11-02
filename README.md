# GR-SATS: Group-Robust Shift-Aware Temperature Scaling

Term paper for CSE 8673 (Machine Learning) at Mississippi State University, Fall 2026, with Dr. Sravani Kurma.

The project asks whether calibration fit on clean images still holds up when dermatology images get corrupted, and whether it holds up for every patient group and not just on average. GR-SATS is a post-hoc calibrator. It gives each image its own temperature based on how far that image is from the training data (a kNN shift score), and it is fit to minimize the worst group's negative log-likelihood instead of the average. Models are DINOv2 and BiomedCLIP, adapted with linear probing or LoRA, on HAM10000 and Fitzpatrick17k with MedMNIST-C corruptions.

## What's in here

| Path | What it is |
| --- | --- |
| `AditPatel_BiomedicalML_CSE_8673/` | The paper (`main.tex`, `references.bib`, ACM class files). This is what gets uploaded to Overleaf. |
| `AditPatel_BiomedicalML_CSE_8673.pdf` | Latest compiled PDF from Overleaf |
| `AditPatel_BiomedicalML_CSE_8673/literature_list.csv` | List of the papers in the literature review |
| `data/DermaMNIST-C.csv`, `data/Fitzpatrick17k-C.csv` | Split and label files used for the data counts in the paper |
| `get_reference_papers.py` | Downloads a PDF for every entry in `references.bib` |
| `Approved_Topic_Phase0.txt`, `Seed_Papers_APatel_detailed_plan.txt` | Phase 0 topic approval and seed paper plan |
| `Phase 2/`, `Phase 3/` | Phase instructions, course template and backups of earlier drafts |
| `Seed paper graphs from connected papers/` | Connected Papers exports for the two seed papers |
| `term_paper_description.txt` | Course description of the term paper |

## Not in the repo

The raw datasets and the reference paper PDFs are left out on purpose (see `.gitignore`). They are several GB, and the dataset licenses don't allow sharing copies.

- HAM10000: download from the Harvard Dataverse (doi:10.7910/DVN/DBW86T) into `data/HAM10000/dataverse_files/`
- Fitzpatrick17k: get the images by following the steps at github.com/mattgroh/fitzpatrick17k and put them in `data/finalfitz17k/`
- Reference PDFs: run `python3 get_reference_papers.py` from the project root. They get saved to `AditPatel_BiomedicalML_CSE_8673/Reference papers/`

## Building the paper

I compile on Overleaf. Upload the contents of `AditPatel_BiomedicalML_CSE_8673/` (without the reference PDFs) and compile with pdfLaTeX. Locally, `latexmk -pdf main.tex` inside that folder should also work if you have a TeX install.

## Status

Phase 2 (methodology and experiment plan) is done. Phase 3 (experiments and results) is in progress, so there is no experiment code here yet.
