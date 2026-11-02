#!/usr/bin/env python3
"""Download a PDF for every entry in references.bib into "Reference papers/".

Run from anywhere:
    python3 get_reference_papers.py

Only uses the Python standard library. Files are saved as <bibkey>.pdf and
existing files are skipped, so it is safe to run again. For arXiv papers the
title on arXiv is compared with the title in references.bib and a warning is
printed if they do not match.
"""

import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "AditPatel_BiomedicalML_CSE_8673"
BIB = PROJECT / "references.bib"
OUT = PROJECT / "Reference papers"

# bib key -> arXiv id, or a direct PDF URL for papers not on arXiv.
SOURCES = {
    "tomani2023dac": "2302.05118",
    "jin2024fairmedfm": "2407.00983",
    "guo2017calibration": "1706.04599",
    "ovadia2019trust": "1906.02530",
    "tomani2021posthoc": "2012.10988",
    "yu2022multidomain": "2206.02757",
    "wang2024cascaded": "2402.09204",
    "roschewitz2025where": "2507.07780",
    "hendrycks2019benchmarking": "1903.12261",
    "disalvo2024medmnistc": "2406.17536",
    "pleiss2017fairness": "1709.02012",
    "liu2019implicit": "1808.10013",
    "sagawa2020groupdro": "1911.08731",
    "salvador2022faircal": "2106.03761",
    "shui2023clusterfocal": "2307.01738",
    "riccilara2023calibration": "2305.05101",
    "zong2023medfair": "2210.01725",
    "shen2025calin": "2506.23298",
    "dominique2025calibration": "2511.07700",
    "lin2026crsbench": "2608.22059",
    "tschandl2018ham10000": "1803.10417",
    "groh2021fitzpatrick": "2104.09957",
    "abhishek2025quality": "https://www.nature.com/articles/s41597-025-04382-5.pdf",
    "irvin2019chexpert": "1901.07031",
    "dosovitskiy2021vit": "2010.11929",
    "oquab2024dinov2": "2304.07193",
    "zhang2025biomedclip": "2303.00915",
    "nguyen2023lvmmed": "2306.11925",
    "he2016resnet": "1512.03385",
    "hu2022lora": "2106.09685",
    # Added with the Phase 1 survey. arXiv ids checked against arXiv titles on 2026-10-04.
    "kull2019dirichlet": "1910.12656",
    "zhang2020mixnmatch": "2003.07329",
    "gupta2021splines": "2006.12800",
    "tomani2022pts": "2102.12182",
    "gal2016dropout": "1506.02142",
    "lakshminarayanan2017ensembles": "1612.01474",
    "muller2019labelsmoothing": "1906.02629",
    "thulasidasan2019mixup": "1905.11001",
    "mukhoti2020focal": "2002.09437",
    "liu2022margin": "2111.15430",
    "kumar2022finetuning": "2202.10054",
    "oh2024calibrated": "2311.01723",
    "murugesan2024robust": "2407.13588",
    "wang2020transferable": "2007.08259",
    "gong2021confidence": "2104.00742",
    "wald2021calibration": "2102.10395",
    "sun2022knn": "2204.06507",
    "zou2023ace": "https://openaccess.thecvf.com/content/ICCV2023/papers/Zou_Adaptive_Calibrator_Ensemble_Navigating_Test_Set_Difficulty_in_Out-of-Distribution_Scenarios_ICCV_2023_paper.pdf",
    "yuksekgonul2023atypicality": "2305.18262",
    "kim2024energy": "2407.12330",
    "kleinberg2017inherent": "1609.05807",
    "hebertjohnson2018multicalibration": "1711.08513",
    "idrissi2022simple": "2110.14503",
    "dutt2024fairtune": "2310.05055",
    "zhang2022improving": "2203.12609",
    "schrouff2022diagnosing": "2202.01034",
    "mehta2023evaluating": "2303.03242",
    "ktena2024generative": "2304.09218",
    "yang2024limits": "2312.10083",
    "gupta2026subgroup": "2608.19078",
    "minderer2021revisiting": "2106.07998",
    "tu2024empirical": "2402.07417",
    "hekler2025beyond": "2506.09593",
    "khan2023fair": "https://proceedings.mlr.press/v225/khan23a/khan23a.pdf",
    "wang2023survey": "2308.01222",
    "shi2024survey": "2407.15851",
    "huang2026uncertainty": "2608.30390",
    "nixon2019measuring": "1904.01685",
    "kumar2019verified": "1909.10155",
    "roelofs2022mitigating": "2012.08668",
    "gruber2022proper": "2203.07835",
    "radford2021clip": "2103.00020",
    # Added from the Connected Papers graphs of the two seeds on 2026-10-06.
    "hekler2023tta": "https://ojs.aaai.org/index.php/AAAI/article/view/26735/26507",
    "yang2023beyond": "2306.04985",
    "tian2024fairdomain": "2407.08813",
}

# Papers with no free PDF link the script can rely on. Get these by hand
# (through the MSU library if they are paywalled) and save them as <bibkey>.pdf.
MANUAL = {
    "platt1999probabilistic": "book chapter, Advances in Large Margin Classifiers (MIT Press, 1999)",
    "zadrozny2002transforming": "KDD 2002, ACM Digital Library",
    "naeini2015bbq": "AAAI 2015, free copy on PubMed Central (PMC4410090)",
    "seyyedkalantari2021underdiagnosis": "Nature Medicine 27 (2021), open access on nature.com",
    "moor2023generalist": "Nature 616 (2023), nature.com",
    "brier1950verification": "Monthly Weather Review 78(1), AMS journals site",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (CSE 8673 reference collector)",
    "Accept": "application/pdf,*/*",
}


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def bib_titles(text):
    """Return {key: title} from a .bib file."""
    titles = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        t = re.search(r"title\s*=\s*\{(.*)\},?\s*$", body, re.M)
        if t:
            titles[key] = t.group(1)
    return titles


def words(s):
    s = re.sub(r"\\['`^\"~]\{?(\w)\}?", r"\1", s)  # strip LaTeX accents
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def arxiv_title(arxiv_id):
    xml = fetch(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    return " ".join(entry.find("a:title", ns).text.split())


def main():
    titles = bib_titles(BIB.read_text(encoding="utf-8"))
    missing = sorted(set(titles) - set(SOURCES) - set(MANUAL))
    if missing:
        print("No source listed for:", ", ".join(missing))

    OUT.mkdir(exist_ok=True)
    failed = []
    for key in titles:
        if key not in SOURCES:
            continue
        dest = OUT / f"{key}.pdf"
        if dest.exists():
            print(f"skip   {key} (already downloaded)")
            continue
        src = SOURCES[key]
        is_arxiv = not src.startswith("http")
        urls = (
            [f"https://arxiv.org/pdf/{src}", f"https://export.arxiv.org/pdf/{src}"]
            if is_arxiv
            else [src]
        )
        try:
            # Check the arXiv title first so a wrong id never saves a wrong paper.
            if is_arxiv:
                found = arxiv_title(src)
                a, b = words(found), words(titles[key])
                if len(a & b) < 0.6 * min(len(a), len(b)):
                    raise ValueError(f"title mismatch, arXiv {src} is '{found}'")
            data, err = None, None
            for url in urls:
                try:
                    data = fetch(url)
                    break
                except Exception as e:
                    err = e
            if data is None:
                raise err
            if not data.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            dest.write_bytes(data)
            print(f"saved  {key} ({len(data) // 1024} KB)")
        except Exception as e:  # keep going so one bad link does not stop the rest
            print(f"FAILED {key}: {e}")
            failed.append(key)
        time.sleep(3)  # arXiv asks for a few seconds between requests

    manual = [k for k in titles if k in MANUAL and not (OUT / f"{k}.pdf").exists()]
    have = sum((OUT / f"{k}.pdf").exists() for k in titles)
    print(f"\nDone. {have} of {len(titles)} papers in '{OUT}'.")
    if manual:
        print("Not downloadable by this script, save these by hand as <bibkey>.pdf:")
        for k in manual:
            print(f"  {k}: {MANUAL[k]}")
    if failed:
        print("Download these by hand:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
