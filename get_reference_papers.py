# downloads a pdf for every paper in references.bib into "Reference papers/"
# saved as <bibkey>.pdf, papers that are already there get skipped
import re
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent / "AditPatel_BiomedicalML_CSE_8673"
OUT = PROJECT / "Reference papers"

# bib key -> arxiv id or a direct pdf link
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
    "hekler2023tta": "https://ojs.aaai.org/index.php/AAAI/article/view/26735/26507",
    "yang2023beyond": "2306.04985",
    "tian2024fairdomain": "2407.08813",
}

# no free link for these, get them by hand and save as <bibkey>.pdf
MANUAL = {
    "platt1999probabilistic": "book chapter, Advances in Large Margin Classifiers (MIT Press, 1999)",
    "zadrozny2002transforming": "KDD 2002, ACM Digital Library",
    "naeini2015bbq": "AAAI 2015, free copy on PubMed Central (PMC4410090)",
    "seyyedkalantari2021underdiagnosis": "Nature Medicine 27 (2021), open access on nature.com",
    "moor2023generalist": "Nature 616 (2023), nature.com",
    "brier1950verification": "Monthly Weather Review 78(1), AMS journals site",
}

keys = re.findall(r"@\w+\{([^,]+),", (PROJECT / "references.bib").read_text(encoding="utf-8"))
OUT.mkdir(exist_ok=True)

for key in keys:
    dest = OUT / f"{key}.pdf"
    if dest.exists():
        continue
    if key not in SOURCES:
        print(f"get by hand: {key} ({MANUAL.get(key, 'no source listed')})")
        continue

    src = SOURCES[key]
    url = src if src.startswith("http") else f"https://arxiv.org/pdf/{src}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        pdf = urllib.request.urlopen(req, timeout=60).read()
        assert pdf.startswith(b"%PDF"), "not a pdf"
        dest.write_bytes(pdf)
        print(f"saved {key}")
    except Exception as e:
        print(f"failed {key}: {e}")
    time.sleep(3)  # arxiv asks for a few seconds between requests

print(f"{sum((OUT / f'{k}.pdf').exists() for k in keys)} of {len(keys)} papers downloaded")
