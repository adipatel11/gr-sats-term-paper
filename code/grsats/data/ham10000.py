import pandas as pd

from grsats.config import DATA

# melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma
MALIGNANT = ["mel", "bcc", "akiec"]


def load_ham():
    df = pd.read_csv(DATA / "DermaMNIST-C.csv")
    # drop the 67 images with no sex or age (like MEDFAIR), leaves 9,948
    df = df[(df["sex"] != "unknown") & df["age"].notna()].reset_index(drop=True)

    # images are split across the two unzipped part_1 / part_2 folders
    folder = DATA / "HAM10000" / "dataverse_files"
    paths = {p.stem: str(p) for p in folder.glob("HAM10000_images_part_*/*.jpg")}

    return pd.DataFrame({
        "id": df["image_id"],
        "lesion": df["lesion_id"],
        "path": [paths.get(i, "") for i in df["image_id"]],  # "" = not found
        "y": df["dx"].isin(MALIGNANT).astype(int),
        "split": df["split"],
        "sex": df["sex"],
        "age": ["60+" if a >= 60 else "<60" for a in df["age"]],
    })
