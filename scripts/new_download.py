import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "./nga_metadata"
IMAGE_DIR = "./nga_images"
OUTPUT_CSV = "final_dataset.csv"

MAX_WORKERS = 16
IMAGE_SIZE = "800,"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/"

def download_csv(filename):
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        return path

    print(f"Downloading {filename}...")
    r = requests.get(BASE_URL + filename)

    with open(path, "wb") as f:
        f.write(r.content)

    return path


objects_path = download_csv("objects.csv")
images_path = download_csv("published_images.csv")
terms_path = download_csv("objects_terms.csv")

print("Loading data...")

df_objects = pd.read_csv(objects_path)
df_images = pd.read_csv(images_path)
df_terms = pd.read_csv(terms_path)

df_images = df_images[
    (df_images["viewtype"] == "primary") &
    (df_images["iiifurl"].notna())
]

df = df_objects.merge(
    df_images,
    left_on="objectid",
    right_on="depictstmsobjectid"
)

df = df[df["isvirtual"] == 0]
df = df.drop_duplicates(subset="objectid")

print("Total usable images:", len(df))

portrait_ids = df_terms[
    df_terms["term"].str.contains("portrait", case=False, na=False)
]["objectid"].unique()

df["is_portrait"] = df["objectid"].isin(portrait_ids)

portraits = df[df["is_portrait"] == True]
non_portraits = df[df["is_portrait"] == False]

print("Portraits:", len(portraits))
print("Non-portraits:", len(non_portraits))

print("Sampling balanced non-portraits...")

grouped = non_portraits.groupby("classification")

samples = []

for cls, group in grouped:
    samples.append(group.sample(n=min(1500, len(group)), random_state=42))

non_portraits_balanced = pd.concat(samples)

print("Balanced non-portraits:", len(non_portraits_balanced))

final_df = pd.concat([portraits, non_portraits_balanced])
final_df = final_df.reset_index(drop=True)

print("Final dataset size:", len(final_df))

final_df[[
    "objectid",
    "title",
    "classification",
    "iiifurl",
    "is_portrait"
]].to_csv(OUTPUT_CSV, index=False)

print("Metadata saved to", OUTPUT_CSV)

def download_image(row):

    object_id = row["objectid"]
    iiifurl = row["iiifurl"]

    filename = f"{object_id}.jpg"
    path = os.path.join(IMAGE_DIR, filename)

    if os.path.exists(path):
        return

    img_url = f"{iiifurl}/full/{IMAGE_SIZE}/0/default.jpg"

    try:
        r = requests.get(img_url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
    except:
        pass


print("Downloading images...")

rows = final_df.to_dict("records")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(tqdm(executor.map(download_image, rows), total=len(rows)))

print("Download complete.")
print("Images saved in:", IMAGE_DIR)