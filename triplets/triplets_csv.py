import pandas as pd
import random
from tqdm import tqdm

INPUT_CSV = "data/final_dataset.csv"
OUTPUT_CSV = "triplets/triplets.csv"

MAX_TRIPLETS = 100000  

df = pd.read_csv(INPUT_CSV)

portraits = df[df["is_portrait"] == True]
non_portraits = df[df["is_portrait"] == False]

portrait_ids = portraits["objectid"].tolist()
non_portrait_ids = non_portraits["objectid"].tolist()

print("Portraits:", len(portrait_ids))
print("Non-portraits:", len(non_portrait_ids))

triplets = []

for _ in tqdm(range(MAX_TRIPLETS)):

    # anchor
    anchor = random.choice(portrait_ids)

    # positive (different portrait)
    positive = random.choice(portrait_ids)
    while positive == anchor:
        positive = random.choice(portrait_ids)

    # negative (mix strategy)
    if random.random() < 0.7:
        negative = random.choice(non_portrait_ids)   # easy
    else:
        negative = random.choice(portrait_ids)       # harder
        while negative == anchor or negative == positive:
            negative = random.choice(portrait_ids)

    triplets.append((anchor, positive, negative))

triplet_df = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])
triplet_df.to_csv(OUTPUT_CSV, index=False)

print(f"Total {len(triplets)} tri[lets] saved to {OUTPUT_CSV}")