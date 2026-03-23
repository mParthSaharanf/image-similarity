import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# -------------------------
# CONFIG
# -------------------------
IMAGE_FOLDER = "./nga_images"     # folder with all images
MODEL_PATH = "embedding_model.pth"         # your trained model
OUTPUT_PATH = "embeddings.npy"   # where to save embeddings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS (same as training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# LOAD MODEL
# -------------------------
from model import EmbeddingModel   # <-- change if your file name differs

model = EmbeddingModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# LOAD IMAGES
# -------------------------
image_paths = []

for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Total images: {len(image_paths)}")

# -------------------------
# GENERATE EMBEDDINGS
# -------------------------
embeddings = []
paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)

            # 🔥 IMPORTANT: embedding extraction
            embedding = model(img)

            embedding = embedding.cpu().numpy().flatten()

            embeddings.append(embedding)
            paths.append(path)

        except Exception as e:
            print(f"Skipping {path}: {e}")

# -------------------------
# SAVE
# -------------------------
embeddings = np.array(embeddings)

np.save(OUTPUT_PATH, {
    "embeddings": embeddings,
    "paths": paths
})

print("Embeddings saved!")