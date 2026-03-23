import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

IMAGE_FOLDER = "data/nga_images"    
MODEL_PATH = "embedding_model.pth"       
OUTPUT_PATH = "embeddings.npy"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


from model import EmbeddingModel  

model = EmbeddingModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

image_paths = []

for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Total images: {len(image_paths)}")

embeddings = []
paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)

            embedding = model(img)

            embedding = embedding.cpu().numpy().flatten()

            embeddings.append(embedding)
            paths.append(path)

        except Exception as e:
            print(f"Skipping {path}: {e}")


embeddings = np.array(embeddings)

np.save(OUTPUT_PATH, {
    "embeddings": embeddings,
    "paths": paths
})

print("Embeddings saved!")