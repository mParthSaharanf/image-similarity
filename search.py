import faiss
import numpy as np
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt

from model import EmbeddingModel
from triplets.triplets_data import transform   # ✅ SAME AS TRAINING

# -----------------------------------
# Paths
# -----------------------------------

INDEX_FILE = "./faiss/image_index.faiss"
PATHS_FILE = "./faiss/image_paths.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# Load FAISS index
# -----------------------------------

index = faiss.read_index(INDEX_FILE)

# -----------------------------------
# Load paths
# -----------------------------------

with open(PATHS_FILE, "rb") as f:
    paths = pickle.load(f)

# -----------------------------------
# Load model (IMPORTANT)
# -----------------------------------

model = EmbeddingModel().to(DEVICE)
model.load_state_dict(torch.load("embedding_model.pth", map_location=DEVICE))
model.eval()

# -----------------------------------
# Search function
# -----------------------------------

def search(query_image, top_k=5):

    img = Image.open(query_image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model(img_tensor)

    embedding = embedding.cpu().numpy().astype("float32")

    faiss.normalize_L2(embedding)

    distances, indices = index.search(embedding, top_k)

    return distances[0], indices[0]

# -----------------------------------
# Visualization
# -----------------------------------

def show_results(query_image, distances, indices):

    plt.figure(figsize=(15,4))

    # query image
    plt.subplot(1, len(indices)+1, 1)
    plt.imshow(Image.open(query_image))
    plt.title("Query")
    plt.axis("off")

    # results
    for i, idx in enumerate(indices):

        img_path = paths[idx]

        plt.subplot(1, len(indices)+1, i+2)
        plt.imshow(Image.open(img_path))
        plt.title(f"{distances[i]:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# -----------------------------------
# Run example
# -----------------------------------

query_image = "./nga_images/15984.jpg"

distances, indices = search(query_image, top_k=6)

print("\nTop similar images:\n")
for i, idx in enumerate(indices):
    print(paths[idx], distances[i])

show_results(query_image, distances, indices)