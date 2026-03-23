import numpy as np
import faiss
import os
import pickle

EMBEDDINGS_FILE = "./embeddings.npy"

INDEX_OUTPUT = "./faiss/image_index.faiss"
PATHS_OUTPUT = "./faiss/image_paths.pkl"

os.makedirs("faiss", exist_ok=True)

data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

embeddings = data["embeddings"].astype("float32")
image_paths = data["paths"]

print("Loaded embeddings:", embeddings.shape)
print("Loaded paths:", len(image_paths))

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

faiss.write_index(index, INDEX_OUTPUT)

with open(PATHS_OUTPUT, "wb") as f:
    pickle.dump(image_paths, f)

print("FAISS index saved:", INDEX_OUTPUT)
print("Paths saved:", PATHS_OUTPUT)