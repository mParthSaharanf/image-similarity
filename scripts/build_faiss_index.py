import numpy as np
import faiss
import os
import pickle

# ------------------------------------------------
# Paths
# ------------------------------------------------

EMBEDDINGS_FILE = "./embeddings.npy"

INDEX_OUTPUT = "./faiss/image_index.faiss"
PATHS_OUTPUT = "./faiss/image_paths.pkl"

os.makedirs("faiss", exist_ok=True)

# ------------------------------------------------
# Load embeddings (NEW FORMAT)
# ------------------------------------------------

data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

embeddings = data["embeddings"].astype("float32")
image_paths = data["paths"]

print("Loaded embeddings:", embeddings.shape)
print("Loaded paths:", len(image_paths))

# ------------------------------------------------
# Normalize embeddings (for cosine similarity)
# ------------------------------------------------

faiss.normalize_L2(embeddings)

# ------------------------------------------------
# Create FAISS index
# ------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

# ------------------------------------------------
# Save index
# ------------------------------------------------

faiss.write_index(index, INDEX_OUTPUT)

# Save paths
with open(PATHS_OUTPUT, "wb") as f:
    pickle.dump(image_paths, f)

print("FAISS index saved:", INDEX_OUTPUT)
print("Paths saved:", PATHS_OUTPUT)