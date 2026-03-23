# image-similarity
This project implements a Content-Based Image Retrieval (CBIR) system designed to find visual similarities in historical paintings and artworks from the [National Gallery of Art (NGA) Open Data](https://github.com/NationalGalleryOfArt/opendata) collection.
By leveraging a foundational Vision Transformer and metric learning, this system maps artworks into a semantic vector space where visually and contextually similar paintings are clustered together.
## Architecture & Approach
The pipeline is built using PyTorch and FAISS, consisting of four main stages:
1. **Dataset Mining**: Downloads NGA metadata and images, automatically categorizing artworks into 'portraits' and 'non-portraits' to create a balanced dataset.
2. **Contrastive Learning**: Generates anchor, positive, and negative image triplets to train the network on subtle artistic similarities (using easy and hard negative mining).
3. **Deep Feature Extraction**: Utilizes a frozen **DINOv2** (`vitb14`) backbone attached to a custom trainable projection head, optimized using `TripletMarginLoss`.
4. **Vector Search**: Embeddings are L2-normalized and indexed using **FAISS** (Inner Product), allowing for blazingly fast cosine-similarity visual searches.
