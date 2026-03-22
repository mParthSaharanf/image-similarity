import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        features = self.backbone(x)  # (B, embedding_dim)
        embedding = self.projection(features)  # (B, 256)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
        return embedding