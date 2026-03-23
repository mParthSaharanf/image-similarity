import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from triplets.triplets_data import TripletDataset, transform
from model import EmbeddingModel

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TripletDataset("triplets.csv", "./nga_images", transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

model = EmbeddingModel().to(DEVICE)

criterion = nn.TripletMarginLoss(margin=1.0)

optimizer = optim.Adam(
    model.projection.parameters(),  
    lr=LR
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=1,
)

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for anchor, positive, negative in tqdm(loader):

        anchor = anchor.to(DEVICE)
        positive = positive.to(DEVICE)
        negative = negative.to(DEVICE)

        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    scheduler.step(avg_loss)

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

torch.save(model.state_dict(), "embedding_model.pth")

print("\nModel saved as embedding_model.pth")