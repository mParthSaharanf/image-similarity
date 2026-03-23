import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGE_DIR = "data/nga_images"
TRIPLETS_CSV = "triplets/triplets.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class TripletDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_image(self, object_id):
        path = os.path.join(self.image_dir, f"{object_id}.jpg")

        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        anchor_id = row["anchor"]
        positive_id = row["positive"]
        negative_id = row["negative"]

        anchor = self.load_image(anchor_id)
        positive = self.load_image(positive_id)
        negative = self.load_image(negative_id)

        return anchor, positive, negative