import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Chemin vers ton dataset (téléchargé depuis Kaggle) ──────────────────────
DATA_PATH = "phase4_text2image/data"

# ── Dataset ──────────────────────────────────────────────────────────────────
class AnimeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        extensions = ['.jpg', '.jpeg', '.png']
        self.images = []
        for ext in extensions:
            self.images.extend(list(Path(data_path).rglob(f"*{ext}")))
        print(f"Dataset : {len(self.images)} images chargées")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
        except:
            image = Image.new("RGB", (64, 64))
        if self.transform:
            image = self.transform(image)
        return image

# ── Transformation standard ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── Visualisation ─────────────────────────────────────────────────────────────
def afficher_grille(data_path, save_path="phase4_text2image/apercu_dataset.png"):
    extensions = ['.jpg', '.jpeg', '.png']
    images = []
    for ext in extensions:
        images.extend(list(Path(data_path).rglob(f"*{ext}")))

    print(f"Nombre total d'images : {len(images)}")

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    echantillon = random.sample(images, 15)

    for i, img_path in enumerate(echantillon):
        try:
            img = mpimg.imread(str(img_path))
            axes[i].imshow(img)
            axes[i].axis('off')
        except:
            axes[i].axis('off')

    plt.suptitle("Apercu dataset Phase 4 — texte vers image", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"Grille sauvegardée : {save_path}")
    print("Screenshot cette image pour ton rapport !")

# ── Test du DataLoader ────────────────────────────────────────────────────────
def tester_dataloader(data_path):
    dataset = AnimeDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    print(f"Shape du batch  : {batch.shape}")
    print(f"Min / Max       : {batch.min():.2f} / {batch.max():.2f}")
    print("DataLoader pret !")
    return dataloader

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    afficher_grille(DATA_PATH)
    dataloader = tester_dataloader(DATA_PATH)