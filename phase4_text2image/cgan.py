import os
import io
import clip
import torch
import torch.nn as nn
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM  = 100
EMBED_DIM   = 512
IMAGE_SIZE  = 64
BATCH_SIZE  = 64
LR          = 0.0002
BETAS       = (0.5, 0.999)
EPOCHS      = 100
MAX_SAMPLES = 20000
CSV_PATH    = "/content/drive/MyDrive/gan-anime/phase4_text2image/all_data.csv"
SAVE_DIR    = "/content/drive/MyDrive/gan-anime/phase4_text2image/output"
CACHE_DIR   = "/content/drive/MyDrive/gan-anime/phase4_text2image/cache"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Device : {DEVICE}")

# ── Dataset 
class SafebooruDataset(Dataset):
    def __init__(self, csv_path, max_samples=20000, transform=None):
        self.transform  = transform
        self.cache_dir  = Path(CACHE_DIR)
        self.data       = []

        print("Lecture du CSV...")
        df = pd.read_csv(csv_path, usecols=["sample_url", "tags"])
        df = df.dropna(subset=["sample_url", "tags"])
        df = df.head(max_samples * 3)  # marge pour les erreurs de téléchargement

        print(f"Téléchargement des images (max {max_samples})...")
        count = 0
        for _, row in df.iterrows():
            if count >= max_samples:
                break

            url      = "https:" + row["sample_url"]
            tags     = row["tags"]
            filename = url.split("/")[-1]
            cache_path = self.cache_dir / filename

            # Télécharger seulement si pas déjà en cache
            if not cache_path.exists():
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(cache_path, "wb") as f:
                            f.write(response.content)
                    else:
                        continue
                except:
                    continue

            self.data.append({
                "image_path": str(cache_path),
                "tags": str(tags)
            })
            count += 1

            if count % 500 == 0:
                print(f"  {count}/{max_samples} images prêtes...")

        print(f"Dataset prêt : {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            image = self.transform(image)

        # Garder les 6 premiers tags comme description
        tags = " ".join(item["tags"].split()[:6])
        return image, tags

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── Generator 
class Generator(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim + embed_dim, 512 * 4 * 4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, text_embed):
        x = torch.cat([z, text_embed], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        return self.model(x)

# ── Discriminator 
class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + embed_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embed):
        x = self.image_encoder(image)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_embed], dim=1)
        return self.classifier(x)

# ── Encodeur CLIP 
def encode_text(texts, clip_model):
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.encode_text(tokens).float()
    return embeddings

# ── Sauvegarde grille
def sauvegarder_grille(G, clip_model, epoch):
    G.eval()
    test_tags = [
        "1girl blue_hair smile",
        "1boy red_eyes serious",
        "blonde_hair long_hair happy",
        "brown_eyes short_hair blush",
    ]
    with torch.no_grad():
        text_embeds = encode_text(test_tags, clip_model)
        z = torch.randn(len(test_tags), LATENT_DIM).to(DEVICE)
        fake_images = G(z, text_embeds)

    fake_images = (fake_images * 0.5 + 0.5).cpu()
    fig, axes = plt.subplots(1, len(test_tags), figsize=(16, 4))
    for i, ax in enumerate(axes):
        img = np.clip(fake_images[i].permute(1, 2, 0).numpy(), 0, 1)
        ax.imshow(img)
        ax.set_title(test_tags[i], fontsize=7)
        ax.axis('off')

    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/epoch_{epoch:03d}.png", dpi=100)
    plt.close()
    print(f"  Grille epoch {epoch} sauvegardée")
    G.train()

# ── Entraînement 
def train():
    print("Chargement CLIP...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    dataset    = SafebooruDataset(CSV_PATH, max_samples=MAX_SAMPLES, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    G = Generator(LATENT_DIM, EMBED_DIM).to(DEVICE)
    D = Discriminator(EMBED_DIM).to(DEVICE)

    opt_G     = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D     = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    criterion = nn.BCELoss()

    losses_G, losses_D = [], []
    print(f"Début entraînement : {EPOCHS} epochs sur {DEVICE}")

    for epoch in range(EPOCHS):
        for real_images, tags in dataloader:
            real_images  = real_images.to(DEVICE)
            batch_size   = real_images.size(0)
            text_embeds  = encode_text(list(tags), clip_model)
            real_labels  = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels  = torch.zeros(batch_size, 1).to(DEVICE)

            # Discriminator
            opt_D.zero_grad()
            loss_D_real = criterion(D(real_images, text_embeds), real_labels)
            z           = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = G(z, text_embeds)
            loss_D_fake = criterion(D(fake_images.detach(), text_embeds), fake_labels)
            loss_D      = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            loss_G = criterion(D(fake_images, text_embeds), real_labels)
            loss_G.backward()
            opt_G.step()

        losses_G.append(loss_G.item())
        losses_D.append(loss_D.item())
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss G: {loss_G.item():.4f} | Loss D: {loss_D.item():.4f}")

        if (epoch + 1) % 10 == 0:
            sauvegarder_grille(G, clip_model, epoch + 1)

    # Sauvegarder modèles
    torch.save(G.state_dict(), f"{SAVE_DIR}/generator.pth")
    torch.save(D.state_dict(), f"{SAVE_DIR}/discriminator.pth")
    print("Modèles sauvegardés !")

    # Courbes de loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses_G, label="Generator")
    plt.plot(losses_D, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de loss — cGAN Phase 4")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/losses.png")
    plt.show()

# ── Main 
if __name__ == "__main__":
    train()