import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from src.models.vae import ConvVAE


# =========================
# Config
# =========================
DATA_PATH = "data/breakout_random.pkl"
OUT_DIR = "artifacts/vae"
DATA_DIR = "data"
SEED = 42


LATENT_DIM = 32
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
BETA = 1.0                # Try 0.5 if recon is blurry
NUM_WORKERS = 0           # Windows safe
SAVE_EVERY_EPOCH = True
SAMPLE_RECON_N = 16


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class FrameDataset(Dataset):
    """
    Loads frames from chunked replay buffer (.npz files)
    obs expected shape: (4,84,84) uint8
    VAE uses ONLY the last frame -> (1,64,64)
    """

    def __init__(self, data_dir: str):
        self.frames = []

        files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("breakout_chunk") and f.endswith(".npz")
        )

        if not files:
            raise RuntimeError("No breakout_chunk_*.npz files found")

        for fname in files:
            path = os.path.join(data_dir, fname)
            data = np.load(path)

            obs = data["obs"]  # (N,4,84,84) uint8

            # Use LAST frame only (standard World Models)
            last_frames = obs[:, -1]  # (N,84,84)

            for frame in last_frames:
                frame = frame.astype(np.float32) / 255.0  # normalize
                frame = frame[None, :, :]                 # (1,84,84)

                # Resize to 64x64
                frame = torch.from_numpy(frame)
                frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0),
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                self.frames.append(frame)

        print(f"[FrameDataset] Loaded {len(self.frames)} frames from chunks")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

# =========================
# Visualization
# =========================
@torch.no_grad()
def save_reconstructions(model, batch, out_path, device):
    model.eval()
    x = batch.to(device)
    x_hat, _, _ = model(x)

    stacked = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    save_image(stacked, out_path, nrow=SAMPLE_RECON_N)
    print(f"[VAE] Saved reconstructions → {out_path}")


# =========================
# Training
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VAE] Device: {device}")

    dataset = FrameDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    model = ConvVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    fixed_batch = next(iter(loader))[:SAMPLE_RECON_N]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        steps = 0

        for x in loader:
            x = x.to(device)

            x_hat, mu, logvar = model(x)
            loss, recon, kl = model.loss_fn(x, x_hat, mu, logvar, beta=BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            steps += 1

        print(
            f"[VAE] Epoch {epoch}/{EPOCHS} | "
            f"loss={total_loss/steps:.6f} "
            f"recon={total_recon/steps:.6f} "
            f"kl={total_kl/steps:.6f}"
        )

        # Save recon images
        recon_path = os.path.join(OUT_DIR, f"recon_epoch_{epoch}.png")
        save_reconstructions(model, fixed_batch, recon_path, device)

        # Save checkpoint
        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(OUT_DIR, f"vae_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[VAE] Saved checkpoint → {ckpt_path}")

    final_path = os.path.join(OUT_DIR, "vae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[VAE] Saved final model → {final_path}")


if __name__ == "__main__":
    main()
