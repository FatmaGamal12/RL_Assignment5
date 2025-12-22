import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from ..models.vae import ConvVAE


# =========================
# Config
# =========================
DATA_PATH = "data/breakout_random.pkl"
OUT_DIR = "artifacts/vae"
SEED = 42

LATENT_DIM = 32
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
BETA = 1.0               # you can try 0.5 or 0.25 if recon looks too blurry
NUM_WORKERS = 0          # Windows safe
SAVE_EVERY_EPOCH = True
SAMPLE_RECON_N = 16      # save a grid of reconstructions


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrameDataset(Dataset):
    """
    Loads frames from your replay buffer pkl: list of (obs, action, reward, done)
    obs is expected shape: (1, 64, 64) float in [0,1]
    """
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            data: List[Tuple[np.ndarray, int, float, bool]] = pickle.load(f)

        self.frames = []
        for (obs, action, reward, done) in data:
            # obs should already be (1,64,64), but we guard anyway
            obs = np.asarray(obs, dtype=np.float32)

            if obs.ndim == 2:
                obs = obs[None, :, :]  # (1,H,W)
            if obs.shape[0] != 1:
                # if accidentally (H,W,1) or something odd:
                obs = obs.reshape(1, obs.shape[-2], obs.shape[-1])

            self.frames.append(obs)

        print(f"[FrameDataset] Loaded {len(self.frames)} frames from {pkl_path}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.frames[idx]  # (1,64,64)
        return torch.from_numpy(x)  # float32


@torch.no_grad()
def save_reconstructions(model, batch, out_path, device):
    model.eval()
    x = batch.to(device)
    x_hat, _, _ = model(x)

    # Make a grid: first originals then reconstructions
    # Shape: (2N, 1, 64, 64)
    stacked = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    save_image(stacked, out_path, nrow=SAMPLE_RECON_N, normalize=False)
    print(f"[VAE] Saved recon grid to {out_path} (top: originals, bottom: reconstructions)")


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VAE] Device: {device}")

    dataset = FrameDataset(DATA_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    model = ConvVAE(latent_dim=LATENT_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # Grab a fixed batch for recon visualization
    fixed_batch = next(iter(loader))[:SAMPLE_RECON_N]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        steps = 0

        for x in loader:
            x = x.to(device)  # (B,1,64,64)

            x_hat, mu, logvar = model(x)
            loss, recon, kl = model.loss_fn(x, x_hat, mu, logvar, beta=BETA)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            steps += 1

        avg_loss = total_loss / steps
        avg_recon = total_recon / steps
        avg_kl = total_kl / steps

        print(f"[VAE] Epoch {epoch}/{EPOCHS} | loss={avg_loss:.6f} recon={avg_recon:.6f} kl={avg_kl:.6f}")

        # Save reconstructions each epoch
        recon_path = os.path.join(OUT_DIR, f"recon_epoch_{epoch}.png")
        save_reconstructions(model, fixed_batch, recon_path, device)

        # Save checkpoint
        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(OUT_DIR, f"vae_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "beta": BETA,
                },
                ckpt_path,
            )
            print(f"[VAE] Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(OUT_DIR, "vae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[VAE] Saved final weights: {final_path}")


if __name__ == "__main__":
    main()
