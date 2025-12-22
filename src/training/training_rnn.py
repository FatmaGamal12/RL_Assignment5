# src/training_rnn.py
import os
import glob
import random
import cv2
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.vae import ConvVAE
from src.models.rnn import MDNRNN


# =========================
# Config
# =========================
SEED = 42

DATA_DIR = "data"                 # contains breakout_chunk_*.npz
VAE_WEIGHTS = "artifacts/vae/vae_final.pt"
OUT_DIR = "artifacts/rnn"

LATENT_DIM = 32
HIDDEN_DIM = 256
N_MIXTURES = 5
N_LAYERS = 1

SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
GRAD_CLIP = 1.0

TRAIN_FRAC = 0.95
NUM_WORKERS = 0

REWARD_COEF = 5.0                 # important for Breakout


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(actions: np.ndarray, action_dim: int) -> np.ndarray:
    oh = np.zeros((actions.shape[0], action_dim), dtype=np.float32)
    oh[np.arange(actions.shape[0]), actions] = 1.0
    return oh


# =========================
# Dataset
# =========================
class LatentSequenceDataset(Dataset):
    """
    Returns:
      z_seq  : (L, D)
      a_seq  : (L, A)
      z_tgt  : (L, D)
      r_tgt  : (L, 1)
    """

    def __init__(
        self,
        z: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        done: np.ndarray,
        action_dim: int,
        seq_len: int,
        indices: np.ndarray = None,
    ):
        self.z = z
        self.a = a
        self.r = r
        self.done = done
        self.action_dim = action_dim
        self.seq_len = seq_len

        max_start = len(z) - (seq_len + 1)
        candidates = np.arange(max_start, dtype=np.int64)

        valid = []
        for i in candidates:
            if not self.done[i : i + seq_len].any():
                valid.append(i)
        valid = np.array(valid, dtype=np.int64)

        self.starts = indices if indices is not None else valid
        print(f"[RNN Dataset] transitions={len(z)} valid_seqs={len(valid)} used={len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        z_seq = self.z[i : i + self.seq_len]
        a_seq = self.a[i : i + self.seq_len]
        z_tgt = self.z[i + 1 : i + self.seq_len + 1]
        r_tgt = self.r[i : i + self.seq_len]

        a_oh = one_hot(a_seq, self.action_dim)

        return (
            torch.from_numpy(z_seq).float(),
            torch.from_numpy(a_oh).float(),
            torch.from_numpy(z_tgt).float(),
            torch.from_numpy(r_tgt).float().unsqueeze(-1),
        )


# =========================
# VAE Encoding
# =========================
@torch.no_grad()
def encode_frames_with_vae(
    vae: ConvVAE,
    frames: np.ndarray,   # (N,1,64,64)
    batch_size: int,
    device: str,
) -> np.ndarray:
    vae.eval()
    z_all = []

    for i in range(0, len(frames), batch_size):
        x = torch.from_numpy(frames[i : i + batch_size]).to(device)
        mu, _ = vae.encode(x)
        z_all.append(mu.cpu().numpy())

    return np.concatenate(z_all, axis=0)


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RNN] Device: {device}")

    # -------------------------
    # Load replay chunks
    # -------------------------
    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "breakout_chunk_*.npz")))
    assert len(chunk_files) > 0, "No replay chunks found!"

    obs, actions, rewards, dones = [], [], [], []

    for path in chunk_files:
        d = np.load(path)
        obs.append(d["obs"])
        actions.append(d["actions"])
        rewards.append(d["rewards"])
        dones.append(d["dones"])

    obs = np.concatenate(obs, axis=0)        # (N,4,84,84)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    dones = np.concatenate(dones, axis=0)

    print(f"[RNN] Loaded transitions: {len(actions)}")

    # -------------------------
    # Preprocess frames for VAE
    # -------------------------
    frames = obs[:, -1]  # last frame only (N,84,84)

    frames_64 = np.zeros((frames.shape[0], 1, 64, 64), dtype=np.float32)
    for i in range(frames.shape[0]):
        frames_64[i, 0] = cv2.resize(frames[i], (64, 64), interpolation=cv2.INTER_AREA) / 255.0

    # -------------------------
    # Load VAE
    # -------------------------
    vae = ConvVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()

    print("[RNN] Encoding frames → z")
    z_all = encode_frames_with_vae(vae, frames_64, batch_size=256, device=device)
    print("[RNN] z shape:", z_all.shape)

    action_dim = int(actions.max()) + 1
    print("[RNN] action_dim:", action_dim)

    # -------------------------
    # Build datasets
    # -------------------------
    temp = LatentSequenceDataset(z_all, actions, rewards, dones, action_dim, SEQ_LEN)
    starts = temp.starts

    rng = np.random.default_rng(SEED)
    rng.shuffle(starts)

    split = int(len(starts) * TRAIN_FRAC)
    train_idx, val_idx = starts[:split], starts[split:]

    train_ds = LatentSequenceDataset(z_all, actions, rewards, dones, action_dim, SEQ_LEN, train_idx)
    val_ds   = LatentSequenceDataset(z_all, actions, rewards, dones, action_dim, SEQ_LEN, val_idx)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # -------------------------
    # Model
    # -------------------------
    model = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        n_mixtures=N_MIXTURES,
        n_layers=N_LAYERS,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # Training
    # -------------------------
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for z_seq, a_seq, z_tgt, r_tgt in train_loader:
            z_seq, a_seq = z_seq.to(device), a_seq.to(device)
            z_tgt, r_tgt = z_tgt.to(device), r_tgt.to(device)

            pi, mu, sigma, r_pred, _ = model(z_seq, a_seq)
            mdn_loss = model.mdn_nll(z_tgt, pi, mu, sigma)
            reward_loss = F.mse_loss(r_pred, r_tgt)

            loss = mdn_loss + REWARD_COEF * reward_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_seq, a_seq, z_tgt, r_tgt in val_loader:
                z_seq, a_seq = z_seq.to(device), a_seq.to(device)
                z_tgt, r_tgt = z_tgt.to(device), r_tgt.to(device)

                pi, mu, sigma, r_pred, _ = model(z_seq, a_seq)
                loss = model.mdn_nll(z_tgt, pi, mu, sigma) + REWARD_COEF * F.mse_loss(r_pred, r_tgt)
                val_loss += loss.item()

        print(f"[RNN] Epoch {epoch}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, os.path.join(OUT_DIR, "rnn_best.pt"))
            print("[RNN] Saved best model")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "rnn_final.pt"))
    print("[RNN] Training complete")


if __name__ == "__main__":
    main()
