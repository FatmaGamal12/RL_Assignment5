import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..models.vae import ConvVAE
from ..models.rnn import MDNRNN


# =========================
# Config
# =========================
SEED = 42

DATA_PATH = "data/breakout_random.pkl"

# Path to your trained VAE weights
VAE_WEIGHTS = "artifacts/vae/vae_final.pt"   # or change to vae_epoch_20.pt if you prefer

# Output directory for RNN artifacts
OUT_DIR = "artifacts/rnn"

# VAE params (must match your VAE)
LATENT_DIM = 32

# RNN params
HIDDEN_DIM = 256
N_MIXTURES = 5
N_LAYERS = 1

# Training params
SEQ_LEN = 32              # 16–64 is common; 32 is safe
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
GRAD_CLIP = 1.0

# Data split
TRAIN_FRAC = 0.95

# Windows-safe DataLoader
NUM_WORKERS = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(actions: np.ndarray, action_dim: int) -> np.ndarray:
    """
    actions: (T,) integers
    returns: (T, action_dim) one-hot float32
    """
    oh = np.zeros((actions.shape[0], action_dim), dtype=np.float32)
    oh[np.arange(actions.shape[0]), actions] = 1.0
    return oh


class LatentSequenceDataset(Dataset):
    """
    Build sequences from the replay buffer:

    For each start index i, we take:
      z_seq    = z[i : i+SEQ_LEN]
      a_seq    = a[i : i+SEQ_LEN]   (one-hot)
      z_target = z[i+1 : i+SEQ_LEN+1]

    We avoid crossing episode boundaries by requiring done=False
    across the sequence window.
    """

    def __init__(
        self,
        z: np.ndarray,         # (N, latent_dim)
        a: np.ndarray,         # (N,) integer actions
        done: np.ndarray,      # (N,) bool
        action_dim: int,
        seq_len: int = 32,
        indices: np.ndarray = None,
    ):
        self.z = z
        self.a = a
        self.done = done
        self.action_dim = action_dim
        self.seq_len = seq_len

        # valid start positions:
        # need i+seq_len < N (because we need z_target up to i+seq_len)
        max_start = len(z) - (seq_len + 1)
        candidates = np.arange(max_start, dtype=np.int64)

        # A sequence [i, i+seq_len] must not include a terminal in the middle
        # We disallow any done=True in positions i .. i+seq_len-1
        valid = []
        for i in candidates:
            if not self.done[i : i + seq_len].any():
                valid.append(i)
        valid = np.array(valid, dtype=np.int64)

        if indices is not None:
            # allow train/val splits by passing a subset of valid indices
            self.starts = indices
        else:
            self.starts = valid

        print(f"[RNN Dataset] total transitions={len(z)} valid sequences={len(valid)} using={len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        z_seq = self.z[i : i + self.seq_len]                      # (L, D)
        a_seq = self.a[i : i + self.seq_len]                      # (L,)
        z_tgt = self.z[i + 1 : i + self.seq_len + 1]              # (L, D)

        a_oh = one_hot(a_seq, self.action_dim)                    # (L, A)

        return (
            torch.from_numpy(z_seq).float(),
            torch.from_numpy(a_oh).float(),
            torch.from_numpy(z_tgt).float(),
        )


@torch.no_grad()
def encode_all_frames_with_vae(
    vae: ConvVAE,
    frames: np.ndarray,     # (N, 1, 64, 64)
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Encode frames -> latent means (mu) as z. (common and stable)
    returns z: (N, latent_dim)
    """
    vae.eval()
    z_list = []

    n = frames.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(frames[start:end]).to(device).float()

        mu, logvar = vae.encode(x)
        # Use mu as deterministic z (standard in world models training pipeline)
        z = mu
        z_list.append(z.detach().cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)
    return z_all


def main():
    set_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RNN] Device: {device}")

    # =========================
    # Load replay buffer
    # =========================
    with open(DATA_PATH, "rb") as f:
        data: List[Tuple[np.ndarray, int, float, bool]] = pickle.load(f)

    # Extract arrays
    frames = []
    actions = []
    dones = []

    for (obs, action, reward, done) in data:
        obs = np.asarray(obs, dtype=np.float32)
        # Expect (1,64,64); guard anyway
        if obs.ndim == 2:
            obs = obs[None, :, :]
        frames.append(obs)
        actions.append(int(action))
        dones.append(bool(done))

    frames = np.stack(frames, axis=0)             # (N,1,64,64)
    actions = np.array(actions, dtype=np.int64)   # (N,)
    dones = np.array(dones, dtype=np.bool_)       # (N,)

    print(f"[RNN] Loaded replay: frames={frames.shape} actions={actions.shape}")

    # Determine action space size (max action id + 1)
    action_dim = int(actions.max()) + 1
    print(f"[RNN] action_dim inferred: {action_dim}")

    # =========================
    # Load VAE and encode frames -> z
    # =========================
    vae = ConvVAE(latent_dim=LATENT_DIM).to(device)
    if not os.path.exists(VAE_WEIGHTS):
        raise FileNotFoundError(
            f"VAE weights not found at {VAE_WEIGHTS}. "
            "Make sure you trained the VAE and the path is correct."
        )

    # Your VAE training saved final weights via state_dict (model.state_dict())
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()

    print("[RNN] Encoding frames with VAE encoder -> z (this may take a bit once)...")
    z_all = encode_all_frames_with_vae(
        vae=vae,
        frames=frames,
        batch_size=256,
        device=device,
    )
    print(f"[RNN] z_all shape: {z_all.shape}")  # (N, latent_dim)

    # =========================
    # Build train/val sequence datasets
    # =========================
    # Create all valid start indices first using a temp dataset
    temp = LatentSequenceDataset(
        z=z_all,
        a=actions,
        done=dones,
        action_dim=action_dim,
        seq_len=SEQ_LEN,
    )
    all_starts = temp.starts

    # Shuffle and split
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_starts)

    split = int(len(all_starts) * TRAIN_FRAC)
    train_idx = all_starts[:split]
    val_idx = all_starts[split:]

    train_ds = LatentSequenceDataset(
        z=z_all, a=actions, done=dones,
        action_dim=action_dim, seq_len=SEQ_LEN,
        indices=train_idx
    )
    val_ds = LatentSequenceDataset(
        z=z_all, a=actions, done=dones,
        action_dim=action_dim, seq_len=SEQ_LEN,
        indices=val_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    # =========================
    # Create and train MDN-RNN
    # =========================
    model = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        n_mixtures=N_MIXTURES,
        n_layers=N_LAYERS,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total = 0.0
        n = 0

        for z_seq, a_seq, z_tgt in loader:
            z_seq = z_seq.to(device)    # (B,L,D)
            a_seq = a_seq.to(device)    # (B,L,A)
            z_tgt = z_tgt.to(device)    # (B,L,D)

            pi, mu, sigma, _ = model(z_seq, a_seq)
            loss = model.mdn_nll(z_tgt, pi, mu, sigma)

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            total += loss.item() * z_seq.size(0)
            n += z_seq.size(0)

        return total / max(1, n)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)

        print(f"[RNN] Epoch {epoch}/{EPOCHS} | train_nll={train_loss:.6f} val_nll={val_loss:.6f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(OUT_DIR, "rnn_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "action_dim": action_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "n_mixtures": N_MIXTURES,
                    "seq_len": SEQ_LEN,
                },
                ckpt_path,
            )
            print(f"[RNN] Saved best checkpoint: {ckpt_path}")

    final_path = os.path.join(OUT_DIR, "rnn_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[RNN] Saved final weights: {final_path}")


if __name__ == "__main__":
    main()
