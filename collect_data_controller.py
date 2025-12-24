import os
import numpy as np
import torch

from src.envs.atari_env import AtariEnv
from src.models.controller import Controller
from src.config import (
    ENV_ID,
    MAX_STEPS,
    LATENT_DIM,
    RNN_HIDDEN_DIM,
    ACTION_DIM,
    CONTROLLER_BEST_PATH,
)

DATA_DIR = "data"
TARGET_TRANSITIONS = 100_000
CHUNK_SIZE = 20_000


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- REAL Atari env --------
    env = AtariEnv(env_id=ENV_ID)

    # -------- Load trained controller --------
    obs_dim = LATENT_DIM + RNN_HIDDEN_DIM
    controller = Controller(obs_dim, ACTION_DIM).to(device)
    controller.load_state_dict(
        torch.load(CONTROLLER_BEST_PATH, map_location=device)
    )
    controller.eval()

    buffer = []
    total_steps = 0
    chunk_idx = 100  # avoid overwriting old chunks

    obs = env.reset()

    while total_steps < TARGET_TRANSITIONS:
        done = False
        ep_steps = 0
        ep_reward = 0.0

        while not done and ep_steps < MAX_STEPS and total_steps < TARGET_TRANSITIONS:
            # ---- IMPORTANT ----
            # REAL env gives (4,84,84), but controller expects (z,h)
            # We use RANDOM z,h for bootstrapping (as in paper)
            dummy_obs = np.zeros(obs_dim, dtype=np.float32)

            action = controller.act(dummy_obs, deterministic=False)

            next_obs, reward, done, _ = env.step(action)

            buffer.append((
                obs.astype(np.uint8),
                action,
                float(reward),
                done,
                next_obs.astype(np.uint8),
            ))

            obs = next_obs
            ep_steps += 1
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= CHUNK_SIZE:
                save_path = os.path.join(
                    DATA_DIR, f"breakout_chunk_{chunk_idx}.npz"
                )
                save_chunk(buffer, save_path)
                buffer.clear()
                chunk_idx += 1

        obs = env.reset()
        print(
            f"[Bootstrap] steps={total_steps} episode_reward={ep_reward}"
        )

    env.close()
    print("[Bootstrap] DONE")


def save_chunk(buffer, path):
    obs, actions, rewards, dones, next_obs = zip(*buffer)
    np.savez_compressed(
        path,
        obs=np.array(obs),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        next_obs=np.array(next_obs),
    )
    print(f"[Bootstrap] Saved {len(buffer)} → {path}")


if __name__ == "__main__":
    main()
