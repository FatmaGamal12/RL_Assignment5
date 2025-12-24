# src/collect_data.py
import os
import numpy as np

from src.envs.atari_env import AtariEnv
from src.utils.replay_buffer import ReplayBuffer


ENV_ID = "BreakoutNoFrameskip-v4"

TARGET_TRANSITIONS = 400_000
CHUNK_SIZE = 20_000
MAX_EPISODE_STEPS = 1000
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    env = AtariEnv(env_id=ENV_ID)
    buffer = ReplayBuffer()

    total_steps = 0
    chunk_idx = 0
    episode_idx = 0

    while total_steps < TARGET_TRANSITIONS:
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while (
            not done
            and total_steps < TARGET_TRANSITIONS
            and ep_steps < MAX_EPISODE_STEPS
        ):
            if np.random.rand() < 0.2:
                action = env.action_space.sample()
            else:
                action = 1  # FIRE or simple heuristic

            next_obs, reward, done, _ = env.step(action)

            buffer.add(
                obs.astype(np.uint8),
                action,
                float(reward),
                done,
                next_obs.astype(np.uint8),
            )

            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            if len(buffer) >= CHUNK_SIZE:
                save_path = os.path.join(
                    DATA_DIR, f"breakout_chunk_{chunk_idx}.npz"
                )
                buffer.save(save_path)
                buffer.clear()
                chunk_idx += 1

        episode_idx += 1
        print(
            f"[Collect] Episode {episode_idx:04d} | "
            f"steps={ep_steps:4d} | "
            f"reward={ep_reward:5.1f} | "
            f"total_steps={total_steps}"
        )

    env.close()
    print("[Collect] DONE collecting data.")


if __name__ == "__main__":
    main()
