# utils/replay_buffer.py
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, obs, action, reward, done, next_obs):
        """
        obs, next_obs: uint8 (4, 84, 84)
        action: int
        reward: float
        done: bool
        """
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def save(self, path):
        """
        Save a chunk safely using compression.
        """
        data = {
            "obs": np.asarray(self.obs, dtype=np.uint8),
            "next_obs": np.asarray(self.next_obs, dtype=np.uint8),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
        }

        np.savez_compressed(path, **data)
        print(f"[ReplayBuffer] Saved {len(self)} transitions to {path}")

    def clear(self):
        self.obs.clear()
        self.next_obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def save(self, path):
        if len(self) == 0:
            print("[ReplayBuffer] WARNING: save() called on empty buffer")
            return

        data = {
            "obs": np.asarray(self.obs, dtype=np.uint8),
            "next_obs": np.asarray(self.next_obs, dtype=np.uint8),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
        }

        np.savez_compressed(path, **data)
        print(f"[ReplayBuffer] Saved {len(self)} transitions to {path}")
