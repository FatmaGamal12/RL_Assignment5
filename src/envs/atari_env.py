import gym
import cv2
import numpy as np
from collections import deque


class AtariEnv:
    """
    Atari environment wrapper with:
    - Grayscale
    - Resize to 84x84
    - Frame stacking (4)
    - uint8 output (NO normalization here)
    """

    def __init__(self, env_id: str):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space

        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Input: RGB frame (H, W, 3)
        Output: uint8 grayscale (84, 84)
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()

        # Gym <0.26 compatibility
        if isinstance(obs, tuple):
            obs = obs[0]

        frame = self.preprocess(obs)

        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        return np.stack(self.frames, axis=0)  # (4, 84, 84) uint8

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        frame = self.preprocess(obs)
        self.frames.append(frame)

        return (
            np.stack(self.frames, axis=0),  # (4, 84, 84)
            float(reward),
            bool(done),
            info,
        )


    def close(self):
        self.env.close()
