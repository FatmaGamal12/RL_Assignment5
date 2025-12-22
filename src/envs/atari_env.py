import gym
import numpy as np
import cv2
from gym.wrappers import RecordVideo


class AtariEnv:
    def __init__(
        self,
        env_id="BreakoutNoFrameskip-v4",
        video_folder="videos",
        record_video=True
    ):
        self.env = gym.make(env_id, render_mode="rgb_array")

        if record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=video_folder,
                episode_trigger=lambda episode_id: True
            )

        self.action_space = self.env.action_space

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[np.newaxis, :, :]

    def reset(self):
        obs, info = self.env.reset()
        return self.preprocess(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        obs = self.preprocess(obs)
        return obs, reward, done, info

    def close(self):
        self.env.close()
