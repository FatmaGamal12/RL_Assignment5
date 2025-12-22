from envs.atari_env import AtariEnv
from utils.replay_buffer import ReplayBuffer
import os

# =========================
# Configuration
# =========================
ENV_ID = "BreakoutNoFrameskip-v4"
EPISODES = 50          # keep small at first
MAX_STEPS = 1000
SAVE_PATH = "data/breakout_random.pkl"


def main():
    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)

    env = AtariEnv(
        env_id=ENV_ID,
        record_video=False  # no need to record random data
    )

    buffer = ReplayBuffer()

    for ep in range(EPISODES):
        obs = env.reset()

        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            buffer.add(obs, action, reward, done)
            obs = next_obs

            if done:
                break

        print(f"Episode {ep + 1}/{EPISODES} finished")

    env.close()
    buffer.save(SAVE_PATH)


if __name__ == "__main__":
    main()
