import os
import gym
import torch
import numpy as np

from gym.wrappers import RecordVideo

from src.models.vae import ConvVAE
from src.models.rnn import MDNRNN
from src.models.controller import Controller


# =========================
# Config
# =========================
ENV_ID = "SpaceInvadersNoFrameskip-v4"   # or BreakoutNoFrameskip-v4
VIDEO_DIR = "videos"

LATENT_DIM = 32
RNN_HIDDEN_DIM = 256
ACTION_DIM = 6

VAE_PATH = "artifacts/vae/vae_final(space) .pt"
RNN_PATH = "artifacts/rnn/rnn_final.pt"
CONTROLLER_PATH = "artifacts/controller/controller_best.pt"

MAX_STEPS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Preprocess frame
# =========================
def preprocess_obs(obs):
    """
    obs: (210,160,3)
    returns: (1,1,64,64)
    """
    import cv2

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)


# =========================
# Main
# =========================
def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # -------- Env with video --------
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        env,
        VIDEO_DIR,
        episode_trigger=lambda e: True,
        name_prefix="world_model_test",
    )

    # -------- Load models --------
    vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    rnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=RNN_HIDDEN_DIM,
    ).to(DEVICE)
    rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
    rnn.eval()

    controller = Controller(
        obs_dim=LATENT_DIM + RNN_HIDDEN_DIM,
        action_dim=ACTION_DIM,
    ).to(DEVICE)
    controller.load_state_dict(torch.load(CONTROLLER_PATH, map_location=DEVICE))
    controller.eval()

    # -------- Episode --------
    obs, _ = env.reset()

    h = torch.zeros((1, 1, RNN_HIDDEN_DIM), device=DEVICE)
    c = torch.zeros((1, 1, RNN_HIDDEN_DIM), device=DEVICE)
    hidden = (h, c)

    z = torch.zeros((1, LATENT_DIM), device=DEVICE)

    total_reward = 0.0

    for step in range(MAX_STEPS):
        # Encode frame
        x = preprocess_obs(obs).to(DEVICE)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            z = mu

        # Controller action
        obs_ctrl = torch.cat([z.squeeze(0), hidden[0][-1, 0]], dim=0)
        action = controller.act(obs_ctrl.cpu().numpy(), deterministic=True)

        # Step env
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        # Update RNN hidden
        a_oh = torch.zeros((1, ACTION_DIM), device=DEVICE)
        a_oh[0, action] = 1.0

        with torch.no_grad():
            _, _, _, _, _, hidden = rnn(
                z.unsqueeze(1), a_oh.unsqueeze(1), hidden
            )

        if done or truncated:
            break

    env.close()
    print(f"Episode reward: {total_reward}")
    print(f"Video saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
