import os
import time
import numpy as np
import torch

from ..envs.world_model_env import WorldModelEnv
from ..models.controller import Controller
from ..config import (
    ENV_ID,
    LATENT_DIM,
    RNN_HIDDEN_DIM,
    POPULATION_SIZE,
    ELITE_FRACTION,
    NOISE_STD,
    CONTROLLER_ITERS,
    EPISODES_PER_EVAL,
    MAX_STEPS,
    CONTROLLER_SAVE_DIR,
    CONTROLLER_BEST_PATH,
    SEED,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_controller(env: WorldModelEnv, controller: Controller, device: str) -> float:
    """
    Run multiple episodes and return average total reward.
    """
    rewards = []

    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            action = controller.act(obs, device=device, deterministic=False)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)

    return float(np.mean(rewards))


def main():
    set_seed(SEED)
    os.makedirs(CONTROLLER_SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Controller] Device: {device}")

    # World-model environment (real rewards, compact state)
    env = WorldModelEnv(
        env_id=ENV_ID,
        latent_dim=LATENT_DIM,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        record_video=False,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_space.n
    print(f"[Controller] obs_dim={obs_dim} action_dim={action_dim}")

    # Base controller
    base_controller = Controller(obs_dim, action_dim).to(device)

    best_reward = -float("inf")
    best_weights = None

    start_time = time.time()

    for it in range(1, CONTROLLER_ITERS + 1):
        rewards = []
        population = []

        for _ in range(POPULATION_SIZE):
            ctrl = Controller(obs_dim, action_dim).to(device)
            ctrl.load_state_dict(base_controller.state_dict())

            # Gaussian noise perturbation
            with torch.no_grad():
                for p in ctrl.parameters():
                    p.add_(NOISE_STD * torch.randn_like(p))

            reward = evaluate_controller(env, ctrl, device)
            rewards.append(reward)
            population.append(ctrl)

        rewards = np.array(rewards)

        elite_count = max(1, int(POPULATION_SIZE * ELITE_FRACTION))
        elite_idx = rewards.argsort()[-elite_count:]
        elites = [population[i] for i in elite_idx]

        mean_reward = rewards.mean()
        max_reward = rewards.max()

        print(
            f"[Controller] Iter {it:02d}/{CONTROLLER_ITERS} | "
            f"mean_reward={mean_reward:.2f} max_reward={max_reward:.2f}"
        )

        # Update base controller = average of elites
        with torch.no_grad():
            for params in zip(*(e.parameters() for e in elites)):
                avg = torch.mean(torch.stack(params), dim=0)
                params[0].copy_(avg)

        # Save best-ever controller
        if max_reward > best_reward:
            best_reward = max_reward
            best_weights = population[elite_idx[np.argmax(rewards[elite_idx])]].state_dict()
            torch.save(best_weights, CONTROLLER_BEST_PATH)
            print(f"[Controller] ✓ New best saved (reward={best_reward:.2f})")

    elapsed = time.time() - start_time
    print(f"[Controller] Training finished in {elapsed/60:.1f} min")
    print(f"[Controller] Best reward achieved: {best_reward:.2f}")
    print(f"[Controller] Best controller saved to: {CONTROLLER_BEST_PATH}")

    env.close()


if __name__ == "__main__":
    main()
