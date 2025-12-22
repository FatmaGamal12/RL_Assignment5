# src/training_controller.py
import os
import time
import numpy as np
import torch

from ..envs.world_model_env import WorldModelEnv
from ..models.controller import Controller
from ..config import (
    LATENT_DIM,
    RNN_HIDDEN_DIM,
    ACTION_DIM,
    POPULATION_SIZE,
    ELITE_FRACTION,
    NOISE_STD,
    CONTROLLER_ITERS,
    EPISODES_PER_EVAL,
    MAX_STEPS,
    CONTROLLER_SAVE_DIR,
    CONTROLLER_BEST_PATH,
    SEED,
    WORLD_TEMPERATURE,
    WORLD_DETERMINISTIC,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_controller(env: WorldModelEnv, controller: Controller, device: str) -> float:
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


def average_elites_into_base(base_controller: Controller, elites: list[Controller]):
    # base_param = mean(elite_params)
    with torch.no_grad():
        base_params = list(base_controller.parameters())
        elite_params = [list(e.parameters()) for e in elites]

        for p_idx in range(len(base_params)):
            stacked = torch.stack([elite_params[e_idx][p_idx].data for e_idx in range(len(elites))], dim=0)
            base_params[p_idx].data.copy_(stacked.mean(dim=0))


def main():
    set_seed(SEED)
    os.makedirs(CONTROLLER_SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Controller] Device: {device}")

    # PURE imagination env
    env = WorldModelEnv(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        rnn_ckpt_path="artifacts/rnn/rnn_best.pt",
        rollout_limit=MAX_STEPS,
        temperature=WORLD_TEMPERATURE,
        deterministic=WORLD_DETERMINISTIC,
        device=device,
    )

    obs_dim = env.obs_dim
    action_dim = ACTION_DIM
    print(f"[Controller] obs_dim={obs_dim} action_dim={action_dim}")

    base_controller = Controller(obs_dim, action_dim).to(device)

    best_reward = -float("inf")
    start_time = time.time()

    for it in range(1, CONTROLLER_ITERS + 1):
        population = []
        rewards = []

        for _ in range(POPULATION_SIZE):
            ctrl = Controller(obs_dim, action_dim).to(device)
            ctrl.load_state_dict(base_controller.state_dict())

            # Add noise to create a candidate
            with torch.no_grad():
                for p in ctrl.parameters():
                    p.add_(NOISE_STD * torch.randn_like(p))

            r = evaluate_controller(env, ctrl, device)
            population.append(ctrl)
            rewards.append(r)

        rewards = np.array(rewards, dtype=np.float32)
        elite_count = max(1, int(POPULATION_SIZE * ELITE_FRACTION))

        elite_idx = rewards.argsort()[-elite_count:]
        elites = [population[i] for i in elite_idx]

        mean_reward = float(rewards.mean())
        max_reward = float(rewards.max())
        best_idx = int(np.argmax(rewards))
        best_ctrl = population[best_idx]

        print(f"[Controller] Iter {it:02d}/{CONTROLLER_ITERS} | mean={mean_reward:.2f} max={max_reward:.2f}")

        # Update base controller from elites
        average_elites_into_base(base_controller, elites)

        # Save best controller ever
        if max_reward > best_reward:
            best_reward = max_reward
            torch.save(best_ctrl.state_dict(), CONTROLLER_BEST_PATH)
            print(f"[Controller] ✓ New best saved: reward={best_reward:.2f} -> {CONTROLLER_BEST_PATH}")

    elapsed = time.time() - start_time
    print(f"[Controller] Training finished in {elapsed/60:.1f} min")
    print(f"[Controller] Best reward: {best_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
