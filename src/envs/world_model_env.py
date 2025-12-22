# src/envs/world_model_env.py
import numpy as np
import torch

from ..models.rnn import MDNRNN


class WorldModelEnv:
    """
    PURE imagination environment.

    Trains the controller ONLY inside the learned world model (MDN-RNN).

    State returned to controller = concat(z_t, h_t)
      z_t : latent (D,)
      h_t : last LSTM hidden (H,)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 4,
        rnn_hidden_dim: int = 256,
        n_mixtures: int = 5,
        n_layers: int = 1,
        rnn_ckpt_path: str = "artifacts/rnn/rnn_best.pt",
        device: str | None = None,
        rollout_limit: int = 1000,
        temperature: float = 1.0,          # >1 more random, <1 more deterministic
        deterministic: bool = False,       # if True: use mean (no noise)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = rnn_hidden_dim
        self.rollout_limit = rollout_limit

        self.temperature = float(temperature)
        self.deterministic = bool(deterministic)

        # --------------------
        # Load RNN World Model
        # --------------------
        self.rnn = MDNRNN(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=rnn_hidden_dim,
            n_mixtures=n_mixtures,
            n_layers=n_layers,
        ).to(self.device)

        ckpt = torch.load(rnn_ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.rnn.load_state_dict(ckpt["model_state"])
        else:
            self.rnn.load_state_dict(ckpt)

        self.rnn.eval()

        # Internal state
        self.z = None              # Tensor (D,)
        self.hidden = None         # (h, c)
        self.steps = 0

        # Controller observation size
        self.obs_dim = latent_dim + rnn_hidden_dim

    # ======================================================
    # Helpers
    # ======================================================
    def _get_h_vec(self) -> np.ndarray:
        h, _ = self.hidden                      # h: (layers, B, H)
        return h[-1, 0].detach().cpu().numpy().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        z = self.z.detach().cpu().numpy().astype(np.float32)
        h = self._get_h_vec()
        return np.concatenate([z, h], axis=0).astype(np.float32)

    def _one_hot(self, action: int) -> torch.Tensor:
        a = torch.zeros(self.action_dim, device=self.device)
        a[action] = 1.0
        return a

    # ======================================================
    # Gym-like API
    # ======================================================
    def reset(self) -> np.ndarray:
        # start from random latent
        self.z = torch.randn(self.latent_dim, device=self.device)

        h0 = torch.zeros((1, 1, self.hidden_dim), device=self.device)
        c0 = torch.zeros((1, 1, self.hidden_dim), device=self.device)
        self.hidden = (h0, c0)

        self.steps = 0
        return self._get_obs()

    @torch.no_grad()
    def step(self, action: int):
        """
        One imagined step using the MDN-RNN.

        Returns:
          obs_next (obs_dim,)
          reward (float)
          done (bool)
          info (dict)
        """
        self.steps += 1

        z_t = self.z.view(1, -1)                       # (B=1, D)
        a_t = self._one_hot(action).view(1, -1)        # (B=1, A)

        # Use the model's built-in sampler (matches training assumptions)
        if self.deterministic:
            # deterministic: use mu of most likely component per dim
            # We approximate by calling forward and choosing max-pi component.
            z_seq = z_t.unsqueeze(1)                   # (1,1,D)
            a_seq = a_t.unsqueeze(1)                   # (1,1,A)

            pi, mu, sigma, r_pred, next_hidden = self.rnn(z_seq, a_seq, hidden=self.hidden)
            self.hidden = next_hidden

            # shapes: pi/mu/sigma = (1,1,M,D)
            pi = pi[:, 0]                              # (1,M,D)
            mu = mu[:, 0]                              # (1,M,D)

            # choose mixture per dim using argmax pi
            # pi_perm: (1,D,M) then argmax over M -> (1,D)
            pi_perm = pi.permute(0, 2, 1)
            idx = torch.argmax(pi_perm, dim=-1)        # (1,D)

            mu_perm = mu.permute(0, 2, 1)              # (1,D,M)
            z_next = torch.gather(mu_perm, 2, idx.unsqueeze(-1)).squeeze(-1)  # (1,D)

            z_next = z_next[0]
            reward = r_pred[:, 0].squeeze().item()

        else:
            # stochastic sampling path (recommended)
            z_next, r_pred, next_hidden = self.rnn.sample_next(
                z_t, a_t, hidden=self.hidden, temperature=self.temperature
            )
            self.hidden = next_hidden
            z_next = z_next[0]
            reward = r_pred.squeeze().item()

        self.z = z_next
        done = self.steps >= self.rollout_limit

        return self._get_obs(), float(reward), bool(done), {}

    # ======================================================
    # Rollout helper (optional)
    # ======================================================
    def rollout(self, controller, horizon: int = 15) -> float:
        obs = self.reset()
        total_reward = 0.0

        for _ in range(horizon):
            action = controller.act(obs)
            obs, r, done, _ = self.step(action)
            total_reward += r
            if done:
                break

        return float(total_reward)
