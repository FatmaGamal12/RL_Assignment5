import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    """
    Simple controller policy for World Models.
    Input: concat(z_t, h_t)  -> (latent_dim + rnn_hidden_dim)
    Output: action logits for discrete actions (e.g., Breakout has 4)

    We keep it small on purpose (World Models controller is often linear/MLP).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim)
        returns logits: (B, action_dim)
        """
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device: str = "cpu", deterministic: bool = False) -> int:
        """
        obs_np: (obs_dim,) numpy
        returns: int action
        """
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)  # (1,obs_dim)
        logits = self.forward(obs)[0]  # (action_dim,)

        if deterministic:
            return int(torch.argmax(logits).item())

        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        return int(action)
