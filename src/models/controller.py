# src/models/controller.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    """
    Controller policy for World Models.
    Input: concat(z_t, h_t) -> (obs_dim)
    Output: action logits -> discrete actions
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
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, device: str | None = None, deterministic: bool = False) -> int:
        self.eval()

        if device is None:
            device = next(self.parameters()).device.type

        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)  # (1, obs_dim)
        logits = self.forward(obs)[0]  # (action_dim,)

        if deterministic:
            return int(torch.argmax(logits).item())

        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        return int(action)
