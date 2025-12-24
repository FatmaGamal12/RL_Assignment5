import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    """
    MDN-RNN (World Models) to model dynamics in latent space.

    Models:
      p(z_{t+1} | z_t, action_t, h_t)   via MDN head
      r_{t} (or r_{t+1})               via reward head

    forward returns:
      pi, mu, sigma : (B,T,M,D)
      r_pred        : (B,T,1)
      next_hidden   : (hT, cT)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_mixtures: int = 5,
        n_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures
        self.n_layers = n_layers

        self.input_dim = latent_dim + action_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # MDN head: outputs pi/mu/log_sigma
        out_dim = 3 * n_mixtures * latent_dim
        self.mdn = nn.Linear(hidden_dim, out_dim)

        # Reward head
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

        self._log_2pi = math.log(2.0 * math.pi)

    def forward(self, z_seq, a_seq, hidden=None):
        """
        z_seq: (B,T,D)
        a_seq: (B,T,A)
        hidden: (h0,c0)

        returns:
          pi, mu, sigma: (B,T,M,D)
          r_pred:        (B,T,1)
          next_hidden:   (hT,cT)
        """
        x = torch.cat([z_seq, a_seq], dim=-1)   # (B,T,D+A)
        out, next_hidden = self.lstm(x, hidden) # out: (B,T,H)

        mdn_out = self.mdn(out)                 # (B,T,3*M*D)
        B, T, _ = mdn_out.shape
        M = self.n_mixtures
        D = self.latent_dim

        mdn_out = mdn_out.view(B, T, 3, M, D)
        pi_logits = mdn_out[:, :, 0]                 # (B,T,M,D)
        mu        = mdn_out[:, :, 1]                 # (B,T,M,D)
        log_sigma = mdn_out[:, :, 2].clamp(-7, 7)    # (B,T,M,D)
        sigma     = torch.exp(log_sigma)

        # Softmax over mixtures per latent dimension
        pi = F.softmax(pi_logits, dim=2)

        # Reward prediction
        r_pred = self.reward_head(out)
        done_logits = self.done_head(out)     # (B,T,1)

        return pi, mu, sigma, r_pred, done_logits, next_hidden


    def mdn_nll(self, z_next, pi, mu, sigma):
        """
        z_next: (B,T,D)
        pi,mu,sigma: (B,T,M,D)
        """
        z_next_e = z_next.unsqueeze(2)  # (B,T,1,D)

        log_prob = -0.5 * (
            ((z_next_e - mu) / (sigma + 1e-8)) ** 2
            + 2.0 * torch.log(sigma + 1e-8)
            + self._log_2pi
        )  # (B,T,M,D)

        log_mix = torch.log(pi + 1e-8) + log_prob
        log_sum = torch.logsumexp(log_mix, dim=2)  # (B,T,D)

        return -log_sum.mean()
    def done_bce(self, done_target, done_logits):
        """
        done_target: (B,T) or (B,T,1)
        done_logits: (B,T,1)
        """
        if done_target.ndim == 2:
            done_target = done_target.unsqueeze(-1)

        return F.binary_cross_entropy_with_logits(
            done_logits,
            done_target.float(),
            reduction="mean",
        )

    def reward_mse(self, r_target, r_pred):
        """
        r_target: (B,T) or (B,T,1)
        r_pred:   (B,T,1)
        """
        if r_target.ndim == 2:
            r_target = r_target.unsqueeze(-1)
        return F.mse_loss(r_pred, r_target, reduction="mean")

    @torch.no_grad()
    def sample_next(self, z_t, a_t, hidden=None, temperature: float = 1.0):
        """
        z_t: (B,D)
        a_t: (B,A)
        returns:
          z_next: (B,D)
          r_pred: (B,1)
          next_hidden
        """
        z_seq = z_t.unsqueeze(1)  # (B,1,D)
        a_seq = a_t.unsqueeze(1)  # (B,1,A)

        pi, mu, sigma, r_pred, next_hidden = self.forward(z_seq, a_seq, hidden=hidden)

        # (B,1,M,D)->(B,M,D)
        pi = pi[:, 0]
        mu = mu[:, 0]
        sigma = sigma[:, 0] * temperature
        r_pred = r_pred[:, 0]  # (B,1)

        B, M, D = mu.shape

        # sample per-dim mixture (simplified)
        pi_perm = pi.permute(0, 2, 1)  # (B,D,M)
        idx = torch.multinomial(pi_perm.reshape(B * D, M), 1).view(B, D)

        mu_perm = mu.permute(0, 2, 1)        # (B,D,M)
        sigma_perm = sigma.permute(0, 2, 1)  # (B,D,M)

        gather_idx = idx.unsqueeze(-1)
        mu_sel = torch.gather(mu_perm, 2, gather_idx).squeeze(-1)
        sigma_sel = torch.gather(sigma_perm, 2, gather_idx).squeeze(-1)

        z_next = mu_sel + sigma_sel * torch.randn_like(mu_sel)
        pi, mu, sigma, r_pred, done_logits, next_hidden = self.forward(
            z_seq, a_seq, hidden=hidden
        )

        done_logits = done_logits[:, 0]   # (B,1)
        return z_next, r_pred, done_logits, next_hidden

