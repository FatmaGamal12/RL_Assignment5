import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    """
    MDN-RNN (World Models) to model dynamics in latent space.

    We model:  p(z_{t+1} | z_t, action_t, h_t)
    using an LSTM + Mixture Density Network output head.

    Inputs (per time step):
      - z_t:      (B, latent_dim)
      - action_t: (B, action_dim)  (one-hot)

    Outputs (per time step):
      - pi:     (B, n_mixtures, latent_dim)
      - mu:     (B, n_mixtures, latent_dim)
      - sigma:  (B, n_mixtures, latent_dim)  (positive)
      - plus LSTM hidden state

    Notes:
      - We output mixture parameters PER latent dimension (common simplification).
      - This is stable and commonly used in student projects.
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

        # MDN head: outputs parameters for pi, mu, log_sigma
        # Total params per time step:
        #   pi:        n_mix * latent_dim
        #   mu:        n_mix * latent_dim
        #   log_sigma: n_mix * latent_dim
        out_dim = 3 * n_mixtures * latent_dim
        self.mdn = nn.Linear(hidden_dim, out_dim)

        # helpful constant
        self._log_2pi = math.log(2.0 * math.pi)

    def forward(self, z_seq, a_seq, hidden=None):
        """
        z_seq: (B, T, latent_dim)
        a_seq: (B, T, action_dim) one-hot
        hidden: optional (h0, c0)

        returns:
          pi, mu, sigma: each (B, T, n_mix, latent_dim)
          next_hidden: (hT, cT)
        """
        x = torch.cat([z_seq, a_seq], dim=-1)  # (B, T, latent+action)
        out, next_hidden = self.lstm(x, hidden)  # out: (B, T, hidden_dim)

        mdn_out = self.mdn(out)  # (B, T, 3*n_mix*latent)

        B, T, _ = mdn_out.shape
        M = self.n_mixtures
        D = self.latent_dim

        mdn_out = mdn_out.view(B, T, 3, M, D)
        pi_logits = mdn_out[:, :, 0]            # (B,T,M,D)
        mu = mdn_out[:, :, 1]                   # (B,T,M,D)
        log_sigma = mdn_out[:, :, 2].clamp(-7, 7)  # stabilize
        sigma = torch.exp(log_sigma)            # positive

        # Convert pi logits to normalized mixture weights.
        # Softmax over mixtures, independently per latent dimension.
        pi = F.softmax(pi_logits, dim=2)        # (B,T,M,D)

        return pi, mu, sigma, next_hidden

    def mdn_nll(self, z_next, pi, mu, sigma):
        """
        Negative log-likelihood for z_next under the mixture model.

        z_next: (B, T, latent_dim)
        pi, mu, sigma: (B, T, n_mix, latent_dim)

        Returns:
          scalar loss (mean over batch/time/dim)
        """
        # Expand z_next to compare with mixture components
        # z_next_e: (B,T,1,D) -> broadcast to (B,T,M,D)
        z_next_e = z_next.unsqueeze(2)

        # log N(z | mu, sigma)
        # = -0.5 * [ ((z-mu)/sigma)^2 + 2*log(sigma) + log(2*pi) ]
        log_prob = -0.5 * (
            ((z_next_e - mu) / (sigma + 1e-8)) ** 2
            + 2.0 * torch.log(sigma + 1e-8)
            + self._log_2pi
        )  # (B,T,M,D)

        # Mix: log sum_k pi_k * exp(log_prob_k)
        # log(pi) + log_prob
        log_mix = torch.log(pi + 1e-8) + log_prob  # (B,T,M,D)

        # logsumexp over mixtures (dim=2)
        log_sum = torch.logsumexp(log_mix, dim=2)  # (B,T,D)

        # NLL = - mean log likelihood
        nll = -log_sum.mean()
        return nll

    @torch.no_grad()
    def sample_next(self, z_t, a_t, hidden=None, temperature: float = 1.0):
        """
        Sample z_{t+1} from the MDN distribution given one step input.

        z_t: (B, latent_dim)
        a_t: (B, action_dim) one-hot
        hidden: optional LSTM hidden state
        temperature: >1 makes more random, <1 more deterministic

        Returns:
          z_next_sample: (B, latent_dim)
          next_hidden: updated hidden state
        """
        z_seq = z_t.unsqueeze(1)  # (B,1,D)
        a_seq = a_t.unsqueeze(1)  # (B,1,A)
        pi, mu, sigma, next_hidden = self.forward(z_seq, a_seq, hidden=hidden)

        # (B,1,M,D) -> (B,M,D)
        pi = pi[:, 0]
        mu = mu[:, 0]
        sigma = sigma[:, 0] * temperature

        B, M, D = mu.shape

        # Choose mixture component per dimension (independent dims simplification)
        # indices: (B,D)
        pi_perm = pi.permute(0, 2, 1)  # (B,D,M)
        idx = torch.multinomial(pi_perm.reshape(B * D, M), 1).view(B, D)  # (B,D)

        # Gather mu/sigma by idx for each dim
        mu_perm = mu.permute(0, 2, 1)        # (B,D,M)
        sigma_perm = sigma.permute(0, 2, 1)  # (B,D,M)

        gather_idx = idx.unsqueeze(-1)  # (B,D,1)
        mu_sel = torch.gather(mu_perm, dim=2, index=gather_idx).squeeze(-1)       # (B,D)
        sigma_sel = torch.gather(sigma_perm, dim=2, index=gather_idx).squeeze(-1) # (B,D)

        eps = torch.randn_like(mu_sel)
        z_next = mu_sel + sigma_sel * eps  # (B,D)

        return z_next, next_hidden
