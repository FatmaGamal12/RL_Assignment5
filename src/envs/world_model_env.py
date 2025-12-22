import numpy as np
import torch

from .atari_env import AtariEnv
from ..models.vae import ConvVAE
from ..models.rnn import MDNRNN


class WorldModelEnv:
    """
    Wraps the REAL Atari environment, but exposes a compact observation:
        obs_controller = concat(z_t, h_t)

    - z_t: VAE latent mean from the current real frame
    - h_t: RNN hidden state updated using (z_{t-1}, action_{t-1})
    - reward/done come from the REAL environment step (not predicted)

    This is the standard pipeline for training the controller:
        real_env -> VAE(z) -> RNN hidden -> controller input
    """

    def __init__(
        self,
        env_id: str = "BreakoutNoFrameskip-v4",
        latent_dim: int = 32,
        rnn_hidden_dim: int = 256,
        n_mixtures: int = 5,
        vae_path: str = "artifacts/vae/vae_final.pt",
        rnn_ckpt_path: str = "artifacts/rnn/rnn_best.pt",
        record_video: bool = False,
        video_folder: str = "videos_controller",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Real environment (gives reward/done)
        self.real_env = AtariEnv(
            env_id=env_id,
            video_folder=video_folder,
            record_video=record_video,
        )
        self.action_space = self.real_env.action_space

        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # Load VAE
        self.vae = ConvVAE(latent_dim=latent_dim).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()

        # Load RNN (checkpoint can be dict or raw state_dict)
        self.rnn = MDNRNN(
            latent_dim=latent_dim,
            action_dim=self.action_space.n,
            hidden_dim=rnn_hidden_dim,
            n_mixtures=n_mixtures,
            n_layers=1,
        ).to(self.device)

        ckpt = torch.load(rnn_ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.rnn.load_state_dict(ckpt["model_state"])
        else:
            self.rnn.load_state_dict(ckpt)
        self.rnn.eval()

        # Internal state
        self.z = None
        self.hidden = None  # (h, c)

        # Controller obs dimension
        self.obs_dim = self.latent_dim + self.rnn_hidden_dim

    @torch.no_grad()
    def _encode_obs_to_z(self, obs: np.ndarray) -> np.ndarray:
        """
        obs is expected shape: (1, 64, 64) float32 in [0,1]
        returns z as numpy shape: (latent_dim,)
        """
        x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # (1,1,64,64)
        mu, logvar = self.vae.encode(x)
        z = mu[0].detach().cpu().numpy().astype(np.float32)  # (D,)
        return z

    def _get_h_vec(self) -> np.ndarray:
        """
        Return last-layer hidden state h_t as numpy shape (hidden_dim,)
        """
        h, c = self.hidden  # h: (layers, B, H)
        h_last = h[-1, 0].detach().cpu().numpy().astype(np.float32)
        return h_last

    def _get_controller_obs(self) -> np.ndarray:
        """
        Return concat(z_t, h_t) as (obs_dim,)
        """
        h_vec = self._get_h_vec()
        return np.concatenate([self.z, h_vec], axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        """
        Resets real env, encodes first frame -> z_0, sets hidden to zeros.
        Returns controller observation: concat(z_0, h_0)
        """
        obs = self.real_env.reset()  # from your AtariEnv: (1,64,64)
        self.z = self._encode_obs_to_z(obs)

        # init RNN hidden state to zeros
        h0 = torch.zeros((1, 1, self.rnn_hidden_dim), device=self.device)
        c0 = torch.zeros((1, 1, self.rnn_hidden_dim), device=self.device)
        self.hidden = (h0, c0)

        return self._get_controller_obs()

    @torch.no_grad()
    def step(self, action: int):
        """
        1) Update RNN hidden using (current z, action)
        2) Step the REAL env -> new frame, reward, done
        3) Encode new frame -> new z
        4) Return concat(new z, new h), reward, done, info
        """
        # 1) update hidden using current z and action
        z_seq = torch.from_numpy(self.z).view(1, 1, -1).float().to(self.device)  # (1,1,D)
        a_oh = np.zeros((self.action_space.n,), dtype=np.float32)
        a_oh[action] = 1.0
        a_seq = torch.from_numpy(a_oh).view(1, 1, -1).float().to(self.device)    # (1,1,A)

        # forward one step to update hidden (we don't need pi/mu/sigma now)
        _, _, _, next_hidden = self.rnn(z_seq, a_seq, hidden=self.hidden)
        self.hidden = next_hidden

        # 2) step real env
        obs, reward, done, info = self.real_env.step(action)

        # 3) encode next obs -> z
        self.z = self._encode_obs_to_z(obs)

        # 4) controller obs
        ctrl_obs = self._get_controller_obs()
        return ctrl_obs, float(reward), bool(done), info

    def close(self):
        self.real_env.close()
