import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for Atari frames.

    Input:
        x : (B, 1, 64, 64)   grayscale, normalized to [0,1]

    Latent:
        z : (B, latent_dim)

    Output:
        x_hat : (B, 1, 64, 64)
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # =========================
        # Encoder
        # =========================
        # (1,64,64) -> (256,4,4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> 32x32x32
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128x8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> 256x4x4
            nn.ReLU(inplace=True),
        )

        self.enc_out_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # =========================
        # Decoder
        # =========================
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # -> 64x64
            nn.Sigmoid(),  # output in [0,1]
        )

    # ======================================================
    # Core VAE functions
    # ======================================================
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    # ======================================================
    # Loss
    # ======================================================
    @staticmethod
    def loss_fn(x, x_hat, mu, logvar, beta: float = 1.0):
        """
        VAE loss = reconstruction + beta * KL

        Reconstruction: MSE (stable for grayscale Atari)
        KL: standard Gaussian prior
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
