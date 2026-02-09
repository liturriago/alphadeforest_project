import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, latent_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, stride=2, padding=1),  # 128x128 → 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),            # 64x64 → 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, 3, stride=2, padding=1),     # 32x32 → 16x16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, embedding_dim, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor):
        z_f = self.encoder(x)
        x_rec = self.decoder(z_f)
        return x_rec, z_f