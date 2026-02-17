"""
Convolutional Autoencoder module for AlphaDeforest.
This module extracts spatial features from satellite imagery embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder (CAE) for spatial feature extraction.
    Compresses high-dimensional embeddings into a compact latent representation.
    """
    def __init__(self, embedding_dim: int = 64, latent_dim: int = 64):
        """
        Initializes the CAE.

        Args:
            embedding_dim (int): Number of input channels (the original embedding depth).
            latent_dim (int): Number of channels in the bottleneck latent representation.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, stride=2, padding=1),  # Input: HxW -> Output: (H/2)x(W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),            # (H/2)x(W/2) -> (H/4)x(W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, 3, stride=2, padding=1),     # (H/4)x(W/4) -> (H/8)x(W/8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, embedding_dim, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CAE.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_rec: Reconstructed tensor of shape (B, C, H, W).
                - z_f: Latent representation of shape (B, latent_dim, H/8, W/8).
        """
        z_f = self.encoder(x)
        x_rec = self.decoder(z_f)
        return x_rec, z_f