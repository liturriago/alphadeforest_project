"""
Main model architecture for AlphaDeforest.
This module combines the Convolutional Autoencoder and the Memory Network
to perform temporal anomaly detection in satellite imagery.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
from alphadeforest.models.cae import ConvolutionalAutoencoder
from alphadeforest.models.memory import MemoryNetwork


class AlphaDeforest(nn.Module):
    """
    AlphaDeforest model for temporal anomaly detection.
    Combines a CAE for spatial feature extraction and an RNN-based Memory Network
    for temporal prediction.
    """
    def __init__(
        self,
        embedding_dim: int = 64,
        latent_dim: int = 64,
        cae_h: int = 16,
        cae_w: int = 16,
        hidden_dim_mem: int = 128
    ):
        """
        Initializes the AlphaDeforest model.

        Args:
            embedding_dim (int): Input feature dimension.
            latent_dim (int): Bottleneck dimension of the CAE.
            cae_h (int): Spatial height of the latent representation.
            cae_w (int): Spatial width of the latent representation.
            hidden_dim_mem (int): Hidden dimension for the Memory Network.
        """
        super().__init__()
        self.cae = ConvolutionalAutoencoder(embedding_dim, latent_dim)
        
        self.latent_dim = latent_dim
        self.cae_h = cae_h
        self.cae_w = cae_w
        
        # Input to memory network is the flattened latent representation
        self.memory_input_dim = latent_dim * cae_h * cae_w
        self.memory = MemoryNetwork(
            input_dim=self.memory_input_dim,
            hidden_dim=hidden_dim_mem
        )

    def forward(self, x_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x_seq (torch.Tensor): Input sequence of shape (B, T, D, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - reconstructions: Reconstructed sequence (B, T, D, H, W).
                - z_f: Spatial latent features (B, T, latent_dim, cae_h, cae_w).
                - z_pred: Predicted latent features for steps 1 to T-1 (B, T-1, latent_dim*cae_h*cae_w).
                - recon_error: Spatial reconstruction error (B, T).
        """
        batch_size, time_steps, d, h, w = x_seq.shape
        
        # 1. Spatial Reconstruction (CAE)
        # Flatten temporal dimension for CAE processing
        x_flat = x_seq.view(batch_size * time_steps, d, h, w)
        x_rec_flat, z_f_flat = self.cae(x_flat)
        
        # Restore temporal dimension
        x_rec = x_rec_flat.view(batch_size, time_steps, d, h, w)
        z_f = z_f_flat.view(batch_size, time_steps, self.latent_dim, self.cae_h, self.cae_w)
        
        # 2. Temporal Prediction (Memory Network)
        # Flatten spatial dimensions for Memory Network
        z_f_seq = z_f.view(batch_size, time_steps, -1)
        
        # Sequential prediction: predict z_t+1 from z_1...z_t
        # We'll take sub-sequences to predict future steps
        z_preds = []
        for t in range(1, time_steps):
            # Context is all steps up to t
            z_context = z_f_seq[:, :t, :]
            z_next_pred = self.memory(z_context)
            z_preds.append(z_next_pred)
            
        z_pred = torch.stack(z_preds, dim=1) if z_preds else torch.empty((batch_size, 0, self.memory_input_dim), device=x_seq.device)
        
        # 3. Anomaly Scores (Simple Reconstruction Error for now)
        recon_error = torch.mean((x_seq - x_rec)**2, dim=(2, 3, 4))
        
        return {
            "reconstructions": x_rec,
            "z_f": z_f,
            "z_pred": z_pred,
            "recon_error": recon_error
        }
