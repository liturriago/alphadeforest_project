import torch
import torch.nn as nn
import torch.nn.functional as F
from .cae import ConvolutionalAutoencoder
from .memory import MemoryNetwork

class AlphaDeforest(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        latent_dim: int = 64,
        cae_h: int = 16,
        cae_w: int = 16,
        hidden_dim_mem: int = 128,
    ):
        super().__init__()

        self.cae = ConvolutionalAutoencoder(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim
        )

        self.z_f_dim = latent_dim * cae_h * cae_w

        self.memory = MemoryNetwork(
            input_dim=self.z_f_dim,
            hidden_dim=hidden_dim_mem
        )

    def forward(self, x_seq: torch.Tensor):
        """
        x_seq: (B, T, D, H, W)
        """
        B, T, _, _, _ = x_seq.shape

        z_f_seq = []
        recon_errors = []
        reconstructions = []

        for t in range(T):
            x_t = x_seq[:, t]
            x_rec, z_f = self.cae(x_t)
            reconstructions.append(x_rec)

            err = F.mse_loss(x_rec, x_t, reduction="none").mean(dim=(1, 2, 3)) 
            recon_errors.append(err)

            z_f_seq.append(z_f.view(B, -1))

        z_f_seq = torch.stack(z_f_seq, dim=1)              # (B, T, Z)
        recon_errors = torch.stack(recon_errors, dim=1)    # (B, T)
        reconstructions = torch.stack(reconstructions, 1)  # (B, T, D, H, W)

        z_pred_seq = []
        for t in range(1, T):
            z_pred = self.memory(z_f_seq[:, :t])
            z_pred_seq.append(z_pred)

        z_pred_seq = torch.stack(z_pred_seq, dim=1)        # (B, T-1, Z)

        return {
            "reconstructions": reconstructions,
            "z_f": z_f_seq,
            "z_pred": z_pred_seq,
            "recon_error": recon_errors,
        }