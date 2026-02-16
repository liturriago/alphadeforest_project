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

        # Dimensión aplanada del vector latente espacial (z_f)
        self.z_f_dim = latent_dim * cae_h * cae_w

        self.memory = MemoryNetwork(
            input_dim=self.z_f_dim,
            hidden_dim=hidden_dim_mem
        )

    def forward(self, x_seq: torch.Tensor):
        """
        x_seq: (B, T, C, H, W) 
        Donde C es Channels (o D en tu notación anterior)
        """
        B, T, C, H, W = x_seq.shape

        # --- OPTIMIZACIÓN 1: Procesamiento Paralelo (Batch Flattening) ---
        # Colapsamos Batch y Time para procesar todas las imágenes juntas
        # (B*T, C, H, W)
        x_flat = x_seq.view(B * T, C, H, W)

        # Pasamos todo por el CAE de una sola vez (Mucho más rápido que un bucle)
        x_rec_flat, z_f_flat = self.cae(x_flat)

        # Restauramos las dimensiones originales
        # Reconstrucciones: (B, T, C, H, W)
        reconstructions = x_rec_flat.view(B, T, C, H, W)
        
        # Latentes: (B, T, Z_dim)
        # Aplanamos las dimensiones espaciales del latente para la memoria
        z_f_seq = z_f_flat.view(B, T, -1)

        # --- OPTIMIZACIÓN 2: Cálculo de Error Vectorizado ---
        # Calculamos MSE sin reducción para mantener las dimensiones
        # (B, T, C, H, W) -> Promediamos sobre (C, H, W) para obtener error por frame
        diff = F.mse_loss(reconstructions, x_seq, reduction="none")
        recon_errors = diff.mean(dim=(2, 3, 4))  # Resultado: (B, T)

        # --- MEMORIA (Secuencial) ---
        # La memoria sigue siendo secuencial porque depende del pasado (Autoregresiva)
        z_pred_seq = []
        
        # Iteramos desde t=1 porque necesitamos contexto previo para predecir
        for t in range(1, T):
            # Pasamos la historia hasta t (z_f_seq[:, :t])
            # Asumimos que tu MemoryNetwork maneja secuencias de longitud variable
            z_pred = self.memory(z_f_seq[:, :t])
            z_pred_seq.append(z_pred)

        # Apilamos las predicciones
        if len(z_pred_seq) > 0:
            z_pred_seq = torch.stack(z_pred_seq, dim=1)  # (B, T-1, Z)
        else:
            # Fallback por si T < 2 (aunque el dataset lo filtra)
            z_pred_seq = torch.empty(B, 0, self.z_f_dim, device=x_seq.device)

        return {
            "reconstructions": reconstructions,
            "z_f": z_f_seq,
            "z_pred": z_pred_seq,
            "recon_error": recon_errors,
        }