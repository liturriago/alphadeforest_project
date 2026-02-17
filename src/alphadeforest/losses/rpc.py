"""
Loss function module for AlphaDeforest.
Implements the Reconstruction-Prediction Composite (RPC) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RPCLoss(nn.Module):
    """
    Reconstruction-Prediction Composite Loss.
    
    Formula: L_total = lambda_rec * L_rec + lambda_pred * L_pred
    """

    def __init__(self, lambda_rec: float = 1.0, lambda_pred: float = 0.5):
        """
        Initializes the RPCLoss.

        Args:
            lambda_rec (float): Weight for the reconstruction loss. Defaults to 1.0.
            lambda_pred (float): Weight for the prediction loss. Defaults to 0.5.
        """
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_pred = lambda_pred

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the composite loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs containing "reconstructions", "z_f", and "z_pred".
            targets (torch.Tensor): Original input sequences used as ground truth for reconstruction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (total_loss, loss_rec, loss_pred).
        """

        x_rec = outputs["reconstructions"]
        z_f = outputs["z_f"]
        z_pred = outputs["z_pred"]

        # 1. Reconstruction Loss (Spatial)
        loss_rec = F.mse_loss(x_rec, targets)

        # 2. Prediction Loss (Temporal Latent Space)
        if z_pred.shape[1] == 0:
            loss_pred = torch.tensor(0.0, device=targets.device)
        else:
            # Shift z_f to get targets for prediction (z_t+1 predicted from z_t)
            z_f_target = z_f[:, 1:]
            loss_pred = F.mse_loss(z_pred, z_f_target)

        # 3. Composite Loss
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_pred * loss_pred
        )

        return total_loss, loss_rec, loss_pred
