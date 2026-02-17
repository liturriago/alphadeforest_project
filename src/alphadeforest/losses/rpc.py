import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RPCLoss(nn.Module):
    """
    Reconstruction–Prediction Composite Loss

    L_total = λ_rec * L_rec + λ_pred * L_pred
    """

    def __init__(self, lambda_rec: float = 1.0, lambda_pred: float = 0.5):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_pred = lambda_pred

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_rec = outputs["reconstructions"]
        z_f = outputs["z_f"]
        z_pred = outputs["z_pred"]

        # ==========================
        # 1. Reconstruction Loss
        # ==========================
        loss_rec = F.mse_loss(x_rec, targets)

        # ==========================
        # 2. Prediction Loss
        # ==========================
        if z_pred.shape[1] == 0:
            loss_pred = torch.zeros(1, device=targets.device)
        else:
            z_f_target = z_f[:, 1:]
            loss_pred = F.mse_loss(z_pred, z_f_target)

        # ==========================
        # 3. Composite Loss
        # ==========================
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_pred * loss_pred
        )

        return total_loss, loss_rec, loss_pred
