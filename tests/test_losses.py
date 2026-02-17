"""
Unit tests for the loss functions in AlphaDeforest.
"""

import torch
from alphadeforest.losses.rpc import RPCLoss


def test_rpc_loss_calculation() -> None:
    """
    Verifies that the RPC loss is calculated correctly as a weighted sum of reconstruction and prediction losses.
    """
    lambda_rec, lambda_pred = 1.0, 0.5
    criterion = RPCLoss(lambda_rec=lambda_rec, lambda_pred=lambda_pred)
    
    batch_size, time_steps, d_flat = 2, 4, 128
    # Simulated data
    targets = torch.randn(batch_size, time_steps, 3, 16, 16)
    outputs = {
        "reconstructions": torch.randn(batch_size, time_steps, 3, 16, 16),
        "z_f": torch.randn(batch_size, time_steps, d_flat), 
        "z_pred": torch.randn(batch_size, time_steps - 1, d_flat)
    }
    
    total_loss, l_rec, l_pred = criterion(outputs, targets)
    
    # Manual verification of the weighted sum
    expected_loss = lambda_rec * l_rec + lambda_pred * l_pred
    assert torch.allclose(total_loss, expected_loss), "Total loss does not match the weighted sum of components"
    assert total_loss.item() > 0, "Loss should be a positive value"


def test_rpc_loss_no_temporal_steps() -> None:
    """
    Tests the edge case where there is only one time step (no prediction possible).
    """
    criterion = RPCLoss()
    outputs = {
        "reconstructions": torch.randn(1, 1, 3, 8, 8),
        "z_f": torch.randn(1, 1, 10),
        "z_pred": torch.empty((1, 0, 10)) # No predictions
    }
    targets = torch.randn(1, 1, 3, 8, 8)
    
    total_loss, _, l_pred = criterion(outputs, targets)
    assert l_pred == 0, "Prediction loss must be 0 if no predictions (z_pred) are provided"