"""
Unit tests for the model architectures in AlphaDeforest.
"""

import torch
from alphadeforest.models.alpha_deforest import AlphaDeforest
from alphadeforest.models.cae import ConvolutionalAutoencoder
from alphadeforest.models.memory import MemoryNetwork
from typing import Dict, Any


def test_cae_shapes(sample_dims: Dict[str, int]) -> None:
    """
    Verifies that the CAE produces the correct reconstruction and latent feature shapes.

    Args:
        sample_dims (Dict[str, int]): Standard dimensions from conftest.
    """
    d = sample_dims
    model = ConvolutionalAutoencoder(embedding_dim=d["D"], latent_dim=32)
    x = torch.randn(d["B"], d["D"], d["H"], d["W"])
    
    x_rec, z_f = model(x)
    
    assert x_rec.shape == x.shape, "CAE reconstruction must match input shape"
    assert z_f.shape[1] == 32, "Incorrect latent dimension depth"
    assert z_f.shape[2] == d["H"] // 8, "Encoder should reduce spatial H by a factor of 8"


def test_memory_network_prediction() -> None:
    """
    Verifies that the MemoryNetwork predicts a single latent vector of the correct size.
    """
    batch, time, d_flat = 2, 5, 128
    model = MemoryNetwork(input_dim=d_flat, hidden_dim=64)
    z_seq = torch.randn(batch, time, d_flat)
    
    z_pred = model(z_seq)
    
    assert z_pred.shape == (batch, d_flat), "Should predict a single latent vector matching input dimension"


def test_alpha_deforest_forward(input_tensor: torch.Tensor, mock_config: Dict[str, Any]) -> None:
    """
    Integration test for the full AlphaDeforest forward pass.

    Args:
        input_tensor (torch.Tensor): Synthetic input sequence from conftest.
        mock_config (Dict[str, Any]): Mock configuration from conftest.
    """
    m_cfg = mock_config["model"]
    model = AlphaDeforest(**m_cfg)
    
    output = model(input_tensor)
    
    assert "reconstructions" in output
    assert "z_pred" in output
    # T-1 because the first prediction requires at least one previous step
    assert output["z_pred"].shape[1] == input_tensor.shape[1] - 1 
    assert output["recon_error"].shape == (input_tensor.shape[0], input_tensor.shape[1])