"""
Memory Network module for AlphaDeforest.
This module uses an LSTM with temporal attention to predict future latent states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism for weighting time steps in a sequence.
    """
    def __init__(self, hidden_dim: int):
        """
        Initializes the Temporal Attention module.

        Args:
            hidden_dim (int): Dimension of the hidden state vectors.
        """
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal attention.

        Args:
            h (torch.Tensor): Sequence of hidden states of shape (B, T, H).

        Returns:
            torch.Tensor: Weighted context vector of shape (B, H).
        """
        scores = torch.tanh(self.attn(h))                # (B, T, H)
        scores = torch.matmul(scores, self.v)            # (B, T)
        weights = F.softmax(scores, dim=1)               # (B, T)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1) # (B, H)
        return context


class MemoryNetwork(nn.Module):
    """
    Memory Network based on Bidirectional LSTM and Temporal Attention.
    Predicts the next latent representation based on past history.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Initializes the Memory Network.

        Args:
            input_dim (int): Dimension of the input latent vectors (flattened).
            hidden_dim (int): Dimension of the LSTM hidden layers. Defaults to 128.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = TemporalAttention(hidden_dim * 2)
        self.predictor = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Memory Network.

        Args:
            z_seq (torch.Tensor): Sequence of latent vectors of shape (B, T, D).

        Returns:
            torch.Tensor: Predicted next latent vector of shape (B, D).
        """
        # z_seq: (B, T, D)
        h, _ = self.lstm(z_seq)
        context = self.attention(h)
        z_pred = self.predictor(context)
        return z_pred