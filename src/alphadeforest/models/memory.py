import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, h: torch.Tensor):
        # h: (B, T, H)
        scores = torch.tanh(self.attn(h))                # (B, T, H)
        scores = torch.matmul(scores, self.v)            # (B, T)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return context

class MemoryNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
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

    def forward(self, z_seq: torch.Tensor):
        # z_seq: (B, T, D)
        h, _ = self.lstm(z_seq)
        context = self.attention(h)
        z_pred = self.predictor(context)
        return z_pred