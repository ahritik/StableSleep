from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, kernel_size=1, bias=True)
        self.score = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, x):  # x: (B, C, T)
        h = torch.tanh(self.proj(x))           # (B, H, T)
        e = self.score(h).squeeze(1)           # (B, T)
        a = torch.softmax(e, dim=-1)           # (B, T)
        ctx = torch.einsum("bct,bt->bc", x, a) # (B, C)
        return ctx, a

class SleepCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 5, hidden: int = 64, attn_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.attn  = TemporalAttention(hidden, hidden=attn_hidden)
        self.fc1   = nn.Linear(hidden, hidden)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Linear(hidden, n_classes)

    def features(self, x):  # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        ctx, a = self.attn(x)  # (B, C)
        return ctx

    def forward(self, x):
        h = self.features(x)
        h = self.drop(F.relu(self.bn3(self.fc1(h))))
        logits = self.head(h)
        return logits
