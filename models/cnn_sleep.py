"""Compact 1D CNN for sleep staging with optional lightweight temporal attention.
We purposely keep BatchNorm (BN) layers because Tent-style TTA adapts BN parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSE(nn.Module):
    """Squeeze temporal dimension → channel-wise gate (SE-style, 1D)."""
    def __init__(self, channels:int, reduction:int=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(-1, keepdim=True)      # temporal squeeze → (B,C,1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))    # channel-wise gate
        return x * w

class SleepCNN(nn.Module):
    """Three conv blocks → global pooling → linear head.
    Shapes assume input (B, C=1, T=fs*epoch_sec). BN is kept for TTA.
    """
    def __init__(self, in_channels=1, num_classes=5, base=32, dropout=0.1, attn=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base,  kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base,        base,  kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base,        base*2,kernel_size=5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.Conv1d(base*2,      base*2,kernel_size=5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base*2,      base*4,kernel_size=3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.Conv1d(base*4,      base*4,kernel_size=3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
        )
        self.attn = TemporalSE(base*4) if attn else nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global average pooling over time
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*4, num_classes, bias=True)
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def init_prior_bias(self, class_counts):
        """Initialize classifier bias to log-priors (stabilizes early training)."""
        import numpy as np
        counts = np.asarray(class_counts, dtype=np.float64) + 1e-6
        probs = counts / counts.sum()
        logit_bias = np.log(probs)
        self.head[-1].bias.data = torch.from_numpy(logit_bias).float().to(self.head[-1].bias.data.device)
