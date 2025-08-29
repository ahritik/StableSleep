# models/tcn_sleep.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpochCNN(nn.Module):
    def __init__(self, in_ch=1, base=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base, base, 7, padding=3),   nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base*2, 5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.Conv1d(base*2, base*2, 5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*4, 3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.Conv1d(base*4, base*4, 3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_dim = base*4

    def forward(self, x):  # x: (B, C, T)
        return self.net(x) # (B, D)

class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation  # 'same' padding (non-causal)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(8, channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):  # (B, C, L)
        r = x
        x = self.conv1(x); x = self.gn1(x); x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x); x = self.gn2(x); x = F.relu(x)
        return x + r

class TCNSequenceHead(nn.Module):
    def __init__(self, in_dim, tcn_channels=128, n_layers=6, dropout=0.1, num_classes=5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, tcn_channels)
        self.blocks = nn.ModuleList([
            ResidualTCNBlock(tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.out_proj = nn.Conv1d(tcn_channels, num_classes, kernel_size=1)

    def forward(self, feats):  # feats: (B, L, D)
        x = self.in_proj(feats)          # (B, L, C)
        x = x.transpose(1, 2)            # (B, C, L)
        for b in self.blocks:
            x = b(x)                     # (B, C, L)
        logits_all = self.out_proj(x).transpose(1, 2)  # (B, L, num_classes)
        center = logits_all.shape[1] // 2
        return logits_all[:, center, :], logits_all

class SleepTCN(nn.Module):
    """
    Sequence model: per-epoch CNN encoder + dilated TCN across L epochs.
    Returns (logits_center, logits_all) to plug into your existing training loop.
    """
    def __init__(self, in_channels=1, num_classes=5, base=16, dropout=0.1,
                 tcn_channels=128, tcn_layers=6):
        super().__init__()
        self.enc = EpochCNN(in_ch=in_channels, base=base, dropout=dropout)
        self.tcn = TCNSequenceHead(self.enc.out_dim, tcn_channels, tcn_layers, dropout, num_classes)
        self.classifier = nn.Linear(tcn_channels, num_classes)  # for bias init hook

    def forward(self, x):  # x: (B, L, C, T)
        B, L, C, T = x.shape
        x = x.reshape(B*L, C, T)
        feat = self.enc(x).view(B, L, -1)    # (B, L, D)
        logits_center, logits_all = self.tcn(feat)
        return logits_center, logits_all
