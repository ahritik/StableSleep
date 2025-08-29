# models/tsn_sleep.py
"""
TinySleepNet-style model:
- Per-epoch CNN encoder -> feature vector
- BiLSTM over a window of consecutive epochs
- Classifier outputs per time step; we use the CENTER step for training/eval
Input x: (B, L, C, T); Output logits_center: (B, num_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpochCNN(nn.Module):
    def __init__(self, in_ch=1, base=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,      base,   7, padding=3), nn.BatchNorm1d(base),   nn.ReLU(),
            nn.Conv1d(base,       base,   7, padding=3), nn.BatchNorm1d(base),   nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base,       base*2, 5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.Conv1d(base*2,     base*2, 5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base*2,     base*4, 3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.Conv1d(base*4,     base*4, 3, padding=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_dim = base*4

    def forward(self, x):  # x: (B, C, T)
        return self.net(x) # (B, D)

class TinySleepNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, base=32, dropout=0.2,
                 use_bilstm=True, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.enc = EpochCNN(in_ch=in_channels, base=base, dropout=dropout)
        self.use_bilstm = use_bilstm
        D = self.enc.out_dim
        if use_bilstm:
            self.rnn = nn.LSTM(input_size=D, hidden_size=lstm_hidden, num_layers=lstm_layers,
                               batch_first=True, bidirectional=True, dropout=0.0)
            head_in = 2*lstm_hidden
        else:
            # simple per-time MLP if you want to avoid LSTM entirely
            self.proj = nn.Linear(D, D)
            head_in = D
        self.classifier = nn.Linear(head_in, num_classes)

    def forward(self, x):  # x: (B, L, C, T)
        B, L, C, T = x.shape
        x = x.reshape(B*L, C, T)
        feat = self.enc(x)           # (B*L, D)
        feat = feat.view(B, L, -1)   # (B, L, D)

        if self.use_bilstm:
            y, _ = self.rnn(feat)    # (B, L, 2H)
        else:
            y = F.relu(self.proj(feat))  # (B, L, D)

        logits_all = self.classifier(y)   # (B, L, num_classes)

        # Return logits at the CENTER step for loss/metrics
        center = L // 2
        logits_center = logits_all[:, center, :]  # (B, num_classes)
        return logits_center, logits_all
