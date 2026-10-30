"""1D Convolutional Neural Network for raw 6-channel HAR time series."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Residual Conv1D block with BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class MotionGuardCNN1D(nn.Module):
    """Deep 1D CNN architecture for 6-channel 3-second HAR waveforms (KU-HAR)."""

    def __init__(
        self,
        in_channels: int = 6,
        sequence_length: int = 300,
        num_classes: int = 18,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
        fc_hidden: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.conv_channels = conv_channels or [64, 128, 256]
        self.kernel_sizes = kernel_sizes or [7, 5, 3]
        self.dropout_rate = dropout
        self.fc_hidden = fc_hidden

        # Multi-scale input stems: capture fast transient spikes and slow gestures
        self.stem_k7 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.stem_k3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        current_channels = 64  # concatenated stems 32 + 32
        
        # Staged convolutional blocks with downsampling
        self.stage1 = ConvBlock1D(current_channels, self.conv_channels[0], kernel_size=self.kernel_sizes[0], stride=2, dropout=dropout)
        self.stage2 = ConvBlock1D(self.conv_channels[0], self.conv_channels[1], kernel_size=self.kernel_sizes[1], stride=2, dropout=dropout)
        self.stage3 = ConvBlock1D(self.conv_channels[1], self.conv_channels[2], kernel_size=self.kernel_sizes[2], stride=2, dropout=dropout)

        # Global pooling: combination of Adaptive Avg Pooling and Max Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[2], fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent embeddings before the classification head."""
        stem1 = self.stem_k7(x)
        stem2 = self.stem_k3(x)
        h = torch.cat([stem1, stem2], dim=1)
        
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        
        pooled = self.global_pool(h).squeeze(-1)  # (B, conv_channels[-1])
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits (B, num_classes)."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    def get_config(self) -> Dict[str, Any]:
        """Returns architecture configuration for exact reconstruction."""
        return {
            "in_channels": self.in_channels,
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "conv_channels": self.conv_channels,
            "kernel_sizes": self.kernel_sizes,
            "dropout": self.dropout_rate,
            "fc_hidden": self.fc_hidden,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MotionGuardCNN1D":
        return cls(
            in_channels=config.get("in_channels", 6),
            sequence_length=config.get("sequence_length", 300),
            num_classes=config.get("num_classes", 18),
            conv_channels=config.get("conv_channels", [64, 128, 256]),
            kernel_sizes=config.get("kernel_sizes", [7, 5, 3]),
            dropout=config.get("dropout", 0.3),
            fc_hidden=config.get("fc_hidden", 128),
        )


class MotionGuardCNNGRU(nn.Module):
    """Optional hybrid CNN-GRU architecture for recurrent temporal dynamics."""

    def __init__(
        self,
        in_channels: int = 6,
        sequence_length: int = 300,
        num_classes: int = 18,
        cnn_channels: int = 64,
        gru_hidden: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.gru_hidden = gru_hidden

        self.conv1 = nn.Conv1d(in_channels, cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 300)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h)))
        
        # Prepare for GRU: (B, C, T') -> (B, T', C)
        h_seq = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h_seq)
        
        # Take last time step or pool
        last_step = gru_out[:, -1, :]  # (B, gru_hidden * 2)
        logits = self.classifier(last_step)
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "cnn_channels": self.cnn_channels,
            "gru_hidden": self.gru_hidden,
        }
