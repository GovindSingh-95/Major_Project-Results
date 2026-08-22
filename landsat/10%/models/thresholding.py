from __future__ import annotations

from torch import Tensor, nn
import torch


class AdaptiveThresholdSelector(nn.Module):
    """Learn a per-pixel threshold surface from the change feature map."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        hidden = max(channels // 2, 64)
        self.threshold_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scalar = nn.Parameter(torch.tensor(1.0))

    def forward(self, cfm: Tensor, base_threshold: float = 0.95) -> tuple[Tensor, Tensor, Tensor]:
        threshold_map = self.threshold_head(cfm)
        adaptive_threshold = 1.0 - threshold_map * self.scalar.clamp_min(0.0)
        adaptive_threshold = adaptive_threshold.clamp(0.0, 1.0)
        confidence = torch.sigmoid(cfm.mean(dim=1, keepdim=True))
        change_mask = (confidence >= adaptive_threshold * base_threshold).float()
        return threshold_map, adaptive_threshold, change_mask
