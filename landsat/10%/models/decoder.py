from __future__ import annotations

from typing import Dict

from torch import Tensor, nn
import torch.nn.functional as F


class DecoderHead(nn.Module):
	def __init__(self, in_channels: int = 256, hidden_channels: int = 128, out_channels: int = 2) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(hidden_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(hidden_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
		)

	def forward(self, features: Tensor, target_size: tuple[int, int]) -> Tensor:
		logits = self.block(features)
		return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class MSTAKDecoder(nn.Module):
	def __init__(self, channels: int = 256, num_classes: int = 56, change_classes: int = 2) -> None:
		super().__init__()
		hidden_channels = max(channels // 2, 64)
		self.semantic_head = DecoderHead(in_channels=channels, hidden_channels=hidden_channels, out_channels=num_classes)
		self.change_head = DecoderHead(in_channels=channels, hidden_channels=hidden_channels, out_channels=change_classes)
		self.transition_head = DecoderHead(in_channels=channels, hidden_channels=hidden_channels, out_channels=num_classes)

	def forward(self, fused_features: Tensor, target_size: tuple[int, int]) -> Dict[str, Tensor]:
		return {
			"change_logits": self.change_head(fused_features, target_size),
			"transition_logits": self.transition_head(fused_features, target_size),
			"semantic_logits": self.semantic_head(fused_features, target_size),
		}

	def semantic_logits(self, features: Tensor, target_size: tuple[int, int]) -> Tensor:
		return self.semantic_head(features, target_size)

