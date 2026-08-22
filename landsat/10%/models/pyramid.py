from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv_norm_relu(in_channels: int, out_channels: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
	)


class MultiScaleFeaturePyramid(nn.Module):
	"""Top-down feature pyramid that projects backbone features to a shared channel space."""

	def __init__(self, in_channels: Sequence[int], out_channels: int = 256) -> None:
		super().__init__()
		self.in_channels = list(in_channels)
		self.out_channels = out_channels
		self.lateral_convs = nn.ModuleList([_conv_norm_relu(channels, out_channels) for channels in self.in_channels])
		self.output_convs = nn.ModuleList(
			[
				nn.Sequential(
					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
					nn.BatchNorm2d(out_channels),
					nn.ReLU(inplace=True),
				)
				for _ in self.in_channels
			]
		)

	def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
		if len(features) != len(self.lateral_convs):
			raise ValueError(f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}")

		laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
		fused: list[Tensor] = [torch.empty(0, device=laterals[0].device)] * len(laterals)
		top_down = laterals[-1]
		fused[-1] = self.output_convs[-1](top_down)
		for index in range(len(laterals) - 2, -1, -1):
			top_down = laterals[index] + F.interpolate(top_down, size=laterals[index].shape[-2:], mode="bilinear", align_corners=False)
			fused[index] = self.output_convs[index](top_down)
		return fused


class PyramidFusion(nn.Module):
	"""Fuse multi-scale tensors to the highest-resolution feature map."""

	def __init__(self, channels: int = 256) -> None:
		super().__init__()
		self.projection = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, features: Sequence[Tensor]) -> Tensor:
		target_size = features[0].shape[-2:]
		resized = [F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False) for feature in features]
		fused = torch.stack(resized, dim=0).sum(dim=0)
		return self.projection(fused)

