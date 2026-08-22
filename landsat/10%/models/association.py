from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ThreeBranchAssociationBlock(nn.Module):
	def __init__(self, channels: int = 256) -> None:
		super().__init__()
		self.ad_branch = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)
		self.cs_branch = nn.Sequential(
			nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)
		self.fs_branch = nn.Sequential(
			nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)
		self.fusion = nn.Sequential(
			nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, t1: Tensor, t2: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
		absolute_difference = torch.abs(t2 - t1)
		cosine_similarity = F.cosine_similarity(t1, t2, dim=1, eps=1e-6).unsqueeze(1)
		spliced = torch.cat([t1, t2], dim=1)

		ad = self.ad_branch(absolute_difference)
		cs = self.cs_branch(cosine_similarity)
		fs = self.fs_branch(spliced)
		fused = self.fusion(torch.cat([ad, cs, fs], dim=1))
		return fused, {"ad": ad, "cs": cs, "fs": fs}


class MultiLevelAssociation(nn.Module):
	def __init__(self, channels: int = 256, levels: int = 4) -> None:
		super().__init__()
		self.blocks = nn.ModuleList([ThreeBranchAssociationBlock(channels=channels) for _ in range(levels)])
		self.fusion = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, features_t1: Sequence[Tensor], features_t2: Sequence[Tensor]) -> tuple[Tensor, List[Dict[str, Tensor]]]:
		if len(features_t1) != len(self.blocks) or len(features_t2) != len(self.blocks):
			raise ValueError("Unexpected number of pyramid levels")

		level_outputs: list[Tensor] = []
		aux: list[Dict[str, Tensor]] = []
		target_size = features_t1[0].shape[-2:]

		for block, t1, t2 in zip(self.blocks, features_t1, features_t2):
			fused, branch_aux = block(t1, t2)
			level_outputs.append(F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False))
			aux.append(branch_aux)

		fused_levels = torch.stack(level_outputs, dim=0).sum(dim=0)
		fused_levels = self.fusion(fused_levels)
		return fused_levels, aux

