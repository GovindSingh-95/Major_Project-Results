from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DualAttentionModule(nn.Module):
	"""Inject text-derived priors into visual features with channel and spatial gates."""

	def __init__(self, channels: int = 256, text_dim: int = 256, reduction: int = 4) -> None:
		super().__init__()
		hidden = max(channels // reduction, 32)
		self.text_projection = nn.Linear(text_dim, channels)
		self.channel_mlp = nn.Sequential(
			nn.Linear(channels * 2, hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, channels),
			nn.Sigmoid(),
		)
		self.spatial_gate = nn.Sequential(
			nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels // reduction),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction, 1, kernel_size=1),
		)
		self.output_norm = nn.BatchNorm2d(channels)

	def forward(self, features: Tensor, text_embeddings: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
		if text_embeddings.dim() != 2:
			raise ValueError("text_embeddings must have shape [num_prompts, dim]")

		batch_size, channels, _, _ = features.shape
		text_context = text_embeddings.mean(dim=0, keepdim=True)
		text_context = self.text_projection(text_context).expand(batch_size, -1)

		visual_context = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
		channel_gate = self.channel_mlp(torch.cat([visual_context, text_context], dim=1)).view(batch_size, channels, 1, 1)

		aligned_text = text_context.view(batch_size, channels, 1, 1)
		cosine_map = F.cosine_similarity(features, aligned_text, dim=1, eps=1e-6).unsqueeze(1)
		spatial_gate = torch.sigmoid(self.spatial_gate(features) + cosine_map)

		enhanced = features * channel_gate * spatial_gate
		enhanced = self.output_norm(enhanced + features)
		return enhanced, {"channel_gate": channel_gate, "spatial_gate": spatial_gate}


class MultimodalDualAttention(nn.Module):
	def __init__(self, channels: int = 256, text_dim: int = 256) -> None:
		super().__init__()
		self.module = DualAttentionModule(channels=channels, text_dim=text_dim)

	def forward(self, features: Tensor, text_embeddings: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
		return self.module(features, text_embeddings)

