from __future__ import annotations

from typing import List

from torch import Tensor, nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
		super().__init__()
		self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample: nn.Module | None = None
		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))

	def forward(self, x: Tensor) -> Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out = out + identity
		out = self.relu(out)
		return out


class ResNet34Backbone(nn.Module):
	"""A self-contained ResNet-34 style backbone for bitemporal feature extraction."""

	def __init__(self, in_channels: int = 3, base_channels: int = 64, frozen_stem: bool = False) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.base_channels = base_channels
		self.stem = nn.Sequential(
			nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(base_channels),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		self.layer1 = self._make_layer(base_channels, base_channels, blocks=3, stride=1)
		self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=4, stride=2)
		self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=6, stride=2)
		self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks=3, stride=2)
		self.feature_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

		if frozen_stem:
			for parameter in self.stem.parameters():
				parameter.requires_grad = False

	def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
		layers = [BasicBlock(in_channels, out_channels, stride=stride)]
		for _ in range(blocks - 1):
			layers.append(BasicBlock(out_channels, out_channels, stride=1))
		return nn.Sequential(*layers)

	def forward_single(self, x: Tensor) -> List[Tensor]:
		x = self.stem(x)
		c2 = self.layer1(x)
		c3 = self.layer2(c2)
		c4 = self.layer3(c3)
		c5 = self.layer4(c4)
		return [c2, c3, c4, c5]

	def forward(self, t1: Tensor, t2: Tensor) -> tuple[list[Tensor], list[Tensor]]:
		return self.forward_single(t1), self.forward_single(t2)


class CLIPVisionBackbone(nn.Module):
	"""Real OpenAI CLIP ViT vision backbone (frozen except the top 2 encoder layers).

	Extracts patch-token features at stride 16 and reshapes them into a spatial
	feature map of shape ``[B, C, H/16, W/16]`` (CLS token removed).

	NOTE: In ``models/mstak.py`` the backbone is attached directly as
	``model.backbone`` so checkpoint keys match the previously-saved
	``best_mstak_clip_weights.pth`` (keys ``backbone.embeddings.*``,
	``backbone.encoder.layers.*``, etc.).
	"""

	def __init__(self, model_name: str = "openai/clip-vit-base-patch16", unfreeze_last: int = 2, patch_size: int = 16) -> None:
		super().__init__()
		from transformers import CLIPVisionModel

		self.model = CLIPVisionModel.from_pretrained(model_name)
		self.patch_size = patch_size

		# Freeze the full backbone, then unfreeze the top ``unfreeze_last`` encoder layers
		# for domain-specific (satellite) spatial tuning.
		for param in self.model.parameters():
			param.requires_grad = False
		for param in self.model.encoder.layers[-unfreeze_last:].parameters():
			param.requires_grad = True

	@property
	def feature_channels(self) -> list[int]:
		# CLIP ViT-B/16 hidden size
		return [self.model.config.hidden_size]

	def forward_single(self, x: Tensor) -> list[Tensor]:
		outputs = self.model(pixel_values=x, interpolate_pos_encoding=True)
		# Strip the CLS token (index 0); keep patch tokens only.
		tokens = outputs.last_hidden_state[:, 1:, :]
		b, n_tokens, c = tokens.shape
		h = x.shape[2] // self.patch_size
		w = x.shape[3] // self.patch_size
		feature_map = tokens.transpose(1, 2).view(b, c, h, w)
		return [feature_map]

	def forward_single_map(self, x: Tensor) -> Tensor:
		return self.forward_single(x)[0]

	def forward(self, t1: Tensor, t2: Tensor) -> tuple[list[Tensor], list[Tensor]]:
		return self.forward_single(t1), self.forward_single(t2)


class SiameseBackbone(nn.Module):
	"""Shared-weight encoder for T1 and T2 imagery."""

	def __init__(self, in_channels: int = 3, base_channels: int = 64, frozen_stem: bool = False) -> None:
		super().__init__()
		self.encoder = ResNet34Backbone(in_channels=in_channels, base_channels=base_channels, frozen_stem=frozen_stem)

	@property
	def feature_channels(self) -> list[int]:
		return self.encoder.feature_channels

	def forward_single(self, x: Tensor) -> list[Tensor]:
		return self.encoder.forward_single(x)

	def forward(self, t1: Tensor, t2: Tensor) -> tuple[list[Tensor], list[Tensor]]:
		return self.encoder(t1, t2)

