from __future__ import annotations

import hashlib
import re
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _tokenize(prompt: str, max_length: int) -> list[str]:
	tokens = re.findall(r"[a-z0-9]+", prompt.lower())
	return tokens[:max_length]


def _hash_token(token: str, vocab_size: int) -> int:
	digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
	return int(digest, 16) % vocab_size


class ClipTextEncoder(nn.Module):
	"""A lightweight CLIP-style text encoder for land-cover prompts."""

	def __init__(
		self,
		prompts: Sequence[str] | None = None,
		embed_dim: int = 256,
		vocab_size: int = 8192,
		max_length: int = 16,
		num_layers: int = 2,
		num_heads: int = 8,
	) -> None:
		super().__init__()
		self.prompts = list(prompts) if prompts is not None else []
		self.embed_dim = embed_dim
		self.vocab_size = vocab_size
		self.max_length = max_length
		self.token_embedding = nn.Embedding(vocab_size, embed_dim)
		self.position_embedding = nn.Parameter(torch.zeros(max_length, embed_dim))
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=embed_dim * 4,
			dropout=0.1,
			activation="gelu",
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.norm = nn.LayerNorm(embed_dim)
		self.projection = nn.Linear(embed_dim, embed_dim)

	def set_prompts(self, prompts: Sequence[str]) -> None:
		self.prompts = list(prompts)

	def _prompts_to_tokens(self, prompts: Sequence[str], device: torch.device) -> Tensor:
		token_ids = torch.zeros(len(prompts), self.max_length, dtype=torch.long, device=device)
		for row, prompt in enumerate(prompts):
			tokens = _tokenize(prompt, self.max_length)
			if not tokens:
				tokens = ["land", "cover"]
			for col, token in enumerate(tokens[: self.max_length]):
				token_ids[row, col] = _hash_token(token, self.vocab_size)
		return token_ids

	def encode(self, prompts: Sequence[str], device: torch.device | None = None) -> Tensor:
		if device is None:
			device = self.position_embedding.device
		tokens = self._prompts_to_tokens(prompts, device=device)
		features = self.token_embedding(tokens)
		position = self.position_embedding[: features.shape[1]].unsqueeze(0)
		features = features + position
		features = self.transformer(features)
		pooled = features.mean(dim=1)
		pooled = self.norm(pooled)
		pooled = self.projection(pooled)
		return F.normalize(pooled, dim=-1)

	def forward(self, prompts: Sequence[str] | None = None) -> Tensor:
		if prompts is None:
			if not self.prompts:
				raise ValueError("No prompts were provided to ClipTextEncoder")
			prompts = self.prompts
		return self.encode(prompts)

