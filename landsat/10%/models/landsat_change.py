from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LandSatChangeModel(nn.Module):
    """LandSat bitemporal *change-map* model.

    Takes a bi-temporal image pair (``a`` = t1, ``b`` = t2) and predicts a single
    5-class change map (0 = unchanged/background, 1..4 = land-cover change classes,
    255 = ignore). The CLIP ViT-B/16 backbone is fully frozen to prevent
    overfitting on small satellite datasets (the same fix that stabilized the
    SECOND results).
    """

    def __init__(self, num_classes: int = 5, clip_model_name: str = "openai/clip-vit-base-patch16") -> None:
        super().__init__()
        self.num_classes = num_classes

        from transformers import CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(clip_model_name)
        # Freeze the entire backbone.
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Conv2d(2305, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        """Encode a single image through CLIP and produce a spatial feature map."""
        outputs = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        tokens = outputs.last_hidden_state[:, 1:, :]  # strip CLS token
        b, n_tokens, c = tokens.shape
        h = x.shape[2] // 16
        w = x.shape[3] // 16
        return tokens.transpose(1, 2).view(b, c, h, w)

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Predict the change map.

        Returns a dict with the key ``change_logits`` (and aliases
        ``semantic_logits`` / ``logits``) at full input resolution.
        """
        fa = self.extract_features(a)
        fb = self.extract_features(b)

        ad = torch.abs(fb - fa)
        cs = F.cosine_similarity(fa, fb, dim=1).unsqueeze(1)
        fs = torch.cat((fa, fb), dim=1)
        fused = torch.cat((ad, cs, fs), dim=1)

        low_res = self.classifier(fused)
        change_logits = F.interpolate(low_res, size=(a.shape[2], a.shape[3]), mode="bilinear", align_corners=True)

        return {
            "change_logits": change_logits,
            "semantic_logits": change_logits,
            "logits": change_logits,
        }


def build_landsat_change_model(num_classes: int = 5, clip_model_name: str = "openai/clip-vit-base-patch16") -> LandSatChangeModel:
    return LandSatChangeModel(num_classes=num_classes, clip_model_name=clip_model_name)