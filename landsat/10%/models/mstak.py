from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MSTAKModel(nn.Module):
    """MST-AK semantic change detection network with a real CLIP ViT-B/16 backbone.

    State-dict compatible with the previously-saved ``pretrained/best_mstak_clip_weights.pth``
    (keys ``backbone.*`` for the raw ``CLIPVisionModel`` and ``classifier.*`` for the
    3-branch fusion classifier head). This mirrors the original working implementation:

        * Frozen CLIP vision backbone, top 2 encoder layers unfrozen.
        * 3-branch interaction: AD (absolute diff), CS (cosine similarity), FS (feature splice).
        * Concat to 2305 channels -> heavy-dropout classifier -> bilinear upsample to input size.

    Forward returns a dict with ``semantic_logits`` (full-resolution 7-class logits) plus
    useful aliases and the low-resolution fused feature map.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        unfreeze_last: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Real pretrained CLIP vision backbone attached directly so checkpoint keys
        # match ``pretrained/best_mstak_clip_weights.pth`` (``backbone.*``).
        from transformers import CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(clip_model_name)

        # 1. Freeze the entire backbone to preserve foundational visual representations.
        #    Fine-tuning the top layers on a tiny 25%-labeled subset caused severe
        #    overfitting (exam F1 collapsed 0.66 -> 0.40 during warmup), so we keep
        #    the CLIP backbone fully frozen and only train the classifier heads.
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 2. Optionally unfreeze the top ``unfreeze_last`` encoder layers (default 0 = none).
        if unfreeze_last > 0:
            for param in self.backbone.encoder.layers[-unfreeze_last:].parameters():
                param.requires_grad = True

        # Heavy-regularization classification head (exact structure of the original MST-AK engine).
        # ``classifier`` predicts the T1 semantic map (kept so pretrained ``classifier.*`` keys load).
        self.classifier = nn.Sequential(
            nn.Conv2d(2305, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.6),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        # Second head predicting the T2 semantic map (dual-head supervision with Label2).
        self.classifier_t2 = nn.Sequential(
            nn.Conv2d(2305, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.6),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        """Encode a single image through CLIP and produce a spatial feature map."""
        outputs = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        # Strip the CLS token (index 0); keep patch tokens only.
        tokens = outputs.last_hidden_state[:, 1:, :]
        b, n_tokens, c = tokens.shape
        h = x.shape[2] // 16
        w = x.shape[3] // 16
        return tokens.transpose(1, 2).view(b, c, h, w)

    def forward(self, t1: Tensor, t2: Tensor, prompts: Sequence[str] | None = None) -> Dict[str, Tensor]:
        f1 = self.extract_features(t1)
        f2 = self.extract_features(t2)

        # 3-branch MST-AK interaction head.
        ad = torch.abs(f2 - f1)
        cs = F.cosine_similarity(f1, f2, dim=1).unsqueeze(1)
        fs = torch.cat((f1, f2), dim=1)

        fused_features = torch.cat((ad, cs, fs), dim=1)
        low_res_t1 = self.classifier(fused_features)
        low_res_t2 = self.classifier_t2(fused_features)

        # Upsample back to input image spatial dimensions.
        target_size = (t1.shape[2], t1.shape[3])
        semantic_t1_logits = F.interpolate(low_res_t1, size=target_size, mode="bilinear", align_corners=True)
        semantic_t2_logits = F.interpolate(low_res_t2, size=target_size, mode="bilinear", align_corners=True)

        return {
            "semantic_logits": semantic_t1_logits,
            "semantic_t1_logits": semantic_t1_logits,
            "semantic_t2_logits": semantic_t2_logits,
            "logits": semantic_t1_logits,
            "seg_logits": semantic_t1_logits,
            "fused_features": fused_features,
        }


def build_mstak(
    in_channels: int = 3,
    num_classes: int = 7,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    unfreeze_last: int = 0,
) -> MSTAKModel:
    return MSTAKModel(
        in_channels=in_channels,
        num_classes=num_classes,
        clip_model_name=clip_model_name,
        unfreeze_last=unfreeze_last,
    )