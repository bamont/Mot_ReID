"""Encodeur de ré-identification : backbone ResNet50 + tête d'embedding L2-normalisee.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = (256, 128)  # (hauteur, largeur) ratio standard d'une bbox pieton en ré-ID
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ReIDEmbeddingNet(nn.Module):
    """ResNet50 (sans la couche fc d'origine) + projection lineaire vers embedding_dim."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.embedding(features)
        return nn.functional.normalize(embedding, p=2, dim=1)


def build_model(embedding_dim: int = 512, pretrained: bool = True) -> ReIDEmbeddingNet:
    return ReIDEmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)


def build_transforms(train: bool) -> transforms.Compose:
    """Transform d'entrainement (avec flip horizontal) ou d'évaluation (déterministe).
    """
    ops = [transforms.Resize(IMAGE_SIZE)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)
