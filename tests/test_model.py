"""Tests de ReIDEmbeddingNet et des transforms.

Utilise systematiquement ``pretrained=False`` : les tests ne doivent pas dependre
d'un telechargement reseau des poids ImageNet (lenteur + flakiness en CI).
"""

import torch
from PIL import Image

from src.reid.model import IMAGE_SIZE, ReIDEmbeddingNet, build_model, build_transforms


class TestReIDEmbeddingNet:
    def test_output_shape_matches_embedding_dim(self):
        model = ReIDEmbeddingNet(embedding_dim=64, pretrained=False)
        x = torch.randn(4, 3, *IMAGE_SIZE)
        out = model(x)
        assert out.shape == (4, 64)

    def test_output_is_l2_normalized(self):
        model = ReIDEmbeddingNet(embedding_dim=32, pretrained=False)
        x = torch.randn(3, 3, *IMAGE_SIZE)
        out = model(x)
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_build_model_helper_returns_same_type(self):
        model = build_model(embedding_dim=16, pretrained=False)
        assert isinstance(model, ReIDEmbeddingNet)


class TestBuildTransforms:
    def _dummy_image(self) -> Image.Image:
        return Image.new("RGB", (64, 32), color=(10, 20, 30))

    def test_train_transform_output_shape(self):
        transform = build_transforms(train=True)
        out = transform(self._dummy_image())
        assert out.shape == (3, *IMAGE_SIZE)

    def test_eval_transform_output_shape(self):
        transform = build_transforms(train=False)
        out = transform(self._dummy_image())
        assert out.shape == (3, *IMAGE_SIZE)

    def test_eval_transform_is_deterministic(self):
        # Pas de flip aleatoire en mode eval -> deux passages doivent etre identiques.
        transform = build_transforms(train=False)
        image = self._dummy_image()
        assert torch.equal(transform(image), transform(image))
