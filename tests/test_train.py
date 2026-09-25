"""Test d'integration leger des boucles d'entrainement (modes random et batch-hard).
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.reid.dataset import TripletMarket1501Dataset
from src.reid.model import build_model, build_transforms
from src.reid.pk_sampling import PKBatchSampler, PKImageDataset
from src.reid.train import (
    train_one_epoch_batch_hard,
    train_one_epoch_random,
    validate_batch_hard,
    validate_random,
)


def _make_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (64, 32), color=color).save(path)
    return path


def _tiny_nested_index(tmp_path: Path) -> dict:
    """identite -> camera -> images, pour le mode random (TripletMarket1501Dataset)."""
    return {
        1: {
            1: [_make_image(tmp_path / "1_c1_a.jpg", (255, 0, 0))],
            2: [_make_image(tmp_path / "1_c2_a.jpg", (200, 0, 0))],
        },
        2: {
            1: [
                _make_image(tmp_path / "2_c1_a.jpg", (0, 255, 0)),
                _make_image(tmp_path / "2_c1_b.jpg", (0, 200, 0)),
            ]
        },
        3: {1: [_make_image(tmp_path / "3_c1_a.jpg", (0, 0, 255))]},
    }


def _tiny_flat_index(tmp_path: Path) -> dict:
    """identite -> images, pour le mode batch-hard (PKImageDataset). 3 identites x 4 images
    (>= k) pour pouvoir former un batch p=3, k=2 sans tirage avec remise."""
    return {
        pid: [_make_image(tmp_path / f"{pid}_{i}.jpg", color) for i in range(4)]
        for pid, color in [(1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 0, 255))]
    }


class TestRandomMiningLoop:
    def test_train_one_epoch_and_validate_run_without_error(self, tmp_path):
        index = _tiny_nested_index(tmp_path)
        device = torch.device("cpu")
        model = build_model(embedding_dim=8, pretrained=False).to(device)
        criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_dataset = TripletMarket1501Dataset(
            index, length=4, transform=build_transforms(train=True), seed=0
        )
        val_dataset = TripletMarket1501Dataset(
            index, length=4, transform=build_transforms(train=False), seed=0
        )
        train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)

        train_loss = train_one_epoch_random(model, train_loader, optimizer, criterion, device)
        val_loss, val_margin_ok = validate_random(model, val_loader, criterion, device)

        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert 0.0 <= val_margin_ok <= 1.0

    def test_training_step_actually_updates_weights(self, tmp_path):
        index = _tiny_nested_index(tmp_path)
        device = torch.device("cpu")
        model = build_model(embedding_dim=8, pretrained=False).to(device)
        criterion = torch.nn.TripletMarginLoss(margin=10.0, p=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        train_dataset = TripletMarket1501Dataset(
            index, length=2, transform=build_transforms(train=True), seed=0
        )
        train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0, drop_last=True)

        before = model.embedding.weight.clone()
        train_one_epoch_random(model, train_loader, optimizer, criterion, device)
        after = model.embedding.weight

        assert not torch.equal(before, after)


class TestBatchHardMiningLoop:
    def _loaders(self, tmp_path):
        flat_index = _tiny_flat_index(tmp_path)
        train_dataset = PKImageDataset(flat_index, transform=build_transforms(train=True))
        val_dataset = PKImageDataset(flat_index, transform=build_transforms(train=False))
        train_sampler = PKBatchSampler(train_dataset, p=3, k=2, batches_per_epoch=2, seed=0)
        val_sampler = PKBatchSampler(val_dataset, p=3, k=2, batches_per_epoch=2, seed=0)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
        return train_loader, val_loader

    def test_train_one_epoch_and_validate_run_without_error(self, tmp_path):
        device = torch.device("cpu")
        model = build_model(embedding_dim=8, pretrained=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader, val_loader = self._loaders(tmp_path)

        train_loss = train_one_epoch_batch_hard(model, train_loader, optimizer, 0.3, device)
        val_loss, val_margin_ok = validate_batch_hard(model, val_loader, 0.3, device)

        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert 0.0 <= val_margin_ok <= 1.0

    def test_training_step_actually_updates_weights(self, tmp_path):
        device = torch.device("cpu")
        model = build_model(embedding_dim=8, pretrained=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_loader, _ = self._loaders(tmp_path)

        before = model.embedding.weight.clone()
        train_one_epoch_batch_hard(model, train_loader, optimizer, margin=10.0, device=device)
        after = model.embedding.weight

        assert not torch.equal(before, after)
