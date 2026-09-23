"""Tests de TripletMarket1501Dataset avec de petites images PNG generées à la volée.

Ce module a besoin de vraies images ouvrables par Pillow puisque
``__getitem__`` charge et transforme les fichiers.
"""

from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms

from src.reid.dataset import TripletMarket1501Dataset


def _make_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color=color).save(path)
    return path


@pytest.fixture
def index(tmp_path):
    """3 identités : deux avec 2 caméras (positif cross-camera possible), une mono-caméra."""
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


class TestTripletMarket1501Dataset:
    def test_len_matches_requested_length(self, index):
        ds = TripletMarket1501Dataset(index, length=17)
        assert len(ds) == 17

    def test_rejects_non_positive_length(self, index):
        with pytest.raises(ValueError):
            TripletMarket1501Dataset(index, length=0)

    def test_getitem_returns_expected_keys(self, index):
        ds = TripletMarket1501Dataset(index, length=5)
        item = ds[0]
        assert set(item) == {"anchor", "positive", "negative", "anchor_id", "negative_id"}
        assert isinstance(item["anchor"], Image.Image)  # pas de transform -> PIL brut
        assert item["anchor_id"] != item["negative_id"]

    def test_transform_is_applied(self, index):
        transform = transforms.Compose([transforms.ToTensor()])
        ds = TripletMarket1501Dataset(index, length=5, transform=transform)
        item = ds[0]
        assert item["anchor"].shape == (3, 8, 8)
        assert item["positive"].shape == (3, 8, 8)
        assert item["negative"].shape == (3, 8, 8)

    def test_same_index_same_epoch_is_deterministic(self, index):
        ds = TripletMarket1501Dataset(index, length=5, seed=42)
        a = ds[2]
        b = ds[2]
        assert a["anchor_id"] == b["anchor_id"]
        assert a["negative_id"] == b["negative_id"]

    def test_set_epoch_changes_sampled_triplets(self, index):
        # Sur assez d'items, changer d'époque doit faire varier au moins un tirage.
        ds = TripletMarket1501Dataset(index, length=20, seed=42)
        epoch0 = [(ds[i]["anchor_id"], ds[i]["negative_id"]) for i in range(20)]
        ds.set_epoch(1)
        epoch1 = [(ds[i]["anchor_id"], ds[i]["negative_id"]) for i in range(20)]
        assert epoch0 != epoch1

    def test_same_seed_and_epoch_reproducible_across_instances(self, index):
        ds_a = TripletMarket1501Dataset(index, length=10, seed=7)
        ds_b = TripletMarket1501Dataset(index, length=10, seed=7)
        results_a = [(ds_a[i]["anchor_id"], ds_a[i]["negative_id"]) for i in range(10)]
        results_b = [(ds_b[i]["anchor_id"], ds_b[i]["negative_id"]) for i in range(10)]
        assert results_a == results_b
