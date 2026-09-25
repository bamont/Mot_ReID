"""Tests de src/reid/pk_sampling.py (dataset PK, sampler, mining batch-hard)."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.reid.pk_sampling import (
    PKBatchSampler,
    PKImageDataset,
    batch_hard_triplet_loss,
    mine_batch_hard,
)


def _make_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color=color).save(path)
    return path


@pytest.fixture
def flat_index(tmp_path):
    """4 identités, nombre d'images variable (certaines < k, pour tester le tirage avec remise)."""
    return {
        1: [_make_image(tmp_path / f"1_{i}.jpg", (255, 0, 0)) for i in range(5)],
        2: [_make_image(tmp_path / f"2_{i}.jpg", (0, 255, 0)) for i in range(5)],
        3: [_make_image(tmp_path / f"3_{i}.jpg", (0, 0, 255)) for i in range(2)],  # < k=4
        4: [_make_image(tmp_path / f"4_{i}.jpg", (255, 255, 0)) for i in range(5)],
    }


class TestPKImageDataset:
    def test_len_is_total_number_of_images(self, flat_index):
        ds = PKImageDataset(flat_index)
        assert len(ds) == 5 + 5 + 2 + 5

    def test_getitem_returns_image_and_pid(self, flat_index):
        ds = PKImageDataset(flat_index)
        image, pid = ds[0]
        assert isinstance(image, Image.Image)
        assert pid in flat_index

    def test_indices_by_pid_covers_every_item_once(self, flat_index):
        ds = PKImageDataset(flat_index)
        all_indices = sorted(i for indices in ds.indices_by_pid.values() for i in indices)
        assert all_indices == list(range(len(ds)))


class TestPKBatchSampler:
    def test_batch_size_is_p_times_k(self, flat_index):
        ds = PKImageDataset(flat_index)
        sampler = PKBatchSampler(ds, p=3, k=4, batches_per_epoch=5, seed=0)
        for batch in sampler:
            assert len(batch) == 12

    def test_yields_requested_number_of_batches(self, flat_index):
        ds = PKImageDataset(flat_index)
        sampler = PKBatchSampler(ds, p=2, k=2, batches_per_epoch=7, seed=0)
        assert len(list(sampler)) == 7
        assert len(sampler) == 7

    def test_each_batch_has_p_distinct_identities_with_k_images_each(self, flat_index):
        ds = PKImageDataset(flat_index)
        sampler = PKBatchSampler(ds, p=3, k=4, batches_per_epoch=5, seed=1)
        for batch in sampler:
            pids = [ds.items[i][1] for i in batch]
            counts = {pid: pids.count(pid) for pid in set(pids)}
            assert len(counts) == 3
            assert all(c == 4 for c in counts.values())

    def test_identity_with_fewer_than_k_images_sampled_with_replacement(self, flat_index):
        # identité 3 n'a que 2 images mais k=4 -> doit quand même produire 4 indices
        ds = PKImageDataset(flat_index)
        sampler = PKBatchSampler(ds, p=4, k=4, batches_per_epoch=3, seed=0)
        for batch in sampler:
            pids = [ds.items[i][1] for i in batch]
            assert pids.count(3) == 4

    def test_same_seed_and_epoch_reproducible(self, flat_index):
        ds = PKImageDataset(flat_index)
        a = list(PKBatchSampler(ds, p=2, k=2, batches_per_epoch=5, seed=42))
        b = list(PKBatchSampler(ds, p=2, k=2, batches_per_epoch=5, seed=42))
        assert a == b

    def test_set_epoch_changes_batches(self, flat_index):
        ds = PKImageDataset(flat_index)
        sampler = PKBatchSampler(ds, p=2, k=2, batches_per_epoch=5, seed=42)
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        assert epoch0 != epoch1

    def test_rejects_k_below_two(self, flat_index):
        with pytest.raises(ValueError):
            PKBatchSampler(PKImageDataset(flat_index), p=2, k=1, batches_per_epoch=1)

    def test_rejects_p_below_two(self, flat_index):
        with pytest.raises(ValueError):
            PKBatchSampler(PKImageDataset(flat_index), p=1, k=2, batches_per_epoch=1)

    def test_rejects_p_larger_than_available_identities(self, flat_index):
        with pytest.raises(ValueError):
            PKBatchSampler(PKImageDataset(flat_index), p=10, k=2, batches_per_epoch=1)

    def test_works_as_dataloader_batch_sampler(self, flat_index):
        ds = PKImageDataset(flat_index, transform=lambda im: torch.tensor(1))
        sampler = PKBatchSampler(ds, p=2, k=2, batches_per_epoch=3, seed=0)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
        batches = list(loader)
        assert len(batches) == 3
        for images, pids in batches:
            assert images.shape[0] == 4
            assert pids.shape[0] == 4


class TestMineBatchHard:
    def test_matches_hand_computed_example_zero_loss(self):
        # 2 identités x 2 images, positifs et négatifs largement separés -> marge respectée.
        embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        pids = torch.tensor([1, 1, 2, 2])

        hardest_positive, hardest_negative = mine_batch_hard(embeddings, pids)
        assert hardest_positive.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert hardest_negative.tolist() == pytest.approx([5.0, 4.0, 4.0, 5.0])

        loss = batch_hard_triplet_loss(embeddings, pids, margin=0.3)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_matches_hand_computed_example_nonzero_loss(self):
        # Positifs et négatifs proches -> la marge n'est pas respectée, loss > 0.
        embeddings = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.6, 0.0], [1.1, 0.0]])
        pids = torch.tensor([1, 1, 2, 2])

        hardest_positive, hardest_negative = mine_batch_hard(embeddings, pids)
        assert hardest_positive.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
        assert hardest_negative.tolist() == pytest.approx([0.6, 0.1, 0.1, 0.6], abs=1e-5)

        loss = batch_hard_triplet_loss(embeddings, pids, margin=0.3)
        assert loss.item() == pytest.approx(0.45, abs=1e-5)

    def test_loss_is_never_negative(self):
        torch.manual_seed(0)
        embeddings = torch.nn.functional.normalize(torch.randn(12, 16), dim=1)
        pids = torch.tensor([i // 3 for i in range(12)])  # 4 identités x 3 images
        loss = batch_hard_triplet_loss(embeddings, pids, margin=0.3)
        assert loss.item() >= 0.0
