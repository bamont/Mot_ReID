"""Batch-hard mining (Hermans et al. 2017, "In Defense of the Triplet Loss").
Solution batch-hard : construire des batches de P identités x K images, calculer les
embeddings de tout le batch en un seul forward pass, puis pour chaque ancre choisir,
parmi les images déjà présentes dans ce batch, le positif le plus eloigné (le plus dur)
et le négatif le plus proche (le plus dur). Le signal de gradient reste utile tant que le
modèle n'a pas encore appris a séparer les cas difficiles, même quand la loss moyenne sur
des triplets aléatoires serait déjà proche de 0.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler


class PKImageDataset(Dataset):
    """Dataset a plat : chaque item est (image, pid), pas de triplet.
    """

    def __init__(self, index: dict[int, list[Path]], transform=None) -> None:
        self.transform = transform
        self.items: list[tuple[Path, int]] = [
            (path, pid) for pid, paths in index.items() for path in paths
        ]
        self.indices_by_pid: dict[int, list[int]] = {}
        for i, (_, pid) in enumerate(self.items):
            self.indices_by_pid.setdefault(pid, []).append(i)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor | Image.Image, int]:
        path, pid = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, pid


class PKBatchSampler(Sampler[list[int]]):
    """Genere des batches de P identites x K images (taille de batch = P*K).
    """

    def __init__(
        self,
        dataset: PKImageDataset,
        p: int,
        k: int,
        batches_per_epoch: int,
        seed: int = 42,
    ) -> None:
        if k < 2:
            raise ValueError(f"k doit etre >= 2 (il faut un positif dans le batch), reçu {k}")
        if p < 2:
            raise ValueError(f"p doit etre >= 2 (il faut un négatif dans le batch), reçu {p}")
        n_ids = len(dataset.indices_by_pid)
        if n_ids < p:
            raise ValueError(f"p={p} identités demandées mais seulement {n_ids} disponibles.")

        self.dataset = dataset
        self.p = p
        self.k = k
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Même convention que TripletMarket1501Dataset.set_epoch : à appeler avant chaque
        époque pour faire varier les batches tirés tout en restant reproductible."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed * 1_000_003 + self._epoch * 97)
        pids = list(self.dataset.indices_by_pid)
        for _ in range(self.batches_per_epoch):
            chosen_pids = rng.sample(pids, self.p)
            batch: list[int] = []
            for pid in chosen_pids:
                candidates = self.dataset.indices_by_pid[pid]
                if len(candidates) >= self.k:
                    batch.extend(rng.sample(candidates, self.k))
                else:
                    batch.extend(rng.choices(candidates, k=self.k))
            yield batch


def mine_batch_hard(embeddings: Tensor, pids: Tensor) -> tuple[Tensor, Tensor]:
    """Pour chaque ancre du batch : distance au positif le plus dur (même id, le plus
    eloigne) et au négatif le plus dur (id different, le plus proche).
    """
    dist = torch.cdist(embeddings, embeddings, p=2)
    same_id = pids.unsqueeze(0) == pids.unsqueeze(1)
    self_mask = torch.eye(len(pids), dtype=torch.bool, device=embeddings.device)

    positive_mask = same_id & ~self_mask
    hardest_positive = (dist * positive_mask).max(dim=1).values

    max_dist = dist.max()
    hardest_negative = (dist + max_dist * same_id.float()).min(dim=1).values

    return hardest_positive, hardest_negative


def batch_hard_triplet_loss(embeddings: Tensor, pids: Tensor, margin: float = 0.3) -> Tensor:
    """Triplet loss (marge dure) appliquée aux paires (positif, négatif) les plus dures
    du batch."""
    hardest_positive, hardest_negative = mine_batch_hard(embeddings, pids)
    return torch.relu(hardest_positive - hardest_negative + margin).mean()
