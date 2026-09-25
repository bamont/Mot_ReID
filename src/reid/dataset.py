"""Dataset PyTorch pour l'entrainement en triplet loss sur Market-1501.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.reid.market1501 import IdentityIndex
from src.reid.triplet_sampling import sample_triplet

# Grand nombre premier arbitraire pour bien disperser (seed, epoch, idx) -> pas de
# propriétés cryptographiques requises, juste éviter les collisions triviales entre époques.
_EPOCH_STRIDE = 1_000_003
_IDX_STRIDE = 97


class TripletMarket1501Dataset(Dataset):
    """Dataset de taille fixe ``length`` qui tire un triplet (ancre, positif, négatif) par item.
    Args:
        index: identite -> camera -> liste de chemins (cf. build_identity_index).
        length: nombre de triplets par époque virtuelle. Une valeur usuelle est
            ``n_identites * k`` (ex: k=4) pour voir chaque identite plusieurs fois par époque
            sans pour autant énumerer toutes les combinaisons possibles.
        transform: transformation torchvision appliquee a chaque image PIL (resize, normalize,
            augmentation...). Si None, les images PIL sont retournees telles quelles.
        seed: graine de base pour la reproductibilité.
        prefer_cross_camera: cf. src/reid/triplet_sampling.py.
    """

    def __init__(
        self,
        index: IdentityIndex,
        length: int,
        transform: Callable[[Image.Image], Tensor] | None = None,
        seed: int = 42,
        prefer_cross_camera: bool = True,
    ) -> None:
        if length <= 0:
            raise ValueError(f"length doit être strictement positif, reçu {length}")
        self.index = index
        self.length = length
        self.transform = transform
        self.seed = seed
        self.prefer_cross_camera = prefer_cross_camera
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """A appeler au début de chaque époque pour faire varier les triplets tirés.
        même convention que ``DistributedSampler.set_epoch`` : sans cet appel, le dataset
        reste utilisable (epoch=0 par défaut) mais retire toujours les mêmes triplets.
        """
        self._epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _rng_for(self, idx: int) -> random.Random:
        seed = self.seed * _EPOCH_STRIDE + self._epoch * _IDX_STRIDE + idx
        return random.Random(seed)

    def _load(self, path: Path) -> Tensor | Image.Image:
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform is not None else image

    def __getitem__(self, idx: int):
        rng = self._rng_for(idx)
        triplet = sample_triplet(rng, self.index, self.prefer_cross_camera)
        return {
            "anchor": self._load(triplet.anchor),
            "positive": self._load(triplet.positive),
            "negative": self._load(triplet.negative),
            "anchor_id": triplet.anchor_id,
            "negative_id": triplet.negative_id,
        }


class ImageListDataset(Dataset):
    """Dataset simple : une liste de chemins d'images, sans notion d'ancre/positif/négatif.
    Utilise a l'évaluation pour extraire les embeddings de query/ et bounding_box_test/,
    ou chaque image est encodée indépendamment (contrairement a l'entrainement, il n'y a
    pas de triplet a construire ici).
    """

    def __init__(
        self,
        paths: list[Path],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor | Image.Image:
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image) if self.transform is not None else image
