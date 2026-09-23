"""Dataset PyTorch pour l'entrainement en triplet loss sur Market-1501.

Choix : echantillonnage "online" plutot qu'une liste de triplets figee.
    Une liste de triplets pre-generee une fois pour toutes limite l'entrainement a un
    nombre fixe de combinaisons ancre/positif/negatif : sur plusieurs dizaines d'epoques,
    le modele finit par revoir les memes triplets. Ici, chaque ``__getitem__`` retire un
    triplet frais via ``sample_triplet`` (src/reid/triplet_sampling.py). Pour rester
    reproductible malgre l'echantillonnage a la volee, chaque (epoch, index) est associe a
    un ``random.Random`` deterministe (via ``set_epoch``, meme convention que les samplers
    distribues de PyTorch) : deux runs avec le meme seed produisent exactement les memes
    triplets, epoque par epoque.

Ce module depend de torch/torchvision/Pillow (contrairement a market1501.py et
triplet_sampling.py qui restent en pur Python testable sans ces dependances lourdes).
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
# proprietes cryptographiques requises, juste eviter les collisions triviales entre epoques.
_EPOCH_STRIDE = 1_000_003
_IDX_STRIDE = 97


class TripletMarket1501Dataset(Dataset):
    """Dataset de taille fixe ``length`` qui tire un triplet (ancre, positif, negatif) par item.

    Args:
        index: identite -> camera -> liste de chemins (cf. build_identity_index).
        length: nombre de triplets par epoque virtuelle. Une valeur usuelle est
            ``n_identites * k`` (ex: k=4) pour voir chaque identite plusieurs fois par epoque
            sans pour autant enumerer toutes les combinaisons possibles.
        transform: transformation torchvision appliquee a chaque image PIL (resize, normalize,
            augmentation...). Si None, les images PIL sont retournees telles quelles.
        seed: graine de base pour la reproductibilite.
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
            raise ValueError(f"length doit etre strictement positif, recu {length}")
        self.index = index
        self.length = length
        self.transform = transform
        self.seed = seed
        self.prefer_cross_camera = prefer_cross_camera
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """A appeler au debut de chaque epoque pour faire varier les triplets tires.

        Meme convention que ``DistributedSampler.set_epoch`` : sans cet appel, le dataset
        reste utilisable (epoch=0 par defaut) mais retire toujours les memes triplets.
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
