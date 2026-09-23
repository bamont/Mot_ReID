"""Echantillonnage de triplets (ancre, positif, négatif) pour l'entrainement en triplet loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from src.reid.market1501 import IdentityIndex, identities_with_min_images


@dataclass(frozen=True)
class Triplet:
    anchor: Path
    positive: Path
    negative: Path
    anchor_id: int
    negative_id: int


def sample_triplet(
    rng: random.Random, index: IdentityIndex, prefer_cross_camera: bool = True
) -> Triplet:
    """Tire un triplet aléatoire à partir de l'index identité -> caméra -> images.

    Leve ``ValueError`` si l'index contient moins de 2 identités exploitables (il faut
    au moins une identité avec >=2 images pour l'ancre/positif, et une autre pour le négatif).
    """
    eligible = identities_with_min_images(index, min_images=2)
    if not eligible:
        raise ValueError("Aucune identité avec au moins 2 images dans l'index.")
    if len(index) < 2:
        raise ValueError("Il faut au moins 2 identités distinctes pour tirer un négatif.")

    anchor_id = rng.choice(eligible)
    anchor_cameras = index[anchor_id]
    anchor_camera = rng.choice(list(anchor_cameras))
    anchor = rng.choice(anchor_cameras[anchor_camera])

    positive = _sample_positive(rng, anchor_cameras, anchor_camera, anchor, prefer_cross_camera)

    negative_id = rng.choice([pid for pid in index if pid != anchor_id])
    negative_cameras = index[negative_id]
    negative_camera = rng.choice(list(negative_cameras))
    negative = rng.choice(negative_cameras[negative_camera])

    return Triplet(
        anchor=anchor,
        positive=positive,
        negative=negative,
        anchor_id=anchor_id,
        negative_id=negative_id,
    )


def _sample_positive(
    rng: random.Random,
    cameras: dict[int, list[Path]],
    anchor_camera: int,
    anchor: Path,
    prefer_cross_camera: bool,
) -> Path:
    other_cameras = [cam for cam in cameras if cam != anchor_camera]

    if prefer_cross_camera and other_cameras:
        chosen_camera = rng.choice(other_cameras)
        return rng.choice(cameras[chosen_camera])

    # Fallback même-caméra : une autre image que l'ancre elle-meme.
    same_camera_candidates = [p for p in cameras[anchor_camera] if p != anchor]
    if same_camera_candidates:
        return rng.choice(same_camera_candidates)

    if other_cameras:
        chosen_camera = rng.choice(other_cameras)
        return rng.choice(cameras[chosen_camera])
    raise ValueError("Impossible de trouver un positif différent de l'ancre.")


def generate_triplets(
    index: IdentityIndex,
    n_triplets: int,
    seed: int = 42,
    prefer_cross_camera: bool = True,
) -> list[Triplet]:
    """Génère une liste FIXE de n_triplets triplets, de facon reproductible (meme seed
    -> memes triplets). Utile pour exporter un échantillon inspectable (sample_triplets.csv)
    ou pour un mode d'entrainement offline simple.
    """
    if n_triplets < 0:
        raise ValueError(f"n_triplets doit être positif, reçu {n_triplets}")
    rng = random.Random(seed)
    return [sample_triplet(rng, index, prefer_cross_camera) for _ in range(n_triplets)]
