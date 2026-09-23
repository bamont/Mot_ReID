"""Parsing et indéxation du dataset Market-1501 pour la ré-identification.

Convention de nommage Market-1501 (ex: ``0002_c1s1_000451_03.jpg``) :
    <pid>_c<camera>s<sequence>_<frame>_<box>.jpg
        pid     : identité de la personne (``-1`` = distracteur, ``0000`` = junk,
                   tous deux à exclure de l'entrainement)
        camera  : numéro de caméra (1 à 6)
        sequence: numéro de séquence vidéo pour cette caméra
        frame   : numéro de frame
        box     : index de la bbox détectée dans cette frame
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<camera>\d+)s(?P<sequence>\d+)_(?P<frame>\d+)_(?P<box>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

# IdentityIndex : pid -> camera -> liste des chemins d'images de cette identite/camera.
IdentityIndex = dict[int, dict[int, list[Path]]]


@dataclass(frozen=True)
class ParsedFilename:
    pid: int
    camera: int
    sequence: int
    frame: int
    box: int


def parse_filename(name: str) -> ParsedFilename:
    """Parse un nom de fichier Market-1501.

    Leve ``ValueError`` si le nom ne suit pas la convention attendue.
    """
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Nom de fichier Market-1501 invalide : {name!r}")
    return ParsedFilename(
        pid=int(match["pid"]),
        camera=int(match["camera"]),
        sequence=int(match["sequence"]),
        frame=int(match["frame"]),
        box=int(match["box"]),
    )


def is_valid_identity(pid: int) -> bool:
    """Exclut les distracteurs (-1) et le junk (0) : ce ne sont pas de vraies identités."""
    return pid > 0


def build_identity_index(image_dir: Path, valid_only: bool = True) -> IdentityIndex:
    """Scanne un dossier Market-1501 (ex: ``bounding_box_train``) et construit l'index.

    Les fichiers dont le nom ne suit pas la convention Market-1501 sont ignorés
    silencieusement.
    """
    index: IdentityIndex = {}
    for path in sorted(image_dir.glob("*.jpg")):
        try:
            parsed = parse_filename(path.name)
        except ValueError:
            continue
        if valid_only and not is_valid_identity(parsed.pid):
            continue
        index.setdefault(parsed.pid, {}).setdefault(parsed.camera, []).append(path)
    return index


def count_images(index: IdentityIndex) -> int:
    return sum(len(paths) for cameras in index.values() for paths in cameras.values())


def identities_with_min_images(index: IdentityIndex, min_images: int = 2) -> list[int]:
    """Identités ayant au moins ``min_images`` images (necessaire pour former un couple ancre/positif)."""
    return [
        pid
        for pid, cameras in index.items()
        if sum(len(paths) for paths in cameras.values()) >= min_images
    ]


def split_identities(
    pids: list[int], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Sépare les identités en train/val (split par IDENTITE, pas par image)."""
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio doit etre dans [0, 1), recu {val_ratio}")
    shuffled = sorted(pids)  # tri d'abord pour un ordre deterministe avant le shuffle
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def subindex(index: IdentityIndex, pids: list[int]) -> IdentityIndex:
    """Sous-ensemble de l'index restreint aux identités données."""
    wanted = set(pids)
    return {pid: cameras for pid, cameras in index.items() if pid in wanted}
