"""Parsing et indexation du dataset Market-1501 pour la ré-identification.

Convention de nommage Market-1501 (ex: ``0002_c1s1_000451_03.jpg``) :
    <pid>_c<camera>s<sequence>_<frame>_<box>.jpg
        pid     : identite de la personne (``-1`` = distracteur, ``0000`` = junk,
                   tous deux a exclure de l'entrainement)
        camera  : numero de camera (1 a 6)
        sequence: numero de sequence video pour cette camera
        frame   : numero de frame
        box     : index de la bbox detectee dans cette frame
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<camera>\d+)s(?P<sequence>\d+)_(?P<frame>\d+)_(?P<box>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

# IdentityIndex : pid -> caméra -> liste des chemins d'images de cette identité/caméra.
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
    """Identités ayant au moins ``min_images`` images (nécessaire pour former un couple ancre/positif)."""
    return [
        pid
        for pid, cameras in index.items()
        if sum(len(paths) for paths in cameras.values()) >= min_images
    ]


def split_identities(
    pids: list[int], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Separe les identités en train/val (split par IDENTITÉ, pas par image). 
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio doit etre dans [0, 1), recu {val_ratio}")
    shuffled = sorted(pids)  # tri d'abord pour un ordre déterministe avant le shuffle
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def subindex(index: IdentityIndex, pids: list[int]) -> IdentityIndex:
    """Sous-ensemble de l'index restreint aux identités données."""
    wanted = set(pids)
    return {pid: cameras for pid, cameras in index.items() if pid in wanted}


def flatten_identity_index(index: IdentityIndex) -> dict[int, list[Path]]:
    """Aplati identité->caméra->images en identité->images (sans distinction de caméra).
    """
    return {
        pid: [path for paths in cameras.values() for path in paths]
        for pid, cameras in index.items()
    }


def load_identity_index_json(path: Path) -> IdentityIndex:
    """Recharge un index ecrit par build_dataset.py (train_identities.json / val_identities.json).
    """
    raw = json.loads(path.read_text())
    return {
        int(pid): {int(camera): [Path(p) for p in paths] for camera, paths in cameras.items()}
        for pid, cameras in raw.items()
    }


@dataclass(frozen=True)
class ParsedImage:
    path: Path
    pid: int
    camera: int


def list_images(image_dir: Path, valid_only: bool = True) -> list[ParsedImage]:
    """Version "a plat" de build_identity_index : une liste (path, pid, camera), sans
    regroupement par identite.
    """
    images = []
    for path in sorted(image_dir.glob("*.jpg")):
        try:
            parsed = parse_filename(path.name)
        except ValueError:
            continue
        if valid_only and not is_valid_identity(parsed.pid):
            continue
        images.append(ParsedImage(path=path, pid=parsed.pid, camera=parsed.camera))
    return images
