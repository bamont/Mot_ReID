"""Conversion des annotations MOT17 (gt.txt) vers le format YOLO.

Règles issues de l'EDA (voir notebooks/01_eda.ipynb) :
- Ne garder que class == 1 (pieton) -- MOT17 annote aussi cyclistes, distracteurs, etc.
- Ne garder que conf == 1 -- conf == 0 signifie "a ignorer" selon le devkit MOT17.
- Clipper les bbox aux dimensions reelles de l'image -- certaines boxes sortent du cadre.
- Splitter train/val par VIDEO DE BASE (ex: "MOT17-02"), jamais par variante
  (MOT17-02-DPM/FRCNN/SDP sont la meme video avec les memes annotations gt --
  les mélanger entre train et val serait une fuite de donnees).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

GT_COLUMNS = [
    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class", "visibility",
]
PEDESTRIAN_CLASS = 1


@dataclass(frozen=True)
class YoloBox:
    """Une annotation au format YOLO : classe + centre/largeur/hauteur normalises [0, 1]."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def base_sequence_name(sequence_dir_name: str) -> str:
    """Extrait le nom de video de base a partir du nom de dossier de sequence.

    >>> base_sequence_name("MOT17-02-FRCNN")
    'MOT17-02'
    >>> base_sequence_name("MOT17-11-DPM")
    'MOT17-11'
    """
    match = re.match(r"(MOT17-\d+)", sequence_dir_name)
    if not match:
        raise ValueError(f"Nom de sequence inattendu, ne matche pas 'MOT17-XX...': {sequence_dir_name!r}")
    return match.group(1)


def parse_gt_line(line: str) -> dict:
    """Parse une ligne brute de gt.txt en dict type."""
    parts = line.strip().split(",")
    if len(parts) < len(GT_COLUMNS):
        raise ValueError(f"Ligne gt.txt malformee (attendu {len(GT_COLUMNS)} champs, recu {len(parts)}): {line!r}")
    values = [float(p) for p in parts[: len(GT_COLUMNS)]]
    return dict(zip(GT_COLUMNS, values))


def should_keep_annotation(ann: dict) -> bool:
    """Applique les filtres class==1 et conf==1 decides pendant l'EDA."""
    return int(ann["class"]) == PEDESTRIAN_CLASS and int(ann["conf"]) == 1


def clip_bbox(
    bb_left: float, bb_top: float, bb_width: float, bb_height: float,
    img_width: int, img_height: int,
) -> tuple[float, float, float, float]:
    """Clippe une bbox (left, top, width, height) aux dimensions de l'image.

    Retourne (left, top, width, height) apres clipping. Une bbox entierement
    hors-cadre (largeur ou hauteur clippee a 0) doit etre ecartee par l'appelant.
    """
    x1 = max(0.0, bb_left)
    y1 = max(0.0, bb_top)
    x2 = min(float(img_width), bb_left + bb_width)
    y2 = min(float(img_height), bb_top + bb_height)
    new_width = max(0.0, x2 - x1)
    new_height = max(0.0, y2 - y1)
    return x1, y1, new_width, new_height


def mot_bbox_to_yolo(
    bb_left: float, bb_top: float, bb_width: float, bb_height: float,
    img_width: int, img_height: int,
) -> YoloBox:
    """Convertit une bbox MOT (left, top, width, height en pixels) en YoloBox normalisee.

    Applique le clipping avant conversion. Leve ValueError si la bbox est
    entierement hors-cadre apres clipping (largeur ou hauteur nulle).
    """
    x1, y1, w, h = clip_bbox(bb_left, bb_top, bb_width, bb_height, img_width, img_height)
    if w <= 0 or h <= 0:
        raise ValueError("Bounding box entierement hors-cadre apres clipping")

    x_center = (x1 + w / 2) / img_width
    y_center = (y1 + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    return YoloBox(class_id=0, x_center=x_center, y_center=y_center, width=norm_width, height=norm_height)


def convert_frame_annotations(
    annotations: list[dict], img_width: int, img_height: int,
) -> list[YoloBox]:
    """Filtre et convertit toutes les annotations d'une frame en boxes YOLO.

    Les annotations qui ne passent pas should_keep_annotation, ou qui sont
    entierement hors-cadre apres clipping, sont silencieusement ecartees.
    """
    boxes = []
    for ann in annotations:
        if not should_keep_annotation(ann):
            continue
        try:
            boxes.append(
                mot_bbox_to_yolo(ann["bb_left"], ann["bb_top"], ann["bb_width"], ann["bb_height"], img_width, img_height)
            )
        except ValueError:
            continue
    return boxes


def split_base_sequences(base_names: list[str], val_fraction: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    """Split train/val PAR VIDEO DE BASE, jamais par variante DPM/FRCNN/SDP.

    A appeler avec des noms deja deduppliques (ex: ["MOT17-02", "MOT17-04", ...]),
    pas avec les noms de dossiers bruts qui incluent le suffixe de detecteur.
    """
    import random

    names = sorted(set(base_names))
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, round(len(names) * val_fraction))
    val_names = sorted(names[:n_val])
    train_names = sorted(names[n_val:])
    return train_names, val_names


def write_yolo_label_file(boxes: list[YoloBox], output_path: Path) -> None:
    """Ecrit un fichier .txt au format YOLO (une ligne par box, vide si aucune box)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for box in boxes:
            f.write(box.to_line() + "\n")
