"""Construit le split train/val et un échantillon de triplets pour la ré-ID Market-1501.

Usage:
    poetry run python -m src.reid.build_dataset
    poetry run python -m src.reid.build_dataset --val-ratio 0.1 --n-sample-triplets 20000 --seed 42

Produit :
    data/processed/market1501_triplets/
        train_identities.json   # {pid: {camera: [chemins...]}} -- pool d'entrainement
        val_identities.json     # identites tenues a l'ecart (jamais vues en train),
                                 # a utiliser pour monitorer un mAP de retrieval pendant l'entrainement
        sample_triplets.csv     # n triplets fixes (anchor,positive,negative,anchor_id,negative_id),
                                 # pour inspection manuelle / illustration dans le rapport
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import MARKET1501_TRAIN_DIR, REID_TRIPLETS_DIR
from src.reid.market1501 import (
    build_identity_index,
    count_images,
    split_identities,
    subindex,
)
from src.reid.triplet_sampling import generate_triplets

TRAIN_IMAGES_DIR = MARKET1501_TRAIN_DIR
OUTPUT_DIR = REID_TRIPLETS_DIR


def _index_to_json(index) -> dict:
    return {
        str(pid): {str(camera): [str(p) for p in paths] for camera, paths in cameras.items()}
        for pid, cameras in index.items()
    }


def write_identity_index(index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_index_to_json(index), indent=2, ensure_ascii=False))


def write_sample_triplets(triplets, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor", "positive", "negative", "anchor_id", "negative_id"])
        for t in triplets:
            writer.writerow([t.anchor, t.positive, t.negative, t.anchor_id, t.negative_id])


def build(
    val_ratio: float,
    n_sample_triplets: int,
    seed: int,
    cross_camera: bool,
) -> None:
    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"{TRAIN_IMAGES_DIR} introuvable. "
            "Voir scripts/download_data.sh pour recuperer Market-1501."
        )

    print(f">> Indexation de {TRAIN_IMAGES_DIR}...")
    full_index = build_identity_index(TRAIN_IMAGES_DIR)
    n_ids = len(full_index)
    n_imgs = count_images(full_index)
    print(f"   {n_ids} identites valides, {n_imgs} images (junk/distracteurs exclus)")

    train_ids, val_ids = split_identities(list(full_index), val_ratio=val_ratio, seed=seed)
    train_index = subindex(full_index, train_ids)
    val_index = subindex(full_index, val_ids)
    print(f"   Split identites : {len(train_ids)} train / {len(val_ids)} val (ratio={val_ratio})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_identity_index(train_index, OUTPUT_DIR / "train_identities.json")
    write_identity_index(val_index, OUTPUT_DIR / "val_identities.json")
    print(f"   Ecrit : {OUTPUT_DIR / 'train_identities.json'}")
    print(f"   Ecrit : {OUTPUT_DIR / 'val_identities.json'}")

    if n_sample_triplets > 0:
        print(f">> Echantillonnage de {n_sample_triplets} triplets (seed={seed})...")
        triplets = generate_triplets(
            train_index, n_sample_triplets, seed=seed, prefer_cross_camera=cross_camera
        )
        write_sample_triplets(triplets, OUTPUT_DIR / "sample_triplets.csv")
        print(f"   Ecrit : {OUTPUT_DIR / 'sample_triplets.csv'}")

    print(f"\nTermine. Sortie dans {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-sample-triplets", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-cross-camera",
        action="store_false",
        dest="cross_camera",
        help="Désactive la préférence pour un positif pris sur une autre caméra.",
    )
    args = parser.parse_args()
    build(
        val_ratio=args.val_ratio,
        n_sample_triplets=args.n_sample_triplets,
        seed=args.seed,
        cross_camera=args.cross_camera,
    )


if __name__ == "__main__":
    main()
