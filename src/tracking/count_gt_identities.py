"""Compte le nombre d'identités réelles (piétons valides, class==1 conf==1)
dans gt.txt pour chaque séquence MOT17, à comparer avec la colonne "n_tracks"
de tracking_summary.csv (nombre d'IDs produits par le tracker) pour
quantifier la fragmentation d'identite : combien de "fausses" identités le
tracker crée en perdant des pistes qu'il devrait maintenir.

Usage :
    python -m src.tracking.count_gt_identities
"""

from __future__ import annotations

from src.config import MOT17_DIR
from src.detection.build_dataset import pick_variant
from src.detection.yolo_conversion import base_sequence_name
from src.tracking.evaluate import load_gt


def count_unique_identities(gt_frames: dict) -> int:
    """Compte les identités distinctes sur l'ensemble des frames.

    Fonction pure, testable avec un dict synthetique (meme format que
    retourne par load_gt : {frame: (ids_array, boxes_array)}).
    """
    all_ids: set[int] = set()
    for ids, _boxes in gt_frames.values():
        all_ids.update(int(i) for i in ids)
    return len(all_ids)


def main() -> None:
    all_sequences = sorted(MOT17_DIR.glob("train/*"))
    base_names = sorted({base_sequence_name(s.name) for s in all_sequences})

    print(f"{'Sequence':12} {'identités réelles':>18}")
    for base in base_names:
        seq_path = pick_variant(all_sequences, base)
        gt_frames = load_gt(seq_path / "gt" / "gt.txt", min_visibility=0.0)
        n_real = count_unique_identities(gt_frames)
        print(f"{base:12} {n_real:>18}")


if __name__ == "__main__":
    main()
