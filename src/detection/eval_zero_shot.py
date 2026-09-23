"""Evalue le modele COCO pre-entraine sans fine-tuning

Usage (Kaggle/Colab) :
    python -m src.detection.eval_zero_shot --pool-dir <chemin_du_pool> --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.detection.kfold import list_base_sequences, prepare_fold, read_manifest


def eval_fold_zero_shot(model: Any, dataset_yaml: Path, imgsz: int = 640) -> dict[str, float]:
    """Evalue `model` (deja charge, non fine-tune) sur un fold, sans entrainement.

    `classes=[0]` restreint les predictions a la classe "person" de COCO,
    qui correspond a notre classe unique "pedestrian".
    """
    metrics = model.val(data=str(dataset_yaml), classes=[0], imgsz=imgsz, verbose=False)
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def aggregate(all_metrics: list[dict[str, float]]) -> tuple[float, float]:
    """Retourne (moyenne, ecart-type population) du mAP50-95 sur tous les folds."""
    values = [m["mAP50-95"] for m in all_metrics]
    mean_v = sum(values) / len(values)
    variance = sum((x - mean_v) ** 2 for x in values) / len(values)
    return mean_v, variance**0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-root", type=Path, default=None, help="Reutilise les folds du k-fold si deja generes"
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--summary-csv", type=Path, default=Path("zero_shot_summary.csv"))
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    manifest_rows = read_manifest(args.pool_dir / "manifest.csv")
    base_names = list_base_sequences(manifest_rows)
    fold_root = args.fold_root or (args.pool_dir.parent / "kfold_runs")

    model = YOLO(args.model)

    all_metrics = []
    for held_out in base_names:
        fold_dir = fold_root / f"fold_{held_out}"
        dataset_yaml = fold_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            print(f"Fold {held_out} absent de {fold_root}, preparation...")
            prepare_fold(args.pool_dir, manifest_rows, held_out, fold_dir)

        m = eval_fold_zero_shot(model, dataset_yaml, args.imgsz)
        m["held_out"] = held_out
        all_metrics.append(m)
        print(f"{held_out}: mAP50-95 (zero-shot, sans fine-tuning) = {m['mAP50-95']:.3f}")

    mean_map, std_map = aggregate(all_metrics)
    print(f"\n{'=' * 50}")
    print(f"Zero-shot ({args.model}, {len(base_names)} folds, AUCUN fine-tuning) :")
    print(f"  mAP50-95 = {mean_map:.3f} +/- {std_map:.3f}")
    print(f"{'=' * 50}")
    print("\nComparer a la baseline fine-tunee (freeze=10, sans domain-augment) : 0.465 +/- 0.158")

    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["held_out", "precision", "recall", "mAP50", "mAP50-95"]
        )
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nDetail par fold ecrit dans {args.summary_csv}")


if __name__ == "__main__":
    main()
