"""Lance détection + tracking + évaluation sur les 7 videos MOT17, avec le
seuil de confiance optimisé sur MOT17-02 (conf=0.4).

Réutilise base_sequence_name/pick_variant de build_dataset.py plutôt que de
dupliquer la logique de sélection de variante (DPM/FRCNN/SDP).

Usage :
    python -m src.tracking.evaluate_all_sequences --device 0
    python -m src.tracking.evaluate_all_sequences --device cpu --conf 0.4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import MOT17_DIR
from src.detection.build_dataset import pick_variant
from src.detection.yolo_conversion import base_sequence_name
from src.tracking.evaluate import build_accumulator, load_gt
from src.tracking.track_video import run_tracking_with_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Seuil optimisé sur MOT17-02 (DECISIONS.md)"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--max-iou", type=float, default=0.5)
    parser.add_argument("--summary-csv", type=Path, default=Path("tracking_summary.csv"))
    return parser.parse_args()


def rows_to_pred_frames(rows) -> dict:
    """Convertit les lignes MOT (liste de tuples) en dict {frame: (ids, boxes)}
    attendu par build_accumulator -- meme format que load_gt/load_pred."""
    frames: dict[int, tuple[list[int], list[list[float]]]] = {}
    for frame, tid, left, top, width, height, _conf in rows:
        ids, boxes = frames.setdefault(frame, ([], []))
        ids.append(tid)
        boxes.append([left, top, width, height])
    return {f: (np.array(ids), np.array(boxes)) for f, (ids, boxes) in frames.items()}


def evaluate_sequence(model, sequence_dir: Path, args: argparse.Namespace) -> dict:
    """Lance tracking + évaluation sur une séquence, retourne les métriques."""
    import motmetrics as mm

    rows = run_tracking_with_model(model, sequence_dir, args.tracker, args.conf, args.device)
    pred_frames = rows_to_pred_frames(rows)
    gt_frames = load_gt(sequence_dir / "gt" / "gt.txt", args.min_visibility)

    acc = build_accumulator(gt_frames, pred_frames, args.max_iou)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"],
        name="tracker",
    )
    row = summary.iloc[0]
    return {
        "sequence": sequence_dir.name,
        "mota": float(row["mota"]),
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "fp": int(row["num_false_positives"]),
        "fn": int(row["num_misses"]),
        "n_tracks": len({r[1] for r in rows}),
    }


def main() -> None:
    args = parse_args()
    from ultralytics import RTDETR

    all_sequences = sorted(MOT17_DIR.glob("train/*"))
    base_names = sorted({base_sequence_name(s.name) for s in all_sequences})
    print(f"{len(base_names)} séquences à évaluer : {base_names}")

    model = RTDETR(args.model)
    all_results = []

    for base in base_names:
        seq_path = pick_variant(all_sequences, base)
        print(f"\n=== {base} ({seq_path.name}) ===")
        metrics = evaluate_sequence(model, seq_path, args)
        all_results.append(metrics)
        print(
            f"  MOTA={metrics['mota']:.1%}  IDF1={metrics['idf1']:.1%}  "
            f"IDs={metrics['id_switches']}  FP={metrics['fp']}  FN={metrics['fn']}  "
            f"identites={metrics['n_tracks']}"
        )

    mota_values = [r["mota"] for r in all_results]
    idf1_values = [r["idf1"] for r in all_results]
    mean_mota, std_mota = (
        sum(mota_values) / len(mota_values),
        (
            sum((x - sum(mota_values) / len(mota_values)) ** 2 for x in mota_values)
            / len(mota_values)
        )
        ** 0.5,
    )
    mean_idf1 = sum(idf1_values) / len(idf1_values)

    print(f"\n{'=' * 50}")
    print(f"Résultat final ({args.model}, conf={args.conf}, {len(base_names)} séquences) :")
    print(f"  MOTA = {mean_mota:.1%} +/- {std_mota:.1%}")
    print(f"  IDF1 moyen = {mean_idf1:.1%}")
    print(f"  min MOTA={min(mota_values):.1%}  max MOTA={max(mota_values):.1%}")
    print(f"{'=' * 50}")

    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sequence", "mota", "idf1", "id_switches", "fp", "fn", "n_tracks"]
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nDetail par séquence écrit dans {args.summary_csv}")


if __name__ == "__main__":
    main()
