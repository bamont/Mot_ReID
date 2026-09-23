"""Teste plusieurs seuils de confiance sur une séquence, sans recharger le
modele a chaque fois (le chargement RT-DETR est le poste de cout fixe,
inutile de le payer 6 fois pour tester 6 seuils).

Usage :
    python -m src.tracking.sweep_confidence \\
        --sequence data/raw/MOT17/train/MOT17-02-FRCNN \\
        --conf-values 0.3 0.35 0.4 0.45 0.5 0.55 \\
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tracking.evaluate import build_accumulator, load_gt
from src.tracking.track_video import run_tracking_with_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sequence", type=Path, required=True, help="Dossier MOT17 contenant img1/ et gt/gt.txt"
    )
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conf-values",
        type=float,
        nargs="+",
        default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        help="Liste des seuils à tester",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Voir evaluate.py : 0.0 par defaut, ne pas filtrer la visibilité du gt pour le tracking",
    )
    parser.add_argument("--max-iou", type=float, default=0.5)
    return parser.parse_args()


def sweep(
    model,
    sequence_dir: Path,
    tracker: str,
    device: str,
    conf_values: list[float],
    gt_frames,
    max_iou: float,
) -> list[dict]:
    """Lance le tracking + évaluation pour chaque seuil, retourne les resultats
    sous forme de liste de dicts (facilement testable/affichable/triable)."""
    import motmetrics as mm

    results = []
    for conf in conf_values:
        rows = run_tracking_with_model(model, sequence_dir, tracker, conf, device)

        pred_frames: dict[int, tuple] = {}
        for frame, tid, left, top, width, height, _conf in rows:
            import numpy as np

            ids, boxes = pred_frames.setdefault(frame, ([], []))
            ids.append(tid)
            boxes.append([left, top, width, height])
        pred_frames = {
            f: (np.array(ids), np.array(boxes)) for f, (ids, boxes) in pred_frames.items()
        }

        acc = build_accumulator(gt_frames, pred_frames, max_iou)
        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"],
            name="tracker",
        )
        row = summary.iloc[0]
        results.append(
            {
                "conf": conf,
                "mota": row["mota"],
                "idf1": row["idf1"],
                "id_switches": int(row["num_switches"]),
                "fp": int(row["num_false_positives"]),
                "fn": int(row["num_misses"]),
                "n_tracks": len({r[1] for r in rows}),
            }
        )
        print(
            f"conf={conf:.2f}  MOTA={row['mota']:.1%}  IDF1={row['idf1']:.1%}  "
            f"IDs={int(row['num_switches'])}  FP={int(row['num_false_positives'])}  "
            f"FN={int(row['num_misses'])}"
        )

    return results


def print_best(results: list[dict]) -> None:
    best = max(results, key=lambda r: r["mota"])
    print(
        f"\nMeilleur seuil : conf={best['conf']:.2f} (MOTA={best['mota']:.1%}, IDF1={best['idf1']:.1%})"
    )


def main() -> None:
    args = parse_args()
    from ultralytics import RTDETR

    model = RTDETR(args.model)
    gt_frames = load_gt(args.sequence / "gt" / "gt.txt", args.min_visibility)

    results = sweep(
        model, args.sequence, args.tracker, args.device, args.conf_values, gt_frames, args.max_iou
    )
    print_best(results)


if __name__ == "__main__":
    main()
