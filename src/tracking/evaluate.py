"""Evalue un fichier de tracking (format MOT) contre les annotations gt.txt,
via py-motmetrics (MOTA, IDF1, faux positifs, ID switches, fragmentations).

Usage :
    python -m src.tracking.evaluate \\
        --gt data/raw/MOT17/train/MOT17-02-FRCNN/gt/gt.txt \\
        --pred runs/track/MOT17-02.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

FrameData = dict[int, tuple[np.ndarray, np.ndarray]]  # frame -> (ids, boxes_ltwh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt", type=Path, required=True, help="gt.txt de la sequence MOT17")
    parser.add_argument("--pred", type=Path, required=True, help="Sortie de track_video.py")
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Defaut 0.0 (pas de filtre) -- voir docstring du module : filtrer la visibilite "
        "ici transformerait a tort des detections de personnes occluses en faux positifs.",
    )
    parser.add_argument(
        "--max-iou", type=float, default=0.5, help="IoU minimal pour apparier gt et pred"
    )
    return parser.parse_args()


def load_gt(gt_path: Path, min_visibility: float) -> FrameData:
    """Charge gt.txt filtre : class==1 (pieton), conf==1, visibility>=seuil."""
    frames: dict[int, tuple[list[int], list[list[float]]]] = {}
    with gt_path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, obj_id, left, top, width, height, conf, cls, vis = row[:9]
            frame_i, obj_id_i = int(frame), int(obj_id)
            cls_i, conf_i, vis_f = int(float(cls)), int(float(conf)), float(vis)

            if cls_i != 1 or conf_i != 1 or vis_f < min_visibility:
                continue

            box = [float(left), float(top), float(width), float(height)]
            ids, boxes = frames.setdefault(frame_i, ([], []))
            ids.append(obj_id_i)
            boxes.append(box)

    return {
        f: (np.array(ids), np.array(boxes) if boxes else np.empty((0, 4)))
        for f, (ids, boxes) in frames.items()
    }


def load_pred(pred_path: Path) -> FrameData:
    """Charge une sortie de tracking"""
    frames: dict[int, tuple[list[int], list[list[float]]]] = {}
    with pred_path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, obj_id, left, top, width, height = row[:6]
            frame_i, obj_id_i = int(frame), int(obj_id)
            box = [float(left), float(top), float(width), float(height)]
            ids, boxes = frames.setdefault(frame_i, ([], []))
            ids.append(obj_id_i)
            boxes.append(box)

    return {
        f: (np.array(ids), np.array(boxes) if boxes else np.empty((0, 4)))
        for f, (ids, boxes) in frames.items()
    }


def build_accumulator(gt_frames: FrameData, pred_frames: FrameData, max_iou: float):
    """Construit l'accumulateur py-motmetrics frame par frame."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_frames) | set(pred_frames))

    for frame in all_frames:
        gt_ids, gt_boxes = gt_frames.get(frame, (np.array([]), np.empty((0, 4))))
        pred_ids, pred_boxes = pred_frames.get(frame, (np.array([]), np.empty((0, 4))))
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou)
        acc.update(gt_ids, pred_ids, distances)

    return acc


def compute_summary(acc) -> str:
    """Calcule et formate MOTA/IDF1/etc. -- format lisible pour le rapport."""
    import motmetrics as mm

    mh = mm.metrics.create()
    metrics = [
        "mota",
        "idf1",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_fragmentations",
        "num_objects",
    ]
    summary = mh.compute(acc, metrics=metrics, name="tracker")
    return mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )


def main() -> None:
    args = parse_args()
    gt_frames = load_gt(args.gt, args.min_visibility)
    pred_frames = load_pred(args.pred)

    if not gt_frames:
        raise ValueError(f"Aucune annotation valide trouvee dans {args.gt} (verifie le filtrage)")
    if not pred_frames:
        raise ValueError(f"Aucune detection trouvee dans {args.pred}")

    acc = build_accumulator(gt_frames, pred_frames, args.max_iou)
    print(compute_summary(acc))


if __name__ == "__main__":
    main()
