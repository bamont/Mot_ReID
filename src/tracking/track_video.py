"""Lance la détection + tracking (RT-DETR + ByteTrack natif d'Ultralytics) sur
une séquence MOT17, produit un fichier de sortie au format MOT standard.

Usage :
    python -m src.tracking.track_video \\
        --sequence data/raw/MOT17/train/MOT17-02-FRCNN \\
        --output runs/track/MOT17-02.txt

Le fichier de sortie (format MOT : frame,id,bb_left,bb_top,w,h,conf,-1,-1,-1)
peut ensuite être comparé à gt/gt.txt de la même séquence pour calculer
MOTA/IDF1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MotRow = tuple[int, int, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        required=True,
        help="Dossier MOT17 contenant img1/ (ex: .../MOT17-02-FRCNN)",
    )
    parser.add_argument("--model", default="rtdetr-l.pt", help="Checkpoint RT-DETR")
    parser.add_argument(
        "--tracker", default="bytetrack.yaml", help="'bytetrack.yaml' ou 'botsort.yaml'"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Seuil de confiance des détections"
    )
    parser.add_argument("--device", default="0", help="'cpu' ou '0' pour le premier GPU")
    parser.add_argument(
        "--output", type=Path, required=True, help="Fichier de sortie au format MOT"
    )
    return parser.parse_args()


def boxes_to_mot_rows(
    frame_idx: int,
    boxes_xywh: np.ndarray,
    track_ids: np.ndarray,
    confs: np.ndarray,
) -> list[MotRow]:
    """Convertit les boites d'une frame (format Ultralytics xywh=centre/largeur/hauteur)
    en lignes MOT (frame, id, bb_left, bb_top, width, height, conf).

    Fonction pure, testable sans modèle ni vidéo réelle.
    """
    rows: list[MotRow] = []
    for (cx, cy, w, h), tid, conf in zip(boxes_xywh, track_ids, confs, strict=False):
        bb_left = float(cx - w / 2)
        bb_top = float(cy - h / 2)
        rows.append((frame_idx, int(tid), bb_left, bb_top, float(w), float(h), float(conf)))
    return rows


def write_mot_file(rows: list[MotRow], output_path: Path) -> None:
    """Ecrit les lignes au format MOT standard (une détection trackée par ligne).

    Colonnes -1,-1,-1 en fin de ligne : champs x/y/z 3D non utilises, requis
    par le format mais ignores par les outils d'evaluation 2D (py-motmetrics).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for frame, tid, left, top, width, height, conf in rows:
            f.write(
                f"{frame},{tid},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.3f},-1,-1,-1\n"
            )


def run_tracking_with_model(
    model, sequence_dir: Path, tracker: str, conf: float, device: str
) -> list[MotRow]:
    """Même logique que run_tracking, mais avec un modèle déjà chargé en mémoire.

    Extrait séparément pour permettre de tester plusieurs seuils de confiance
    sans recharger le modèle à chaque fois (voir sweep_confidence.py).
    """
    img_dir = sequence_dir / "img1"
    if not any(img_dir.glob("*.jpg")):
        raise FileNotFoundError(f"Aucune image .jpg trouvee dans {img_dir}")

    all_rows: list[MotRow] = []

    results = model.track(
        source=str(img_dir),
        tracker=tracker,
        classes=[0],  # person (COCO) == notre classe "pedestrian"
        conf=conf,
        device=device,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(results, start=1):
        if result.boxes is None or result.boxes.id is None:
            continue  # aucune piste active sur cette frame
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        all_rows.extend(boxes_to_mot_rows(frame_idx, boxes_xywh, track_ids, confs))

    return all_rows


def run_tracking(
    sequence_dir: Path, model_name: str, tracker: str, conf: float, device: str
) -> list[MotRow]:
    """Lance RT-DETR + tracker sur toutes les frames de la sequence, dans l'ordre.

    Nécessite un vrai modèle chargé -- pas testé unitairement.
    """
    from ultralytics import RTDETR

    model = RTDETR(model_name)
    return run_tracking_with_model(model, sequence_dir, tracker, conf, device)


def main() -> None:
    args = parse_args()
    rows = run_tracking(args.sequence, args.model, args.tracker, args.conf, args.device)
    write_mot_file(rows, args.output)

    n_tracks = len({row[1] for row in rows})
    n_frames = len({row[0] for row in rows})
    print(f"{len(rows)} détections trackées sur {n_frames} frames, {n_tracks} identités uniques")
    print(f"Ecrit dans {args.output}")


if __name__ == "__main__":
    main()
