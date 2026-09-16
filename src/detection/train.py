"""Entraine YOLOv8 sur le dataset MOT17

Usage local (test rapide, CPU) :
    poetry run python -m src.detection.train --epochs 3 --device cpu

Usage sur Colab/Kaggle (GPU) :
    python -m src.detection.train --epochs 50 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import DATA_PROCESSED

DATASET_YAML = DATA_PROCESSED / "mot17_yolo" / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Checkpoint de depart (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device", default="cpu",
        help="'cpu', '0' pour le premier GPU, '0,1' pour plusieurs GPUs",
    )
    parser.add_argument(
        "--freeze", type=int, default=None,
        help=(
            "Nombre de couches a geler depuis le debut du backbone (ex: 10). "
        ),
    )
    parser.add_argument("--name", default="baseline", help="Nom du run (dossier dans runs/detect/)")
    parser.add_argument(
        "--data", type=Path, default=DATASET_YAML,
        help="Chemin vers dataset.yaml (par defaut: %(default)s)",
    )
    parser.add_argument("--lr0", type=float, default=0.001, help="Learning rate initial")
    parser.add_argument("--optimizer", default="AdamW", help="'auto' pour laisser Ultralytics choisir")

    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--erasing", type=float, default=0.0)
    
    parser.add_argument(
        "--domain-augment", action="store_true",
        help="Active l'augmentation a la volée nuit/flou (monkey-patch Albumentations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"dataset.yaml introuvable a {args.data}. "
            "launchez d'abord le script de preparation du dataset src/detection/build_dataset.py"
        )
        
    if args.domain_augment:
        from src.detection.onthefly_augment import patch_ultralytics_albumentations
 
        patched = patch_ultralytics_albumentations()
        if not patched:
            print("ATTENTION : --domain-augment demande mais le patch a echoue (ultralytics introuvable ?)")
        else:
            print("Pipeline Albumentations patche : MotionBlur + RandomGamma/RandomBrightnessContrast actifs")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        lr0=args.lr0,
        optimizer=args.optimizer,
        mosaic=args.mosaic,
        scale=args.scale,
        erasing=args.erasing,
        freeze=args.freeze,
    )


if __name__ == "__main__":
    main()
