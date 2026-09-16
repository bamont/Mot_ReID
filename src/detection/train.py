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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 
    # --- Modele et duree ---
    parser.add_argument("--model", default="yolo26n.pt", help="Checkpoint de depart")
    parser.add_argument("--epochs", type=int, default=20, help="Le pic etait a l'epoch 1-6 au 1er benchmark")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping : epochs sans amelioration avant arret")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="'cpu', '0' pour le 1er GPU")
    parser.add_argument("--name", default="run", help="Nom du run (dossier dans runs/detect/)")
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument(
        "--freeze", type=int, default=None,
        help=(
            "Nombre de couches a geler depuis le debut du backbone (ex: 10). "
            "But : preserver les features COCO utiles plutot que les ecraser en "
            "fine-tunant sur seulement 6 scenes MOT17 -- cf. DECISIONS.md, le "
            "1er et 2e benchmark montrent une degradation continue post-epoch 1."
        ),
    )
    parser.add_argument("--lr0", type=float, default=0.001, help="Learning rate initial")
    parser.add_argument("--optimizer", default="AdamW", help="'auto' pour laisser Ultralytics choisir")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--erasing", type=float, default=0.0)
    parser.add_argument(
        "--domain-augment", action="store_true",
        help="Active l'augmentation a la volee nuit/flou (monkey-patch Albumentations)",
    )
 
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    if not args.data.exists():
        raise FileNotFoundError(
            f"dataset.yaml introuvable a {args.data}. "
            "As-tu lance build_dataset.py, et le chemin est-il correct pour cet environnement ?"
        )
 
    if args.domain_augment:
        from src.detection.onthefly_augment import enable_domain_augment, supports_native_augmentations
 
        native = supports_native_augmentations()
        if native is True:
            print("Robustesse nuit/flou activee via le mecanisme natif Ultralytics (augmentations=)")
        elif native is False:
            print("Mecanisme natif indisponible sur cette version d'Ultralytics -- repli sur le monkey-patch")
        else:
            print("ATTENTION : --domain-augment demande mais ultralytics introuvable")
 
    model = YOLO(args.model)
    train_kwargs = dict(
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
    if args.domain_augment:
        train_kwargs = enable_domain_augment(train_kwargs)
 
    model.train(**train_kwargs)
 
 
if __name__ == "__main__":
    main()
 